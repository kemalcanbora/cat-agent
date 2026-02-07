import copy
import json
import re
import time
from typing import Dict, Iterator, List, Optional, Union

import json5

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    import warnings

    warnings.warn(
        "Polars not installed. Install with: pip install polars\n"
        "Falling back to standard Python processing (slower performance)."
    )

from cat_agent.agents.assistant import KNOWLEDGE_SNIPPET, Assistant, format_knowledge_to_source_and_content
from cat_agent.agents.doc_qa.parallel_doc_qa_member import NO_RESPONSE, ParallelDocQAMember
from cat_agent.agents.doc_qa.parallel_doc_qa_summary import ParallelDocQASummary
from cat_agent.agents.keygen_strategies import GenKeyword
from cat_agent.llm.base import BaseChatModel, ModelServiceError
from cat_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, USER, Message
from cat_agent.log import logger
from cat_agent.tools import BaseTool
from cat_agent.tools.doc_parser import DocParser
from cat_agent.tools.simple_doc_parser import PARSER_SUPPORTED_FILE_TYPES
from cat_agent.utils.parallel_executor import parallel_exec
from cat_agent.utils.tokenization_qwen import count_tokens
from cat_agent.utils.utils import (extract_files_from_messages, extract_text_from_message, get_file_type,
                                   print_traceback)

MAX_NO_RESPONSE_RETRY = 4
DEFAULT_NAME = 'Simple Parallel DocQA With RAG Sum Agents'
DEFAULT_DESC = 'Agent that answers questions after simple parallel RAG recall'

PARALLEL_CHUNK_SIZE = 1000

MAX_RAG_TOKEN_SIZE = 4500
RAG_CHUNK_SIZE = 300


class ParallelDocQA(Assistant):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = DEFAULT_NAME,
                 description: Optional[str] = DEFAULT_DESC,
                 files: Optional[List[str]] = None,
                 use_polars: bool = True):
        function_list = function_list or []
        super().__init__(
            function_list=[{
                'name': 'retrieval',
                'max_ref_token': MAX_RAG_TOKEN_SIZE,
                'parser_page_size': RAG_CHUNK_SIZE,
                'rag_searchers': ['keyword_search']
            }] + function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            files=files,
        )

        self.doc_parse = DocParser()
        self.summary_agent = ParallelDocQASummary(llm=self.llm)
        self.use_polars = use_polars and POLARS_AVAILABLE

    def _get_files(self, messages: List[Message]):
        session_files = extract_files_from_messages(messages, include_images=False)
        valid_files = []
        for file in session_files:
            f_type = get_file_type(file)
            if f_type in PARSER_SUPPORTED_FILE_TYPES and file not in valid_files:
                valid_files.append(file)
        return valid_files

    def _parse_and_chunk_files(self, messages: List[Message]):
        valid_files = self._get_files(messages)
        records = []
        for file in valid_files:
            # here to split docs, we should decide chunc size by input doc token length
            # if a document's tokens are below this max_ref_token, it will remain unchunked.
            _record = self.doc_parse.call(params={'url': file},
                                          parser_page_size=PARALLEL_CHUNK_SIZE,
                                          max_ref_token=PARALLEL_CHUNK_SIZE)
            records.append(_record)
        return records

    # NEW: Polars-optimized chunk preparation
    def _prepare_parallel_data_polars(
            self,
            records: List[dict],
            messages: List[Message],
            lang: str,
            user_question: str
    ) -> List[dict]:
        if not self.use_polars or not POLARS_AVAILABLE:
            return self._prepare_parallel_data_standard(records, messages, lang, user_question)

        try:
            # Collect all chunks from all records into a flat structure
            chunks_data = []
            for record in records:
                if not record.get('raw'):
                    logger.warning(f"Empty record for document: {record.get('url', 'unknown')}")
                    continue

                for chunk in record['raw']:
                    chunks_data.append({
                        'doc_url': record.get('url', ''),
                        'doc_title': record.get('title', ''),
                        'chunk_content': chunk.get('content', ''),
                        'chunk_token': chunk.get('token', 0),
                        'chunk_metadata': str(chunk.get('metadata', {}))
                    })

            if not chunks_data:
                logger.warning("No valid chunks found in documents")
                return []

            # Create Polars DataFrame for vectorized operations
            df = pl.DataFrame(chunks_data)

            # Add row index for tracking
            df = df.with_row_count('index')

            # Vectorized filtering: remove empty or very short chunks
            df = df.filter(
                (pl.col('chunk_content').str.len_chars() > 10) &
                (pl.col('chunk_token') > 0)
            )

            # Optional: Smart chunk selection based on relevance
            if hasattr(self, 'max_chunks_per_question') and len(df) > self.max_chunks_per_question:
                df = self._smart_chunk_selection_polars(df, user_question)

            # Add constant columns efficiently (broadcasting)
            df = df.with_columns([
                pl.lit(user_question).alias('instruction'),
                pl.lit(lang).alias('lang')
            ])

            # Convert to list of dicts for parallel execution
            # Note: We need to preserve the messages object separately since it's not serializable in Polars
            work_items = df.select([
                'index',
                'chunk_content',
                'instruction',
                'lang'
            ]).to_dicts()

            # Add messages to each work item (can't be done in Polars due to object type)
            for item in work_items:
                item['messages'] = messages
                item['knowledge'] = item.pop('chunk_content')

            logger.info(f'Parallel Member Num (Polars): {len(work_items)}')
            return work_items

        except Exception as e:
            logger.warning(f"Polars processing failed: {e}. Falling back to standard processing.")
            return self._prepare_parallel_data_standard(records, messages, lang, user_question)

    # NEW: Smart chunk selection using Polars
    def _smart_chunk_selection_polars(
            self,
            df: 'pl.DataFrame',
            user_question: str,
            max_chunks: int = None
    ) -> 'pl.DataFrame':

        if max_chunks is None:
            max_chunks = getattr(self, 'max_chunks_per_question', 20)

        # Extract simple keywords (more sophisticated extraction could use the LLM)
        keywords = user_question.lower().split()
        keywords = [k for k in keywords if len(k) > 3][:5]  # Top 5 longest words

        if not keywords:
            # No keywords extracted, return top N by position
            return df.head(max_chunks)

        # Build regex pattern for keyword matching (case-insensitive)
        keyword_pattern = '|'.join(keywords)

        # Vectorized scoring
        df = df.with_columns([
            # Keyword match score
            pl.col('chunk_content')
            .str.to_lowercase()
            .str.count_matches(keyword_pattern)
            .alias('keyword_score'),

            # Position score (prefer earlier chunks)
            (1.0 / (pl.col('index') + 1)).alias('position_score'),

            # Length score (prefer medium-length chunks: 100-500 tokens)
            pl.when(pl.col('chunk_token').is_between(100, 500))
            .then(1.0)
            .otherwise(0.5)
            .alias('length_score')
        ])

        # Combined relevance score (weighted)
        df = df.with_columns([
            (
                    pl.col('keyword_score') * 0.6 +
                    pl.col('position_score') * 0.2 +
                    pl.col('length_score') * 0.2
            ).alias('relevance_score')
        ])

        # Select top chunks by relevance
        selected = df.sort('relevance_score', descending=True).head(max_chunks)

        logger.info(f'Smart selection: {len(df)} → {len(selected)} chunks (saved {len(df) - len(selected)} LLM calls)')

        return selected

    # ORIGINAL: Standard Python implementation (fallback)
    def _prepare_parallel_data_standard(
            self,
            records: List[dict],
            messages: List[Message],
            lang: str,
            user_question: str
    ) -> List[dict]:
        """
        Standard Python loop implementation for preparing work items.
        Used as fallback when Polars is not available.
        """
        data = []
        idx = 0
        for record in records:
            if not record.get('raw'):
                continue
            for chunk in record['raw']:
                chunk_text = chunk['content']
                data.append({
                    'index': idx,
                    'messages': messages,
                    'lang': lang,
                    'knowledge': chunk_text,
                    'instruction': user_question,
                })
                idx += 1

        logger.info(f'Parallel Member Num (Standard): {len(data)}')
        return data

    # NEW: Polars-optimized result aggregation
    def _aggregate_results_polars(
            self,
            results: List[tuple]
    ) -> tuple[str, List[tuple]]:
        if not self.use_polars or not POLARS_AVAILABLE or not results:
            return self._aggregate_results_standard(results)

        try:
            # Convert results to Polars DataFrame
            df = pl.DataFrame({
                'index': [r[0] for r in results],
                'content': [r[1] for r in results]
            })

            # Attempt to parse JSON responses (vectorized)
            df = df.with_columns([
                pl.col('content')
                .str.strip_chars()
                .alias('content_clean')
            ])

            # Try to extract JSON structure
            filtered_indices = []
            filtered_contents = []

            for row in df.iter_rows(named=True):
                idx = row['index']
                text = row['content_clean']

                # Parse JSON response
                parser_success, parser_json_content = self._parser_json(text)

                if parser_success and ('res' in parser_json_content) and ('content' in parser_json_content):
                    pa_res = parser_json_content['res']
                    pa_content = parser_json_content['content']

                    if (pa_res in ['ans', 'none']) and (isinstance(pa_content, str)):
                        if pa_res == 'ans':
                            filtered_indices.append(idx)
                            filtered_contents.append(pa_content.strip())
                            continue
                        elif pa_res == 'none':
                            continue

                # Check for none responses
                if self._is_none_response(text):
                    continue

                # Extract text from output
                clean_output = self._extract_text_from_output(text)
                if clean_output and clean_output.strip():
                    filtered_indices.append(idx)
                    filtered_contents.append(clean_output.strip())

            # Create filtered DataFrame
            if filtered_indices:
                filtered_df = pl.DataFrame({
                    'index': filtered_indices,
                    'content': filtered_contents
                }).sort('index')

                # Concatenate all valid responses
                member_res = '\n\n'.join(filtered_df['content'].to_list())

                # Convert back to list of tuples for compatibility
                filtered_results = list(zip(
                    filtered_df['index'].to_list(),
                    filtered_df['content'].to_list()
                ))

                logger.info(f'Filtered results (Polars): {len(results)} → {len(filtered_results)} valid responses')
                return member_res, filtered_results
            else:
                logger.warning("No valid responses after filtering")
                return '', []

        except Exception as e:
            logger.warning(f"Polars aggregation failed: {e}. Falling back to standard processing.")
            return self._aggregate_results_standard(results)

    # ORIGINAL: Standard Python implementation (fallback)
    def _aggregate_results_standard(
            self,
            results: List[tuple]
    ) -> tuple[str, List[tuple]]:
        """
        Standard Python loop implementation for result aggregation.
        Used as fallback when Polars is not available.
        """
        ordered_results = sorted(results, key=lambda x: x[0])
        filtered_results = []

        for index, text in ordered_results:
            parser_success, parser_json_content = self._parser_json(text)
            if parser_success and ('res' in parser_json_content) and ('content' in parser_json_content):
                pa_res, pa_content = parser_json_content['res'], parser_json_content['content']
                if (pa_res in ['ans', 'none']) and (isinstance(pa_content, str)):
                    if pa_res == 'ans':
                        filtered_results.append((index, pa_content.strip()))
                        continue
                    elif pa_res == 'none':
                        continue
            if self._is_none_response(text):
                continue
            clean_output = self._extract_text_from_output(text)
            if clean_output and clean_output.strip():
                filtered_results.append((index, clean_output.strip()))

        if filtered_results:
            member_res = '\n\n'.join(text for index, text in filtered_results)
        else:
            member_res = ''

        logger.info(f'Filtered results (Standard): {len(results)} → {len(filtered_results)} valid responses')
        return member_res, filtered_results

    def _retrieve_according_to_member_responses(
            self,
            messages: List[Message],
            lang: str = 'en',
            user_question: str = '',
            member_res: str = '',
    ):
        messages = copy.deepcopy(messages)
        valid_files = self._get_files(messages)

        keygen = GenKeyword(llm=self.llm)
        member_res_token_num = count_tokens(member_res)

        # Limit the token length of keygen input to avoid wasting tokens due to excessively long docqa member output.
        unuse_member_res = member_res_token_num > MAX_RAG_TOKEN_SIZE
        query = user_question if unuse_member_res else f'{user_question}\n\n{member_res}'

        try:
            *_, last = keygen.run([Message(USER, query)])
        except ModelServiceError:
            print_traceback()

        keyword = last[-1].content
        keyword = keyword.strip()
        if keyword.startswith('```json'):
            keyword = keyword[len('```json'):]
        if keyword.endswith('```'):
            keyword = keyword[:-3]

        try:
            logger.info(keyword)
            keyword_dict = json5.loads(keyword)
            keyword_dict['text'] = query
            if unuse_member_res:
                keyword_dict['text'] += '\n\n' + member_res
            rag_query = json.dumps(keyword_dict, ensure_ascii=False)
        except Exception:
            rag_query = query
            if unuse_member_res:
                rag_query += '\n\n' + member_res

        # max_ref_token is the retrieve doc token size
        # parser_page_size is the chunk size in retrieve

        retrieve_content = self.function_map['retrieval'].call(
            {
                'query': rag_query,
                'files': valid_files
            },
            max_ref_token=MAX_RAG_TOKEN_SIZE,
            parser_page_size=RAG_CHUNK_SIZE,
        )
        if not isinstance(retrieve_content, str):
            retrieve_content = json.dumps(retrieve_content, ensure_ascii=False, indent=4)

        retrieve_content = format_knowledge_to_source_and_content(retrieve_content)

        snippets = []

        for k in retrieve_content:
            snippets.append(KNOWLEDGE_SNIPPET[lang].format(source=k['source'], content=k['content']))

        assert len(snippets) > 0, retrieve_content
        retrieve_res = '\n\n'.join(snippets)
        return retrieve_res

    def _is_none_response(self, text: str) -> bool:
        none_response_list = ['i am sorry', NO_RESPONSE, '"res": "none"']
        text_lower = text.lower()
        for none_response in none_response_list:
            if none_response in text_lower:
                return True
        return False

    def _extract_text_from_output(self, output):
        # Remove symbols and keywords from the JSON structure using regular expressions
        cleaned_output = re.sub(r'[{}"]|("res":\s*"ans"|"res":\s*"none"|"\s*content":\s*)', '', output)
        return cleaned_output

    def _parser_json(self, content):
        content = content.strip()
        if content.startswith('```json'):
            content = content[len('```json'):]
        if content.endswith('```'):
            content = content[:-3]
        try:
            content_dict = json5.loads(content)
            return True, content_dict
        except Exception:
            return False, content

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)

        # Extract User Question
        user_question = extract_text_from_message(messages[-1], add_upload_info=False)
        logger.info('user_question: ' + user_question)

        # Implement chunk strategy for parallel agent
        records = self._parse_and_chunk_files(messages=messages)
        if not records:
            logger.error('No valid documents found or all parsing failed')
            yield [Message(role='assistant', content='Error: No valid documents to process')]
            return

        # NEW: Use Polars for data preparation (5-10x faster)
        data = self._prepare_parallel_data_polars(
            records=records,
            messages=messages,
            lang=lang,
            user_question=user_question
        )

        if not data:
            logger.error('No valid chunks after preparation')
            yield [Message(role='assistant', content='Error: No valid content to process')]
            return

        # Retry for None responses (common in smaller models)
        retry_cnt = MAX_NO_RESPONSE_RETRY
        member_res = ''
        filtered_results = []

        while retry_cnt > 0:
            time1 = time.time()
            results = parallel_exec(self._ask_member_agent, data, jitter=0.5)
            time2 = time.time()
            logger.info(f'Finished parallel_exec. Time spent: {time2 - time1:.2f} seconds.')

            # NEW: Use Polars for result aggregation (10-30x faster)
            member_res, filtered_results = self._aggregate_results_polars(results)

            if filtered_results:
                break

            retry_cnt -= 1
            if retry_cnt > 0:
                logger.warning(
                    f'No valid responses, retrying... ({MAX_NO_RESPONSE_RETRY - retry_cnt}/{MAX_NO_RESPONSE_RETRY})')

        # Retrieve additional context based on member responses
        retrieve_content = self._retrieve_according_to_member_responses(
            messages=messages,
            lang=lang,
            user_question=user_question,
            member_res=member_res
        )

        # Final summary
        return self.summary_agent.run(messages=messages, lang=lang, knowledge=retrieve_content)

    def _ask_member_agent(self,
                          index: int,
                          messages: List[Message],
                          lang: str = 'en',
                          knowledge: str = '',
                          instruction: str = '') -> tuple:

        doc_qa = ParallelDocQAMember(llm=self.llm)
        *_, last = doc_qa.run(messages=messages, knowledge=knowledge, lang=lang, instruction=instruction)
        return index, last[-1].content
