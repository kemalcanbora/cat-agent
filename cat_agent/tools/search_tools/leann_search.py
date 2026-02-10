from typing import List, Tuple, Optional, Dict

import json
import logging
import os
import shutil

from cat_agent.log import logger
from cat_agent.settings import DEFAULT_WORKSPACE
from cat_agent.tools.base import register_tool
from cat_agent.tools.doc_parser import Record
from cat_agent.tools.search_tools.base_search import BaseSearch


@register_tool('leann_search')
class LeannSearch(BaseSearch):
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.rebuild_rag: Optional[bool] = None
        if cfg is not None:
            self.rebuild_rag = cfg.get('rebuild_rag', None)
            self.leann_top_k = cfg.get('leann_top_k', 100)

    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        try:
            from leann import LeannBuilder, LeannSearcher  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'LEANN is not installed. Please install it first, e.g.: `pip install leann`.\n'
                'See https://github.com/yichuan-w/LEANN for details.'
            ) from e

        self._configure_logging()

        index_path, meta_path = self._index_paths()
        os.makedirs(os.path.dirname(index_path), exist_ok=True)


        if self.rebuild_rag:
            logger.info(f"[LeannSearch] Building LEANN index at {index_path}")
            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model = "BAAI/bge-base-en-v1.5",
                embedding_mode = "sentence-transformers",
                is_compact=True,
                is_recompute=False,
            )

            chunk_metadata = self._add_docs_to_builder(builder, docs)
            self._remove_existing_index(index_path)

            builder.build_index(index_path)
            logger.info(f"[LeannSearch] Built LEANN index at {index_path}")

            self._save_metadata(meta_path, chunk_metadata)
        else:
            logger.info(
                f"[LeannSearch] Reusing existing LEANN index/storage at {index_path} (rebuild_rag={self.rebuild_rag})"
            )
            chunk_metadata = self._load_metadata(meta_path)

            if not chunk_metadata:
                chunk_metadata = self._collect_chunks_from_docs(docs)

        searcher = LeannSearcher(index_path)
        print("[LeannSearch] Number of chunks available for search:", len(chunk_metadata))
        if self.leann_top_k == 0:
            logger.info("[LeannSearch] No chunks available for search; returning empty result.")
            return []

        logger.info(f"[LeannSearch] Running LEANN search for query={query!r}, top_k={self.leann_top_k}")
        results = searcher.search(query,
                                  top_k=self.leann_top_k,
                                  complexity=32,
                                  beam_width=1,
                                  prune_ratio=0.0,
                                  recompute_embeddings=False,
                                  )

        chunk_and_score: List[Tuple[str, int, float]] = []
        for result in results:
            meta = getattr(result, 'metadata', None) or getattr(result, 'meta', None) or {}
            if not isinstance(meta, dict):
                continue

            url = meta.get('url')
            chunk_id = meta.get('chunk_id')

            score = getattr(result, 'score', None)
            if score is None and isinstance(result, dict):
                score = result.get('score', 0.0)

            if url is None or chunk_id is None:
                continue

            chunk_and_score.append((url, int(chunk_id), float(score or 0.0)))

        chunk_and_score.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"[LeannSearch] Retrieved {len(chunk_and_score)} scored chunks from LEANN.")
        logger.info(f"[LeannSearch] Finished LEANN search using index at {index_path}")
        return chunk_and_score

    @staticmethod
    def _configure_logging() -> None:
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("leann").setLevel(logging.WARNING)
        logging.getLogger("leann_backend_hnsw").setLevel(logging.WARNING)
        logging.getLogger("leann_backend_hnsw.hnsw_backend").setLevel(logging.WARNING)

    @staticmethod
    def _index_paths() -> Tuple[str, str]:
        root = os.path.join(DEFAULT_WORKSPACE, 'storage', 'leann_indexes')
        index_path = os.path.join(root, 'rag_index.leann')
        meta_path = os.path.join(root, 'rag_index.meta.json')
        return index_path, meta_path

    @staticmethod
    def _remove_existing_index(index_path: str) -> None:
        if not os.path.exists(index_path):
            return

        logger.info(f"[LeannSearch] Removing existing index at {index_path} before rebuild.")
        try:
            if os.path.isdir(index_path):
                shutil.rmtree(index_path, ignore_errors=True)
            else:
                os.remove(index_path)
        except OSError:
            logger.warning(
                f"[LeannSearch] Failed to remove previous index at {index_path}, proceeding anyway."
            )

    @staticmethod
    def _add_docs_to_builder(builder, docs: List[Record]) -> List[Tuple[str, int]]:
        chunk_metadata: List[Tuple[str, int]] = []
        for doc in docs:
            for chunk_id, page in enumerate(doc.raw):
                metadata = {'url': doc.url, 'chunk_id': chunk_id}
                builder.add_text(page.content, metadata=metadata)
                chunk_metadata.append((doc.url, chunk_id))
        return chunk_metadata

    @staticmethod
    def _collect_chunks_from_docs(docs: List[Record]) -> List[Tuple[str, int]]:
        chunk_metadata: List[Tuple[str, int]] = []
        for doc in docs:
            for chunk_id, _page in enumerate(doc.raw):
                chunk_metadata.append((doc.url, chunk_id))
        return chunk_metadata

    @staticmethod
    def _save_metadata(meta_path: str, chunk_metadata: List[Tuple[str, int]]) -> None:
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"chunks": chunk_metadata}, f)
        except OSError:
            logger.warning(f"[LeannSearch] Failed to persist metadata to {meta_path}; reuse mode may be impaired.")

    @staticmethod
    def _load_metadata(meta_path: str) -> List[Tuple[str, int]]:
        if not os.path.exists(meta_path):
            return []

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, TypeError):
            logger.warning(f"[LeannSearch] Failed to load metadata from {meta_path}; falling back to docs.")
            return []

        loaded = data.get("chunks", [])
        if not isinstance(loaded, list):
            return []

        chunk_metadata: List[Tuple[str, int]] = []
        for entry in loaded:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                chunk_metadata.append((entry[0], int(entry[1])))
        return chunk_metadata
