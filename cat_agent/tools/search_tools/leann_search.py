from typing import List, Tuple, Optional, Dict

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
    """
    Vector search backed by LEANN.

    This searcher builds a LEANN index over the provided `docs` and uses it
    to rank chunks by semantic similarity.

    It is implemented as a separate searcher so that:
    - Existing FAISS-based search (in `vector_search.py`) remains untouched.
    - Hybrid search can include LEANN alongside other searchers.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # Whether to rebuild the LEANN index on each call.
        #   True  -> always rebuild from current docs.
        #   False -> reuse any existing index at `index_path` (user must ensure it is up to date).
        #   None  -> default to rebuild (current behavior).
        self.rebuild_rag: Optional[bool] = None
        if cfg is not None:
            self.rebuild_rag = cfg.get('rebuild_rag', None)

    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        """
        Rank chunks using LEANN and return (url, chunk_id, score) tuples
        sorted by ascending score (more relevant first).
        """
        try:
            from leann import LeannBuilder, LeannSearcher  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'LEANN is not installed. Please install it first, e.g.: `pip install leann`.\n'
                'See https://github.com/yichuan-w/LEANN for details.'
            ) from e

        # Reduce noisy logs from dependencies so LEANN logs stay readable.
        # - sentence-transformers: model loading / cache warnings
        # - leann / hnsw backend: detailed HNSW conversion debug logs
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("leann").setLevel(logging.WARNING)
        logging.getLogger("leann_backend_hnsw").setLevel(logging.WARNING)
        logging.getLogger("leann_backend_hnsw.hnsw_backend").setLevel(logging.WARNING)

        # Flatten all chunks and remember mapping from LEANN index entries
        # back to (doc.url, chunk_id).
        chunk_metadata = []  # list of (url, chunk_id)

        # Create (or reuse) a workspace directory for LEANN indexes.
        # We persist the index under the framework's workspace so repeated
        # calls can optionally reuse it when `rebuild_rag` is False.
        leann_root = os.path.join(DEFAULT_WORKSPACE, 'storage', 'leann_indexes')
        os.makedirs(leann_root, exist_ok=True)
        index_path = os.path.join(leann_root, 'rag_index.leann')

        logger.info(f"[LeannSearch] Building LEANN index at {index_path}")

        try:
            # Decide whether we need to (re)build the index.
            must_rebuild = (self.rebuild_rag is True) or (self.rebuild_rag is None) or (not os.path.exists(index_path))

            if must_rebuild:
                builder = LeannBuilder(backend_name='hnsw')

                total_chunks = 0
                for doc in docs:
                    for chunk_id, page in enumerate(doc.raw):
                        text = page.content
                        # Store mapping so we can reconstruct (url, chunk_id) later.
                        metadata = {'url': doc.url, 'chunk_id': chunk_id}
                        builder.add_text(text, metadata=metadata)
                        chunk_metadata.append((doc.url, chunk_id))
                        total_chunks += 1

                logger.info(
                    f"[LeannSearch] Added {total_chunks} chunks from {len(docs)} docs to LEANN builder."
                )

                # Build (or rebuild) the index on disk.
                if os.path.exists(index_path):
                    logger.info(f"[LeannSearch] Removing existing index at {index_path} before rebuild.")
                    try:
                        # LEANN stores index as a directory or single file depending on backend.
                        if os.path.isdir(index_path):
                            shutil.rmtree(index_path, ignore_errors=True)
                        else:
                            os.remove(index_path)
                    except OSError:
                        logger.warning(
                            f"[LeannSearch] Failed to remove previous index at {index_path}, proceeding anyway."
                        )

                builder.build_index(index_path)
                logger.info(f"[LeannSearch] Built LEANN index at {index_path}")
            else:
                logger.info(
                    f"[LeannSearch] Reusing existing LEANN index/storage at {index_path} "
                    f"(rebuild_rag={self.rebuild_rag})"
                )

            # Search with LEANN (whether index was rebuilt or reused).
            searcher = LeannSearcher(index_path)
            # We ask LEANN for as many results as we have chunks; BaseSearch
            # will take care of trimming based on max_ref_token.
            top_k = len(chunk_metadata) if chunk_metadata else 0
            if top_k == 0:
                logger.info("[LeannSearch] No chunks available for search; returning empty result.")
                return []

            logger.info(f"[LeannSearch] Running LEANN search for query={query!r}, top_k={top_k}")
            results = searcher.search(query, top_k=top_k)

            # LEANN search results are expected to expose metadata that we
            # attached during `add_text`. We convert them to the interface
            # expected by BaseSearch: (url, chunk_id, score).
            chunk_and_score: List[Tuple[str, int, float]] = []
            for r in results:
                # The exact result structure is defined by LEANN. We assume
                # it provides `.metadata` and `.score` attributes; if the
                # library returns dicts instead, access them as keys.
                meta = getattr(r, 'metadata', None) or getattr(r, 'meta', None) or {}
                if isinstance(meta, dict):
                    url = meta.get('url')
                    chunk_id = meta.get('chunk_id')
                else:
                    # Fallback – if metadata is missing or unexpected, skip.
                    continue

                score = getattr(r, 'score', None)
                if score is None and isinstance(r, dict):
                    score = r.get('score', 0.0)
                if url is None or chunk_id is None:
                    continue
                # LEANN usually returns higher score = more similar; BaseSearch
                # expects earlier entries to be more relevant, so we keep the
                # natural ordering from LEANN and just pass the score through.
                chunk_and_score.append((url, int(chunk_id), float(score)))

            logger.info(f"[LeannSearch] Retrieved {len(chunk_and_score)} scored chunks from LEANN.")
            return chunk_and_score
        finally:
            # We intentionally do NOT delete the index_path so that it can live
            # under the workspace. This makes it inspectable and paves the way
            # for future reuse across calls.
            logger.info(f"[LeannSearch] Finished LEANN search using index at {index_path}")

