from typing import List, Tuple, Optional, Dict

import logging
import os
import shutil

from cat_agent.log import logger
from cat_agent.settings import DEFAULT_WORKSPACE
from cat_agent.tools.base import register_tool
from cat_agent.tools.doc_parser import Record
from cat_agent.tools.search_tools.base_search import BaseSearch
import json

@register_tool('leann_search')
class LeannSearch(BaseSearch):
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.rebuild_rag: Optional[bool] = None
        if cfg is not None:
            self.rebuild_rag = cfg.get('rebuild_rag', None)

    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        """
        Rank chunks using LEANN and return (url, chunk_id, score) tuples
        sorted by descending score (more relevant first).
        """
        try:
            from leann import LeannBuilder, LeannSearcher  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'LEANN is not installed. Please install it first, e.g.: `pip install leann`.\n'
                'See https://github.com/yichuan-w/LEANN for details.'
            ) from e

        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("leann").setLevel(logging.WARNING)
        logging.getLogger("leann_backend_hnsw").setLevel(logging.WARNING)
        logging.getLogger("leann_backend_hnsw.hnsw_backend").setLevel(logging.WARNING)

        chunk_metadata: List[Tuple[str, int]] = []

        leann_root = os.path.join(DEFAULT_WORKSPACE, 'storage', 'leann_indexes')
        os.makedirs(leann_root, exist_ok=True)
        index_path = os.path.join(leann_root, 'rag_index.leann')
        meta_path = os.path.join(leann_root, 'rag_index.meta.json')

        must_rebuild = (self.rebuild_rag is True) or (self.rebuild_rag is None) or (not os.path.exists(index_path))

        if must_rebuild:
            logger.info(f"[LeannSearch] Building LEANN index at {index_path}")
            builder = LeannBuilder(backend_name='hnsw')

            total_chunks = 0
            for doc in docs:
                for chunk_id, page in enumerate(doc.raw):
                    text = page.content
                    metadata = {'url': doc.url, 'chunk_id': chunk_id}
                    builder.add_text(text, metadata=metadata)
                    chunk_metadata.append((doc.url, chunk_id))
                    total_chunks += 1

            if os.path.exists(index_path):
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

            builder.build_index(index_path)
            logger.info(f"[LeannSearch] Built LEANN index at {index_path}")

            # Persist chunk metadata for reuse mode
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"chunks": chunk_metadata}, f)
            except OSError:
                logger.warning(f"[LeannSearch] Failed to persist metadata to {meta_path}; reuse mode may be impaired.")
        else:
            logger.info(
                f"[LeannSearch] Reusing existing LEANN index/storage at {index_path} (rebuild_rag={self.rebuild_rag})"
            )
            # Load persisted metadata to recover top_k when reusing
            try:
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        loaded = data.get("chunks", [])
                        if isinstance(loaded, list):
                            # ensure it is List[Tuple[str, int]]
                            for entry in loaded:
                                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                                    chunk_metadata.append((entry[0], int(entry[1])))
            except (OSError, ValueError, TypeError):
                logger.warning(f"[LeannSearch] Failed to load metadata from {meta_path}; falling back to docs.")

            if not chunk_metadata:
                for doc in docs:
                    for chunk_id, _page in enumerate(doc.raw):
                        chunk_metadata.append((doc.url, chunk_id))

        searcher = LeannSearcher(index_path)

        top_k = len(chunk_metadata)
        if top_k == 0:
            logger.info("[LeannSearch] No chunks available for search; returning empty result.")
            return []

        logger.info(f"[LeannSearch] Running LEANN search for query={query!r}, top_k={top_k}")
        results = searcher.search(query, top_k=top_k)

        chunk_and_score: List[Tuple[str, int, float]] = []
        for r in results:
            meta = getattr(r, 'metadata', None) or getattr(r, 'meta', None) or {}
            if isinstance(meta, dict):
                url = meta.get('url')
                chunk_id = meta.get('chunk_id')
            else:
                continue

            score = getattr(r, 'score', None)
            if score is None and isinstance(r, dict):
                score = r.get('score', 0.0)

            if url is None or chunk_id is None:
                continue

            chunk_and_score.append((url, int(chunk_id), float(score)))

        # LEANN typically returns descending relevance; ensure ordering explicitly
        chunk_and_score.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"[LeannSearch] Retrieved {len(chunk_and_score)} scored chunks from LEANN.")
        logger.info(f"[LeannSearch] Finished LEANN search using index at {index_path}")
        return chunk_and_score

