import time
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from ingestion.base import Document
from core.retrieval_result import RetrievalResult
from retrieval.pg_vector_store import PGVectorStore


class PGHybridRetriever:
    """
    Production-oriented hybrid retriever for RAG V3.

    Combines:

        Dense retrieval:
            PostgreSQL + pgvector

        Sparse retrieval:
            BM25 over chunks stored in PostgreSQL

    This class is intentionally separate from the existing
    FAISS-based HybridRetriever used by RAG V1/V2.

    Retrieval flow:

        Query
          |
          +--------------------+
          |                    |
          v                    v
      pgvector               BM25
          |                    |
          +---------+----------+
                    |
                    v
             Merge candidates
                    |
                    v
             RetrievalResult
    """

    def __init__(
        self,
        vector_store: PGVectorStore,
        auto_refresh: bool = False,
    ):
        """
        Args:
            vector_store:
                PGVectorStore instance used for dense retrieval.

            auto_refresh:
                If True, rebuilds the BM25 index before every search.

                Useful during development when documents may be added
                externally.

                For production, prefer False and explicitly call
                refresh() after document ingestion.
        """

        self.vector_store = vector_store
        self.auto_refresh = auto_refresh

        self._documents: List[Document] = []
        self._document_ids: List[int] = []

        self._bm25: Optional[BM25Okapi] = None

        self._indexed_chunk_count = 0

        self.refresh()

    # ==========================================================
    # TOKENIZATION
    # ==========================================================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Basic BM25 tokenizer.

        Keeps tokenization consistent between:
            - stored document chunks
            - incoming queries
        """

        if not text:
            return []

        return text.lower().split()

    # ==========================================================
    # REFRESH BM25 INDEX
    # ==========================================================

    def refresh(self) -> None:
        """
        Rebuild the BM25 index from all document chunks currently
        stored in PostgreSQL.

        Call this after inserting or deleting documents.

        Example:

            pg_store.add(...)
            retriever.refresh()
        """

        start_time = time.perf_counter()

        rows = self.vector_store.get_all_chunks()

        documents = []
        document_ids = []

        for row in rows:

            document = Document(
                text=row.content,
                metadata={
                    "source": row.filename,
                    "filename": row.filename,
                    "document_id": row.document_id,
                    "chunk_id": row.id,
                    "chunk_index": row.chunk_index,
                    **(
                        row.chunk_metadata
                        if isinstance(row.chunk_metadata, dict)
                        else {}
                    ),
                },
            )

            documents.append(document)
            document_ids.append(row.id)

        self._documents = documents
        self._document_ids = document_ids

        if documents:

            tokenized_documents = [
                self._tokenize(document.text)
                for document in documents
            ]

            self._bm25 = BM25Okapi(
                tokenized_documents
            )

        else:

            self._bm25 = None

        self._indexed_chunk_count = len(documents)

        refresh_latency = (
            time.perf_counter() - start_time
        )

        print("\n========== BM25 REFRESH ==========")
        print(
            "PostgreSQL Chunks :",
            self._indexed_chunk_count
        )
        print(
            "Refresh Latency   :",
            f"{refresh_latency:.4f} sec"
        )
        print("=" * 40)

    # ==========================================================
    # SEARCH
    # ==========================================================

    def hybrid_search(
        self,
        query: str,
        vector_store: Optional[PGVectorStore] = None,
        embedder=None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Perform hybrid dense + sparse retrieval.

        Args:
            query:
                User/re-written query.

            vector_store:
                Optional PGVectorStore.
                If omitted, the store passed during construction is used.

            embedder:
                Embedder used to generate the query embedding.

            top_k:
                Number of candidates retrieved from each retriever.

        Returns:
            RetrievalResult containing merged candidates.
        """

        start_time = time.perf_counter()

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if embedder is None:
            raise ValueError(
                "embedder is required for vector retrieval."
            )

        vector_store = (
            vector_store
            if vector_store is not None
            else self.vector_store
        )

        # ------------------------------------------------------
        # Optional BM25 refresh
        # ------------------------------------------------------

        if self.auto_refresh:
            self.refresh()

        # ------------------------------------------------------
        # Dense Retrieval
        # ------------------------------------------------------

        query_vector = embedder.embed_query(
            query
        )

        vector_scores, vector_ids = (
            vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
            )
        )

        # ------------------------------------------------------
        # Sparse Retrieval
        # ------------------------------------------------------

        result = RetrievalResult()

        if self._bm25 is None or not self._documents:

            bm25_scores = np.array([])

            bm25_indices = []

        else:

            tokenized_query = self._tokenize(
                query
            )

            bm25_scores = self._bm25.get_scores(
                tokenized_query
            )

            bm25_indices = np.argsort(
                bm25_scores
            )[::-1][:top_k]

        # ------------------------------------------------------
        # Candidate map
        #
        # Key = PostgreSQL chunk ID
        #
        # This is important because pgvector returns database
        # IDs rather than FAISS positional indices.
        # ------------------------------------------------------

        candidates: Dict[int, Dict] = {}

        # ------------------------------------------------------
        # Add pgvector results
        # ------------------------------------------------------

        for score, chunk_id in zip(
            vector_scores,
            vector_ids,
        ):

            chunk_id = int(chunk_id)

            if chunk_id <= 0:
                continue

            candidates[chunk_id] = {
                "vector_score": float(score),
                "bm25_score": 0.0,
            }

        # ------------------------------------------------------
        # Add BM25 results
        # ------------------------------------------------------

        for index in bm25_indices:

            index = int(index)

            if index < 0 or index >= len(
                self._document_ids
            ):
                continue

            chunk_id = self._document_ids[index]

            if chunk_id in candidates:

                candidates[chunk_id][
                    "bm25_score"
                ] = float(
                    bm25_scores[index]
                )

            else:

                candidates[chunk_id] = {
                    "vector_score": 0.0,
                    "bm25_score": float(
                        bm25_scores[index]
                    ),
                }

        # ------------------------------------------------------
        # Retrieve actual PostgreSQL chunks
        # ------------------------------------------------------

        candidate_ids = list(
            candidates.keys()
        )

        if not candidate_ids:

            result.retrieval_latency = (
                time.perf_counter() - start_time
            )

            return result

        rows = vector_store.get_chunks(
            candidate_ids
        )

        row_map = {
            row.id: row
            for row in rows
        }

        # ------------------------------------------------------
        # Build RetrievalResult
        # ------------------------------------------------------

        added_ids = set()

        for chunk_id in candidate_ids:

            if chunk_id in added_ids:
                continue

            row = row_map.get(
                chunk_id
            )

            if row is None:
                continue

            scores = candidates[
                chunk_id
            ]

            chunk = Document(
                text=row.content,
                metadata={
                    "source": row.filename,
                    "filename": row.filename,
                    "document_id": row.document_id,
                    "chunk_id": row.id,
                    "chunk_index": row.chunk_index,
                    **(
                        row.chunk_metadata
                        if isinstance(
                            row.chunk_metadata,
                            dict
                        )
                        else {}
                    ),
                },
            )

            result.add_chunk(
                chunk=chunk,
                vector_score=scores[
                    "vector_score"
                ],
                bm25_score=scores[
                    "bm25_score"
                ],
            )

            added_ids.add(
                chunk_id
            )

        # ------------------------------------------------------
        # Latency
        # ------------------------------------------------------

        result.retrieval_latency = (
            time.perf_counter() - start_time
        )

        # ------------------------------------------------------
        # Monitoring
        # ------------------------------------------------------

        print("\n===== PG HYBRID RETRIEVAL =====")
        print(
            "Query           :",
            query
        )
        print(
            "BM25 Chunks     :",
            self._indexed_chunk_count
        )
        print(
            "Vector Results  :",
            len(vector_ids)
        )
        print(
            "BM25 Results    :",
            len(bm25_indices)
        )
        print(
            "Merged Results  :",
            result.retrieved_count
        )
        print(
            "Retrieval Time  :",
            f"{result.retrieval_latency:.4f} sec"
        )

        for i, item in enumerate(
            result.retrieved_chunks,
            start=1,
        ):

            print(
                f"\nChunk {i}"
            )

            print(
                "Vector :",
                item.vector_score
            )

            print(
                "BM25   :",
                item.bm25_score
            )

            print(
                "Source :",
                item.chunk.metadata.get(
                    "filename",
                    "unknown"
                )
            )

        print(
            "=" * 40
        )

        return result

    # ==========================================================
    # REFRESH CHECK
    # ==========================================================

    def refresh_if_needed(self) -> None:
        """
        Refresh BM25 only when the number of PostgreSQL chunks
        differs from the currently indexed BM25 corpus.

        Useful when documents can be added after the retriever
        was created.
        """

        current_count = (
            self.vector_store.count()
        )

        if current_count != self._indexed_chunk_count:

            print(
                "\nBM25 index is stale."
            )

            print(
                "Indexed chunks :",
                self._indexed_chunk_count
            )

            print(
                "Database chunks:",
                current_count
            )

            self.refresh()

    # ==========================================================
    # PROPERTIES
    # ==========================================================

    @property
    def indexed_chunk_count(self) -> int:
        """
        Number of chunks currently represented
        in the BM25 index.
        """

        return self._indexed_chunk_count