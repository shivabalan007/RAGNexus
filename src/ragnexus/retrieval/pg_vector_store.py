from typing import List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from database.connection import SessionLocal
from database.vector_models import DocumentChunk


class PGVectorStore:
    """
    PostgreSQL + pgvector vector store for RAG V3.

    Responsibilities:
    - Insert document chunks and their embeddings.
    - Perform vector similarity search using pgvector.
    - Delete all chunks belonging to a document.
    - Check whether a document already exists.
    - Report stored vector count.

    This class is independent of the existing FAISS VectorStore
    used by RAG V1 and V2.
    """

    EMBEDDING_DIM = 384

    def __init__(self):
        pass


    def add(self,document_id: str, filename: str, content_hash: str, chunks: Sequence, embeddings: np.ndarray,) -> int:
        """
        Store document chunks and their embeddings in PostgreSQL.

        Args:
            document_id:
                Stable ID representing one uploaded document.

            filename:
                Original uploaded filename.

            chunks:
                Sequence of Document objects.

            embeddings:
                NumPy array with shape:
                (number_of_chunks, 384)

        Returns:
            Number of inserted chunks.
        """

        if not document_id:
            raise ValueError("document_id cannot be empty.")

        if not filename:
            raise ValueError("filename cannot be empty.")

        
        if not content_hash:
            raise ValueError("content_hash cannot be empty.")

        if chunks is None:
            raise ValueError("chunks cannot be None.")

        embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2-dimensional. "
                f"Received shape: {embeddings.shape}"
            )

        if embeddings.shape[1] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.EMBEDDING_DIM}, "
                f"received {embeddings.shape[1]}."
            )

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match "
                f"embedding count ({len(embeddings)})."
            )

        if len(chunks) == 0:
            return 0

        db: Session = SessionLocal()

        try:

            # Prevent accidental duplicate document insertion.
            existing = (
                db.query(DocumentChunk.id)
                .filter(
                    DocumentChunk.document_id == document_id
                )
                .first()
            )

            if existing:
                raise ValueError(
                    f"Document '{document_id}' already exists."
                )

            rows = []

            for index, (chunk, embedding) in enumerate(
                zip(chunks, embeddings)
            ):

                content = getattr(chunk, "text", None)

                if not content:
                    raise ValueError(
                        f"Chunk {index} has no text content."
                    )

                metadata = getattr(chunk, "metadata", None)

                vector = embedding.astype(
                    np.float32
                ).tolist()

                rows.append(
                    DocumentChunk(
                        document_id=document_id,
                        filename=filename,
                        content_hash=content_hash,
                        content=content,
                        chunk_index=index,
                        embedding=vector,
                        chunk_metadata=metadata,
                    )
                )

            db.add_all(rows)
            db.commit()

            return len(rows)

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()


    def search(self, query_vector: np.ndarray, top_k: int = 10,) -> Tuple[List[float], List[int]]:
        """
        Search PostgreSQL using pgvector cosine distance.

        Returns:
            scores:
                Similarity scores where higher is better.

            indices:
                Database row IDs corresponding to the results.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        # Accept either:
        # (384,)
        # or
        # (1, 384)
        if query_vector.ndim == 2:

            if query_vector.shape[0] != 1:
                raise ValueError(
                    "search() expects a single query vector."
                )

            query_vector = query_vector[0]

        if query_vector.ndim != 1:
            raise ValueError(
                f"Query vector must be 1-dimensional. "
                f"Received shape: {query_vector.shape}"
            )

        if query_vector.shape[0] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Expected query dimension "
                f"{self.EMBEDDING_DIM}, "
                f"received {query_vector.shape[0]}."
            )

        vector = query_vector.tolist()

        db: Session = SessionLocal()

        try:

            # pgvector cosine distance:
            #
            # smaller distance = more similar
            #
            # Convert to similarity:
            #
            # similarity = 1 - cosine_distance

            distance = DocumentChunk.embedding.cosine_distance(
                vector
            )

            statement = (
                select(
                    DocumentChunk.id,
                    distance.label("distance"),
                )
                .order_by(distance)
                .limit(top_k)
            )

            results = db.execute(statement).all()

            scores = []
            indices = []

            for row in results:

                similarity = 1.0 - float(row.distance)

                scores.append(similarity)
                indices.append(row.id)

            return scores, indices

        finally:
            db.close()


    def get_chunks(self,ids: Sequence[int],) -> List[DocumentChunk]:
        """
        Retrieve stored document chunks by database IDs.
        """

        if not ids:
            return []

        db: Session = SessionLocal()

        try:

            statement = (
                select(DocumentChunk)
                .where(
                    DocumentChunk.id.in_(list(ids))
                )
            )

            rows = db.execute(statement).scalars().all()

            # Preserve the order returned by search().
            row_map = {
                row.id: row
                for row in rows
            }

            return [
                row_map[row_id]
                for row_id in ids
                if row_id in row_map
            ]

        finally:
            db.close()

    def get_all_chunks(self) -> List[DocumentChunk]:
        """
        Retrieve all document chunks stored in PostgreSQL. Used by the V3 BM25 retriever to build its lexical corpus.
        """
    
        db: Session = SessionLocal()
    
        try:
            statement = (select(DocumentChunk).order_by(DocumentChunk.id.asc()))
    
            return db.execute(statement).scalars().all()
    
        finally:
            db.close()

    
    def delete_document(
        self,
        document_id: str,
    ) -> int:
        """
        Delete every chunk belonging to a document.

        Returns:
            Number of deleted rows.
        """

        if not document_id:
            raise ValueError("document_id cannot be empty.")

        db: Session = SessionLocal()

        try:

            result = db.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id == document_id
                )
            )

            db.commit()

            return result.rowcount

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def document_hash_exists(self, content_hash: str) -> bool:
        """
        Check whether a document with the given content has already exists in PostgreSQL.
        """

        if not content_hash:
            return False

        db: Session = SessionLocal()

        try:
            result = (db.query(DocumentChunk.id).filter(DocumentChunk.content_hash == content_hash).first())

            return result is not None

        finally:
            db.close()

    def get_document_by_hash(self, content_hash: str):
        if not content_hash:
            return None

        db: Session = SessionLocal()

        try:
            result = (
                db.query(DocumentChunk.document_id).filter(DocumentChunk.content_hash == content_hash).first()
            )

            return result[0] if result else None

        finally:
            db.close()

    def document_exists(
        self,
        document_id: str,
    ) -> bool:
        """
        Check whether a document already exists.
        """

        if not document_id:
            return False

        db: Session = SessionLocal()

        try:

            result = (
                db.query(DocumentChunk.id)
                .filter(
                    DocumentChunk.document_id == document_id
                )
                .first()
            )

            return result is not None

        finally:
            db.close()

    def count(self) -> int:
        """
        Return total number of stored document chunks.
        """

        db: Session = SessionLocal()

        try:
            return db.query(DocumentChunk).count()

        finally:
            db.close()


    def document_chunk_count(
        self,
        document_id: str,
    ) -> int:
        """
        Return the number of chunks belonging to one document.
        """

        if not document_id:
            return 0

        db: Session = SessionLocal()

        try:

            return (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.document_id == document_id
                )
                .count()
            )

        finally:
            db.close()