from __future__ import annotations
from pathlib import Path
from usearch.index import Index

import sqlite3
import numpy as np


class MultiVectorIndex:
    """
    Multivector index storing token vectors in a usearch HNSW index
    and metadata (string key, document id, token id) in a SQLite database.
    """

    def __init__(
        self, path: str, ndim: int, metric: str = "ip", dtype: str = "f32"
    ):
        """
        Create a new multivector index.

        Args:
            path: Path for usearch index file. SQLite DB will be created at path with .db suffix.
            ndim: Dimensionality of vectors.
            metric: Similarity metric (e.g., 'ip', 'l2').
            dtype: Data type for vectors (e.g., 'f32', 'f16').
        """
        self.index_path = Path(path)
        self.db_path = self.index_path.with_suffix(".db")
        if self.index_path.exists() or self.db_path.exists():
            raise FileExistsError(
                f"Index or database already exists at '{path}'"
            )
        # Ensure parent directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize usearch index and save an empty index file
        self.index = Index(ndim=ndim, metric=metric, dtype=dtype)
        self.index.save(str(self.index_path))

        # Initialize SQLite database for metadata
        self.db = sqlite3.connect(str(self.db_path))
        self.db.execute(
            """
            CREATE TABLE tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                doc_id INTEGER NOT NULL,
                token_id INTEGER NOT NULL
            )
            """
        )
        self.db.commit()
        self.ndim = ndim

    @classmethod
    def load(cls, path: str) -> MultiVectorIndex:
        """
        Load existing multivector index from disk.

        Args:
            path: Path to usearch index file. Expects SQLite DB at path with .db suffix.
        """
        index_path = Path(path)
        db_path = index_path.with_suffix(".db")
        if not index_path.exists():
            raise FileNotFoundError(f"Usearch index not found at '{path}'")
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite DB not found at '{db_path}'")
        inst = cls.__new__(cls)
        inst.index_path = index_path
        inst.db_path = db_path
        inst.index = Index.restore(str(index_path), view=False)
        inst.db = sqlite3.connect(str(db_path))
        inst.ndim = inst.index.ndim
        return inst

    def add(self, key: str, doc_id: int, vectors: list[np.ndarray]) -> None:
        """
        Add a sequence of token vectors for a document.

        Args:
            key: String key for the document.
            doc_id: Numeric document identifier.
            vectors: List of 1D numpy arrays of length ndim.
        """
        ids = []
        vecs = []
        cur = self.db.cursor()
        for token_id, vec in enumerate(vectors):
            arr = np.asarray(vec, dtype=float)
            if arr.ndim != 1 or arr.shape[0] != self.ndim:
                raise ValueError(
                    f"Vector shape {arr.shape} does not match index ndim={self.ndim}"
                )
            cur.execute(
                "INSERT INTO tokens (key, doc_id, token_id) VALUES (?, ?, ?)",
                (key, doc_id, token_id),
            )
            vid = cur.lastrowid
            ids.append(int(vid))
            vecs.append(arr)
        self.db.commit()
        self.index.add(ids, np.array(vecs))

    def add_batch(self, items: list[tuple[str, int, list[np.ndarray]]]) -> None:
        """
        Add multiple documents at once.

        Args:
            items: List of tuples (key, doc_id, vectors).
        """
        ids = []
        vecs = []
        cur = self.db.cursor()
        for key, doc_id, vectors in items:
            for token_id, vec in enumerate(vectors):
                arr = np.asarray(vec, dtype=float)
                if arr.ndim != 1 or arr.shape[0] != self.ndim:
                    raise ValueError(
                        f"Vector shape {arr.shape} does not match index ndim={self.ndim}"
                    )
                cur.execute(
                    "INSERT INTO tokens (key, doc_id, token_id) VALUES (?, ?, ?)",
                    (key, doc_id, token_id),
                )
                vid = cur.lastrowid
                ids.append(int(vid))
                vecs.append(arr)
        self.db.commit()
        self.index.add(ids, vecs)

    def search(
        self, vectors: list[np.ndarray], k: int
    ) -> list[list[tuple[str, int, int, float]]]:
        """
        Search the index with query vectors.

        Args:
            vectors: List of numpy 1D arrays of length ndim.
            k: Number of nearest neighbors per query vector.

        Returns:
            A list (len = number of queries) of lists of tuples:
            (key, doc_id, token_id, score), sorted by score descending.
        """
        # Perform search in usearch
        matches = self.index.search(vectors, k)
        # Extract ids and dists (supporting different attribute names)
        ids = getattr(matches, "ids", None) or getattr(matches, "keys", None)
        dists = getattr(matches, "dists", None) or getattr(
            matches, "distances", None
        )
        results: list[list[tuple[str, int, int, float]]] = []
        cur = self.db.cursor()
        for row_ids, row_dists in zip(ids, dists):
            row_res: list[tuple[str, int, int, float]] = []
            for vid, dist in zip(row_ids, row_dists):
                cur.execute(
                    "SELECT key, doc_id, token_id FROM tokens WHERE id = ?",
                    (int(vid),),
                )
                rec = cur.fetchone()
                if rec:
                    row_res.append((rec[0], rec[1], rec[2], float(dist)))
            results.append(row_res)
        return results

    def __getitem__(self, doc_id: int) -> list[np.ndarray]:
        """
        Retrieve all token vectors for a given document id, in token order.
        """
        cur = self.db.cursor()
        cur.execute(
            "SELECT id FROM tokens WHERE doc_id = ? ORDER BY token_id",
            (doc_id,),
        )
        vids = [int(r[0]) for r in cur.fetchall()]
        if not vids:
            return []
        # Fetch vectors from usearch
        vecs = self.index.get(vids)
        return list(vecs)

    def rerank(
        self,
        query_vectors: list[np.ndarray],
        candidate_doc_ids: list[int],
        k: int,
    ) -> list[tuple[str, int, float]]:
        """
        Rerank candidate documents using ColBERT maxsim.

        Args:
            query_vectors: List of numpy 1D arrays (query token embeddings).
            candidate_doc_ids: List of document ids to rerank.
            k: Number of top documents to return.

        Returns:
            List of tuples (key, doc_id, score), sorted by score descending.
        """
        scores: list[tuple[str, int, float]] = []
        cur = self.db.cursor()
        for doc_id in candidate_doc_ids:
            # Retrieve string key for doc
            cur.execute(
                "SELECT key FROM tokens WHERE doc_id = ? LIMIT 1", (doc_id,)
            )
            rec = cur.fetchone()
            if not rec:
                continue
            key = rec[0]
            # Retrieve doc vectors
            doc_vecs = self.__getitem__(doc_id)
            if not doc_vecs:
                continue
            arr = np.stack(doc_vecs, axis=0)  # (n_tokens, ndim)
            # Compute maxsim score
            score = 0.0
            for q in query_vectors:
                q_arr = np.asarray(q, dtype=float)
                sims = arr.dot(q_arr)
                score += float(np.max(sims))
            scores.append((key, doc_id, score))
        # Sort and return top-k
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:k]

    def save(self) -> None:
        """
        Save index and metadata to disk.
        """
        self.db.commit()
        self.index.save(str(self.index_path))
