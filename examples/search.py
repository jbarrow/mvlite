import modal
from urllib.request import urlopen
from pathlib import Path
from mvlite.index import MultiVectorIndex

import numpy as np


def create_index(pdf_path: str = "https://arxiv.org/pdf/1706.03762"):
    """Build a multivector index from a PDF."""
    model = modal.Cls.from_name("colpali-encoder", "Model")()
    data = (
        urlopen(pdf_path).read()
        if pdf_path.startswith("http")
        else Path(pdf_path).read_bytes()
    )
    pages = np.array(model.encode.remote(data))
    ndim = pages[0].shape[-1]
    mvi = MultiVectorIndex("modal.usearch", ndim)
    for doc_id, vecs in enumerate(pages):
        mvi.add(str(doc_id), doc_id, vecs)
    mvi.save()


def maxsim():
    """ColBERT-style maxsim: initial retrieval + rerank."""
    model = modal.Cls.from_name("colpali-encoder", "Model")()
    mvi = MultiVectorIndex.load("modal.usearch")
    q_vs = model.encode.remote(["attention equation"])[0]
    q_arr = q_vs.numpy() if hasattr(q_vs, "numpy") else np.array(q_vs)
    hits = mvi.search(q_arr, k=10)
    cands = sorted({doc for hit in hits for _, doc, _, _ in hit})
    for key, doc_id, score in mvi.rerank(q_arr, cands, k=5):
        print(f"Page {doc_id}: {score:.4f}")


if __name__ == "__main__":
    # create_index()
    maxsim()
