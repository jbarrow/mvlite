# `mvlite` - Fast Multi-vector Indexing and Retrieval

Build local or scalable RAG applications easily.
A single durable object per user provides free partitioning for RAG.

## Overview

`mvlite` is a python library focused on making multivector indexing, reranking, and retrieval fast on your local machine.
For loads up to, say, a few million pages, this should be the fastest and easiest way to run ColPali/ColBERT/ColQwen/etc.

It's built atop uSearch.

## Installation

To install `mvlite`, run:

```bash
pip install mvlite
```

## Usage

```py
from mvlite import MultiVectorIndex

index = MultiVectorIndex("demo.mvlite", ndims=3, nvectors=2)
index.add([[],[]])
index.add([[],[]])

index.search([[],[]])
```

## Usage with ColBERT/ColPali
