from __future__ import annotations
from typing import Protocol
from usearch.index import Index

import numpy


class PoolingStrategy(Protocol):
    def pool(self, vectors: list[numpy.ndarray]) -> numpy.ndarray: ...


class MeanPooling:
    def pool(self, vectors: list[numpy.ndarray]) -> numpy.ndarray:
        return numpy.mean(vectors, axis=0)


class MultiVectorIndex:
    def __init__(
        self,
        ndim: int,
        nvectors: int,
        pooling_strategy: PoolingStrategy,
        quantization: str,
    ) -> None:
        self.index = Index(ndim=ndim)
        self.nvectors = nvectors
        self.pooling_strategy = pooling_strategy
        self.quantization = quantization

    def add(self, index: int, vectors: list[numpy.ndarray]) -> None:
        pass

    def add_batch(
        self, indices: list[int], vectors: list[numpy.ndarray]
    ) -> None:
        pass

    def search(
        self, vectors: list[numpy.ndarray], k: int
    ) -> list[list[tuple[int, float]]]:
        return []

    def __getitem__(self, index: int) -> list[numpy.ndarray]:
        return []

    def rerank(
        self, query: numpy.ndarray, valuses: list[numpy.ndarray], k: int
    ) -> list[list[tuple[int, float]]]:
        return []

    def save(self):
        pass

    @classmethod
    def load(cls, file: str) -> MultiVectorIndex:
        return
