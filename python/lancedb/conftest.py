import os
import time 

import numpy as np
import pytest
from pydantic import PrivateAttr

from .embeddings import EmbeddingFunctionRegistry, TextEmbeddingFunction, get_registry

# import lancedb so we don't have to in every example


@pytest.fixture(autouse=True)
def doctest_setup(monkeypatch, tmpdir):
    # disable color for doctests so we don't have to include
    # escape codes in docstrings
    monkeypatch.setitem(os.environ, "NO_COLOR", "1")
    # Explicitly set the column width
    monkeypatch.setitem(os.environ, "COLUMNS", "80")
    # Work in a temporary directory
    monkeypatch.chdir(tmpdir)

class MockEmbeddingAPI:
    """
    Dummy class representing an embedding API that is rate limited
    """
    def __init__(self, rate_limit=0, time_unit=60):
        self.rate_limit = rate_limit
        self.time_unit = time_unit
        self.request_count = 0
        self.window_start_time = time.time()
    
    def embed(self, texts):
        if not self.rate_limit: # no rate limit
            return self._process_api_request(texts)

        current_time = time.time()

        if current_time - self.window_start_time > self.time_unit:
            self.window_start_time = current_time
            self.request_count = 0

        if self.request_count < self.rate_limit:
            self.request_count += 1
            return self._process_api_request(texts)
        else:
            raise Exception("429") # too many requests
        
    def _process_api_request(self, texts):
        return [self._compute_one_embedding(row) for row in texts]

    def _compute_one_embedding(self, row):
        emb = np.random.rand(10)
        emb /= np.linalg.norm(emb)
        return emb

@get_registry().register("test")
class MockTextEmbeddingFunction(TextEmbeddingFunction):
    """
    Return the hash of the first 10 characters
    """
    def generate_embeddings(self, texts):
        return [self._compute_one_embedding(row) for row in texts]

    def _compute_one_embedding(self, row):
        emb = np.array([float(hash(c)) for c in row[:10]])
        emb /= np.linalg.norm(emb)
        return emb

    def ndims(self):
        return 10

@get_registry().register("test_rate_limited")
class MockRateLimitedTextEmbeddingFunction(TextEmbeddingFunction):
    """
    Mock Ebedding function that calls a rate limited API. 
    Limits are set to 1 request per 0.1 sec to facilitate testing.
    """
    _model: MockEmbeddingAPI = PrivateAttr(default=MockEmbeddingAPI(rate_limit=1, time_unit=0.1))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_embeddings(self, texts):
        rs = self._model.embed(texts)
        return rs

    def ndims(self):
        return 10