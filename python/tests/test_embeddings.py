#  Copyright 2023 LanceDB Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import sys
import time

import lance
import pytest
import numpy as np
import pyarrow as pa

import lancedb
from lancedb.conftest import MockTextEmbeddingFunction
from lancedb.embeddings import (
    EmbeddingFunctionConfig,
    EmbeddingFunctionRegistry,
    get_registry,
    with_embeddings,
)
from lancedb.pydantic import LanceModel, Vector


def mock_embed_func(input_data):
    return [np.random.randn(128).tolist() for _ in range(len(input_data))]


def test_with_embeddings():
    for wrap_api in [True, False]:
        if wrap_api and sys.version_info.minor >= 11:
            # ratelimiter package doesn't work on 3.11
            continue
        data = pa.Table.from_arrays(
            [
                pa.array(["foo", "bar"]),
                pa.array([10.0, 20.0]),
            ],
            names=["text", "price"],
        )
        data = with_embeddings(mock_embed_func, data, wrap_api=wrap_api)
        assert data.num_columns == 3
        assert data.num_rows == 2
        assert data.column_names == ["text", "price", "vector"]
        assert data.column("text").to_pylist() == ["foo", "bar"]
        assert data.column("price").to_pylist() == [10.0, 20.0]


def test_embedding_function(tmp_path):
    registry = EmbeddingFunctionRegistry.get_instance()

    # let's create a table
    table = pa.table(
        {
            "text": pa.array(["hello world", "goodbye world"]),
            "vector": [np.random.randn(10), np.random.randn(10)],
        }
    )
    conf = EmbeddingFunctionConfig(
        source_column="text",
        vector_column="vector",
        function=MockTextEmbeddingFunction(),
    )
    metadata = registry.get_table_metadata([conf])
    table = table.replace_schema_metadata(metadata)

    # Write it to disk
    lance.write_dataset(table, tmp_path / "test.lance")

    # Load this back
    ds = lance.dataset(tmp_path / "test.lance")

    # can we get the serialized version back out?
    configs = registry.parse_functions(ds.schema.metadata)

    conf = configs["vector"]
    func = conf.function
    actual = func.compute_query_embeddings("hello world")

    # And we make sure we can call it
    expected = func.compute_query_embeddings("hello world")

    assert np.allclose(actual, expected)

def test_rate_limited_embedding_function(tmp_path):

    def _get_schema(model):
        class Schema(LanceModel):
            text: str = model.SourceField()
            vector: Vector(model.ndims()) = model.VectorField()
        
        return Schema

    model_builder = get_registry().get("test_rate_limited")
    model = model_builder.create() # without rate limiting
    
    db = lancedb.connect(tmp_path)
    
    table = db.create_table("test_without_limit", schema=_get_schema(model))

    table.add([{"text": "hello world"}]) # 
    assert len(table) == 1
    time.sleep(0.1)

    # Should hit the rate limit and throw an exception as we are not handling rate limiting
    #### The RateLimitHandler should handle this but it gets re-initilaized
    with pytest.raises(Exception) as e:
        table.add([{"text": "hello world"}])
        table.add([{"text": "hello world"}]) 
    e.value == "429"
   
    '''
    model = model_builder.create(rate_limit=1, time_unit=0.1) # handle rate limiting in the embedding function
    table = db.create_table("test_with_limit", schema=_get_schema(model))

    table.add([{"text": "hello"}, {"text": "bye"}])
    table.add([{"text": "hello"}, {"text": "bye"}])
    table.add([{"text": "hello"}, {"text": "bye"}])
    assert len(table) == 3
    '''

    


