#  Copyright (c) 2023. LanceDB Developers
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
import os
from abc import ABC, abstractmethod
from typing import List, Union
from tqdm import tqdm

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, Field, PrivateAttr

from .utils import TEXT, retry_with_exponential_backoff
from .fine_tuner import QADataset
import lancedb

class EmbeddingFunction(BaseModel, ABC):
    """
    An ABC for embedding functions.

    All concrete embedding functions must implement the following:
    1. compute_query_embeddings() which takes a query and returns a list of embeddings
    2. get_source_embeddings() which returns a list of embeddings for the source column
    For text data, the two will be the same. For multi-modal data, the source column
    might be images and the vector column might be text.
    3. ndims method which returns the number of dimensions of the vector column
    """

    __slots__ = ("__weakref__",)  # pydantic 1.x compatibility
    max_retries: int = (
        7  # Setitng 0 disables retires. Maybe this should not be enabled by default,
    )
    _ndims: int = PrivateAttr()

    @classmethod
    def create(cls, **kwargs):
        """
        Create an instance of the embedding function
        """
        return cls(**kwargs)

    @abstractmethod
    def compute_query_embeddings(self, *args, **kwargs) -> List[np.array]:
        """
        Compute the embeddings for a given user query
        """
        pass

    @abstractmethod
    def compute_source_embeddings(self, *args, **kwargs) -> List[np.array]:
        """
        Compute the embeddings for the source column in the database
        """
        pass

    def compute_query_embeddings_with_retry(self, *args, **kwargs) -> List[np.array]:
        """
        Compute the embeddings for a given user query with retries
        """
        return retry_with_exponential_backoff(
            self.compute_query_embeddings, max_retries=self.max_retries
        )(
            *args,
            **kwargs,
        )

    def compute_source_embeddings_with_retry(self, *args, **kwargs) -> List[np.array]:
        """
        Compute the embeddings for the source column in the database with retries
        """
        return retry_with_exponential_backoff(
            self.compute_source_embeddings, max_retries=self.max_retries
        )(*args, **kwargs)

    def sanitize_input(self, texts: TEXT) -> Union[List[str], np.ndarray]:
        """
        Sanitize the input to the embedding function.
        """
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pa.Array):
            texts = texts.to_pylist()
        elif isinstance(texts, pa.ChunkedArray):
            texts = texts.combine_chunks().to_pylist()
        return texts

    def safe_model_dump(self):
        from ..pydantic import PYDANTIC_VERSION

        if PYDANTIC_VERSION.major < 2:
            return dict(self)
        return self.model_dump()

    @abstractmethod
    def ndims(self):
        """
        Return the dimensions of the vector column
        """
        pass

    def SourceField(self, **kwargs):
        """
        Creates a pydantic Field that can automatically annotate
        the source column for this embedding function
        """
        return Field(json_schema_extra={"source_column_for": self}, **kwargs)

    def VectorField(self, **kwargs):
        """
        Creates a pydantic Field that can automatically annotate
        the target vector column for this embedding function
        """
        return Field(json_schema_extra={"vector_column_for": self}, **kwargs)

    def __eq__(self, __value: object) -> bool:
        if not hasattr(__value, "__dict__"):
            return False
        return vars(self) == vars(__value)

    def __hash__(self) -> int:
        return hash(frozenset(vars(self).items()))
    
    def finetune(self, dataset: QADataset, *args, **kwargs):
        """
        Finetune the embedding function on a dataset
        """
        raise NotImplementedError("Finetuning is not supported for this embedding function")
    
    def evaluate(self, dataset: QADataset, path=None, *args, **kwargs):
        """
        Evaluate the embedding function on a dataset
        """
        raise NotImplementedError("Evaluation is not supported for this embedding function")


class EmbeddingFunctionConfig(BaseModel):
    """
    This model encapsulates the configuration for a embedding function
    in a lancedb table. It holds the embedding function, the source column,
    and the vector column
    """

    vector_column: str
    source_column: str
    function: EmbeddingFunction


class TextEmbeddingFunction(EmbeddingFunction):
    """
    A callable ABC for embedding functions that take text as input
    """

    def compute_query_embeddings(self, query: str, *args, **kwargs) -> List[np.array]:
        return self.compute_source_embeddings(query, *args, **kwargs)

    def compute_source_embeddings(self, texts: TEXT, *args, **kwargs) -> List[np.array]:
        texts = self.sanitize_input(texts)
        return self.generate_embeddings(texts)

    @abstractmethod
    def generate_embeddings(
        self, texts: Union[List[str], np.ndarray]
    ) -> List[np.array]:
        """
        Generate the embeddings for the given texts
        """
        pass

    def evaluate(self, dataset: QADataset, path=None, *args, **kwargs):
        """
        Evaluate the embedding function on a dataset. This calculates the hit-rate for the
        top-k retrieved documents for each query in the dataset. Assumes that the first
        relevant document is the expected document.
        Pro - Should work for any embedding model
        Con - Only returns simple hit-rate, does not return more detailed metrics
        if possible, should be implemented in the downstream embedding model class if custom 
        metrics are needed.
        Parameters
        ----------
        dataset: QADataset
            The dataset to evaluate on
        
        Returns
        -------
        dict
            The evaluation results
        """
        corpus = dataset.corpus
        queries = dataset.queries
        relevant_docs = dataset.relevant_docs
        path = path or os.path.join(os.getcwd(), "eval")
        db = lancedb.connect(path)
        class Schema(lancedb.pydantic.LanceModel):
            id: str
            text: str = self.SourceField()
            vector: lancedb.pydantic.Vector(self.ndims()) = self.VectorField()
        
        retriever = db.create_table("eval", schema=Schema, mode="overwrite")
        pylist = [{"id": str(k), "text": v} for k, v in corpus.items()]
        retriever.add(pylist)

        eval_results = []
        for query_id, query in tqdm(queries.items()):
            retrieved_nodes = retriever.search(query).to_list()
            retrieved_ids = [node["id"]  for node in retrieved_nodes]
            expected_id = relevant_docs[query_id][0]
            is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

            eval_result = {
                "is_hit": is_hit,
                "retrieved": retrieved_ids,
                "expected": expected_id,
                "query": query_id,
            }
            eval_results.append(eval_result)

        return eval_results