import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry

cohere = EmbeddingFunctionRegistry.get_instance().get("openai").create(rate_limit=1)

class TextModel(LanceModel):
    text: str = cohere.SourceField()
    vector: Vector(cohere.ndims()) =  cohere.VectorField()

data = [ { "text": "hello world" },
        { "text": "goodbye world" }]

db = lancedb.connect("~/.lancedb")
tbl = db.create_table("test", schema=TextModel, mode="overwrite")

tbl.add(data)
