import os
from typing import List

import polars as pl
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

if os.getcwd() != "/home/onyxia/work/funathon-project2":
    os.chdir("funathon-project2")

load_dotenv()


client_llmlab = OpenAI(
    base_url=os.environ["LLMLAB_URL"],
    api_key=os.environ["LLMLAB_API_KEY"],
)

client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    port=os.environ["QDRANT_API_PORT"],
    check_compatibility=False,
)


# Models
EMB_MODEL_NAME = "qwen3-embedding-8b"  # Embedding model
GEN_MODEL_NAME = "gemma4-26b-moe"  # Generative model

# Qdrant
COLLECTION_NAME = "nace-collection"
RETRIEVER_LIMIT = 5  # Number of candidates returned by the vector search

# Generation
TEMPERATURE = 0.1  # Low temperature → more deterministic, reproducible outputs

# Evaluation
SAMPLE_SIZE = 100  # Number of activities to evaluate (increase for more robust results)

# Q1


def get_embeddings(
    txt_to_embed,
    client_llmlab,
    emb_model: str,
) -> List[float]:
    try:
        response = client_llmlab.embeddings.create(
            model=EMB_MODEL_NAME, input=txt_to_embed
        )

        return response.data[0].embedding

    except Exception as e:
        raise RuntimeError(f"Embedding failed for doc {txt_to_embed}: {str(e)}")


activity = "Installation, maintenance and repair of residential air conditioning systems for private customers"

get_embeddings(activity, client_llmlab, emb_model=EMB_MODEL_NAME)


# Q2
# need polars
search_embedding = get_embeddings(activity, client_llmlab, emb_model=EMB_MODEL_NAME)

points = client_qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=search_embedding,
    limit=RETRIEVER_LIMIT,
)


points_df = (
    pl.DataFrame(points.model_dump())
    .unnest()
    .unnest()
    .select(["id", "score", "code", "text"])
)

points_df[0]
