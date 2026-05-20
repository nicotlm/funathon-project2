import os
from typing import List

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


emb_model = "qwen3-embedding-8b"
COLLECTION_NAME = "nace-collection-test"


def get_embeddings(
    txt_to_embed,
    client_llmlab,
    emb_model: str,
) -> List[float]:
    try:
        response = client_llmlab.embeddings.create(model=emb_model, input=txt_to_embed)

        return response.data[0].embedding

    except Exception as e:
        raise RuntimeError(f"Embedding failed for doc {txt_to_embed}: {str(e)}")


activity = "Installation, maintenance and repair of residential air conditioning systems for private customers"

get_embeddings(activity, client_llmlab, emb_model=emb_model)
