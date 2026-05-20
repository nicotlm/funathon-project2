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


# Q4

SYSTEM_PROMPT = """\
You are an expert classifier for the NACE 2.1 nomenclature (Statistical Classification of Economic Activities in the European Community).

Given a company activity description and a short list of candidate NACE codes, your job is to pick the single most appropriate code from the candidates — or to declare the activity not codable if the description is too ambiguous.

Always reply with a valid JSON object matching the requested schema. No explanations, no extra text.
"""

USER_PROMPT_TEMPLATE = """\
## Activity to classify
{activity}

## Candidate NACE codes and their explanatory notes
{proposed_nace_descriptions}

## Rules
- Pick exactly one code from this list: [{proposed_nace_codes}]. Do not invent codes outside the list.
- If several activities are mentioned, only consider the first one.
- If the description is too vague to decide, return `nace_code: null` and `codable: false`.

## Output — valid JSON only
{{
  "nace_code": "<one code from the candidate list, or null>",
  "codable": <true | false>,
  "confidence": <float between 0.0 and 1.0>
}}
"""

import json

user_prompt = USER_PROMPT_TEMPLATE.format(
    activity=activity,
    proposed_nace_descriptions="## " + "\n\n## ".join(points_df["text"]),
    proposed_nace_codes=", ".join(points_df["code"]),
)

response = client_llmlab.chat.completions.create(
    model=GEN_MODEL_NAME,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
    temperature=TEMPERATURE,
    response_format={"type": "json_object"},
)

llm_response = json.loads(response.choices[0].message.content)
print(json.dumps(llm_response, indent=2))


# Load data
import duckdb

con = duckdb.connect(database=":memory:")

con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

query_definition = f"""
SELECT *
FROM read_parquet(
  'https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet'
)
USING SAMPLE {SAMPLE_SIZE}
"""

annotations = con.sql(query_definition).to_df().to_dict(orient="records")
print(f"Dataset loaded: {len(annotations)} rows")
print(f"Keys: {list(annotations[0].keys())}")
annotations[:2]

# Pipeline with polars


def run_rag_pipeline(activity: str) -> dict:
    """
    Run the full RAG pipeline for a single activity label.

    Parameters
    ----------
    activity : str
        Free-text economic activity label to be coded.

    Returns
    -------
    dict with keys:
        - nace_code (str | None) : predicted NACE code
        - codable (bool)        : True if the label could be coded
        - confidence (float)    : confidence score (0–1)
        - retrieved_codes (list): candidates returned by the retriever
    """
    # --- Step 1: Embedding ---
    emb_response = client_llmlab.embeddings.create(model=EMB_MODEL_NAME, input=activity)
    embedding = emb_response.data[0].embedding

    # --- Step 2: Retrieval ---
    points = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=RETRIEVER_LIMIT,
    )

    points_df = (
        pl.DataFrame(points.model_dump())
        .unnest()
        .unnest()
        .select(["id", "score", "code", "text"])
    )

    # --- Step 3: Prompt construction ---
    user_prompt = USER_PROMPT_TEMPLATE.format(
        activity=activity,
        proposed_nace_descriptions="## " + "\n\n## ".join(points_df["text"]),
        proposed_nace_codes=", ".join(points_df["code"]),
    )

    # --- Step 4: LLM inference ---
    gen_response = client_llmlab.chat.completions.create(
        model=GEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )

    result = json.loads(gen_response.choices[0].message.content)
    # Keep retrieved candidates for retriever evaluation
    result["retrieved_codes"] = points_df["code"]

    return result


annotations_df = pl.DataFrame(annotations)

results_df = annotations_df.with_columns(
    pl.col("label")
    .map_elements(
        lambda a: run_rag_pipeline(a),
        return_dtype=pl.Struct(
            {
                "nace_code": pl.Utf8,
                "codable": pl.Boolean,
                "confidence": pl.Float64,
                "retrieved_codes": pl.List(pl.String),
            }
        ),
    )
    .alias("pred")
).unnest()

# Metrics
results_df = results_df.with_columns(
    retriever_hit=pl.col("code").is_in(pl.col("retrieved_codes")),
    pipeline_correct=pl.col("code") == pl.col("nace_code"),
).with_columns(
    llm_correct_given_retriever=pl.when(pl.col("retriever_hit"))
    .then(pl.col("pipeline_correct"))
    .otherwise(None),
)

# Q1
results_df["retriever_hit"].mean()
