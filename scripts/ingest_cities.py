"""Create a Pinecone index and upload Wanderlust's curated city profiles.

Run only after filling .env with your own HKUST GenAI and Pinecone details:
    .venv/bin/python scripts/ingest_cities.py
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def required(name: str) -> str:
    value = os.getenv(name)
    if not value or "YOUR-" in value:
        raise RuntimeError(f"Missing {name}. Fill it in .env before running this script.")
    return value


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def profile_text(city: dict) -> str:
    return "\n".join(
        [
            f"City: {city['city']}, {city['country']}",
            f"Region: {city['region']}",
            f"Description: {city['description']}",
            f"Vibes: {', '.join(city['vibes'])}",
            f"Travel styles: {', '.join(city['styles'])}",
            f"Budget level: {city['budget_level']}",
        ]
    )


def main() -> None:
    hkust_key = required("HKUST_GENAI_API_KEY")
    hkust_base_url = os.getenv("HKUST_GENAI_BASE_URL", "https://hkust.azure-api.net/hkust-genai/v1/")
    pinecone_key = required("PINECONE_API_KEY")
    index_name = required("PINECONE_INDEX_NAME")
    embedding_model = required("HKUST_GENAI_EMBEDDING_MODEL")
    namespace = os.getenv("PINECONE_NAMESPACE", "cities-v1")
    dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    cities = json.loads((PROJECT_ROOT / "data" / "cities.seed.json").read_text())
    client = OpenAI(
        api_key=hkust_key,
        base_url=hkust_base_url,
        default_headers={"api-key": hkust_key},
    )
    pinecone = Pinecone(api_key=pinecone_key)

    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while not pinecone.describe_index(index_name).status["ready"]:
            print("Waiting for Pinecone index to become ready...")
            time.sleep(2)

    index = pinecone.Index(index_name)
    texts = [profile_text(city) for city in cities]
    embeddings = client.embeddings.create(input=texts, model=embedding_model).data

    vectors = []
    for city, embedding, text in zip(cities, embeddings, texts):
        metadata = {**city, "search_text": text}
        vectors.append({"id": slug(f"{city['city']}-{city['country']}"), "values": embedding.embedding, "metadata": metadata})

    index.upsert(vectors=vectors, namespace=namespace)
    print(f"Uploaded {len(vectors)} city profiles to {index_name}/{namespace}.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f"Configuration error: {error}") from error
