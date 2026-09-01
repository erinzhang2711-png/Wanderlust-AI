"""Create a Pinecone index and upload Wanderlust city profiles.

Run only after filling .env with your own HKUST GenAI and Pinecone details:
    .venv/bin/python scripts/ingest_cities.py
"""

import json
import os
import re
import time
import argparse
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


def chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_cities() -> list[dict]:
    seed_path = PROJECT_ROOT / "data" / "cities.seed.json"
    geonames_path = PROJECT_ROOT / "data" / "cities.geonames.json"
    cities = json.loads(seed_path.read_text())
    if geonames_path.exists():
        cities.extend(json.loads(geonames_path.read_text()))
    else:
        print("GeoNames dataset not found; uploading the 16 curated seed cities only.")

    unique: dict[tuple[str, str], dict] = {}
    for city in cities:
        key = (city["city"].casefold(), city["country"].casefold())
        unique.setdefault(key, city)
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Wanderlust city profiles to Pinecone.")
    parser.add_argument("--start", type=int, default=0, help="Resume from a zero-based city offset.")
    args = parser.parse_args()
    if args.start < 0:
        raise SystemExit("--start must be zero or greater.")

    hkust_key = required("HKUST_GENAI_API_KEY")
    hkust_base_url = os.getenv("HKUST_GENAI_BASE_URL", "https://hkust.azure-api.net/hkust-genai/v1/")
    pinecone_key = required("PINECONE_API_KEY")
    index_name = required("PINECONE_INDEX_NAME")
    embedding_model = os.getenv("HKUST_GENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    namespace = os.getenv("PINECONE_NAMESPACE", "cities-v1")
    dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    all_cities = load_cities()
    cities = all_cities[args.start :]
    if not cities:
        print(f"No remaining cities to upload (dataset contains {len(all_cities)} profiles).")
        return
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
    uploaded = 0
    for city_batch in chunks(cities, 96):
        texts = [profile_text(city) for city in city_batch]
        embeddings = client.embeddings.create(input=texts, model=embedding_model).data
        vectors = []
        for city, embedding, text in zip(city_batch, embeddings, texts):
            metadata = {**city, "search_text": text}
            city_key = f"{city['city']}-{city['country']}"
            vector_id = city.get("id") or f"seed-{slug(city_key)}"
            vectors.append({"id": vector_id, "values": embedding.embedding, "metadata": metadata})
        index.upsert(vectors=vectors, namespace=namespace)
        uploaded += len(vectors)
        print(f"Uploaded {args.start + uploaded}/{len(all_cities)} city profiles...")

    print(f"Uploaded {args.start + uploaded}/{len(all_cities)} city profiles to {index_name}/{namespace}.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f"Configuration error: {error}") from error
