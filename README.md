# Wanderlust AI

An independent Streamlit travel-planning app. It preserves the original recommendation, itinerary, hotel, map, and memory-stamp flows while using a redesigned visual system inspired by the Wanderlust portfolio demos.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run app2.py
```

Populate `.streamlit/secrets.toml` before starting. The actual secrets file is ignored by Git.

To preview the landing/profile screen without credentials, run `WANDERLUST_DEMO_MODE=true streamlit run app2.py`. This mode is visual-only; it does not make API calls.

## Required services

| Service | Used for | What to configure |
| --- | --- | --- |
| HKUST GenAI API | City matching, trip concepts, itinerary generation, photo analysis | Subscription API key and model names |
| Pinecone | Vector search over the city knowledge base | API key and an existing index name |
| RapidAPI | Travel Advisor attractions/restaurants/hotels and weather | API key with subscriptions to the required APIs |

The existing `travel-world-openai` Pinecone index is not transferred with this code. Create and populate an index under your own account before production use, or ask for the source dataset/export from the original owner.

## API checklist

1. In the HKUST API Developer Portal, subscribe to `hkust-genai-v1`. Copy one subscription key and use the default base URL and model names from `.env.example`.
2. Create a Pinecone project and index. Record its key and name, then ingest the city vectors/dataset.
3. In RapidAPI, subscribe to the Travel Advisor and weather APIs used by this app, then create a key.
4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in only your own values.

## Build your city dataset

`data/cities.seed.json` is Wanderlust's own starter dataset: 16 curated city profiles with city, country, description, vibes, travel styles, and budget level. For global coverage, generate up to 5,000 additional city records from GeoNames (CC BY 4.0):

```bash
.venv/bin/python scripts/build_geonames_cities.py --limit 5000
```

The generated `data/cities.geonames.json` stays local and is ignored by Git. Its source attribution is recorded in `data/SOURCES.md`.

After filling in your HKUST GenAI and Pinecone values in `.env`, run:

```bash
.venv/bin/python scripts/ingest_cities.py
```

The script creates the `PINECONE_INDEX_NAME` index when needed, generates embeddings with your `HKUST_GENAI_EMBEDDING_MODEL`, and uploads the records to `PINECONE_NAMESPACE`. It is safe to run again: the same city IDs are updated rather than duplicated.

## RapidAPI subscriptions

RapidAPI uses one API key, but each API must have its own subscription. Start only with free hard-limit plans and monitor the quota in the RapidAPI dashboard. The original Travel Advisor host may require a separate plan; do not assume an existing key includes hotel, restaurant, attraction, or weather access.

## Deploy

Push this folder to a new GitHub repository. In Streamlit Community Cloud, set the entry point to `app2.py`, then add the same secret values in **App settings → Secrets**. Never commit a real key or paste it into GitHub Issues, chat, or front-end code.
