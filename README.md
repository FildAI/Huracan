# Huracan

Huracan is a search engine for B2B outreach. You ask in plain language, for example
*"fintech startups in the EU with 50–200 employees"*. It returns the matching companies,
then the people who work at them.

It was built in 2025 for [Fild AI](https://github.com/FildAI), a deep-research tool for
finding B2B leads, as the search behind its prospect lists. It is published as is, under the
MIT licence.

## How a query works

```
query ──► gpt-4o-mini: which columns matter, which keywords
            │
            ▼
         vocab collections (one per column, MiniLM embeddings)
         "EU"   → Germany, France, Netherlands, …
         "SaaS" → the industry labels that really exist in the data
            │
            ▼
         gpt-4o-mini: MUST / SHOULD filter built only from real values
            │
            ▼
         Qdrant: dense search (text-embedding-3-small) under that filter
            │   no hits → retry without the filter
            ▼
         dedupe by company id or domain ──► companies
            │
            ▼
         MongoDB: people at those companies ──► contacts
```

Search engines that pass a language model's guess straight into a filter break on
values the data doesn't use: "USA" against "United States", "software" against
"Computer Software". Huracan keeps one small vector collection with every distinct
value of each filterable column (`name`, `primary_industry`, `country`, `size`).
Each keyword is matched against that collection first. The model then builds the
filter from values that are known to exist.

## Run it

You need Python 3.11, Docker and an OpenAI API key.

```bash
cp .env.example .env          # fill in the keys
docker compose -f docker-compose.infra.yml up -d   # Qdrant + MongoDB
pip install -r requirements.txt
uvicorn main:app --reload     # http://localhost:8000/docs
```

Or run the API in Docker too: `docker compose up --build`.

### Load data

Companies come from a CSV with these columns: `name`, `description`,
`primary_industry`, `type`, `size`, `location`, `country`, `domain`, `linkedin_url`.

```bash
python scripts/upload_bulk.py path/to/companies.csv      # a file or a folder of CSVs
python scripts/vocab_index.py                             # build the vocab collections
python scripts/upload_people_mongodb.py path/to/people/  # CSVs with "Company Name", "Full Name"
```

No data ships with the repo.

## API

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/companies/search?query=…&limit=10` | Companies for a plain-language query |
| `GET` | `/people/search?query=…&companies_limit=50&limit=100` | Companies, then up to 5,000 people at them |
| `POST` | `/people/search/by-companies` | People for a list of company names |
| `GET` | `/health`, `/companies/health`, `/people/health` | Health checks |

## Layout

```
main.py                     FastAPI app
routes/                     /companies and /people endpoints
core/indexes/companies.py   filtered dense search, fallback, dedupe
core/indexes/vocab_bm25.py  vocab collections and filter building
core/vector_config.py       which fields are embedded and with which model
core/services/              bulk upload and people lookup
utils/ai.py                 the two structured-output prompts
scripts/                    loaders; scripts/experiments/ holds the early BM25
                            and ten-embedding prototypes (README-multi-embed.md)
```

## Status

Not maintained. The code is a snapshot of a working prototype from June 2025. Pin the
versions in `requirements.txt` if you run it.

## Licence

[MIT](LICENSE)
