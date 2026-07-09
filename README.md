# DataStory AI

DataStory AI is an end-to-end web application that turns a CSV dataset into an interactive data-analysis workflow. It is designed as a junior data analyst experience: users can upload data, inspect quality issues, clean the dataset, explore patterns, train machine-learning models, compare results, and export a structured analysis report.

The product is built for a deployed frontend plus hosted API setup. The frontend is a Vite/React application intended for Vercel, while the backend is a FastAPI service that can run on Render, Docker, or any Python web-hosting platform.

## Core Capabilities

- CSV upload and bundled sample datasets
- Dataset profiling with health score, missingness, duplicates, column types, warnings, and possible targets
- Configurable cleaning workflow with preview and apply steps
- Exploratory data analysis with numeric summaries, categorical distributions, missing-value views, outlier checks, and correlations
- Target selection and automatic task detection for classification or regression
- Model selection, training, tuning, and comparison across multiple scikit-learn estimators
- Results leaderboard with best-model details and export
- AI report workspace with sectioned analysis and Markdown report export
- Optional AI/RAG features through OpenRouter and FAISS-backed knowledge retrieval
- Light and dark UI modes

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React, TypeScript, Vite |
| Styling | Tailwind CSS, custom theme tokens |
| UI | Framer Motion, Lucide icons |
| Charts | Plotly.js |
| State/Data | React Router, React Query, local storage |
| Backend | FastAPI, Pydantic |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, imbalanced-learn |
| AI Integration | OpenRouter API |
| Retrieval | FAISS |
| Storage | SQLite |

## Repository Structure

```text
datastory_ai/
  backend/                  FastAPI API, routers, schemas, services, ML pipeline
  data/sample_datasets/     bundled demo CSV datasets
  web/                      Vite React frontend for Vercel
  requirements.txt          backend Python dependencies
  Dockerfile                backend container image
  Procfile                  backend process command
  render.yaml               Render backend blueprint
  runtime.txt               Python runtime hint
```

## Deployment Overview

DataStory AI is deployed as two pieces:

1. **Frontend on Vercel**
   - Root directory: `web`
   - Framework: Vite
   - Build command: `npm run build`
   - Output directory: `dist`
   - Environment variable: `VITE_API_BASE`

2. **Backend on Render or another Python host**
   - Start command: `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Health check: `/healthz`
   - Environment variables: `DATASTORY_DB`, `CORS_ORIGINS`, optional `OPENROUTER_API_KEY`
   - Persistent storage is recommended for SQLite and uploaded/runtime data

## Frontend Environment

Set this in Vercel for the `web` project:

| Variable | Required | Description |
| --- | --- | --- |
| `VITE_API_BASE` | Yes | Public URL of the deployed backend API. Example: `https://datastory-api.onrender.com` |

If the API is served through the same domain under `/api`, `VITE_API_BASE` can be set to `/api`.

## Backend Environment

Set these on the backend hosting platform:

| Variable | Required | Description |
| --- | --- | --- |
| `DATASTORY_DB` | Recommended | SQLite database path. Use a persistent disk path in production. |
| `CORS_ORIGINS` | Yes | Comma-separated list of allowed frontend origins, including the Vercel URL. |
| `OPENROUTER_API_KEY` | Optional | Enables AI explanations and AI-assisted report generation. |

Example production values:

```text
DATASTORY_DB=/var/data/datastory.db
CORS_ORIGINS=https://your-datastory-app.vercel.app
OPENROUTER_API_KEY=your_key_if_enabled
```

## API Surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Backend health check |
| `GET` | `/samples` | List bundled sample datasets |
| `POST` | `/samples/{sample_id}` | Load a sample dataset |
| `POST` | `/upload` | Upload a CSV dataset |
| `GET` | `/profile/{dataset_id}` | Retrieve dataset profile |
| `POST` | `/clean/{dataset_id}` | Preview or apply cleaning operations |
| `GET` | `/eda/{dataset_id}` | Retrieve EDA summaries and chart data |
| `POST` | `/select-target/{dataset_id}` | Select a target and detect task type |
| `GET` | `/models` | List available models for the task |
| `POST` | `/train/{dataset_id}` | Train selected models |
| `GET` | `/results/{dataset_id}` | Retrieve model results |
| `POST` | `/chat/{dataset_id}` | Ask questions about the dataset |
| `POST` | `/report/{dataset_id}` | Generate an AI-assisted report |

## Production Notes

- The frontend is static and deploys cleanly to Vercel from the `web` directory.
- The backend should be deployed separately because model training, file upload, Pandas processing, and SQLite storage need a Python server runtime.
- Uploaded datasets are currently held by the backend process during active analysis. Use a single backend instance for simple deployments, or add durable object/file storage before horizontal scaling.
- SQLite is suitable for demos and small deployments. Put the database file on persistent disk so records survive restarts.
- Model training is CPU-bound. Hosting plans should be sized according to expected dataset size and number of concurrent users.
- `OPENROUTER_API_KEY` is optional; core profiling, cleaning, EDA, modeling, leaderboard, and Markdown report export work without it.

## Deployment Status

The project includes the required frontend Vercel config in `web/vercel.json`, backend process files, environment examples, and ignore rules for generated build/cache/database artifacts.
