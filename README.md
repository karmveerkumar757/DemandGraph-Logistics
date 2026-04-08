# AI-Based Logistics Optimization System

An academic-grade, modular web application for transportation network optimization focused on two core AI modules:

- Transformer-based demand forecasting for zone and time-slot demand estimation.
- Graph-based route optimization with configurable distance and travel-time objectives.

## Team

- Shivam Gangwar
- Kramveer Kumar
- Rishav Pal
- Sayantan Mandal

## Architecture

The codebase is organized as a shared Python package consumed by both runtime surfaces:

- `apps/api`: FastAPI application for typed backend services.
- `apps/dashboard`: Streamlit dashboard for logistics visualizations.
- `src/logistics_optimization`: shared domain, ML, database, service, and dashboard client code.
- `scripts`: operational utilities for database setup, seeding, training, and evaluation.
- `tests`: lightweight regression coverage for critical modules.

## Core Modules

### Demand Forecasting

- NYC Taxi preprocessing pipeline for time-based zone demand aggregation.
- Transformer encoder model for learning `Y(t+1) = f(X(t))`.
- Training and inference services with MAE and RMSE evaluation.

### Route Optimization

- Pluggable weighted graph model `G = (V, E)`.
- Dijkstra and A* search with configurable distance and travel-time weights.
- Route evaluation for total distance and average travel time.

## Future-Proofing

Extension interfaces are defined for:

- Fleet Allocation
- ETA Prediction

This keeps future academic phases decoupled from the current forecasting and routing modules.

## Local Development

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -e .[dev]
   ```

2. Initialize the SQLite database:

   ```bash
   python scripts/init_db.py
   python scripts/seed_demo_data.py
   ```

3. Start the API:

   ```bash
   uvicorn apps.api.main:app --reload
   ```

4. Start the Streamlit dashboard:

   ```bash
   streamlit run apps/dashboard/Home.py
   ```

## Docker

Run the full stack with:

```bash
docker compose up --build
```

The API runs on `http://localhost:8000` and the dashboard runs on `http://localhost:8501`.

