# Fert Recon

React + Vite + Leaflet frontend and FastAPI backend for Morocco fertilizer recommendations and EE-powered maps.

## Frontend
```bash
npm install
npm run dev           # localhost:5173
```
Set `VITE_API_BASE` to point at your backend in prod (defaults to `/api` locally).

## Backend
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
```
Provide EE creds via `GOOGLE_APPLICATION_CREDENTIALS` (or upload through `/api/auth/service-account` in the UI). Ensure `public/data/fertimap_grid.csv` and `server/fertimap_rf_models.joblib` are present; retrain surrogates with:
```bash
python - <<'PY'
from server.fertimap_service import train_local_models
train_local_models()
PY
```

## Deploy hints
- Frontend: build `npm run build` (output `dist`), serve statically; set `VITE_API_BASE=https://your-backend`.
- Backend: deploy `uvicorn server.main:app` (Railway/Fly/Render/etc.), install `server/requirements.txt`, keep model/data files accessible, add EE auth env vars.
