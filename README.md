# FastAPI Model Serving Template

A production-minded starter service for serving a small scikit-learn classifier through
FastAPI. It includes health and metadata endpoints, request validation, and a stable
prediction schema that can be swapped for a PyTorch or ONNX model.

## What it demonstrates

- Typed request and response contracts
- Model initialization separated from HTTP routes
- Health and metadata endpoints for deployment probes
- A local deterministic model with no external dataset download

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
uvicorn src.api:app --reload
~~~

Use the interactive API docs at http://127.0.0.1:8000/docs.

Example request body:

~~~json
{"features": [0.2, -1.1, 0.7, 1.4]}
~~~

## Structure

- src/model.py — local classifier wrapper
- src/api.py — FastAPI application

## License

Apache-2.0
