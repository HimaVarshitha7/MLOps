# Task 6 — Observations

1. **Online vs Batch Inference:** Online inference predicts one record when a result is needed. Batch inference predicts a group of records together.
2. **Streaming vs Batch Inference:** Streaming processes incoming records one at a time; batch inference processes a fixed collection together. This project simulates streaming with a one-second delay.
3. **Why FastAPI?** FastAPI exposes predictions through HTTP endpoints, validates input with Pydantic, and provides interactive Swagger docs at `/docs`.
4. **Why reuse the same trained model?** Reusing `model.pkl` keeps preprocessing and predictions consistent across online, batch, streaming, and API inference and avoids unnecessary retraining.
