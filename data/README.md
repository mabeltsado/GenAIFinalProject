# data/

Place your survey CSV files here. **Do not commit real customer data.**

## Expected CSV schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `review_id` | string | ✅ | Unique identifier (e.g. `R001`) |
| `review_text` | string | ✅ | The free-text customer review |
| `rating` | integer 1–5 | ✅ | Numeric satisfaction rating |
| `customer_segment` | string | optional | e.g. `residential`, `commercial` |
| `date` | string / date | optional | Submission date |

## Sample data

A built-in 30-row synthetic dataset is available without uploading a file.
Click **"Load Sample Dataset"** in the app to use it.

## Generating your own synthetic data

You can use Claude or any LLM to generate synthetic survey reviews. Example prompt:

> Generate 50 synthetic electricity customer survey responses in CSV format with columns:
> review_id, review_text, rating (1-5), customer_segment.
> Cover billing, outages, the mobile app, smart meters, and customer service.
> Use only fictional details — no real names, addresses, or account numbers.
