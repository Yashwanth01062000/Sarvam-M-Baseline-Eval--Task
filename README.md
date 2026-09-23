# Sarvam-M Baseline Evaluation

This project implements the Sarvam-M baseline evaluation described in the internship task.

## Scope

- 100 prompts total
- 25 factual QA
- 25 translation
- 25 summarization
- 25 reasoning/math
- English, Hindi, and Tamil
- Temperature fixed at 0

## Important

The assignment requires checking the current Sarvam API endpoint and model name before a live run. The defaults in `.env.example` are based on the starter assignment and must be verified against current official documentation.

The repository does not contain a real API key.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set your real key locally:

```text
SARVAM_API_KEY=YOUR_REAL_KEY
```

Never commit `.env`.

## Validate

```powershell
python -m pytest
```

## Test API connection

```powershell
python -m src.run_eval --test-connection
```

## Run all 100 prompts

```powershell
python -m src.run_eval
```

This creates `data/results.csv`.

## Score

```powershell
python -m src.scoring
```

Translation gets chrF automatically. Factual QA and reasoning/math get automatic binary scoring.

### Summarization manual scoring

Create `data/manual_summary_scores.csv` with:

```text
id,faithfulness_score,fluency_score,review_notes
```

Enter 1-5 scores for the summarization rows, then run:

```powershell
python -m src.scoring --manual-summary data/manual_summary_scores.csv
```

## Analysis

```powershell
python -m src.analyze
```

Creates:

- `data/summary.csv`
- `data/failures.csv`

## One-command pipeline

After the API connection is verified:

```powershell
python scripts/run_all.py
```

Note: summarization scores are blank until manual scores are supplied.

## Outputs

- `data/prompts.csv`
- `data/results.csv`
- `data/scores.csv`
- `data/summary.csv`
- `data/failures.csv`
- `reports/report.md`

## Security

- API key comes from `SARVAM_API_KEY`.
- `.env` is ignored by Git.
- The key is never written to result CSV files.
- Do not paste a real key into source code.

## Acceptance checks

The project validates 100 rows, unique IDs, four task categories with 25 rows each, required CSV columns, and traceability from scores back to result IDs.
