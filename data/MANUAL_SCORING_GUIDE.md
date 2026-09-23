# Manual summarization scoring

After the real API run, review each summarization output.

- Faithfulness: 1–5
- Fluency: 1–5

Fill `manual_summary_scores.csv`, then run:

```powershell
python -m src.scoring --manual-summary data/manual_summary_scores.csv
```

Do not invent scores before the model has actually been run.
