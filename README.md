# AI security + evals + W&B demo

The same support assistant, with and without guardrails. Try attacks live, run a labeled eval suite on both modes, compare the results, and track both runs in Weights & Biases.

```
User prompt → input guard → model → tool policy → output check → eval → W&B
```

| File | What it shows |
|---|---|
| `security.py` | The app and its defenses: hardened vs naive system prompt, LLM input guard, email tool policy, and canary secrets with an exact-match leak check |
| `evals.py` | 14 labeled cases in 6 categories, three scores per case (guard correct, safe, judge pass), summary metrics |
| `tracking.py` | One W&B run per mode, grouped per comparison: metrics, a table of every case, pass rate by category, confusion matrix |
| `app.py` | Streamlit UI on one page: security test, eval results, charts, W&B links |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Without `OPENAI_API_KEY` the app runs as a **guardrail simulation**: the input guard is a keyword filter, Protected replies are placeholders, and Vulnerable replies are scripted (marked `[Simulated]`) to show what a leak or a rogue email looks like. Leak and email counts in simulation are therefore staged, not measured, and only the guard decision is scored. With a key, a real model answers and an LLM judge grades each outcome.

If the judge model is unavailable, the guard fails closed and blocks every request. The app shows a judge-error warning when that happens, so check `JUDGE_MODEL` with one live request before presenting.

Keys are read from `.streamlit/secrets.toml` or environment variables:

```toml
OPENAI_API_KEY = "sk-..."
WANDB_API_KEY = "..."                    # optional
WANDB_PROJECT = "ai-security-evals"      # optional
WANDB_ENTITY = "your-username"           # optional: W&B user or team that receives the runs
```

## Demo script

1. **Security test**: pick *Poisoned document* and click **Send to both modes**. Both pipelines run at once and appear side by side. In Protected, the input guard stops the request. In Vulnerable, every defense is off, and the model may email the customer's data to `evil.example` without writing any reply, so the user never sees it happen.
2. **Run evals**: *Quick eval* runs 6 representative cases for a fast, cheap live demo; *Run full eval suite* runs all 14. Both run in both modes.
3. **Charts**: headline pass rate, where attacks were stopped, pass rate by category, and the guard's confusion matrix.
4. **W&B**: open the two runs. They share a group and differ in `config.mode`, so W&B compares them side by side.

In simulation mode the Spanish jailbreak gets past the keyword filter. That's intended: it shows why the live guard is an LLM.
