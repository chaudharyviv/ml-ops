"""Weights & Biases logging: one W&B run per mode, grouped per comparison.

In W&B, open the project, group by "group" (or filter on config.mode), and the
Protected and Vulnerable runs of each comparison sit side by side.
"""
from __future__ import annotations

from typing import Any

import evals
import security


def log_run(
    api_key: str,
    project: str,
    group: str,
    mode: str,
    live: bool,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    entity: str = "",
) -> dict[str, str]:
    """Returns {"url": ...} on success or {"error": ...}."""
    if not api_key:
        return {"error": "W&B not configured: set WANDB_API_KEY to log runs."}
    try:
        import wandb

        # The key goes straight to this run. wandb.login() would also copy it
        # into ~/.netrc (plain text) and append it again on every run.
        with wandb.init(
            entity=entity or None,   # None = the key's default team
            project=project,
            group=group,
            name=f"{mode.lower()}-{group}",
            job_type="eval",
            reinit="finish_previous",
            settings=wandb.Settings(api_key=api_key, silent=True),
            # Config is what W&B filters and groups on: put the experiment's knobs here
            config={
                "mode": mode,
                "protected": mode == "Protected",
                "live": live,
                "app_model": security.APP_MODEL,
                "judge_model": security.JUDGE_MODEL,
                "prompt_version": security.PROMPT_VERSION,
                "n_cases": len(results),
            },
        ) as run:
            cases = wandb.Table(columns=[
                "case_id", "category", "attack", "prompt", "response",
                "blocked", "stopped_by", "leaks", "unauthorized", "passed", "judge_reason",
            ])
            for r in results:
                cases.add_data(
                    r["case_id"], r["category"], r["attack"], r["prompt"], r["response"],
                    r["blocked"], r["stopped_by"], ", ".join(r["leaks"]), r["unauthorized"],
                    r["passed"], r["judge_reason"],
                )

            by_category = wandb.Table(columns=["category", "pass_rate"])
            for cat in evals.CATEGORIES:
                rows = [r for r in results if r["category"] == cat]
                if not rows:   # a subset (e.g. the quick eval) may not cover every category
                    continue
                by_category.add_data(cat, 100 * sum(r["passed"] for r in rows) / len(rows))

            run.log({
                **{k: v for k, v in summary.items() if v is not None},
                "cases": cases,
                "pass_rate_by_category": wandb.plot.bar(
                    by_category, "category", "pass_rate", title=f"Pass rate by category ({mode})"
                ),
                "guard_confusion": wandb.plot.confusion_matrix(
                    y_true=[int(r["attack"]) for r in results],
                    preds=[int(r["blocked"]) for r in results],
                    class_names=["benign", "attack"],
                    title=f"Input guard: actual vs blocked ({mode})",
                ),
            })
            return {"url": run.url}
    except Exception as exc:
        return {"error": f"W&B logging failed: {exc}"}
