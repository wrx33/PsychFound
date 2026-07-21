from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from psychfound.config import load_config, resolve_path

from .grpo import train_grpo
from .sft import train_sft


def run_pipeline(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    start_at: str | None = None,
    stop_after: str | None = None,
) -> list[dict[str, object]]:
    source = Path(config_path).resolve()
    root = source.parent.parent
    config = load_config(source, strict_env=False)
    stages = config.get("stages", [])
    names = [stage["name"] for stage in stages]
    if start_at and start_at not in names:
        raise ValueError(f"unknown start stage: {start_at}")
    if stop_after and stop_after not in names:
        raise ValueError(f"unknown stop stage: {stop_after}")
    active = start_at is None
    results = []
    for stage in stages:
        if stage["name"] == start_at:
            active = True
        if not active or not stage.get("enabled", True):
            continue
        stage_config = resolve_path(stage["config"], root)
        trainer = train_sft if stage["type"] == "sft" else train_grpo
        result = trainer(stage_config, dry_run=dry_run)
        result["name"] = stage["name"]
        results.append(result)
        if stage.get("export_model_as"):
            os.environ[stage["export_model_as"]] = str(Path(str(result["model_output"])).resolve())
        if stage["name"] == stop_after:
            break
    if not dry_run and int(os.environ.get("RANK", "0")) == 0:
        run_dir = resolve_path(config.get("run_dir", "outputs/runs"), root)
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (run_dir / f"run-{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results
