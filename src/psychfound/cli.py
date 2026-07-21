from __future__ import annotations

import argparse
import json

from .data import prepare_rl_data, validate_sharegpt
from .evaluation import evaluate_predictions
from .inference import generate
from .training import merge_adapter, run_pipeline, train_grpo, train_sft


def _show(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psychfound", description="PsychFound research pipeline")
    parser.add_argument("--version", action="version", version="PsychFound 1.0.0")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-data")
    validate.add_argument("--input", required=True)

    prepare = commands.add_parser("prepare-rl")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--task", choices=["diagnosis", "medication"], required=True)
    prepare.add_argument("--precision", choices=["category", "subtype"], required=True)
    prepare.add_argument("--train-size", type=int, required=True)
    prepare.add_argument("--test-size", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-input-chars", type=int, default=320)

    for name in ("train-sft", "train-grpo"):
        train = commands.add_parser(name)
        train.add_argument("--config", required=True)
        train.add_argument("--dry-run", action="store_true")

    merge = commands.add_parser("merge-adapter")
    merge.add_argument("--base-model", required=True)
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--output", required=True)

    pipeline = commands.add_parser("pipeline")
    pipeline.add_argument("--config", default="configs/pipeline.yaml")
    pipeline.add_argument("--dry-run", action="store_true")
    pipeline.add_argument("--start-at")
    pipeline.add_argument("--stop-after")

    infer = commands.add_parser("infer")
    infer.add_argument("--config", default="configs/inference.yaml")
    infer.add_argument("--prompt", required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-data":
            _show(validate_sharegpt(args.input))
        elif args.command == "prepare-rl":
            _show(prepare_rl_data(args.input, args.output, task=args.task, precision=args.precision,
                                  train_size=args.train_size, test_size=args.test_size, seed=args.seed,
                                  max_input_chars=args.max_input_chars))
        elif args.command == "train-sft":
            _show(train_sft(args.config, dry_run=args.dry_run))
        elif args.command == "train-grpo":
            _show(train_grpo(args.config, dry_run=args.dry_run))
        elif args.command == "merge-adapter":
            _show(merge_adapter(args.base_model, args.adapter, args.output))
        elif args.command == "pipeline":
            _show(run_pipeline(args.config, dry_run=args.dry_run, start_at=args.start_at, stop_after=args.stop_after))
        elif args.command == "infer":
            print(generate(args.config, args.prompt))
        elif args.command == "evaluate":
            _show(evaluate_predictions(args.predictions))
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0
