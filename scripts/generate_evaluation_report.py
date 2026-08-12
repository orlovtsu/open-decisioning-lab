import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decisioning.evaluation import build_holdout_report
from decisioning.model import train_model
from decisioning.reporting import build_visual_report
from decisioning.synthetic import SyntheticConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a synthetic holdout evaluation report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--output", type=Path, default=Path("reports/holdout_report.json"))
    args = parser.parse_args()

    config = SyntheticConfig(seed=args.seed, rows=args.rows)
    model = train_model(config)
    report = build_holdout_report(model, config)
    build_visual_report(model, config, args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "rows": args.rows,
                "overall": report.overall,
                "thresholds": report.thresholds,
                "cohorts": report.cohorts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
