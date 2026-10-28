"""Run an experiment and score it (see ``ovannot.runner.run`` for the run folder layout).

Usage::

    uv run python scripts/run.py --exp configs/experiments/<name>.yaml [--subset dev|test]
        [--limit N] [--workers K] [--only DATASET ...] [key.sub=value ...]

Re-running the same command resumes the run: images already annotated are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ovannot.runner.run import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True, help="experiment yaml")
    parser.add_argument("--subset", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="first N images per dataset")
    parser.add_argument("--workers", type=int, default=8, help="images annotated in parallel")
    parser.add_argument("--only", nargs="+", metavar="DATASET", help="restrict the datasets")
    parser.add_argument("overrides", nargs="*", help="experiment overrides, key.sub=value")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run_dir = run_experiment(
        args.exp, args.subset, args.limit, args.workers, args.only, args.overrides
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
