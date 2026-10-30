#!/usr/bin/env python
"""Run Stage 1 -> 2 -> 3 and write results/REPORT.md.  Resumable (LLM calls are cached)."""
from _cli import base_parser, guarded, load

ap = base_parser(__doc__)
ap.add_argument("--budget-only", action="store_true", help="print the rough token budget and exit")
args = ap.parse_args()
cfg = load(args)
from halluguard.budget import print_budget  # noqa: E402

print_budget(cfg)
if args.budget_only:
    raise SystemExit(0)
from halluguard.experiments import stage1, stage2, stage3  # noqa: E402
from halluguard.experiments.common import load_resources  # noqa: E402
from halluguard.report import build_report  # noqa: E402


def main():
    res = load_resources(cfg)
    stage1.run(cfg, res)
    stage2.run(cfg, res)
    stage3.run(cfg, res)
    print("\nReport:", build_report(cfg))


guarded(main)
