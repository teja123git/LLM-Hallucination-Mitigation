#!/usr/bin/env python
"""Stage 2: hallucination detection benchmarks (HaluEval / RAGTruth / FEVER), calibration, ablations, error analysis."""
from _cli import base_parser, guarded, load

ap = base_parser(__doc__)
ap.add_argument("--n-dev", type=int, help="HaluEval dev questions (calibration)")
ap.add_argument("--n-test", type=int, help="HaluEval test questions")
ap.add_argument("--skip-fever", action="store_true")
ap.add_argument("--skip-ragtruth", action="store_true")
ap.add_argument("--reuse-halueval", action="store_true", help="re-use results/stage2/features_halueval.jsonl (no LLM calls for the HaluEval part)")
args = ap.parse_args()
cfg = load(args)
if args.reuse_halueval:
    cfg.stage2.reuse_halueval = True
if args.n_dev:
    cfg.stage2.halueval.n_dev = args.n_dev
if args.n_test:
    cfg.stage2.halueval.n_test = args.n_test
if args.skip_fever:
    cfg.stage2.fever.n = 0
if args.skip_ragtruth:
    cfg.stage2.ragtruth.n = 0
from halluguard.experiments import stage2  # noqa: E402

guarded(lambda: stage2.run(cfg))
