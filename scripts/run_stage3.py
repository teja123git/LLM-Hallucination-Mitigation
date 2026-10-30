#!/usr/bin/env python
"""Stage 3: controlled ablation A-G (+F-CoVe): generation, verification, correction, judge grading, statistics, figures."""
from _cli import base_parser, guarded, load

ap = base_parser(__doc__)
ap.add_argument("--n", type=int, help="questions per dataset (default: stage3.n_per_dataset)")
ap.add_argument("--systems", help="comma-separated subset, e.g. A,B,E,F  (A,B,C,D,E,F,G,F_cove)")
ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (judge-free metrics only; saves the judge model's quota)")
args = ap.parse_args()
cfg = load(args)
from halluguard.experiments import stage3  # noqa: E402

guarded(lambda: stage3.run(cfg, systems=args.systems.split(",") if args.systems else None, n=args.n, use_judge=False if args.no_judge else None))
