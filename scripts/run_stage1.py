#!/usr/bin/env python
"""Stage 1: datasets, leak-free knowledge base, retrieval quality, Baseline A (LLM only) and B (hybrid RAG)."""
from _cli import base_parser, guarded, load

ap = base_parser(__doc__)
ap.add_argument("--n", type=int, help="questions per dataset (default: stage1.n_per_dataset)")
args = ap.parse_args()
cfg = load(args)
if args.n:
    cfg.stage1.n_per_dataset = args.n
from halluguard.experiments import stage1  # noqa: E402

guarded(lambda: stage1.run(cfg))
