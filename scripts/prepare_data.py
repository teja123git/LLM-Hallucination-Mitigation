#!/usr/bin/env python
"""Download / validate all datasets and pre-build the knowledge base (optional - everything also happens on demand)."""
from _cli import base_parser, load

ap = base_parser(__doc__)
ap.add_argument("--check", action="store_true", help="print the schema and basic statistics of every dataset (use this if a loader complains)")
args = ap.parse_args()
cfg = load(args)
from halluguard import data as D  # noqa: E402

hv, tq = D.load_halueval_qa(cfg), D.load_truthfulqa(cfg)
print(f"HaluEval-QA : {len(hv):>6} questions | columns {list(hv[0])}")
print(f"TruthfulQA  : {len(tq):>6} questions | with a Wikipedia source: {sum(bool(x['wiki_titles']) for x in tq)} "
      f"({len({t for x in tq for t in x['wiki_titles']})} unique pages)")
rt = D.load_ragtruth(cfg, "test", ("QA",))
print(f"RAGTruth    : {len(rt):>6} test QA responses | hallucinated: {sum(x['hallucinated'] for x in rt)}")
try:
    fv = D.load_fever(cfg)
    from collections import Counter
    print(f"FEVER       : {len(fv):>6} claims | {dict(Counter(x['label'] for x in fv))} | with gold evidence: {sum(bool(x['evidence']) for x in fv)}")
    if args.check:
        ex = next(x for x in fv if x["evidence"])
        print("  example:", ex["claim"], "->", ex["label"], "|", ex["evidence"][:2])
except Exception as e:
    print(f"FEVER       : unavailable ({e}). Stage 2 will skip it (or set stage2.fever.n=0).")
if not args.check:
    from halluguard.experiments.common import load_resources
    res = load_resources(cfg)
    print(f"Knowledge base built: {res.retriever.size} passages; Wikipedia pages for TruthfulQA: {len(res.wiki_pages)}")
