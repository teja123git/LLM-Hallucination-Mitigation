#!/usr/bin/env python
"""Create the tiny offline datasets in data/toy (used by mock mode / tests).  Needs the real datasets in data/raw
(they are downloaded on first use).  Excerpts keep the original schemas; FEVER-like claims are synthetic."""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from halluguard.config import load_config  # noqa: E402
from halluguard import data as D  # noqa: E402
from halluguard.utils import split_sentences, write_jsonl  # noqa: E402

cfg = load_config()
rng = random.Random(7)
out = ROOT / "data" / "toy"

hv = D.load_halueval_qa(cfg)
write_jsonl(out / "halueval_qa.jsonl", rng.sample(hv, 120))

tq = [t for t in D.load_truthfulqa(cfg)]
write_jsonl(out / "truthfulqa.jsonl", rng.sample(tq, 40))

rt = D.load_ragtruth(cfg, "test", ("QA",))
pos = [r for r in rt if r["hallucinated"]]
neg = [r for r in rt if not r["hallucinated"]]
sel = rng.sample(pos, 20) + rng.sample(neg, 30)
for r in sel:
    r.pop("prompt", None)
write_jsonl(out / "ragtruth_qa.jsonl", sel)

# synthetic FEVER-like claims built from HaluEval paragraphs
rows = []
pool = rng.sample(hv, 60)
for i, it in enumerate(pool[:40]):
    sents = split_sentences(it["knowledge"])
    if not sents:
        continue
    s = sents[0]
    kind = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"][i % 3]
    if kind == "SUPPORTS":
        rows.append({"id": f"fever_toy_{i}", "claim": s, "label": kind, "evidence": [{"title": "", "text": s}], "source": "fever"})
    elif kind == "REFUTES":
        rows.append({"id": f"fever_toy_{i}", "claim": "It is not true that " + s[0].lower() + s[1:], "label": kind, "evidence": [{"title": "", "text": s}], "source": "fever"})
    else:
        other = split_sentences(pool[-1 - i]["knowledge"])
        rows.append({"id": f"fever_toy_{i}", "claim": (other or [s])[0], "label": kind, "evidence": [], "source": "fever"})
write_jsonl(out / "fever.jsonl", rows)
print({p.name: sum(1 for _ in open(p)) for p in out.glob("*.jsonl")})
