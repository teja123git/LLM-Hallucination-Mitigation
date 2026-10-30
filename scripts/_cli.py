"""Shared CLI plumbing for the scripts (works without `pip install -e .`)."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from halluguard.config import load_config  # noqa: E402
from halluguard.llm import DailyQuotaExceeded  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", help="extra YAML profile merged over configs/default.yaml (e.g. configs/quick.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. --set llm.model=llama-3.3-70b-versatile")
    ap.add_argument("--mock", action="store_true", help="OFFLINE smoke test with fake LLM/NLI and toy data (numbers are meaningless)")
    return ap


def load(args):
    if args.mock:
        os.environ["HALLUGUARD_MOCK"] = "1"
    return load_config(args.config, args.set, mock=args.mock)


def guarded(fn):
    """Run ``fn``; turn the daily-quota error into a friendly message and exit code 3 (progress is cached)."""
    try:
        return fn()
    except DailyQuotaExceeded as e:
        print(f"\n[!] {e}\nNothing is lost: re-run the same command later and it will continue from the cache.", file=sys.stderr)
        sys.exit(3)
