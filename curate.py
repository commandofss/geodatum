#!/usr/bin/env python3
"""Curate train.jsonl with CuratorKIT, then verify the export is safe to train on.

Usage: python curate.py [input.jsonl] [output_dir]
Checks: row count kept, every <tool_call> survived, no message text changed.
"""
import json
import sys

from curatorkit import Curator, CuratorConfig

SRC = sys.argv[1] if len(sys.argv) > 1 else "train.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output/clean_messages"

Curator(CuratorConfig(
    dataset=SRC,
    clean=True,
    clean_transforms={"strip_html": False},  # default strips <tool_call>/<tool_result> tags
    dedup="exact",
    export_formats=["messages"],
    output_dir=OUT,
)).run().print_summary()

load = lambda p, key: [json.loads(l)[key] for l in open(p, encoding="utf-8")]
raw = load(SRC, "messages")
out = load(f"{OUT}/sft_messages.jsonl", "messages")

changed = sum([m["content"] for m in r] != [m["content"] for m in o]
              for r, o in zip(raw, out))
calls = lambda rows: sum("<tool_call>" in json.dumps(r) for r in rows)
print(f"rows in/out: {len(raw)}/{len(out)} | rows with <tool_call> in/out: {calls(raw)}/{calls(out)}"
      f" | rows with changed text: {changed}")
ok = len(raw) == len(out) and calls(raw) == calls(out) and changed == 0
print("CURATION CHECK:", "PASS" if ok else "FAIL (do not train on this export)")
sys.exit(0 if ok else 1)
