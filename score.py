#!/usr/bin/env python3
"""GeoDatum scorer. Runs a model on test.jsonl and checks every answer against the converter.

Use the SAME command for the untrained model (before) and the trained model (after):
  python score.py --test test.jsonl --label before --limit 100 --out before.json
  python score.py --test test.jsonl --label after --adapter ./geodatum_curated --limit 100 --out after.json

Metrics (all checked automatically, no judge model):
  tool_args_correct   complete requests: valid <tool_call> with the right grid, datum, easting, northing
  end_to_end_correct  complete requests: right call AND final answer has the converter's lat/lon
                      (and warns 'outside Nigeria' exactly when the converter says so)
  ask_correct         missing datum/grid or unsupported zone: asks, makes no tool call, gives no coordinates
  clean_stop_rate     every generation ended on an end-of-sequence token (did not ramble to the length limit)
"""
import argparse
import json
import re
from collections import defaultdict

from geodatum_converter import convert_to_latlon
from generate_dataset import SYSTEM_PROMPT

CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
COMPLETE = ("complete_inside", "complete_outside")


def parse_call(text):
    m = CALL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return {"_invalid": True}


def args_ok(call, meta):
    if not call or call.get("name") != "convert_to_latlon":
        return False
    a = call.get("arguments") or {}
    try:
        return (a.get("grid") == meta["grid"] and a.get("datum") == "Minna"
                and abs(float(a["easting"]) - meta["easting"]) < 1e-6
                and abs(float(a["northing"]) - meta["northing"]) < 1e-6)
    except Exception:
        return False


def score_row(row, gen):
    meta, user = row["meta"], row["messages"][0]["content"]
    hist = [{"role": "user", "content": user}]
    reply1, clean1 = gen(hist)
    call = parse_call(reply1)
    out = {"scenario": meta["scenario"], "error_type": meta.get("error_type", "none"),
           "clean": clean1, "reply1": reply1}
    lat, lon = str(meta["expected_lat"]), str(meta["expected_lon"])
    if meta["scenario"] in COMPLETE:
        out["args_ok"] = args_ok(call, meta)
        out["final_ok"] = False
        if out["args_ok"]:
            a = call["arguments"]
            res = convert_to_latlon(a["easting"], a["northing"], a["grid"], "Minna")
            hist += [{"role": "assistant", "content": reply1},
                     {"role": "user", "content": "<tool_result>" + json.dumps(res) + "</tool_result>"}]
            reply2, clean2 = gen(hist)
            out["clean"] = clean1 and clean2
            out["reply2"] = reply2
            low = reply2.lower()
            says_outside = "outside" in low
            out["final_ok"] = (lat in reply2 and lon in reply2
                               and says_outside == (not meta["expected_inside_bbox"]))
    else:
        out["ask_ok"] = call is None and "?" in reply1 and lat not in reply1 and lon not in reply1
    return out


def rate(rows, key):
    xs = [r[key] for r in rows if key in r]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


def summarize(results):
    comp = [r for r in results if r["scenario"] in COMPLETE]
    ask = [r for r in results if r["scenario"] not in COMPLETE]
    summary = {}
    for name, rows, key in (("tool_args_correct", comp, "args_ok"),
                            ("end_to_end_correct", comp, "final_ok"),
                            ("ask_correct", ask, "ask_ok"),
                            ("clean_stop_rate", results, "clean")):
        v, n = rate(rows, key)
        summary[name] = {"value": v, "n": n}
    by = defaultdict(list)
    for r in results:
        by[r["scenario"]].append(r)
    summary["by_scenario"] = {}
    for sc, rows in sorted(by.items()):
        key = "final_ok" if sc in COMPLETE else "ask_ok"
        v, n = rate(rows, key)
        summary["by_scenario"][sc] = {"value": v, "n": n}
    outs = defaultdict(list)
    for r in comp:
        if r["scenario"] == "complete_outside":
            outs[r["error_type"]].append(r)
    summary["outside_by_error_type"] = {k: dict(zip(("value", "n"), rate(v, "final_ok")))
                                        for k, v in sorted(outs.items())}
    return summary


class HFModel:
    def __init__(self, adapter=None, model_name="CohereLabs/tiny-aya-global", max_new=200):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.max_new = max_new
        self.tok = AutoTokenizer.from_pretrained(model_name)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        named = self.tok.convert_tokens_to_ids("<EOS_TOKEN>")
        end_resp = self.tok.convert_tokens_to_ids("<|END_RESPONSE|>")
        ids = {self.tok.eos_token_id}
        for t in (named, end_resp):
            if t is not None and t != self.tok.unk_token_id:
                ids.add(t)
        self.eos_ids = sorted(i for i in ids if i is not None)
        print("tokenizer eos id:", self.tok.eos_token_id, "| <EOS_TOKEN> id:", named,
              "| <|END_RESPONSE|> id:", end_resp, "| stopping on:", self.eos_ids)

    def __call__(self, messages):
        chat = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        enc = self.tok.apply_chat_template(chat, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                  eos_token_id=self.eos_ids,
                                  pad_token_id=self.tok.pad_token_id or self.eos_ids[0])
        new = out[0][enc["input_ids"].shape[1]:]
        clean = bool(len(new)) and new[-1].item() in self.eos_ids
        return self.tok.decode(new, skip_special_tokens=True).strip(), clean


class StubModel:
    """Stand-ins used only to test the scorer itself."""
    def __init__(self, kind, rows):
        self.kind, self.rows = kind, {r["messages"][0]["content"]: r for r in rows}

    def __call__(self, messages):
        row = self.rows[messages[0]["content"]]
        if self.kind == "gold":                       # replays the reference answers
            return row["messages"][len(messages)]["content"], True
        if self.kind == "asker":                      # always asks, never calls the tool
            return "Which grid and datum are these on?", True
        return "That is about latitude 8.1, longitude 3.9.", False   # guesser: no tool, no asking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--label", default="run")
    ap.add_argument("--adapter")
    ap.add_argument("--model", default="hf", choices=["hf", "stub-gold", "stub-asker", "stub-guesser"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    gen = HFModel(args.adapter) if args.model == "hf" else StubModel(args.model.split("-")[1], rows)
    results = []
    for i, row in enumerate(rows, 1):
        results.append(score_row(row, gen))
        if i % 20 == 0:
            print(f"  scored {i}/{len(rows)}", flush=True)
    s = summarize(results)
    print(f"\n=== {args.label}: {len(rows)} test rows ===")
    for k in ("tool_args_correct", "end_to_end_correct", "ask_correct", "clean_stop_rate"):
        v = s[k]["value"]
        print(f"{k:20s} {'n/a' if v is None else f'{v:.1%}':>7s}   (n={s[k]['n']})")
    print("by scenario:", {k: (None if v['value'] is None else round(v['value'], 3), v['n'])
                           for k, v in s["by_scenario"].items()})
    print("outside rows by mistake type:", {k: (round(v['value'], 3), v['n'])
                                            for k, v in s["outside_by_error_type"].items()})
    if args.out:
        json.dump({"label": args.label, "adapter": args.adapter, "summary": s, "rows": results},
                  open(args.out, "w"), indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
