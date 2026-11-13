#!/usr/bin/env python3
"""Score saved GeoDatum generations with AuditKIT and compare before vs after.

No GPU needed: it reads before.json / after.json written by score.py (same test rows).
For each metric, AuditKIT's exact_match compares a short normalised outcome string
(what the model did) with the expected string (what the converter says it should do).
Generation and parsing are GeoDatum's own code; scoring and comparison are AuditKIT's.

  python audit_eval.py --test test.jsonl --before before.json --after after.json --out audit_results.json

Safety checks (added after a caching bug made "after" repeat the "before" numbers):
  * every ak.evaluate call runs with cache=False and a distinct model name per run;
  * each AuditKIT score is re-computed directly and must match, otherwise the script stops;
  * each AuditKIT score is compared row by row with the flags score.py saved.
"""
import argparse
import json
import re
import sys

import auditkit as ak

from score import parse_call

METRICS = ("tool_args_correct", "end_to_end_correct", "ask_correct", "clean_stop_rate")
SCOREPY_FLAG = {"tool_args_correct": "args_ok", "end_to_end_correct": "final_ok",
                "ask_correct": "ask_ok", "clean_stop_rate": "clean"}


def fmt(x):
    return f"{float(x):.6f}"


def call_sig(call):
    if not call or call.get("name") != "convert_to_latlon":
        return "NOCALL"
    a = call.get("arguments") or {}
    try:
        return f"CALL {a.get('grid')} {a.get('datum')} {fmt(a['easting'])} {fmt(a['northing'])}"
    except Exception:
        return "BADCALL"


def views(row, out):
    """{metric: (what the model did, what it should have done)} for the metrics that apply."""
    meta = row["meta"]
    lat, lon = str(meta["expected_lat"]), str(meta["expected_lon"])
    v = {"clean_stop_rate": ("CLEAN" if out["clean"] else "UNCLEAN", "CLEAN")}
    if meta["scenario"] in ("complete_inside", "complete_outside"):
        want_call = f"CALL {meta['grid']} Minna {fmt(meta['easting'])} {fmt(meta['northing'])}"
        got = call_sig(parse_call(out["reply1"]))
        v["tool_args_correct"] = (got, want_call)
        r2 = out.get("reply2")
        if r2 is None:
            final = "NOFINAL"
        else:
            final = ("COORDS" if lat in r2 and lon in r2 else "NOCOORDS") + " " + \
                    ("WARN" if "outside" in r2.lower() else "NOWARN")
        want = "COORDS " + ("NOWARN" if meta["expected_inside_bbox"] else "WARN")
        v["end_to_end_correct"] = (got + " " + final, want_call + " " + want)
    else:
        r1 = out["reply1"]
        if parse_call(r1) is not None:
            shown = "CALL"
        elif "?" in r1 and lat not in r1 and lon not in r1:
            shown = "ASK"
        else:
            shown = "NOASK"
        v["ask_correct"] = (shown, "ASK")
    return v


def build(rows, outs, metric):
    """Samples for AuditKIT plus what the model did, the target, and score.py's own flag per row."""
    samples, lookup, targets, flags = [], {}, {}, {}
    for i, (row, out) in enumerate(zip(rows, outs)):
        v = views(row, out).get(metric)
        if v:
            key = f"ROW{i:05d}"
            samples.append(ak.Sample(input=key, target=v[1]))
            lookup[key], targets[key] = v[0], v[1]
            flags[key] = bool(out.get(SCOREPY_FLAG[metric]))
    return samples, lookup, targets, flags


def make_model(lookup, name):
    def model(prompts):
        res = []
        for p in prompts:
            m = re.search(r"ROW\d{5}", p)
            res.append(lookup.get(m.group(0), "") if m else "")
        return res
    # AuditKIT fingerprints the callable by name; give each run its own identity.
    model.__name__ = model.__qualname__ = name
    return model


def headline_value(res):
    h = res.headline
    v = h.get("exact_match", next(iter(h.values()))) if isinstance(h, dict) else h
    if isinstance(v, dict):
        v = v.get("mean", next(iter(v.values())))
    return float(v)


def run_metric(label, metric, rows, outs):
    samples, lookup, targets, flags = build(rows, outs, metric)
    if not samples:
        return None
    res = ak.evaluate(samples, make_model(lookup, f"geodatum_{label}_{metric}"),
                      ["exact_match"], cache=False)
    n = len(samples)
    audit = headline_value(res)
    direct = sum(lookup[k] == targets[k] for k in lookup) / n
    if abs(audit - direct) > 1e-9:
        sys.exit(f"STOP: AuditKIT gave {audit:.4f} for {label}/{metric} but a direct count gives "
                 f"{direct:.4f}. Do not use these numbers; paste this message to Claude.")
    disagree = sum((lookup[k] == targets[k]) != flags[k] for k in lookup)
    return {"res": res, "n": n, "audit": audit, "direct": direct,
            "scorepy": sum(flags.values()) / n, "disagree": disagree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--before", default="before.json")
    ap.add_argument("--after", default="after.json")
    ap.add_argument("--out", default="audit_results.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    data = {k: json.load(open(p))["rows"] for k, p in (("before", args.before), ("after", args.after))}
    n = len(data["before"])
    assert len(data["after"]) == n, "before and after must cover the same test rows"
    rows = rows[:n]

    runs, table = {"before": {}, "after": {}}, {}
    for label, outs in data.items():
        for m in METRICS:
            r = run_metric(label, m, rows, outs)
            if r:
                runs[label][m] = r

    print(f"\nAuditKIT exact_match on {n} test rows")
    print(f"{'metric':22s}{'before':>9s}{'after':>9s}{'n':>6s}   check vs score.py")
    for m in METRICS:
        if m in runs["before"] and m in runs["after"]:
            b, a = runs["before"][m], runs["after"][m]
            bad = b["disagree"] + a["disagree"]
            check = "same on every row" if bad == 0 else f"DIFFERS on {bad} row(s)"
            table[m] = {"before": b["audit"], "after": a["audit"], "n": b["n"],
                        "scorepy_before": b["scorepy"], "scorepy_after": a["scorepy"],
                        "rows_differing_from_scorepy": bad}
            print(f"{m:22s}{b['audit']:9.1%}{a['audit']:9.1%}{b['n']:6d}   {check}")
    for m in table:
        try:
            print(f"\n--- ak.compare: {m} ---")
            print(ak.compare(runs["before"][m]["res"], runs["after"][m]["res"]).summary())
        except Exception as e:
            print("compare failed:", repr(e))
    json.dump(table, open(args.out, "w"), indent=1)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
