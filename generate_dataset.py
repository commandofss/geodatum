#!/usr/bin/env python3
"""GeoDatum dataset generator (first version).

Writes JSONL rows: {"messages": [...], "meta": {...}}.
- Tool calls are plain text: <tool_call>{...}</tool_call> in an assistant message.
- Tool results come back as a USER message: <tool_result>{...}</tool_result>.
- No system message is stored. Pass SYSTEM_PROMPT to AlignTune's system_prompt argument.
Usage:
  python generate_dataset.py --n 2000 --seed 1 --out train.jsonl
  python generate_dataset.py --n 300 --seed 2 --out test.jsonl --exclude train.jsonl
"""
import argparse
import json
import random

from geodatum_converter import GRIDS, convert_to_latlon, latlon_to_grid

SYSTEM_PROMPT = (
    "You are GeoDatum, an assistant for Nigerian survey coordinates. "
    "You convert coordinates on the Minna datum to WGS84 latitude and longitude. "
    "The only valid grid names are exactly: UTM 31N, UTM 32N, West Belt, Mid Belt, East Belt. "
    "Do not call the tool until you know the datum, the grid, the easting and the northing. "
    "If the datum or the grid is missing, ask for it; never guess. "
    "If the user names a grid that is not in the list, say it is not available and ask which "
    "listed grid to use. "
    "Never calculate a conversion yourself: call the tool by writing "
    '<tool_call>{"name": "convert_to_latlon", "arguments": {"easting": ..., "northing": ..., '
    '"grid": ..., "datum": "Minna"}}</tool_call>. '
    "When a tool result says inside_nigeria_bbox is false, warn the user that the point is "
    "outside Nigeria and ask them to check the grid and datum."
)

GRID_ALIASES = {
    "UTM 31N": ["UTM zone 31N", "UTM 31N", "UTM zone 31 north"],
    "UTM 32N": ["UTM zone 32N", "UTM 32N", "UTM zone 32 north"],
    "West Belt": ["Nigeria West Belt", "West Belt"],
    "Mid Belt": ["Nigeria Mid Belt", "Mid Belt"],
    "East Belt": ["Nigeria East Belt", "East Belt"],
}
# Rough longitude bands, only to make example inputs look realistic.
LON_RANGE = {"UTM 31N": (2.7, 6.5), "UTM 32N": (5.5, 12.5), "West Belt": (2.7, 6.5),
             "Mid Belt": (6.0, 10.5), "East Belt": (9.5, 14.7)}
LAT_RANGE = (4.3, 13.8)

COORD_TEMPLATES = ["easting {e}, northing {n}", "E {e} N {n}", "X = {e}, Y = {n}", "E: {e}, N: {n}"]
COMPLETE = [
    "Convert {coords}, {grid} on the Minna datum, to latitude and longitude.",
    "{coords}. Minna datum, {grid}. What are the lat/long?",
    "Datum: Minna. Grid: {grid}. Please convert {coords} to latitude/longitude.",
    "I have a beacon at {coords} ({grid}, Minna datum). Where is it in lat/lon?",
]
MISSING_GRID = [
    "Convert {coords} on the Minna datum to latitude and longitude.",
    "Datum is Minna. What is {coords} in lat/long?",
    "Please convert {coords} (Minna) to lat/lon.",
]
MISSING_DATUM = [
    "Convert {coords}, {grid}, to latitude and longitude.",
    "What are the lat/long for {coords} in {grid}?",
    "Please convert {coords} ({grid}) to lat/lon.",
]
MISSING_BOTH = [
    "Convert {coords} to latitude and longitude.",
    "What is {coords} in lat/long?",
    "Please convert {coords} to lat/lon.",
]
ASK_GRID = [
    "Which grid are these coordinates in? The same numbers give different places in "
    "UTM 31N, UTM 32N and the West, Mid and East Belts.",
    "I need the grid before I can convert these: UTM 31N, UTM 32N, or the West, Mid or East Belt?",
]
ASK_DATUM = [
    "Which datum are these coordinates on? I can convert coordinates on the Minna datum.",
    "What datum is this on? I can only convert from the Minna datum, so please confirm it.",
]
ASK_BOTH = [
    "Which datum and which grid are these on? I can convert Minna datum coordinates in "
    "UTM 31N, UTM 32N, or the West, Mid or East Belt.",
]
ASK_UNSUPPORTED = [
    "I don't have Minna datum conversions for UTM zone 33N. I can use UTM 31N, UTM 32N, or the "
    "West, Mid or East Belt. Which grid should I use?",
]
ANSWER_OK = [
    "Using {grid} on the Minna datum, E {e}, N {n} converts to latitude {lat}, longitude {lon} (WGS84).",
    "Result: latitude {lat}, longitude {lon} (WGS84), from {grid}, Minna datum.",
]
ANSWER_OUT = [
    "Warning: this converts to latitude {lat}, longitude {lon} (WGS84), which is outside Nigeria. "
    "Please check that the grid ({grid}) and the datum are correct before using it.",
    "This point falls outside Nigeria (latitude {lat}, longitude {lon}, WGS84). "
    "Check the grid ({grid}) and datum, because the coordinates may belong to a different grid.",
]


def s(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def rnd(x, dec):
    return round(x, 2) if dec else int(round(x))


def u(text):
    return {"role": "user", "content": text}


def a(text):
    return {"role": "assistant", "content": text}


def coords_text(e, n):
    return random.choice(COORD_TEMPLATES).format(e=s(e), n=s(n))


def tool_call(e, n, grid):
    payload = {"name": "convert_to_latlon",
               "arguments": {"easting": e, "northing": n, "grid": grid, "datum": "Minna"}}
    return "<tool_call>" + json.dumps(payload) + "</tool_call>"


def tool_result(res):
    return "<tool_result>" + json.dumps(res) + "</tool_result>"


def final_answer(res, e, n, grid):
    tpl = random.choice(ANSWER_OK if res["inside_nigeria_bbox"] else ANSWER_OUT)
    return tpl.format(grid=grid, e=s(e), n=s(n), lat=res["latitude"], lon=res["longitude"])


def sample_inside(grid=None):
    grid = grid or random.choice(list(GRIDS))
    dec = random.random() < 0.4
    lo, hi = LON_RANGE[grid]
    for _ in range(200):
        lat, lon = random.uniform(*LAT_RANGE), random.uniform(lo, hi)
        e, n = latlon_to_grid(lat, lon, grid)
        e, n = rnd(e, dec), rnd(n, dec)
        res = convert_to_latlon(e, n, grid)
        if "error" not in res and res["inside_nigeria_bbox"]:
            return grid, e, n, res
    raise RuntimeError("could not sample an inside point")


def sample_outside():
    for _ in range(500):
        grid = random.choice(list(GRIDS))
        dec = random.random() < 0.4
        if grid.startswith("UTM"):
            e, n = random.uniform(100000, 900000), random.uniform(0, 2000000)
        else:
            e, n = random.uniform(0, 1300000), random.uniform(0, 1800000)
        e, n = rnd(e, dec), rnd(n, dec)
        res = convert_to_latlon(e, n, grid)
        if "error" not in res and not res["inside_nigeria_bbox"]:
            return grid, e, n, res
    raise RuntimeError("could not sample an outside point")


def drop_leading_digit(x):
    """Simulate a dropped first digit of the whole-number part (keeps any decimals)."""
    ip, _, fp = s(x).partition(".")
    ip = ip[1:].lstrip("0") or None
    if ip is None:
        return None
    return float(ip + "." + fp) if fp else int(ip)


def sample_mistake(kind):
    """Start from a valid Nigerian point, corrupt it the way people do, keep only
    results that land outside the Nigeria box so the warning is truthful.
    kind: swapped (E/N swapped), dropped_digit, wrong_grid (label from another grid)."""
    for _ in range(1000):
        grid, e, n, _res = sample_inside()
        g2 = grid
        if kind == "swapped":
            e2, n2 = n, e
        elif kind == "dropped_digit":
            if random.random() < 0.5:
                e2, n2 = e, drop_leading_digit(n)
            else:
                e2, n2 = drop_leading_digit(e), n
        else:
            g2 = random.choice([g for g in GRIDS if g != grid])
            e2, n2 = e, n
        if e2 is None or n2 is None:
            continue
        res = convert_to_latlon(e2, n2, g2)
        if "error" not in res and not res["inside_nigeria_bbox"]:
            return g2, e2, n2, res
    raise RuntimeError(f"could not sample a {kind} mistake")


def meta(scenario, first_action, grid, e, n, res, missing=None, error_type="none"):
    return {"scenario": scenario, "first_action": first_action, "grid": grid,
            "easting": e, "northing": n, "missing": missing or [],
            "expected_lat": res["latitude"], "expected_lon": res["longitude"],
            "expected_inside_bbox": res["inside_nigeria_bbox"], "error_type": error_type}


def make_complete(inside, kind="noise"):
    if inside:
        (grid, e, n, res), et = sample_inside(), "none"
    elif kind == "noise":
        (grid, e, n, res), et = sample_outside(), "noise"
    else:
        (grid, e, n, res), et = sample_mistake(kind), kind
    alias = random.choice(GRID_ALIASES[grid])
    user = random.choice(COMPLETE).format(coords=coords_text(e, n), grid=alias)
    msgs = [u(user), a(tool_call(e, n, grid)), u(tool_result(res)), a(final_answer(res, e, n, grid))]
    return {"messages": msgs, "meta": meta("complete_inside" if inside else "complete_outside",
                                           "tool_call", grid, e, n, res, error_type=et)}


def finish(msgs, reply, grid, e, n, res):
    """Sometimes continue the conversation after the user answers the question."""
    if random.random() < 0.5:
        msgs += [u(reply), a(tool_call(e, n, grid)), u(tool_result(res)),
                 a(final_answer(res, e, n, grid))]
    return msgs


def make_missing(kind):
    grid, e, n, res = sample_inside()
    alias = random.choice(GRID_ALIASES[grid])
    c = coords_text(e, n)
    if kind == "grid":
        user, ask = random.choice(MISSING_GRID).format(coords=c), random.choice(ASK_GRID)
        reply, missing = random.choice(["It's {g}.", "{g}.", "They are in {g}."]).format(g=alias), ["grid"]
    elif kind == "datum":
        user, ask = random.choice(MISSING_DATUM).format(coords=c, grid=alias), random.choice(ASK_DATUM)
        reply, missing = random.choice(["Minna.", "Minna datum.", "It's the Minna datum."]), ["datum"]
    else:
        user, ask = random.choice(MISSING_BOTH).format(coords=c), random.choice(ASK_BOTH)
        reply, missing = f"Minna datum, {alias}.", ["datum", "grid"]
    msgs = finish([u(user), a(ask)], reply, grid, e, n, res)
    return {"messages": msgs, "meta": meta(f"missing_{kind}", "ask", grid, e, n, res, missing)}


def make_unsupported():
    grid, e, n, res = sample_inside()
    c = coords_text(e, n)
    user = f"Convert {c}, Minna datum, UTM zone 33N, to latitude and longitude."
    reply = random.choice(GRID_ALIASES[grid]) + "."
    msgs = finish([u(user), a(random.choice(ASK_UNSUPPORTED))], reply, grid, e, n, res)
    return {"messages": msgs, "meta": meta("unsupported_zone", "ask", grid, e, n, res, ["grid"])}


SCENARIOS = [
    (lambda: make_complete(True), 35),
    (lambda: make_complete(False, "noise"), 5),
    (lambda: make_complete(False, "swapped"), 3.3),
    (lambda: make_complete(False, "dropped_digit"), 3.3),
    (lambda: make_complete(False, "wrong_grid"), 3.4),
    (lambda: make_missing("grid"), 15), (lambda: make_missing("datum"), 15),
    (lambda: make_missing("both"), 10), (make_unsupported, 10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="train.jsonl")
    ap.add_argument("--exclude", help="JSONL whose first user messages must not be reused")
    args = ap.parse_args()
    random.seed(args.seed)

    seen = set()
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as f:
            seen = {json.loads(line)["messages"][0]["content"] for line in f}

    builders, weights = zip(*SCENARIOS)
    rows, attempts = [], 0
    while len(rows) < args.n and attempts < args.n * 20:
        attempts += 1
        ex = random.choices(builders, weights)[0]()
        key = ex["messages"][0]["content"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(ex)

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    counts = {}
    for ex in rows:
        counts[ex["meta"]["scenario"]] = counts.get(ex["meta"]["scenario"], 0) + 1
    et = {}
    for ex in rows:
        k = ex["meta"]["error_type"]
        et[k] = et.get(k, 0) + 1
    print(f"Wrote {len(rows)} rows to {args.out}: {counts}")
    print(f"Error types: {et}")


if __name__ == "__main__":
    main()
