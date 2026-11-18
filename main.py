"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts the GTA VI launch-day Twitch category viewers, hours 2..24, as five
quantiles each, from the case's hour-1 print and the comparables it ships.

Model (fit at run time from the case's own comparables, so it adapts to the
final case's 42 launches, practice launches included):
  1. Shape: for each hour h, the ratio viewers_h / viewers_1 across the
     "comparable"-tier launches (rolling midnight-local console launches, no
     early-access contamination) follows a strong level effect: the bigger
     hour 1, the smaller the peak ratio (corr log(p1) vs log(peak/p1) ~ -0.94).
     So per hour we fit log(ratio_h) = a_h + b_h * log(p1) by least squares
     over the comparables with hour-1 >= 1000 viewers, and predict the ratio
     for this case's hour-1 print.
  2. Blend: the regression shape is blended (80/20, in log space) with the
     fixed median shape of the same comparables, which stabilises the fit.
  3. The effective hour-1 used for the shape is capped at 400,000 viewers:
     above that the shape is the 400k shape (the biggest launches' evening
     sits near hour 1, it does not collapse), while the absolute forecast
     still scales with the true hour-1 print. Ratios are clamped to
     [0.60, 15].
  4. Spread: q_tau = median * exp(z_tau * 0.10 * sqrt(min(h-1, 9))),
     z = [-1.645, -0.674, 0, 0.674, 1.645]. The 0.10 spread (capped growth)
     was tuned on the four practice cases; it is narrower than the house
     baseline's 0.15 because the shape already carries most of the signal.
  5. Fallback: if the comparables cannot be used at all (missing, too few,
     or any error), the program falls back to the house persistence baseline,
     so it always writes a valid forecast.

Practice validation (local pinball harness, log scale, 5 quantiles, hours
2..24; answers fetched once from SullyGnome, the cases' own source):
  house baseline:  p00 0.7437  p01 0.2868  p02 0.2146  p03 0.3033  mean 0.3871
  this model:      p00 0.1158  p01 0.1207  p02 0.1092  p03 0.2113  mean 0.1393
  (64% better than the baseline on the mean; leave-one-out mean 0.1383,
  each practice case scored with the other three practice launches
  included in the fit, as the final case includes them.)

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so the sys.path line below puts it
back first. -s drops user site-packages. Runs in <1 s per case on 1 CPU.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (tuned on the four practice cases) ---
P1_FLOOR = 1000.0        # comparables with hour-1 below this are ignored
P1_CAP = 400000.0        # effective hour-1 for the shape is capped here
BLEND_W = 0.8            # weight of the p1-adaptive shape (rest: median shape)
SIGMA = 0.10             # log-spread per sqrt-hour, capped at sqrt(9)
SPREAD_CAP_H = 9         # sqrt(h-1) is capped at sqrt(SPREAD_CAP_H)
RATIO_LO = 0.60
RATIO_HI = 15.0
BASELINE_SIGMA = 0.15    # fallback persistence baseline spread


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def tier_of(comp: dict) -> str:
    m = re.search(r"Tier:\s*([a-z\-]+)", comp.get("unlock") or "")
    return m.group(1) if m else "unknown"


def release_day_curve(comp: dict):
    """{hour: viewers} for the 24 hours of the comparable's day_start, or None."""
    ds = (comp.get("day_start") or "")[:10]
    if not ds:
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        frm = row.get("from") or ""
        if frm[:10] == ds:
            try:
                h = int(frm[11:13]) + 1
            except (ValueError, IndexError):
                continue
            v = row.get("viewers")
            if isinstance(v, (int, float)) and v is not None:
                by_hour[h] = float(v)
    if 1 not in by_hour or not by_hour[1] or by_hour[1] <= 0:
        return None
    return by_hour


def usable_curves(case: dict):
    """[(p1, {hour: viewers})] for comparable-tier launches, hour-1 >= floor,
    at most 3 missing hours in 2..24."""
    curves = []
    for comp in case.get("comparables") or []:
        if tier_of(comp) != "comparable":
            continue
        by_hour = release_day_curve(comp)
        if by_hour is None:
            continue
        p1 = by_hour[1]
        if p1 < P1_FLOOR:
            continue
        missing = sum(
            1 for h in range(2, 25)
            if h not in by_hour or by_hour[h] is None
        )
        if missing > 3:
            continue
        curves.append((p1, by_hour))
    return curves


def fit_shape(curves):
    """Per-hour least-squares fit of log(ratio_h) on log(p1).
    Returns ({h: (a, b)}, {h: median_ratio})."""
    params = {}
    medians = {}
    for h in range(2, 25):
        xs, ys = [], []
        ratios = []
        for p1, by_hour in curves:
            if h in by_hour and by_hour[h] is not None and by_hour[h] > 0:
                xs.append(math.log(p1))
                ys.append(math.log(by_hour[h] / p1))
                ratios.append(by_hour[h] / p1)
        ratios.sort()
        medians[h] = ratios[len(ratios) // 2] if ratios else 1.0
        n = len(xs)
        if n < 5:
            params[h] = (0.0, 0.0)
            continue
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
        params[h] = (my - b * mx, b)
    return params, medians


def shape_for(p1: float, params, medians) -> dict:
    """Blended ratio-to-hour-1 shape for this case's hour-1 print."""
    lp = math.log(min(max(1.0, p1), P1_CAP))
    shape = {}
    for h in range(2, 25):
        a, b = params[h]
        adaptive = math.exp(a + b * lp)
        r = math.exp(
            BLEND_W * math.log(max(1e-9, adaptive))
            + (1.0 - BLEND_W) * math.log(max(1e-9, medians[h]))
        )
        shape[h] = min(RATIO_HI, max(RATIO_LO, r))
    return shape


def baseline_forecast(p1: float, hours):
    """House persistence baseline (fallback): flat at p1, 0.15 spread."""
    p1 = max(1.0, float(p1))
    out = {}
    for h in hours:
        out[h] = [
            p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1)) for z in Z
        ]
    return out


def model_forecast(p1: float, shape: dict, hours):
    out = {}
    for h in hours:
        med = max(1.0, float(p1) * shape[h])
        s = SIGMA * math.sqrt(min(h - 1, SPREAD_CAP_H))
        out[h] = [med * math.exp(z * s) for z in Z]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    hours = [row["hour"] for row in case["forecast"]]

    try:
        curves = usable_curves(case)
        if len(curves) < 5:
            raise ValueError("too few usable comparables")
        params, medians = fit_shape(curves)
        shape = shape_for(p1, params, medians)
        quantiles = model_forecast(p1, shape, hours)
    except Exception:
        # Never fail a case: fall back to the house baseline.
        quantiles = baseline_forecast(p1, hours)

    rows = []
    for h in hours:
        qs = quantiles[h]
        # Enforce non-decreasing, > 0, <= 1e9.
        qs = [min(1e9, max(0.01, float(q))) for q in qs]
        qs = sorted(qs)
        rows.append({"hour": h, "q": [round2(q) for q in qs]})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
