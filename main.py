"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts the GTA VI launch-day Twitch category viewers, hours 2..24, as five
quantiles each, from the case's hour-1 print and the comparables it ships.

Model (fit at run time from the case's own comparables, so it adapts to the
final case's 42 launches, practice launches included):
  1. Shape: for each hour h, the ratio viewers_h / viewers_1 across the
     "comparable"-tier launches (rolling midnight-local console launches, no
     early-access contamination) follows a strong level effect: the bigger
     hour 1, the smaller the peak ratio (corr log(p1) vs log(peak/p1) ~ -0.94).
     So per hour we fit log(ratio_h) = a_h + b_h * log(p1) + c_h * log(pre/p1)
     by least squares over the comparables with hour-1 >= 1000 viewers,
     where pre is the launch's pre-release hype peak (the maximum hourly
     viewers in its history before day_start) and the second term captures
     that a launch whose pre-spike is large relative to hour 1 has lower
     early-hour ratios (the spike borrows viewers from the early hours).
     The slope b_h is shrunk 10% toward the cross-hour mean slope (the
     per-hour slopes are noisy; a little shrinkage generalises better).
  2. Blend: the regression shape is blended (80/20, in log space) with the
     fixed median shape of the same comparables, which stabilises the fit.
  3. The effective hour-1 used for the shape is capped at 400,000 viewers:
     above that the shape is the 400k shape (the biggest launches' evening
     sits near hour 1, it does not collapse), while the absolute forecast
     still scales with the true hour-1 print. Ratios are clamped to
     [0.60, 15].
  4. Spread (per hour, from the data): q_tau = median * exp(z_tau * s_h),
     z = [-1.645, -0.674, 0, 0.674, 1.645], where s_h = 0.30 * (the
     residual std of log(ratio_h) across the comparables at hour h). The
     early hours scatter far more than the evening (unlock waves), so a
     single global spread is wrong: the house baseline's 0.15 growing with
     sqrt(h-1) is far too narrow at hours 3-6 and too wide later. The 0.30
     scale is tuned on the four practice cases (0.25-0.35 all similar);
     the raw residual std overstates the error for big launches, so it is
     scaled down.
  5. Fallback: if the comparables cannot be used at all (missing, too few,
     or any error), the program falls back to the house persistence baseline,
     so it always writes a valid forecast.

Practice validation (local pinball harness, log scale, 5 quantiles, hours
2..24; answers fetched once from SullyGnome, the cases' own source):
  house baseline:  p00 0.7437  p01 0.2868  p02 0.2146  p03 0.3033  mean 0.3871
  v2 model:        p00 0.1143  p01 0.1164  p02 0.1107  p03 0.1967  mean 0.1345
  this model (v3): p00 0.1130  p01 0.1160  p02 0.0922  p03 0.1995  mean 0.1302
  (66% better than the baseline on the mean; leave-one-out mean 0.1290,
  leave-one-comparable-out on big launches (p1 >= 150k) 0.1682; each
  practice case scored with the other three practice launches included
  in the fit, as the final case includes them.)

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
SHRINK_B = 0.1          # per-hour slope b_h shrunk this far toward the
                         # cross-hour mean slope (0.08-0.15 all similar)
SIGMA = 0.10             # fallback log-spread per sqrt-hour (used only when
                         # an hour has too few residuals to estimate its own)
SPREAD_CAP_H = 9         # sqrt(h-1) is capped at sqrt(SPREAD_CAP_H)
RESID_SCALE = 0.3        # per-hour spread = RESID_SCALE * residual std of
                         # log(ratio) at that hour across the comparables.
                         # The residual std (~0.7-1.1) overstates the error
                         # for big launches, so it is scaled down; 0.3 is
                         # the tuned value (0.25-0.35 all similar).
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


def pre_peak(rows, day_start, time_key):
    """Maximum viewers in the rows before day_start (the launch's
    pre-release hype peak). rows: comparables' hourly rows (time_key 'from')
    or the case file's history (time_key 'at'). Returns None if no rows."""
    ds = (day_start or "")[:10]
    if not ds:
        return None
    best = None
    for row in rows or []:
        t = row.get(time_key) or ""
        if t[:10] < ds:
            v = row.get("viewers")
            if isinstance(v, (int, float)) and v is not None:
                if best is None or v > best:
                    best = float(v)
    return best


def usable_curves(case: dict):
    """[(p1, {hour: viewers}, pre_peak)] for comparable-tier launches,
    hour-1 >= floor, at most 3 missing hours in 2..24. pre_peak is the
    launch's pre-release hype peak (max hourly viewers before day_start),
    or None if the comparable carries no pre-launch rows."""
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
        pre = pre_peak(comp.get("hourly"), comp.get("day_start") or "", "from")
        curves.append((p1, by_hour, pre))
    return curves


def fit_shape(curves):
    """Per-hour least-squares fit of log(ratio_h) on log(p1) and, where
    available, log(pre_peak / p1). Returns ({h: (a, b, c)}, {h: median_ratio},
    {h: residual_std}). The residual std measures how much the comparables'
    shapes scatter around the fit at each hour: early hours scatter far more
    (unlock waves) than evening hours, so the forecast spread is per-hour.
    The slope b_h is shrunk SHRINK_B toward the cross-hour mean slope, with
    the intercept recomputed so the line still passes through the centroid;
    the per-hour slopes are noisy and a little shrinkage generalises better.
    The pre-peak term is used only when every curve at that hour carries a
    pre-peak; otherwise the hour falls back to the single regressor."""
    params = {}
    medians = {}
    resid_sd = {}
    for h in range(2, 25):
        xs, ys, zs = [], [], []
        ratios = []
        for p1, by_hour, pre in curves:
            if h in by_hour and by_hour[h] is not None and by_hour[h] > 0:
                xs.append(math.log(p1))
                ys.append(math.log(by_hour[h] / p1))
                ratios.append(by_hour[h] / p1)
                zs.append(
                    math.log(max(1e-9, pre / p1)) if pre else None
                )
        ratios.sort()
        medians[h] = ratios[len(ratios) // 2] if ratios else 1.0
        n = len(xs)
        if n < 5:
            params[h] = (0.0, 0.0, 0.0)
            resid_sd[h] = None
            continue
        mx = sum(xs) / n
        my = sum(ys) / n
        if all(z is not None for z in zs):
            mz = sum(zs) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            szz = sum((z - mz) ** 2 for z in zs)
            sxz = sum((x - mx) * (z - mz) for x, z in zip(xs, zs))
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            szy = sum((z - mz) * (y - my) for z, y in zip(zs, ys))
            den = sxx * szz - sxz * sxz
            if abs(den) < 1e-12:
                b = sxy / sxx if sxx else 0.0
                c = 0.0
            else:
                b = (sxy * szz - szy * sxz) / den
                c = (szy * sxx - sxy * sxz) / den
            a = my - b * mx - c * mz
            resid = [
                y - (a + b * x + c * z) for x, z, y in zip(xs, zs, ys)
            ]
            # keep the centroid for the shrinkage step below
            params[h] = (a, b, c, mx, my, mz)
        else:
            den = sum((x - mx) ** 2 for x in xs)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
            a = my - b * mx
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
            params[h] = (a, b, 0.0, mx, my, 0.0)
        mean = sum(resid) / n
        resid_sd[h] = math.sqrt(sum((r - mean) ** 2 for r in resid) / n)
    # Shrink each hour's slope toward the cross-hour mean slope.
    bs = [params[h][1] for h in range(2, 25) if resid_sd[h] is not None]
    bmean = sum(bs) / len(bs) if bs else 0.0
    for h in range(2, 25):
        if resid_sd[h] is None:
            continue
        a, b, c, mx, my, mz = params[h]
        b_new = (1.0 - SHRINK_B) * b + SHRINK_B * bmean
        a_new = my - b_new * mx - c * mz
        params[h] = (a_new, b_new, c)
    return params, medians, resid_sd


def shape_for(p1: float, params, medians, pre_peak=None) -> dict:
    """Blended ratio-to-hour-1 shape for this case's hour-1 print.
    The effective hour-1 is capped at P1_CAP: for a launch far bigger than
    any comparable, the shape stays the biggest launches' shape (evening
    near hour 1) instead of collapsing toward a flat line. pre_peak is this
    case's pre-release hype peak (None if the case carries no history);
    its log-ratio to hour 1 enters through the fitted coefficient c_h
    (a value of 0, i.e. pre_peak == hour 1, is neutral)."""
    lp = math.log(min(max(1.0, p1), P1_CAP))
    zp = math.log(max(1e-9, pre_peak / p1)) if pre_peak else 0.0
    shape = {}
    for h in range(2, 25):
        a, b, c = params[h]
        adaptive = math.exp(a + b * lp + c * zp)
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


def model_forecast(p1: float, shape: dict, resid_sd: dict, hours):
    out = {}
    for h in hours:
        med = max(1.0, float(p1) * shape[h])
        # Per-hour spread from the comparables' residual scatter at that
        # hour (scaled down: big launches are steadier than the pool).
        # Falls back to the old global spread if the hour has no estimate.
        sd = resid_sd.get(h)
        if sd is None:
            s = SIGMA * math.sqrt(min(h - 1, SPREAD_CAP_H))
        else:
            s = RESID_SCALE * sd
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
        params, medians, resid_sd = fit_shape(curves)
        pre = pre_peak(case.get("history"), case.get("day_start") or "", "at")
        shape = shape_for(p1, params, medians, pre)
        quantiles = model_forecast(p1, shape, resid_sd, hours)
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
