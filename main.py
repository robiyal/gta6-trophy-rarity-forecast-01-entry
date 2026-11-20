"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Model, in counts, not rates
---------------------------
Exophase prints a rate, but a rate is a ratio: earned / tracked.  The
page refreshes on its own schedule, so between refreshes the printed
rates sit still while players keep earning, and at a refresh the tracked
count jumps and every rate but the 100% one drops.  A model that only
looks at rates cannot see that; a model that looks at counts can.

For each read we recover the earned count of every trophy from the
printed rate and the tracked count, then fit two saturating growth
curves in log time (x = ln(1 + hours since release)):

  * log(tracked) against x -- new players join, and the influx decays;
  * log(earned_i) against x, per trophy -- players reach the trophy,
    then the pool that will ever earn it runs out.

Each fit is a straight line in log time (a constant when the line
wants to fall: earned and tracked never decrease), fitted on the last
24 hours of reads when there are enough of them.  A straight line in
log time is already a saturating curve in clock time -- growth
decelerates as ln(1 + hours) flattens -- and it extrapolates stably:
a concave quadratic fitted on a narrow window bends wildly four days
out and undershot both a power-law truth and a saturating truth in
synthetic tests, while the line stayed within 1.5x of both.  The
median rate at the target is 100 * earned / tracked from the two
lines.  The quadratic is not thrown away: where it exists and is
concave, half the log gap between its prediction and the line's joins
the spread, because two reasonable models disagreeing is itself
uncertainty about the curve's shape.

Spread
------
Quantiles sit on the logit scale around the median.  The spread is the
honest uncertainty of the two curves: each fit's prediction standard
error at the target (which grows as the target moves past the data),
half the log gap between the line's prediction and a concave
quadratic's where the two disagree about the curve's shape, plus a
random-walk term of 0.5 logit units per square-root day, the same size
the house baseline uses.  The total is floored at 0.6 logit units: even a perfectly flat history cannot promise that Exophase will
not refresh -- the rules record one refresh multiplying the tracked
count by six -- and capped at 2.5.  Refreshes push rates down, never
up, so the downward spread carries one more term: how long the page
has sat unchanged in the case's own history.  A refresh is overdue
after a long freeze, and the term saturates at 1.0 logit unit for a
freeze of many hours; where reads change regularly, as they do through
the final, the term is near zero and the band stays symmetric.  The
100% trophy is special-cased: it survived a sixfold dilution at 100%,
so its forecast stays at the top with only a small downward tail.

Everything is read from the case file at runtime: trophy list, release
date, history, horizon and levels.  Nothing about the trophies is
hardcoded.  Standard library only, no network, deterministic.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Logit spread of a rate's move, per square-root day (random walk).
SIGMA_RW = 0.5
# Floor on the total logit spread: a flat history still faces a refresh.
SIGMA_FLOOR = 0.6
# Cap on the total logit spread: a wild fit must not print absurd bands.
SIGMA_CAP = 2.5
# Refresh risk on the downside: saturating size, in logit units, and the
# freeze length that brings it most of the way there, in hours.  The
# rules record the page frozen for about four and a half hours, then a
# refresh multiplying the tracked count by six -- so a long freeze means
# a refresh is overdue and the next one can cut rates severalfold.
REFRESH_SIGMA = 1.0
REFRESH_TAU_HOURS = 4.0
# Fits use the most recent reads only; launch-night reads are stale.
WINDOW_HOURS = 24.0


def parse_time(text):
    """Parse an ISO 8601 timestamp, with or without a Z suffix."""
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def parse_release(case):
    """The game's release instant, from the case file, else the first read."""
    game = case.get("game") or {}
    raw = game.get("release_date")
    if raw:
        try:
            return parse_time(raw)
        except ValueError:
            pass
    history = case.get("history") or []
    if history:
        return parse_time(history[0]["at"])
    return parse_time(case["at"])


def log_time(moment, release):
    """ln(1 + hours since release); reads before release sit at zero."""
    hours = (moment - release).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    """logit of a percent, clamped to 0.01..99.99 first (0 and 100 have none)."""
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg):
    """Inverse logit, as a percent, computed without overflow."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def _solve(matrix, rhs):
    """Solve a small dense linear system; None when it is singular."""
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for r in range(col + 1, n):
            f = aug[r][col] / pv
            if f:
                for c in range(col, n + 1):
                    aug[r][c] -= f * aug[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        rest = aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))
        out[i] = rest / aug[i][i]
    return out


def _least_squares(design, ys, target_row):
    """Least squares for one design matrix.

    Returns (prediction, prediction standard error, coefficients), or
    None when the design is singular.  The standard error carries the
    +1 of a new observation, so it covers process noise as well as the
    fit's own error, and it grows as the target moves past the data.
    """
    p = len(design[0])
    n = len(ys)
    ata = [[sum(r[i] * r[j] for r in design) for j in range(p)] for i in range(p)]
    aty = [sum(r[i] * y for r, y in zip(design, ys)) for i in range(p)]
    beta = _solve(ata, aty)
    if beta is None:
        return None
    pred = sum(b * v for b, v in zip(beta, target_row))
    ssr = 0.0
    for row, y in zip(design, ys):
        fitted = sum(b * v for b, v in zip(beta, row))
        ssr += (y - fitted) ** 2
    dof = n - p
    sigma2 = ssr / dof if dof > 0 else 0.0
    w = _solve(ata, list(target_row))
    if w is None:
        return None
    quad = sum(v * wi for v, wi in zip(target_row, w))
    se = math.sqrt(max(0.0, sigma2 * (1.0 + quad)))
    return pred, se, beta


def fit_log_curve(xs, ys, x_target):
    """Fit log y against x with a line, else a mean; probe a quadratic.

    Returns (prediction, prediction standard error, quadratic prediction
    or None).  The line is the forecast: it extrapolates stably.  The
    quadratic is only a probe -- fitted when there are at least three
    distinct x values and it bends downward (growth saturates) -- and
    its disagreement with the line measures shape uncertainty.  A line
    that wants to fall becomes a mean, since the counted thing never
    decreases.
    """
    n = len(xs)
    distinct = len(set(xs))
    quad_pred = None
    if n >= 3 and distinct >= 3:
        got = _least_squares(
            [[1.0, x, x * x] for x in xs], ys, [1.0, x_target, x_target * x_target]
        )
        if got is not None and got[2][2] <= 0.0:
            quad_pred = got[0]
    if n >= 2 and distinct >= 2:
        got = _least_squares([[1.0, x] for x in xs], ys, [1.0, x_target])
        if got is not None and got[2][1] >= 0.0:
            return got[0], got[1], quad_pred
    mean = sum(ys) / n
    if n > 1:
        var = sum((y - mean) ** 2 for y in ys) / (n - 1)
        se = math.sqrt(var * (1.0 + 1.0 / n))
    else:
        se = 0.0
    return mean, se, quad_pred


def _shape_sigma(pred, quad_pred):
    """Half the log gap between the line and the quadratic probe.

    Zero when the probe does not exist or agrees with the line: two
    models agreeing is no extra uncertainty.
    """
    if quad_pred is None:
        return 0.0
    return 0.5 * abs(quad_pred - pred)


def forecast_case(case):
    """Forecast one case; returns {"trophies": [{"id", "q": [...]}]}."""
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zvals = [NormalDist().inv_cdf(tau) for tau in levels]
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    release = parse_release(case)
    x_target = log_time(target, release)

    if not history:
        # No reads at all: a flat prior, wide spread, never a crash.
        return {
            "trophies": [
                {"id": t["id"], "q": _quantiles(1.0, 1.5, 1.5, zvals)}
                for t in trophies
            ]
        }

    window = [
        r for r in history if parse_time(r["at"]) >= at - timedelta(hours=WINDOW_HOURS)
    ]
    if len(window) < 3:
        window = history
    reads = sorted(window, key=lambda r: r["at"])

    # How long the page has sat unchanged: the last read whose tracked
    # count or rates differ from the read before it.  A long freeze
    # means a refresh is overdue, and a refresh cuts rates.
    last_change = parse_time(reads[0]["at"])
    prev_key = None
    for read in reads:
        key = (read["players_tracked"], tuple(read["rates"]))
        if prev_key is not None and key != prev_key:
            last_change = parse_time(read["at"])
        prev_key = key
    frozen_hours = max(0.0, (at - last_change).total_seconds() / 3600.0)
    refresh_sigma = REFRESH_SIGMA * (1.0 - math.exp(-frozen_hours / REFRESH_TAU_HOURS))

    xs = [log_time(parse_time(r["at"]), release) for r in reads]
    tracked = [float(r["players_tracked"]) for r in reads]
    log_tracked = [math.log(max(1.0, t)) for t in tracked]
    t_pred, t_se, t_quad = fit_log_curve(xs, log_tracked, x_target)
    t_shape = _shape_sigma(t_pred, t_quad)
    last_tracked = tracked[-1]
    tracked_pred = max(last_tracked, math.exp(t_pred))

    rows = []
    for i, trophy in enumerate(trophies):
        pairs = []
        last_earned = None
        last_rate = None
        for j, read in enumerate(reads):
            rates = read["rates"]
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            earned = math.floor(rate / 100.0 * tracked[j] + 0.5)
            last_earned = earned
            last_rate = rate
            pairs.append((xs[j], math.log(max(0.5, float(earned)))))
        if last_rate is not None and last_rate >= 99.99:
            # Survived a sixfold dilution at 100%: stays at the top.
            rows.append({"id": trophy["id"], "q": [99.9, 99.98, 100, 100, 100]})
            continue
        if len(pairs) >= 2:
            e_pred, e_se, e_quad = fit_log_curve(
                [p[0] for p in pairs], [p[1] for p in pairs], x_target
            )
            e_shape = _shape_sigma(e_pred, e_quad)
            earned_pred = math.exp(e_pred)
        else:
            e_se = 0.0
            e_shape = 0.0
            earned_pred = float(last_earned) if last_earned is not None else 0.0
        if last_earned is not None:
            earned_pred = max(earned_pred, float(last_earned))
        earned_pred = min(earned_pred, tracked_pred)
        if last_earned is None:
            # Never seen in any read: a typical rate, wide spread.
            median, sigma = 1.0, 1.5
            sigma_down = sigma
        else:
            median = 100.0 * earned_pred / tracked_pred
            sigma = math.sqrt(
                e_se**2
                + t_se**2
                + e_shape**2
                + t_shape**2
                + (SIGMA_RW * math.sqrt(horizon_days)) ** 2
            )
            sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
            # The refresh term widens the downside only.
            sigma_down = math.sqrt(sigma**2 + refresh_sigma**2)
        rows.append(
            {"id": trophy["id"], "q": _quantiles(median, sigma_down, sigma, zvals)}
        )
    return {"trophies": rows}


def _quantiles(median, sigma_down, sigma_up, zvals):
    """Five quantiles of a rate, on the logit scale, non-decreasing.

    The downside spread carries the refresh risk; the upside does not.
    """
    center = logit(median)
    out = []
    prev = None
    for z in zvals:
        spread = sigma_down if z < 0 else sigma_up
        q = clamp(round4(logistic(center + z * spread)), 0, 100)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
