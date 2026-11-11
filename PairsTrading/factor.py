"""
Relative Gross Margin Momentum factor for the pluggable pairs-trading strategy.

Replace PairsTrading/factor.py with this file; main.py does not change.

CONTRACT (used by main.py)
    Factor(algorithm, param)
    .name: short label for SUMMARY
    .series(symbol): [(usable_from, value), ...], sorted chronologically
    .summary(): counts for SUMMARY

FACTOR
    GM_i(q) = (Revenue_i(q) - COGS_i(q)) / Revenue_i(q)
    F_i(q)  = GM_i(q) - GM_i(q-4)

The strategy's existing main.py computes F_i - F_j, optionally standardizes
F across active stocks, and regresses adjusted price gaps on the pair factor.

Morningstar's OperationRatios.GrossMargin.ThreeMonths supplies the quarterly
margin. We identify the underlying financial quarter by
FinancialStatements.PeriodEndingDate.ThreeMonths and compare it to the same
fiscal quarter roughly one year earlier (allowing for 52/53-week calendars).

Point-in-time safeguards:
  * Read historical Fundamental snapshots, not today's fundamentals for old dates.
  * Use only the first observed value for each fiscal period.
  * A new value becomes usable only AFTER both its observed snapshot date and
    the reported filing date (when provided), conservatively one day later.
  * Missing prior-year quarters produce no factor rather than a fabricated zero.
  * Cache snapshots and fetch incrementally; do not recalculate the full history.

Data: QuantConnect Morningstar US Fundamental Data, no extra SEC subscription.
"""

from AlgorithmImports import *
from datetime import date, datetime, timedelta
import math


# Project parameters can override these via main.py's param(name, default).
DEFAULTS = {
    "GM_LOOKBACK_DAYS": 1100,       # ~3 years: covers prior-year and recent factor history
    "GM_REFRESH_DAYS": 7,          # don't rerequest daily snapshots on repeated calls
    "GM_YOY_TOLERANCE_DAYS": 45,   # accommodates shifting fiscal quarter ends / 53-week years
}


def _as_date(value):
    """Convert Python or pythonnet .NET date objects to a Python date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date(int(value.year), int(value.month), int(value.day))
    except (AttributeError, ValueError, TypeError):
        try:
            return date(int(value.Year), int(value.Month), int(value.Day))
        except (AttributeError, ValueError, TypeError):
            return None


def _valid_date(value):
    result = _as_date(value)
    return result if result is not None and result.year > 1900 else None


def _quarterly_margin(fundamental):
    """Three-month gross margin. Missing/invalid data is NOT treated as zero."""
    field = fundamental.operation_ratios.gross_margin
    # has_value tests the default (three-month) period and avoids treating
    # absent values, usually zero in the data provider, as genuine margins.
    if not field.has_value:
        return None
    number = float(field.three_months)
    return number if math.isfinite(number) else None


def _quarter_end(fundamental):
    """The period end to which the three-month financial statement relates."""
    try:
        return _valid_date(fundamental.financial_statements.period_ending_date.three_months)
    except (AttributeError, ValueError, TypeError):
        return None


def _filing_date(fundamental):
    """Publication date of the corresponding three-month financial statement."""
    try:
        return _valid_date(fundamental.financial_statements.file_date.three_months)
    except (AttributeError, ValueError, TypeError):
        # Availability is still tied to historical snapshot dates, not today's data.
        return None


def _previous_year_margin(period_end, known_margins, tolerance):
    """Find margin of matching fiscal quarter one year earlier, or None.

    known_margins maps fiscal-quarter-end dates to (available_date, margin).
    Adjacent quarters (~90 days apart) cannot be mistaken for the prior year.
    """
    target = period_end - timedelta(days=365)
    choices = [q for q in known_margins
               if 365 - tolerance <= (period_end - q).days <= 365 + tolerance]
    if not choices:
        return None
    closest = min(choices, key=lambda q: abs((q - target).days))
    return known_margins[closest][1]


class Factor:
    """Year-over-year quarterly gross margin momentum, per equity."""

    def __init__(self, algorithm, param):
        self.algo = algorithm
        for name, default in DEFAULTS.items():
            setattr(self, name, param(name, default))
        if self.GM_LOOKBACK_DAYS < 500:
            raise ValueError("GM_LOOKBACK_DAYS must be >= 500 to form YoY changes")
        if self.GM_REFRESH_DAYS < 1:
            raise ValueError("GM_REFRESH_DAYS must be >= 1")
        if not 0 <= self.GM_YOY_TOLERANCE_DAYS < 60:
            raise ValueError("GM_YOY_TOLERANCE_DAYS must be between 0 and 59")

        self.name = "gross_margin_yoy"
        self._series = {}   # symbol -> list[(usable_from datetime, YoY delta)]
        self._margins = {}  # symbol -> {quarter_end date: (usable_from datetime, margin)}
        self._fetched = {}  # symbol -> last successful request end datetime
        self._stats = {
            "rows": 0, "quarters": 0, "yoy": 0,
            "missing_margin": 0, "missing_period": 0,
            "missing_previous": 0, "errors": 0,
        }

    def series(self, symbol):
        """Return time-stamped, point-in-time YoY margins for `symbol`."""
        now = self.algo.time
        last = self._fetched.get(symbol)
        if last is not None and now - last < timedelta(days=self.GM_REFRESH_DAYS):
            return self._series.get(symbol, [])

        start = (last - timedelta(days=3)) if last is not None else \
                (now - timedelta(days=self.GM_LOOKBACK_DAYS))
        if start >= now:
            return self._series.get(symbol, [])

        try:
            # Typed history returns Fundamental objects rather than an enormous
            # DataFrame of nested corporate fields. History works even when the
            # symbol is not currently subscribed to the algorithm.
            rows = self.algo.history[Fundamental](symbol, start, now, Resolution.DAILY)
            margins = self._margins.setdefault(symbol, {})
            values = self._series.setdefault(symbol, [])

            for row in rows:
                self._stats["rows"] += 1
                period_end = _quarter_end(row)
                if period_end is None:
                    self._stats["missing_period"] += 1
                    continue
                if period_end in margins:
                    continue  # filled daily snapshots; process each fiscal quarter once

                try:
                    margin = _quarterly_margin(row)
                except Exception as error:
                    # Some Morningstar period fields were retired in 2026.
                    self._stats["errors"] += 1
                    if self._stats["errors"] <= 5:
                        self.algo.log(f"FACTOR {symbol.value} gross_margin: {error}")
                    continue
                if margin is None:
                    self._stats["missing_margin"] += 1
                    continue

                observed = _valid_date(row.end_time)
                if observed is None:
                    continue
                filing = _filing_date(row)
                # One-day cushion after publication/observation, protecting
                # against intraday filing timestamps and backtest lookahead.
                usable_from = datetime.combine(
                    max(d for d in (observed, filing) if d is not None) + timedelta(days=1),
                    datetime.min.time(),
                )
                if usable_from > now:
                    # History should not contain future data, but don't insert
                    # a not-yet-published quarter if it does.
                    continue

                previous = _previous_year_margin(
                    period_end, margins, self.GM_YOY_TOLERANCE_DAYS
                )
                margins[period_end] = (usable_from, margin)
                self._stats["quarters"] += 1
                if previous is None:
                    self._stats["missing_previous"] += 1
                    continue

                values.append((usable_from, margin - previous))
                self._stats["yoy"] += 1

            values.sort(key=lambda item: item[0])
            self._fetched[symbol] = now
        except Exception as error:
            self._stats["errors"] += 1
            if self._stats["errors"] <= 5:
                name = getattr(symbol, "value", str(symbol))
                self.algo.log(f"FACTOR {name} Fundamental: {error}")

        return self._series.get(symbol, [])

    def summary(self):
        usable = sum(bool(v) for v in self._series.values())
        return (f"gm_stocks={usable}/{len(self._series)} "
                f"gm_quarters={self._stats['quarters']} "
                f"gm_yoy={self._stats['yoy']} "
                f"gm_errors={self._stats['errors']} "
                f"gm_missing_prev={self._stats['missing_previous']} "
                f"gm_missing_margin={self._stats['missing_margin']}")
