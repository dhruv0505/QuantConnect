"""
The pluggable factor for main.py: FUNDAMENTALS version (drop-in replacement for the text factor).
main.py doesn't need to change.

CONTRACT (all main.py relies on)
  class Factor:
      name: str                          short label for the SUMMARY log
      __init__(self, algorithm, param)   algorithm = the running QCAlgorithm;
                                         param(name, default) reads a project parameter,
                                         typed like the default
      series(self, symbol)               [(usable_from, value), ...] sorted by time.
                                         usable_from = first algorithm time the value may be
                                         used (no lookahead). Empty = no data.
      summary(self)                      short string of counts for the SUMMARY log

THIS FACTOR: Morningstar US Fundamental data from QuantConnect, read with
history[Fundamental](symbol, start, end). Settings (each can be a project parameter):

  FUND_METRICS   comma-separated metric names, averaged into one number per stock.
                 Choices: gross_margin, ebitda_margin, operation_margin, net_margin,
                          revenue_growth
                 Examples: "gross_margin"   or   "gross_margin,ebitda_margin"
  FUND_TRANSFORM "yoy"   = change versus the same period a year earlier (default). Margins
                           are ratios, so this is the change in margin. Removes seasonality
                           and matches the pair's relative price move better than a level.
                 "level" = the raw value, no change calculation.
  FUND_PERIOD    which window of each field to read first: "twelve_months" (trailing
                 twelve months, default) or "three_months" or "value" (Morningstar's default
                 window). If a window is missing, "value" is tried next.

A new value is stored only when the number CHANGES, i.e. when a new filing arrives. These
fields only change at filings, which is what makes this safe. Do not add daily-moving fields
such as P/E or market cap here (the series would fill with one row per day, and a price-based
value would leak the price into the factor).

TIMING: the new Morningstar dataset stamps ratio-type fields with the filing date. A value is
usable from the day AFTER its row's end_time, because filings can arrive after the close.

If a field was retired by Morningstar it throws NotSupportedException; the message names the
replacement. The first error per metric is written to the log as "FACTOR ...".
"""
from AlgorithmImports import *
from datetime import timedelta

# This factor's settings. Each can be overridden by a project parameter of the same name.
DEFAULTS = {
    "FUND_METRICS": "gross_margin",   # comma-separated, see the list above
    "FUND_TRANSFORM": "yoy",          # "yoy" or "level"
    "FUND_PERIOD": "twelve_months",   # "twelve_months", "three_months" or "value"
    "FILING_LOOKBACK_DAYS": 900,      # history fetched on first use; >1 year so YoY exists
    "YOY_MIN_DAYS": 330,              # the "year ago" value must be this many days earlier...
    "YOY_MAX_DAYS": 400,              # ...and at most this many (closest to 365 is chosen)
}

# metric name -> (group on the Fundamental object, property name). Check the names with
# autocomplete if one fails; to add a metric, add a line here.
METRICS = {
    "gross_margin": ("operation_ratios", "gross_margin"),
    "ebitda_margin": ("operation_ratios", "ebitda_margin"),
    "operation_margin": ("operation_ratios", "operation_margin"),
    "net_margin": ("operation_ratios", "net_margin"),
    "revenue_growth": ("operation_ratios", "revenue_growth"),
}
TRANSFORMS = ("yoy", "level")


class Factor:
    """Fundamentals factor (margin change or level) per stock."""

    def __init__(self, algorithm, param):
        self.algo = algorithm
        for name, default in DEFAULTS.items():
            setattr(self, name, param(name, default))
        self._metrics = [m.strip() for m in self.FUND_METRICS.split(",") if m.strip()]
        unknown = [m for m in self._metrics if m not in METRICS]
        if not self._metrics or unknown:
            raise ValueError(f"FUND_METRICS must be a comma list from {tuple(METRICS)}, "
                             f"got {self.FUND_METRICS!r}")
        if self.FUND_TRANSFORM not in TRANSFORMS:
            raise ValueError(f"FUND_TRANSFORM must be one of {TRANSFORMS}, got {self.FUND_TRANSFORM!r}")
        # Window of each field to try, in order, without repeats.
        self._periods = list(dict.fromkeys([self.FUND_PERIOD, "value"]))
        self.name = f"fund/{'+'.join(self._metrics)}/{self.FUND_TRANSFORM}"
        # Filled incrementally so each day of history is read once:
        # symbol -> [(usable_from, raw level)], symbol -> fetched-up-to time,
        # symbol -> [(usable_from, factor value)]
        self._levels, self._fetched, self._series = {}, {}, {}
        self._warned = set()
        self._stats = {"rows": 0, "levels": 0, "no_value": 0, "no_data": 0, "errors": 0}

    # ---------- Contract ----------

    def series(self, symbol):
        """[(usable_from, value)] for the stock, sorted. Only history since the last call is
        fetched and read."""
        now = self.algo.time
        start = self._fetched.get(symbol, now - timedelta(days=self.FILING_LOOKBACK_DAYS))
        if start < now:
            levels = self._levels.setdefault(symbol, [])
            added = False
            try:
                for row in self.algo.history[Fundamental](symbol, start, now):
                    self._stats["rows"] += 1
                    if not row.has_fundamental_data:
                        self._stats["no_data"] += 1
                        continue
                    level = self._level(row)
                    if level is None:
                        self._stats["no_value"] += 1
                        continue
                    if levels and levels[-1][1] == level:   # unchanged: no new filing yet
                        continue
                    when = (getattr(row, "end_time", None) or row.time) + timedelta(days=1)
                    if levels and when <= levels[-1][0]:
                        continue
                    levels.append((when, level))
                    self._stats["levels"] += 1
                    added = True
            except Exception as error:
                self._stats["errors"] += 1
                if self._stats["errors"] <= 5:
                    self.algo.log(f"FACTOR {symbol.value} history: {error}")
            self._fetched[symbol] = now
            if added or symbol not in self._series:
                self._series[symbol] = self._transform(levels)
        return self._series.get(symbol, [])

    def summary(self):
        return f"fund={self._stats} fund_stocks={sum(1 for s in self._series.values() if s)}"

    # ---------- Helpers ----------

    def _warn(self, metric, error):
        """Log the first error per metric (the message names the replacement if one exists)."""
        self._stats["errors"] += 1
        if metric not in self._warned:
            self._warned.add(metric)
            self.algo.log(f"FACTOR metric {metric}: {error}")

    def _read(self, field, metric):
        """One number from a Morningstar field, trying each window in turn. None if missing."""
        for window in self._periods:
            try:
                value = float(getattr(field, window))
            except Exception as error:
                if "retired" in str(error).lower() or "NotSupported" in type(error).__name__:
                    self._warn(metric, error)
                continue
            if value == value and value not in (0.0, float("inf"), float("-inf")):
                return value        # 0.0 is Morningstar's "missing", NaN/inf are unusable
        return None

    def _level(self, row):
        """Raw level for one daily row: the average of the chosen metrics, or None if any
        of them is missing (so a composite always uses the same ingredients)."""
        values = []
        for metric in self._metrics:
            group, attr = METRICS[metric]
            try:
                field = getattr(getattr(row, group), attr)
            except Exception as error:
                self._warn(metric, error)
                return None
            value = self._read(field, metric)
            if value is None:
                return None
            values.append(value)
        return sum(values) / len(values)

    def _transform(self, levels):
        """Raw levels -> factor values. "yoy": this value minus the value about a year earlier
        (closest to 365 days back within the allowed window), stamped when the newer value
        became usable. The first year of data has no YoY value."""
        if self.FUND_TRANSFORM == "level":
            return list(levels)
        out = []
        for when, value in levels:
            best = None
            for earlier_when, earlier_value in levels:
                gap = (when - earlier_when).days
                if self.YOY_MIN_DAYS <= gap <= self.YOY_MAX_DAYS:
                    if best is None or abs(gap - 365) < abs(best[0] - 365):
                        best = (gap, earlier_value)
            if best is not None:
                out.append((when, value - best[1]))
        return out
