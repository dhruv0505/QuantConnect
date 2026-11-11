"""
Distance-method pairs trading with a pluggable factor (LEAN, Python).
Paste into a QuantConnect cloud project as main.py, next to factor.py.
Run mode: SIGNAL_MODE (g / ag / factor).

DEFINITIONS (i, j order is fixed at formation)
  V_i(t)    = P_i(t) / P_i(t0)                growth of $1, adjusted prices
  G_ij(t)   = V_i(t) - V_j(t)                 signed gap
  D_ij      = (1/T) * sum_t G_ij(t)^2         raw squared distance, not demeaned
  z_ij(t)   = (G_ij(t) - mu_ij) / sigma_ij    mu, sigma frozen over formation window
  AG_ij(t)  = G_ij(t) / (V_i(t) + V_j(t))     adjusted gap
  z_ag(t)   = (AG_ij(t) - mu_ag) / sigma_ag   mu, sigma frozen over formation window
  F_i       = the stock's factor value, from factor.py (Factor.series; the contract is
              there). Replace factor.py to test a different factor without touching this file.
  dF_ij(q)  = F_i(q) - F_j(q)
  Model     : AG_p(q) = alpha + beta * dF_p(q) + e    pooled OLS, last 4 quarters
  Residual  : AG(t) - (alpha + beta * dF latest known)
  z_resid   = (residual - mean_form) / std_form       frozen between refits

Signal by SIGNAL_MODE (three runs, so each change is measured separately):
  "g"      : z_ij                                   baseline, raw gap
  "ag"     : z_ag                                   adjusted gap, no factor
  "factor" : z_resid if |t(beta)| >= MIN_ABS_T_STAT, otherwise z_ag
"ag" vs "g" isolates the change of gap measure; "factor" vs "ag" isolates the factor.
In "factor" the fallback is z_ag, not z_ij as in the original spec, so that run only ever
uses AG and matches "ag" exactly in quarters where the t-stat gate fails.

Entry event filter (ENTRY_EVENT_DAYS > 0): a pair doesn't open if either stock filed an
8-K with a material item (EVENT_ITEMS: earnings, deals, executive changes, impairments...)
within the last ENTRY_EVENT_DAYS calendar days. A gap opened by news is likely justified.
It applies in every SIGNAL_MODE and is independent of the factor.

No lookahead: History only returns data with end_time <= the algorithm clock. SEC records
are stamped at midnight of the filing date, but filings often land after the close, so a
filing only counts from the next day, and factor values are used only from their
usable_from time. Every normalization constant is frozen at formation or refit.
"""
from AlgorithmImports import *
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from factor import Factor

# All tunables. Any of these can be overridden by a QC project parameter of the same name.
# The factor's own settings are in factor.py and are overridden the same way.
DEFAULTS = {
    "N_UNIVERSE": 1000, "PREFILTER_MULT": 1.5, "MIN_PRICE": 5.0,
    "FORMATION_DAYS": 252, "N_PAIRS": 20,
    "ONE_PAIR_PER_STOCK": True,  # False = a stock may be in several pairs; orders are netted
    "SAME_INDUSTRY_ONLY": True,  # both stocks in the same Morningstar industry group
    "ENTRY_Z": 2.0, "EXIT_Z": 0.5, "STOP_Z": 6.0, "MAX_HOLD_DAYS": 60,  # stop 6 beat 4 in testing
    "BLOCK_AFTER_STOP": True,  # False = a stopped pair may re-enter once |z| is back below STOP_Z
    "GROSS_LEVERAGE": 2.0, "SLIPPAGE": 0.0005,
    "SIGNAL_MODE": "factor",   # "g" = baseline, "ag" = adjusted gap only, "factor" = AG + factor
    "REQUIRE_FACTOR": True,    # a stock needs a factor value to join a pair; keep the same
                               # for all runs so they trade the same pairs
    "MIN_ABS_T_STAT": 1.0, "STANDARDIZE_FACTOR": True,
    "ENTRY_EVENT_DAYS": 0,     # >0 = skip entries if either stock filed a material 8-K this recently
    "EVENT_ITEMS": "1.01,1.02,1.03,2.01,2.02,2.05,2.06,3.01,4.01,4.02,5.01,5.02",
    "EVENT_REANCHOR": False,    # True = when an 8-K window closes, treat the gap's move over it as justified
    "REANCHOR_MIN_SIGMA": 1.0,  # ...but only if that move exceeds this many formation stds (else noise)
    "EVENT_REFRESH_DAYS": 5,    # re-anchoring looks up pair stocks' 8-Ks every this many trading days
    "AG_VARIANT": "end",       # "end" = quarter-end AG, "avg" = quarter-average AG
    "FILING_LAG_DAYS": 45,     # F(q) is read as of q_end + lag, capped at today
    "START_DATE": "2018-01-01", "END_DATE": "",  # empty = latest available
}
EXCHANGES = ("NYS", "NAS", "ASE")   # NYSE, NASDAQ, AMEX
COMMON_STOCK = "ST00000001"         # Morningstar security type for common shares
SIGNAL_MODES = ("g", "ag", "factor")


# ---------- Pure math (no LEAN dependency, unit-tested) ----------

def growth(prices):
    """V(t) = P(t) / P(t0). prices: T x N array."""
    return prices / prices[0]


def adjusted_gap(v_i, v_j):
    """AG = (V_i - V_j) / (V_i + V_j)."""
    return (v_i - v_j) / (v_i + v_j)


def pair_distances(v):
    """N x N matrix of D_ij from T x N V, using SSD = sum Vi^2 + sum Vj^2 - 2 sum ViVj."""
    sq = (v * v).sum(axis=0)
    return (sq[:, None] + sq[None, :] - 2.0 * (v.T @ v)) / v.shape[0]


def rank_pairs(d, groups=None):
    """Upper-triangle (i, j) index pairs by ascending distance; optionally same group only."""
    i, j = np.triu_indices(d.shape[0], 1)
    if groups is not None:
        keep = (groups[i] == groups[j]) & (groups[i] > 0)
        i, j = i[keep], j[keep]
    order = np.argsort(d[i, j], kind="stable")
    return zip(i[order].tolist(), j[order].tolist())


def pick_pairs(ranked, n, one_per_stock, usable):
    """Greedy selection by ascending distance. usable(k) is evaluated lazily."""
    used, out = set(), []
    for a, b in ranked:
        if len(out) == n:
            break
        if one_per_stock and (a in used or b in used):
            continue
        if usable(a) and usable(b):
            out.append((a, b))
            used.update((a, b))
    return out


def fit_ols(x, y):
    """OLS of y on a constant and x (n x k). Returns alpha, betas, t-stats, r2, n; or None."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, k = x.shape[0] if x.ndim == 2 else 0, x.shape[1] if x.ndim == 2 else 0
    if n <= k + 1 or np.any(x.std(axis=0) < 1e-12):
        return None
    design = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.pinv(design.T @ design)
    coef = xtx_inv @ design.T @ y
    resid = y - design @ coef
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    se = np.sqrt(np.maximum(sse / (n - k - 1) * np.diag(xtx_inv), 1e-30))
    return {"alpha": float(coef[0]), "beta": coef[1:], "t": coef[1:] / se[1:],
            "r2": 1.0 - sse / sst if sst > 0 else 0.0, "n": n}


def last_quarter_ends(today, n):
    """The n most recent completed calendar quarter-end dates, oldest first."""
    year, quarter, out = today.year, (today.month - 1) // 3, []
    for _ in range(n):
        if quarter == 0:
            year, quarter = year - 1, 4
        month = quarter * 3
        out.append(date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1))
        quarter -= 1
    return out[::-1]


def as_of(series, when):
    """Latest value from [(time, value), ...] (sorted) with time <= when, else None."""
    value = None
    for t, v in series:
        if t > when:
            break
        value = v
    return value


def net_weights(symbols, legs):
    """Net target weight per symbol from (long, short, weight) pair legs."""
    weights = {s: 0.0 for s in symbols}
    for long_leg, short_leg, weight in legs:
        if long_leg in weights:
            weights[long_leg] += weight
        if short_leg in weights:
            weights[short_leg] -= weight
    return {s: 0.0 if abs(w) < 1e-12 else w for s, w in weights.items()}


def standardize(values):
    """Cross-sectional z-score of a {key: value} dict."""
    if len(values) < 2:
        return {}
    arr = np.array(list(values.values()), float)
    sd = arr.std(ddof=1)
    return {k: (v - arr.mean()) / sd for k, v in values.items()} if sd > 0 else {}


# ---------- Algorithm ----------

class Pair:
    """A pair plus its frozen formation constants and live state."""

    def __init__(self, a, b, p0a, p0b, mu, sigma, ag_form, form_dates, formed):
        self.a, self.b, self.key = a, b, (a, b)
        self.p0a, self.p0b = p0a, p0b
        self.mu, self.sigma = mu, sigma                 # G stats over formation
        self.ag_form, self.form_dates, self.formed = ag_form, form_dates, formed
        self.ag_mu, self.ag_sigma = ag_form.mean(), ag_form.std(ddof=1)   # AG stats
        self.side, self.held = 0, 0                     # side: +1 = long i, short j
        self.entry = None                               # weight, notional, entry prices
        self.blocked = self.broken = False
        self.model = None                               # alpha, beta, df, res_mean, res_std
        # Re-anchoring: shifts to the pair's normal gap justified by 8-K events.
        self.shift_g = self.shift_ag = 0.0
        self.event = None                               # open window: pre-event gap, end time
        self.event_seen = None                          # latest 8-K date already handled
        self.recent = []                                # last few (date, G, AG) evaluations


class FactorPairs(QCAlgorithm):

    def initialize(self):
        for name, default in DEFAULTS.items():
            setattr(self, name, self._param(name, default))
        self.set_start_date(datetime.strptime(self.START_DATE, "%Y-%m-%d"))
        if self.END_DATE:
            self.set_end_date(datetime.strptime(self.END_DATE, "%Y-%m-%d"))
        self.set_cash(1_000_000)

        if self.SIGNAL_MODE not in SIGNAL_MODES:
            raise ValueError(f"SIGNAL_MODE must be one of {SIGNAL_MODES}, got {self.SIGNAL_MODE!r}")

        self.factor = Factor(self, self._param)   # factor.py
        self._event_items = {i.strip() for i in self.EVENT_ITEMS.split(",") if i.strip()}

        self.pairs, self._candidates, self._groups = [], [], {}
        self._cand_key = self._form_key = self._refit_key = None
        self._subs, self._dirty = set(), set()
        # Material 8-K dates, filled by short history requests (a live SECReport8K subscription
        # per pair stock ran out of memory): symbol -> [filing date], symbol -> fetched-up-to.
        self._events, self._events_fetched = {}, {}
        self._event_skips = self._event_errors = 0
        self._reanchor_stats = {"applied": 0, "below_threshold": 0, "no_baseline": 0}
        if self.EVENT_REANCHOR and self.ENTRY_EVENT_DAYS <= 0:
            raise ValueError("EVENT_REANCHOR needs ENTRY_EVENT_DAYS > 0 (it sets the window length)")
        self.trades, self.refits = [], []
        self._days = self._active_sum = self._max_positions = 0
        self._gross_sum, self._factor_days, self._raw_days = 0.0, 0, 0
        self._max_abs_z = 0.0

        self.universe_settings.resolution = Resolution.DAILY
        self.add_universe(self._select)
        spy = self.add_equity("SPY", Resolution.MINUTE).symbol
        self.schedule.on(self.date_rules.every_day(spy),
                         self.time_rules.before_market_close(spy, 30), self._daily)

    def _param(self, name, default):
        raw = self.get_parameter(name)
        if raw is None or raw == "":
            return default
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes")
        return type(default)(raw)

    def _select(self, fundamental):
        """Records candidates once each January and July. Subscribes to nothing itself."""
        key = (self.time.year, self.time.month)
        if self.time.month not in (1, 7) or key == self._cand_key:
            return Universe.UNCHANGED
        rows = [f for f in fundamental if f.has_fundamental_data and f.price > self.MIN_PRICE]
        total = len(rows)
        rows = [f for f in rows
                if f.security_reference.exchange_id in EXCHANGES
                and f.security_reference.security_type == COMMON_STOCK
                and f.security_reference.is_primary_share
                and not f.security_reference.is_depositary_receipt]
        rows.sort(key=lambda f: f.dollar_volume, reverse=True)
        rows = rows[:int(self.N_UNIVERSE * self.PREFILTER_MULT)]
        self._candidates = [f.symbol for f in rows]
        self._groups = {f.symbol: f.asset_classification.morningstar_industry_group_code
                        for f in rows} if self.SAME_INDUSTRY_ONLY else {}
        self._cand_key = key
        self.log(f"UNIVERSE {self.time:%Y-%m-%d}: {total} priced, {len(rows)} kept")
        return Universe.UNCHANGED

    def _daily(self):
        now = self.time
        half = (now.year, now.month <= 6)
        if now.month in (1, 7) and half != self._form_key and self._cand_key == (now.year, now.month):
            self._form_key = half
            self._form()
        elif now.month in (2, 5, 8, 11) and (now.year, now.month) != self._refit_key:
            self._refit_key = (now.year, now.month)
            self._refit()
        self._evaluate()

    # ---------- Formation ----------

    def _bars(self, symbols, days):
        """Adjusted daily closes and volumes, indexed by trading date, columns by symbol."""
        h = self.history(symbols, self.time - timedelta(days=days), self.time, Resolution.DAILY,
                         data_normalization_mode=DataNormalizationMode.ADJUSTED)
        if h.empty:
            return pd.DataFrame(), pd.DataFrame()
        close, volume = h["close"].unstack(0), h["volume"].unstack(0)
        # Daily bars can end at 16:00 or at the next midnight; both map to the trading date.
        close.index = (close.index - pd.Timedelta(seconds=1)).normalize()
        volume.index = close.index
        return close, volume

    def _form(self):
        t = self.FORMATION_DAYS
        close, volume = self._bars(self._candidates, int(t * 1.6) + 10)
        if len(close) < t:
            self.log(f"FORMATION skipped: {len(close)} rows of history")
            return
        close = close.iloc[-t:].dropna(axis=1)
        adv = (close * volume[close.columns].iloc[-t:]).iloc[-63:].mean()
        keep = adv.nlargest(self.N_UNIVERSE).index
        close = close[sorted(keep, key=str)]     # stable (i, j) order across formations
        symbols, prices = list(close.columns), close.values

        v = growth(prices)
        d = pair_distances(v)
        groups = None
        if self.SAME_INDUSTRY_ONLY:
            groups = np.array([self._groups.get(s) or 0 for s in symbols])
        if self.REQUIRE_FACTOR:
            usable = lambda k: as_of(self.factor.series(symbols[k]), self.time) is not None
        else:
            usable = lambda k: True
        picks = pick_pairs(rank_pairs(d, groups), self.N_PAIRS, self.ONE_PAIR_PER_STOCK, usable)

        previous = {p.key: p for p in self.pairs}
        pairs = []
        for a, b in picks:
            gap = v[:, a] - v[:, b]
            sigma = gap.std(ddof=1)
            if sigma < 1e-9:
                continue
            pair = Pair(symbols[a], symbols[b], prices[0, a], prices[0, b], gap.mean(), sigma,
                        adjusted_gap(v[:, a], v[:, b]), close.index.values, self.time.date())
            if pair.ag_sigma < 1e-9:
                continue
            kept = previous.pop(pair.key, None)
            if kept:                              # re-selected: keep the open position
                pair.side, pair.held, pair.entry = kept.side, kept.held, kept.entry
            pairs.append(pair)
        for dropped in previous.values():
            self._close(dropped, "reformation")
        self._flush()
        self.pairs = pairs
        self._sync_subscriptions()
        self.log(f"FORMATION {self.time:%Y-%m-%d}: N={len(symbols)} " + "; ".join(
            f"{p.a.value}/{p.b.value} {d[symbols.index(p.a), symbols.index(p.b)]:.5f}" for p in pairs))
        self._refit()

    def _sync_subscriptions(self):
        wanted = {s for p in self.pairs for s in p.key}
        for symbol in self._subs - wanted:
            self.remove_security(symbol)
        for symbol in wanted - self._subs:
            security = self.add_security(symbol, Resolution.MINUTE,
                                         data_normalization_mode=DataNormalizationMode.ADJUSTED)
            security.set_slippage_model(ConstantSlippageModel(self.SLIPPAGE))
            # Headroom so a 2.0 gross target is not rejected at the default 2x margin limit.
            security.set_leverage(max(4.0, 2.0 * self.GROSS_LEVERAGE))
        self._subs = wanted

    # ---------- 8-K events ----------

    def _record_event(self, symbol, row):
        """Store the filing date of a material 8-K (one with an item in EVENT_ITEMS)."""
        report = row.report
        if report is None or str(report.form_type or "").upper() != "8-K":
            return
        items = {str(i).strip() for i in (report.items or [])}
        dates = self._events.setdefault(symbol, [])
        if items & self._event_items and row.end_time not in dates:
            dates.append(row.end_time)

    def _refresh_events(self, symbol):
        """Fetch 8-Ks filed since the last fetch for this stock (first time: the last
        ENTRY_EVENT_DAYS + 5 days). The dataset is added only for the request, then removed."""
        now = self.time
        start = self._events_fetched.get(symbol, now - timedelta(days=self.ENTRY_EVENT_DAYS + 5))
        if start >= now:
            return
        dataset = self.add_data(SECReport8K, symbol, Resolution.DAILY).symbol
        try:
            for row in self.history[SECReport8K](dataset, start, now, Resolution.DAILY):
                self._record_event(symbol, row)
        except Exception as error:
            self._event_errors += 1
            if self._event_errors <= 5:
                self.log(f"FILINGS {symbol.value} SECReport8K: {error}")
        finally:
            self.remove_security(dataset)
        self._events_fetched[symbol] = now

    def _event_dates(self, symbol):
        """Material 8-K dates usable by now: an 8-K counts from the day after filing."""
        now = self.time
        keep = timedelta(days=self.ENTRY_EVENT_DAYS + 5)
        dates = self._events.get(symbol, [])
        dates[:] = [d for d in dates if now - d <= keep]
        return [d for d in dates if d + timedelta(days=1) <= now]

    def _recent_event(self, symbol):
        """True if the stock filed a material 8-K within the last ENTRY_EVENT_DAYS days.
        Fetches up to now first, so entry decisions always see every filing to date."""
        self._refresh_events(symbol)
        window = timedelta(days=self.ENTRY_EVENT_DAYS)
        return any(self.time - d <= window for d in self._event_dates(symbol))

    def _track_events(self, pair, g, ag):
        """Re-anchoring. A new 8-K for either stock opens a window (or extends an open one)
        and records the gap from the last evaluation before the filing date. When the window
        closes, the gap's move over it is taken as justified by the news and added to the
        pair's normal level, if it exceeds REANCHOR_MIN_SIGMA formation stds."""
        now = self.time
        window = timedelta(days=self.ENTRY_EVENT_DAYS)
        latest = max((d for s in pair.key for d in self._event_dates(s)), default=None)
        if latest is not None and (pair.event_seen is None or latest > pair.event_seen):
            pair.event_seen = latest
            if pair.event is None:
                before = [r for r in pair.recent if r[0] < latest.date()]
                pair.event = {"pre": before[-1] if before else None, "until": latest + window}
            else:
                pair.event["until"] = max(pair.event["until"], latest + window)
        if pair.event is None or now <= pair.event["until"]:
            return
        pre, pair.event = pair.event["pre"], None
        if pre is None:                     # filed before the pair had any evaluations
            self._reanchor_stats["no_baseline"] += 1
            return
        move_g, move_ag = g - pre[1], ag - pre[2]
        move, scale = (move_g, pair.sigma) if self.SIGNAL_MODE == "g" else (move_ag, pair.ag_sigma)
        if abs(move) >= self.REANCHOR_MIN_SIGMA * scale:
            pair.shift_g += move_g
            pair.shift_ag += move_ag
            self._reanchor_stats["applied"] += 1
        else:
            self._reanchor_stats["below_threshold"] += 1

    # ---------- Factor regression ----------

    def _snapshot(self, symbols, when):
        """Latest known factor value per symbol at `when`, standardized if configured."""
        raw = {s: as_of(self.factor.series(s), when) for s in symbols}
        raw = {s: v for s, v in raw.items() if v is not None}
        return standardize(raw) if self.STANDARDIZE_FACTOR else raw

    def _factor_diffs(self, pair, snapshot):
        """Regressors for a pair, or None if any is missing.
        A different factor goes in factor.py. To use several at once, append more
        differences here; fit_ols already handles k factors, but the gate and the model
        dictionary in _refit use beta[0] only and would need extending too."""
        if pair.a not in snapshot or pair.b not in snapshot:
            return None
        return [snapshot[pair.a] - snapshot[pair.b]]

    def _refit(self):
        """Pooled OLS over the last 4 quarters and all live pairs, then refresh residual stats."""
        live = [p for p in self.pairs if not p.broken]
        for pair in live:
            pair.model = None
        if self.SIGNAL_MODE != "factor" or not live:
            return
        now = self.time
        quarters = last_quarter_ends(now.date(), 4)
        symbols = sorted({s for p in live for s in p.key}, key=str)
        close, _ = self._bars(symbols, (now.date() - quarters[0]).days + 100)
        if close.empty:
            return
        lag = timedelta(days=self.FILING_LAG_DAYS)
        snapshots = [self._snapshot(symbols, min(datetime.combine(q, datetime.min.time()) + lag, now))
                     for q in quarters]
        current = self._snapshot(symbols, now)

        xs, ys, history = [], [], {}
        for pair in live:
            ag = adjusted_gap(close[pair.a] / pair.p0a, close[pair.b] / pair.p0b).dropna()
            history[pair.key] = []
            for q, snapshot in zip(quarters, snapshots):
                diffs = self._factor_diffs(pair, snapshot)
                history[pair.key].append(diffs[0] if diffs else np.nan)
                in_quarter = ag.loc[pd.Timestamp(q.year, q.month - 2, 1):pd.Timestamp(q)]
                if diffs is None or in_quarter.empty:
                    continue
                xs.append(diffs)
                ys.append(in_quarter.iloc[-1] if self.AG_VARIANT == "end" else in_quarter.mean())

        fit = fit_ols(xs, ys)
        if fit is None:
            self.log(f"REFIT {now:%Y-%m-%d}: not estimable, n={len(ys)}; all pairs raw")
            return
        beta, t_stat = float(fit["beta"][0]), float(fit["t"][0])
        self.refits.append((now.date(), fit["alpha"], beta, t_stat, fit["r2"], fit["n"]))
        self.plot("Factor", "beta", beta)
        self.plot("Factor", "t(beta)", t_stat)

        factor_pairs = 0
        if abs(t_stat) >= self.MIN_ABS_T_STAT:
            q_dates = np.array(quarters, dtype="datetime64[ns]")
            for pair in live:
                diffs = self._factor_diffs(pair, current)
                steps = pd.Series(history[pair.key], dtype=float).ffill().bfill().values
                if diffs is None or np.isnan(steps).any():
                    continue
                # dF is a step function between quarters across the formation window.
                index = np.searchsorted(q_dates, pair.form_dates, side="right") - 1
                residual = pair.ag_form - (fit["alpha"] + beta * steps[np.clip(index, 0, 3)])
                res_std = residual.std(ddof=1)
                if res_std > 1e-9:
                    pair.model = {"alpha": fit["alpha"], "beta": beta, "df": diffs[0],
                                  "res_mean": residual.mean(), "res_std": res_std}
                    factor_pairs += 1
        self.log(f"REFIT {now:%Y-%m-%d}: alpha={fit['alpha']:.5f} beta={beta:.5f} t={t_stat:.2f} "
                 f"R2={fit['r2']:.3f} n={fit['n']} factor_pairs={factor_pairs}/{len(live)}")

    # ---------- Trading ----------

    def _signal(self, pair):
        v_a = self.securities[pair.a].price / pair.p0a
        v_b = self.securities[pair.b].price / pair.p0b
        # shift_g / shift_ag are 0 unless EVENT_REANCHOR moved the pair's normal level.
        if self.SIGNAL_MODE == "g":
            return (v_a - v_b - pair.mu - pair.shift_g) / pair.sigma
        model = pair.model
        if model is None:                    # "ag" mode, or "factor" mode with the gate failed
            self._raw_days += 1
            return (adjusted_gap(v_a, v_b) - pair.ag_mu - pair.shift_ag) / pair.ag_sigma
        self._factor_days += 1
        residual = adjusted_gap(v_a, v_b) - (model["alpha"] + model["beta"] * model["df"])
        return (residual - model["res_mean"] - pair.shift_ag) / model["res_std"]

    def _tradable(self, symbol):
        if symbol not in self.securities:
            return False
        security = self.securities[symbol]
        last = security.get_last_data()
        return (security.is_tradable and security.has_data and security.price > 0
                and last is not None and (self.time - last.end_time).days <= 5)

    def _close(self, pair, reason):
        if pair.side == 0:
            return
        # Per-pair P&L estimate before costs. A shared stock's holding can't be split by pair.
        entry = pair.entry
        return_a = float(self.securities[pair.a].price) / entry["price_a"] - 1.0
        return_b = float(self.securities[pair.b].price) / entry["price_b"] - 1.0
        pnl = pair.side * entry["notional"] * (return_a - return_b)
        self.trades.append({"pair": f"{pair.a.value}/{pair.b.value}", "reason": reason,
                            "held": pair.held, "pnl": pnl})
        self._dirty.update(pair.key)
        pair.side, pair.held, pair.entry = 0, 0, None

    def _flush(self):
        """Send net targets for changed stocks, so closing one pair leaves others' legs alone."""
        if not self._dirty:
            return
        legs = []
        for pair in self.pairs:
            if pair.side:
                long_leg, short_leg = pair.key if pair.side > 0 else pair.key[::-1]
                legs.append((long_leg, short_leg, pair.entry["weight"]))
        targets = net_weights(self._dirty, legs)
        self.set_holdings([PortfolioTarget(s, w) for s, w in targets.items()])
        self._dirty.clear()

    def _break(self, pair, reason):
        self.log(f"BROKEN {self.time:%Y-%m-%d} {pair.a.value}/{pair.b.value}: {reason}")
        self._close(pair, "broken")
        pair.broken = True

    def on_data(self, data):
        for symbol, event in data.delistings.items():
            if event.type == DelistingType.WARNING:
                for pair in self.pairs:
                    if symbol in pair.key and not pair.broken:
                        self._break(pair, f"delisting warning for {symbol.value}")
        self._flush()

    def _evaluate(self):
        active = [p for p in self.pairs if not p.broken and not p.blocked]
        today = self.time.date()
        for pair in self.pairs:
            if pair.broken or pair.blocked or pair.formed == today:
                continue                # new subscriptions have no price yet on formation day
            if not (self._tradable(pair.a) and self._tradable(pair.b)):
                self._break(pair, f"tradable: {pair.a.value}={self._tradable(pair.a)} "
                                  f"{pair.b.value}={self._tradable(pair.b)}")
                continue
            if self.EVENT_REANCHOR:
                # Refresh every EVENT_REFRESH_DAYS evaluations (daily lookups for every pair
                # stock would be far too slow). A filing may be noticed late, but its window
                # and pre-filing gap come from the filing date and the stored evaluations.
                if self._days % self.EVENT_REFRESH_DAYS == 0:
                    self._refresh_events(pair.a)
                    self._refresh_events(pair.b)
                v_a = self.securities[pair.a].price / pair.p0a
                v_b = self.securities[pair.b].price / pair.p0b
                g, ag = float(v_a - v_b), float(adjusted_gap(v_a, v_b))
                self._track_events(pair, g, ag)
                pair.recent = (pair.recent + [(today, g, ag)])[-40:]
            z = self._signal(pair)
            self._max_abs_z = max(self._max_abs_z, abs(z))
            if pair.side:
                pair.held += 1
                if -pair.side * z <= self.EXIT_Z:   # also catches a jump through zero
                    self._close(pair, "converged")
                elif abs(z) >= self.STOP_Z:
                    self._close(pair, "stop")
                    pair.blocked = self.BLOCK_AFTER_STOP
                elif pair.held >= self.MAX_HOLD_DAYS:
                    self._close(pair, "max_hold")
                    pair.blocked = self.BLOCK_AFTER_STOP
            elif self.ENTRY_Z <= abs(z) < self.STOP_Z:
                if self.ENTRY_EVENT_DAYS > 0 and (self._recent_event(pair.a)
                                                  or self._recent_event(pair.b)):
                    self._event_skips += 1          # gap likely news-driven: don't trade it
                    continue
                pair.side = -1 if z > 0 else 1      # z > 0: i is rich, short i, long j
                weight = self.GROSS_LEVERAGE / (2.0 * max(1, len(active)))
                pair.entry = {"weight": weight,
                              "notional": weight * float(self.portfolio.total_portfolio_value),
                              "price_a": float(self.securities[pair.a].price),
                              "price_b": float(self.securities[pair.b].price)}
                self._dirty.update(pair.key)
        self._flush()
        self._days += 1
        self._active_sum += len(active)
        self._gross_sum += float(self.portfolio.total_absolute_holdings_cost
                                 / self.portfolio.total_portfolio_value)
        positions = sum(1 for holding in self.portfolio.values() if holding.invested)
        self._max_positions = max(self._max_positions, positions)

    def on_end_of_algorithm(self):
        trades, days = self.trades, max(1, self._days)
        count = max(1, len(trades))
        reasons = {r: sum(t["reason"] == r for t in trades)
                   for r in ("converged", "stop", "max_hold", "broken", "reformation")}
        self.log(f"SUMMARY signal_mode={self.SIGNAL_MODE} factor={self.factor.name} "
                 f"entry_event_days={self.ENTRY_EVENT_DAYS} event_skips={self._event_skips} "
                 f"reanchor={self.EVENT_REANCHOR} reanchors={self._reanchor_stats} "
                 f"stop_z={self.STOP_Z} block={self.BLOCK_AFTER_STOP} trades={len(trades)} "
                 f"win_rate={sum(t['pnl'] > 0 for t in trades) / count:.3f} "
                 f"avg_hold={sum(t['held'] for t in trades) / count:.1f} "
                 f"avg_active_pairs={self._active_sum / days:.1f} "
                 f"stop_rate={(reasons['stop'] + reasons['max_hold']) / count:.3f} exits={reasons} "
                 f"avg_gross={self._gross_sum / days:.2f} max_positions={self._max_positions} "
                 f"max_abs_z={self._max_abs_z:.2f} eval_days={self._days} "
                 f"factor_share={self._factor_days / max(1, self._factor_days + self._raw_days):.3f} "
                 f"event_errors={self._event_errors} {self.factor.summary()}")
        if self.refits:
            betas = [r[2] for r in self.refits]
            self.log(f"SUMMARY refits={len(betas)} beta_mean={np.mean(betas):.5f} "
                     f"beta_std={np.std(betas):.5f} "
                     f"share_abs_t_ge_2={np.mean([abs(r[3]) >= 2 for r in self.refits]):.3f}")
