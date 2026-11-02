"""
Distance-method pairs trading with a lexical-density filter (LEAN, Python).
Paste into a QuantConnect cloud project as main.py. Baseline vs factor: USE_FACTOR.

DEFINITIONS (i, j order is fixed at formation)
  V_i(t)    = P_i(t) / P_i(t0)                growth of $1, adjusted prices
  G_ij(t)   = V_i(t) - V_j(t)                 signed gap
  D_ij      = (1/T) * sum_t G_ij(t)^2         raw squared distance, not demeaned
  z_ij(t)   = (G_ij(t) - mu_ij) / sigma_ij    mu, sigma frozen over formation window
  AG_ij(t)  = G_ij(t) / (V_i(t) + V_j(t))     adjusted gap
  dLD_ij(q) = LD_i(q) - LD_j(q)               report_sentiment.lexical_density
  Model     : AG_p(q) = alpha + beta * dLD_p(q) + e   pooled OLS, last 4 quarters
  Residual  : AG(t) - (alpha + beta * dLD latest known)
  z_resid   = (residual - mean_form) / std_form       frozen between refits

Signal: z_resid if |t(beta)| >= MIN_ABS_T_STAT, otherwise raw z_ij.
No lookahead: History only returns data with end_time <= the algorithm clock, and
every normalization constant is frozen at formation or refit.
"""
from AlgorithmImports import *
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

# All tunables. Any of these can be overridden by a QC project parameter of the same name.
DEFAULTS = {
    "N_UNIVERSE": 1000, "PREFILTER_MULT": 1.5, "MIN_PRICE": 5.0,
    "FORMATION_DAYS": 252, "N_PAIRS": 20,
    "ONE_PAIR_PER_STOCK": True,  # False = a stock may be in several pairs; orders are netted
    "SAME_INDUSTRY_ONLY": False,
    "ENTRY_Z": 2.0, "EXIT_Z": 0.5, "STOP_Z": 4.0, "MAX_HOLD_DAYS": 60,
    "GROSS_LEVERAGE": 2.0, "SLIPPAGE": 0.0005,
    "USE_FACTOR": True,        # False = baseline (raw z-score)
    "REQUIRE_BRAIN": True,     # keep True for both runs so they trade the same pairs
    "MIN_ABS_T_STAT": 1.0, "STANDARDIZE_LD": True,
    "LD_VARIANT": "all",       # "all" = 10-K and 10-Q, "10k" = 10-K only
    "AG_VARIANT": "end",       # "end" = quarter-end AG, "avg" = quarter-average AG
    "FILING_LAG_DAYS": 45,     # LD(q) is read as of q_end + lag, capped at today
    "START_DATE": "2018-01-01", "END_DATE": "",  # empty = latest available
}
EXCHANGES = ("NYS", "NAS", "ASE")   # NYSE, NASDAQ, AMEX
COMMON_STOCK = "ST00000001"         # Morningstar security type for common shares


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
        self.mu, self.sigma = mu, sigma
        self.ag_form, self.form_dates, self.formed = ag_form, form_dates, formed
        self.side, self.held = 0, 0                     # side: +1 = long i, short j
        self.entry = None                               # weight, notional, entry prices
        self.blocked = self.broken = False
        self.model = None                               # alpha, beta, dld, res_mean, res_std


class LexicalDensityPairs(QCAlgorithm):

    def initialize(self):
        for name, default in DEFAULTS.items():
            setattr(self, name, self._param(name, default))
        self.set_start_date(datetime.strptime(self.START_DATE, "%Y-%m-%d"))
        if self.END_DATE:
            self.set_end_date(datetime.strptime(self.END_DATE, "%Y-%m-%d"))
        self.set_cash(1_000_000)

        variants = {"all": BrainCompanyFilingLanguageMetricsAll,
                    "10k": BrainCompanyFilingLanguageMetrics10K}
        self._ld_type = variants[self.LD_VARIANT]

        self.pairs, self._candidates, self._groups = [], [], {}
        self._cand_key = self._form_key = self._refit_key = None
        self._subs, self._brain, self._ld, self._dirty = set(), {}, {}, set()
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
        self._ld = {}
        usable = (lambda k: bool(self._ld_series(symbols[k]))) if self.REQUIRE_BRAIN else (lambda k: True)
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
        for symbol in [s for s in self._brain if s not in wanted]:
            self.remove_security(self._brain.pop(symbol))
        for symbol in self._subs - wanted:
            self.remove_security(symbol)
        for symbol in wanted - self._subs:
            security = self.add_security(symbol, Resolution.MINUTE,
                                         data_normalization_mode=DataNormalizationMode.ADJUSTED)
            security.set_slippage_model(ConstantSlippageModel(self.SLIPPAGE))
            # Headroom so a 2.0 gross target is not rejected at the default 2x margin limit.
            security.set_leverage(max(4.0, 2.0 * self.GROSS_LEVERAGE))
        self._subs = wanted

    # ---------- Factor ----------

    def _ld_series(self, symbol):
        """[(end_time, lexical density)] known by now. Empty means no Brain data for the stock."""
        if symbol not in self._ld:
            if symbol not in self._brain:
                self._brain[symbol] = self.add_data(self._ld_type, symbol, Resolution.DAILY).symbol
            rows = self.history[self._ld_type](self._brain[symbol], self.time - timedelta(days=600),
                                               self.time, Resolution.DAILY)
            series = []
            for row in rows:
                metrics = row.report_sentiment
                value = metrics.lexical_density if metrics is not None else None
                if value is not None:
                    series.append((row.end_time, float(value)))
            self._ld[symbol] = sorted(series, key=lambda item: item[0])
        return self._ld[symbol]

    def _snapshot(self, symbols, when):
        """Latest known LD per symbol at `when`, standardized if configured."""
        raw = {s: as_of(self._ld_series(s), when) for s in symbols}
        raw = {s: v for s, v in raw.items() if v is not None}
        return standardize(raw) if self.STANDARDIZE_LD else raw

    def _factor_diffs(self, pair, snapshot):
        """Regressors for a pair, or None if any is missing.
        EXTENSION POINT: append more differences here (CapEx/assets, revenue change,
        text-change measures). fit_ols already handles k factors; the gate and the
        model dictionary in _refit use beta[0] only and would need extending too."""
        if pair.a not in snapshot or pair.b not in snapshot:
            return None
        return [snapshot[pair.a] - snapshot[pair.b]]

    def _refit(self):
        """Pooled OLS over the last 4 quarters and all live pairs, then refresh residual stats."""
        live = [p for p in self.pairs if not p.broken]
        for pair in live:
            pair.model = None
        if not self.USE_FACTOR or not live:
            return
        self._ld = {}
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
                # dLD is a step function between quarters across the formation window.
                index = np.searchsorted(q_dates, pair.form_dates, side="right") - 1
                residual = pair.ag_form - (fit["alpha"] + beta * steps[np.clip(index, 0, 3)])
                res_std = residual.std(ddof=1)
                if res_std > 1e-9:
                    pair.model = {"alpha": fit["alpha"], "beta": beta, "dld": diffs[0],
                                  "res_mean": residual.mean(), "res_std": res_std}
                    factor_pairs += 1
        self.log(f"REFIT {now:%Y-%m-%d}: alpha={fit['alpha']:.5f} beta={beta:.5f} t={t_stat:.2f} "
                 f"R2={fit['r2']:.3f} n={fit['n']} factor_pairs={factor_pairs}/{len(live)}")

    # ---------- Trading ----------

    def _signal(self, pair):
        v_a = self.securities[pair.a].price / pair.p0a
        v_b = self.securities[pair.b].price / pair.p0b
        model = pair.model
        if model is None:
            self._raw_days += 1
            return (v_a - v_b - pair.mu) / pair.sigma
        self._factor_days += 1
        residual = adjusted_gap(v_a, v_b) - (model["alpha"] + model["beta"] * model["dld"])
        return (residual - model["res_mean"]) / model["res_std"]

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
            z = self._signal(pair)
            self._max_abs_z = max(self._max_abs_z, abs(z))
            if pair.side:
                pair.held += 1
                if -pair.side * z <= self.EXIT_Z:   # also catches a jump through zero
                    self._close(pair, "converged")
                elif abs(z) >= self.STOP_Z:
                    self._close(pair, "stop")
                    pair.blocked = True
                elif pair.held >= self.MAX_HOLD_DAYS:
                    self._close(pair, "max_hold")
                    pair.blocked = True
            elif self.ENTRY_Z <= abs(z) < self.STOP_Z:
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
        self.log(f"SUMMARY use_factor={self.USE_FACTOR} trades={len(trades)} "
                 f"win_rate={sum(t['pnl'] > 0 for t in trades) / count:.3f} "
                 f"avg_hold={sum(t['held'] for t in trades) / count:.1f} "
                 f"avg_active_pairs={self._active_sum / days:.1f} "
                 f"stop_rate={(reasons['stop'] + reasons['max_hold']) / count:.3f} exits={reasons} "
                 f"avg_gross={self._gross_sum / days:.2f} max_positions={self._max_positions} "
                 f"max_abs_z={self._max_abs_z:.2f} eval_days={self._days} "
                 f"factor_share={self._factor_days / max(1, self._factor_days + self._raw_days):.3f}")
        if self.refits:
            betas = [r[2] for r in self.refits]
            self.log(f"SUMMARY refits={len(betas)} beta_mean={np.mean(betas):.5f} "
                     f"beta_std={np.std(betas):.5f} "
                     f"share_abs_t_ge_2={np.mean([abs(r[3]) >= 2 for r in self.refits]):.3f}")
