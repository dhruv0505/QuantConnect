# Task: Build a Pairs-Trading Strategy with a Lexical-Density Filter in QuantConnect (LEAN, Python)

You are building a QuantConnect LEAN algorithm (Python) from scratch. Work in a new project directory with `main.py` (the algorithm), a `research.ipynb` (optional sanity checks), and a short `README.md`. Use the `lean` CLI if it is installed (`lean project-create`, `lean backtest`); otherwise write files that can be pasted into the QuantConnect cloud IDE. Read the QuantConnect docs for anything you are unsure about (universe selection, `History`, Brain datasets) rather than guessing API signatures.

---

## 1. Strategy overview

This is a **statistical-arbitrage pairs strategy** (the "distance method", Gatev, Goetzmann & Rouwenhorst 2006) with one addition: a **fundamental text factor (lexical density of 10-K/10-Q filings)** used to decide whether a price gap between two paired stocks is "justified" or likely to mean-revert.

Baseline idea, in plain terms:
1. Build a liquid universe of US stocks (NYSE, AMEX, NASDAQ).
2. Every 6 months, find the 20 pairs of stocks whose normalized prices moved most closely together over the prior year.
3. Every day, measure how far each pair has drifted apart. When the gap is unusually large, go long the underperformer and short the outperformer, expecting convergence.
4. **Problem:** some gaps are permanent (one company genuinely started winning). **Fix:** regress the gap on the difference in lexical density between the two companies' latest filings. Only trade the part of the gap the factor does *not* explain.

---

## 2. Formal definitions (these replace the informal terms in my original notes)

Use exactly these definitions. Put them in a module-level docstring and as named functions so the code is auditable.

**Universe (`U`).** Common stocks listed on NYSE, AMEX, or NASDAQ (use QC's `Exchange` / `PrimaryExchange` fine-fundamental fields). Liquidity filter at each formation date:
- `HasFundamentalData == True`
- price > $5 (config)
- 63-trading-day average daily dollar volume in the top `N_UNIVERSE` names (default 1000, config) — this is how "illiquid stocks are removed"
- Must have the Brain filing metrics (section 5) available, otherwise exclude.
- Exclude ETFs/ADRs/preferreds.

**Growth of $1, `V_i(t)`.** For stock *i*, using split/dividend-adjusted close prices and a formation window starting at date `t0`:
`V_i(t) = P_i(t) / P_i(t0)`.
This is what my notes called "simulate return if you put in 1 dollar 1 year ago." Note it is a *level* (starts at 1.0), not a percent return.

**Gap (the "stock gap"), `G_ij(t)`.** `G_ij(t) = V_i(t) − V_j(t)`. Signed. Pair ordering (i, j) is fixed at formation.

**Distance (the "mean squared of stock gap").** Over the formation window of `T` trading days (default 252):
`D_ij = (1/T) * Σ_t G_ij(t)^2`
Equivalently the sum of squared deviations divided by T (SSD/T). Do **not** demean the gap, this is raw squared distance as in Gatev et al. Because it is a monotone transform of SSD, rank by SSD.

**Pair selection.** Select the 20 pairs with the smallest `D_ij`, subject to:
- each stock may appear in at most one pair (greedy by ascending D, skip pairs whose stocks are already used). Make this a config flag, default True, to avoid concentrated exposure.
- optional flag `SAME_INDUSTRY_ONLY` (default False): require same Morningstar industry group. Implement it, but leave off by default so we can compare.

**Re-formation schedule.** Re-select pairs every 6 months (first trading day of January and July). On re-formation, close all positions in pairs that are dropped and reset normalization (`t0` = start of the new 252-day formation window).

**Normalized spread / z-score (the "deviation calculation," done daily).**
`z_ij(t) = (G_ij(t) − μ_ij) / σ_ij`
where `μ_ij` and `σ_ij` are the mean and standard deviation of `G_ij` **over the formation window** (frozen between re-formations, no lookahead).

**Adjusted gap (from my notes: "gap / summed return").**
`AG_ij(t) = G_ij(t) / (V_i(t) + V_j(t))`
i.e. the gap as a fraction of the combined growth of $1 in each stock. This makes gaps comparable across pairs with very different overall performance. (Note: `V_i + V_j` is the sum of the two growth-of-$1 levels, not the sum of percent returns.)

**Factor: lexical density, `LD_i(q)`.** For stock *i* and quarter *q*, use the lexical density metric from the **Brain Language Metrics on Company Filings (BLMCF)** dataset on QuantConnect. Per the Quantpedia write-up (https://quantpedia.com/strategies/how-to-use-lexical-density-of-company-filings, source paper Hanicova, Kalus, Vojtko, SSRN 3921091): lexical density measures how many *information-carrying* words (content words such as nouns, verbs, adjectives, adverbs, as opposed to function words like "the", "of", "and") a filing contains; high lexical density = more information-carrying words. Conceptually `LD = (# lexical words) / (# total words)`.
- **Look up the exact field name and definition in the QC Brain BLMCF docs** (https://www.quantconnect.com/datasets/brain-language-metrics-company-filings) and implement against what Brain actually provides. Do not re-compute it from raw text unless the dataset is unavailable.
- Use the **10-Q/10-K-appropriate series** the dataset offers (it has all-filings and 10-K-only variants; pick one, document the choice, and make it a config option).
- Use the value that was **known at the time**: the factor for quarter *q* only becomes usable on the filing's publication/data timestamp, not the period-end date. This is critical for avoiding lookahead.

**Factor difference.** `ΔLD_ij(q) = LD_i(q) − LD_j(q)`, using the same (i, j) ordering as the gap. Optionally cross-sectionally standardize `LD` (z-score across the universe) at each quarter before differencing; make this a config flag (default True).

**Regression (the "run regression every quarter for the past year").** With only 4 quarters of history per pair, a per-pair regression has 4 data points and is meaningless. Instead use a **pooled panel regression** across all active pairs:
- Sample: for each of the last 4 completed quarters `q` and each active pair `p`, one observation `(ΔLD_p(q), AG_p(q_end))` where `AG_p(q_end)` is the adjusted gap on the last trading day of quarter `q` (also implement a variant using the quarter-average of daily AG; config flag).
- Model: `AG_p(q) = α + β · ΔLD_p(q) + ε_p(q)`
- Fit with OLS (numpy / statsmodels is fine; check availability in QC).
- Refit every quarter, immediately after the new filings' data timestamps have passed (e.g. first trading day of Feb, May, Aug, Nov, or on a rolling "N days after quarter-end" rule; config).
- Store `α, β, R², t-stat(β), n_obs` each refit and log them.

**Predicted vs. residual gap.**
- `AG_hat_p(t) = α + β · ΔLD_p(latest known)`
- `residual_p(t) = AG_p(t) − AG_hat_p(t)`
- **The trading signal uses the residual**, not the raw gap: the factor tells us what gap is "fair"; we trade deviations from that.
- Standardize: `z_resid_p(t) = (residual_p(t) − mean_resid_form) / std_resid_form`, with the mean/std computed over the formation window using the *fitted* model (frozen between refits). 
- **Gate:** only use the factor-adjusted signal if `β` is statistically meaningful (config `MIN_ABS_T_STAT`, default 1.0; below it, fall back to the raw z-score `z_ij`). Log which mode each pair is in.

**CapEx (stray note in my draft).** I wrote "CapEx: Capital Expenditure" as a reminder of a candidate *second factor*. Do NOT implement it in v1. Structure the factor code so additional factors (CapEx / assets from QC fundamentals, revenue change, 10-K/10-Q text-change measures) can be added later as extra columns in the regression (`AG ~ α + Σ β_k ΔF_k`). Leave a clearly marked extension point.

---

## 3. Trading rules

- **Position sizing:** dollar-neutral per pair. Allocate `1/N_active_pairs` of gross target exposure to each pair, half long / half short. Default gross leverage 2.0 (config).
- **Entry:** if `z ≥ +ENTRY_Z` (default 2.0): the pair's first stock (i) is rich → short i, long j. If `z ≤ −ENTRY_Z`: long i, short j. (`z` means the factor-adjusted z-score when the gate passes, else the raw z-score.)
- **Exit (convergence):** close when `|z| ≤ EXIT_Z` (default 0.5).
- **Stop-loss / divergence:** close when `|z| ≥ STOP_Z` (default 4.0), or after `MAX_HOLD_DAYS` (default 60 trading days). This directly addresses the "gap genuinely widens" failure mode. Flag the pair as `blocked` until the next re-formation after a stop-out.
- **Daily evaluation** of z-scores at a fixed time (e.g. 30 min before close) using adjusted data; place orders with `SetHoldings` / `MarketOrder` as appropriate. Be careful with short availability and use QC's `Securities[sym].IsTradable`.
- Handle delistings, symbol changes, halted names, and missing data defensively: if either leg of a pair can't be traded or priced, close the other leg and drop the pair until re-formation.
- Transaction costs: use QC's default fee model and a conservative slippage model (e.g. `ConstantSlippageModel` ~ 5 bps). Do not disable costs.

---

## 4. Implementation notes (QuantConnect specifics)

- Use `AddUniverse` with coarse + fine selection (or the current QC universe API; check docs) with **6-month re-selection** for the pair formation and a monthly cheap refresh only if needed. Don't recompute O(N²) pair distances daily.
- **Compute the pairwise distance matrix vectorized with NumPy.** For N=1000 and T=252, build matrix `V` (T×N), then `D = (V² summed)` via the identity `SSD_ij = Σ V_i² + Σ V_j² − 2 Σ V_i V_j` (a single matrix multiply). Take the upper triangle, sort, and pick greedily. Do not write nested Python loops.
- Request history with `self.History(symbols, 252, Resolution.Daily)` using adjusted data normalization; verify adjustment mode explicitly and document it.
- Add the Brain BLMCF dataset for each selected stock (`self.AddData(BrainCompanyFilingLanguageMetrics...)` or the current class names; look up exact API) and cache the latest known lexical-density value per symbol with its availability timestamp. Remove subscriptions for dropped symbols to stay within data limits.
- **No lookahead anywhere:** prices up to and including the decision day only, factors only from data timestamps ≤ the decision day, normalization constants frozen at formation/refit.
- Put all tunables in a single config block at the top of `main.py` using `self.GetParameter(...)` so they can be swept in QC optimization: `N_UNIVERSE, FORMATION_DAYS, N_PAIRS, ENTRY_Z, EXIT_Z, STOP_Z, MAX_HOLD_DAYS, GROSS_LEVERAGE, SAME_INDUSTRY_ONLY, USE_FACTOR, MIN_ABS_T_STAT, STANDARDIZE_LD, LD_VARIANT, AG_VARIANT`.
- Backtest range: start 2018-01-01 (Brain BLMCF coverage begins ~2010 but give ample warm-up; check the docs), end at the latest available date. Starting cash $1,000,000.
- Use `self.Debug`/`self.Log` sparingly; log one line per re-formation (selected pairs, distances) and per regression refit (α, β, t-stat, R², n_obs).

---

## 5. Required comparison runs (build this in from the start)

The point of the project is to find out whether the lexical-density filter helps. Implement it so a single flag (`USE_FACTOR`) switches modes, and produce results for:
1. **Baseline:** pure distance-method pairs, raw z-score signal.
2. **Factor-adjusted:** same pairs and parameters, signal = regression residual z-score (with the t-stat gate).
Report side by side: CAGR, annualized vol, Sharpe, max drawdown, win rate, average holding period, number of trades, average pairs active, and the % of stop-outs. Also report the time series of `β` and its t-stat so we can see whether the factor relationship is stable or noise.

---

## 6. Deliverables and working style

1. `main.py`: the full algorithm, clean and commented, with the formal definitions above as docstrings.
2. `README.md`: one page, covering the definitions, config options, how to run, and known limitations.
3. A short results summary comparing baseline vs. factor-adjusted.
4. Before writing code, **state your plan and flag any ambiguity or API uncertainty** (especially exact Brain dataset class/field names, and fine-fundamental field names for exchange/industry). When something in this spec is underdetermined, pick the most defensible option, make it a config flag, and note it in the README. Do not silently invent API names; check the docs.
5. Run a backtest and fix errors until it completes. Sanity-check: pairs should look like economically related companies (e.g. beverage, banks, utilities), gross exposure should be near target, and the position count should never exceed `2 × N_PAIRS` stocks.
6. Do not overstate results. If the factor-adjusted version does not beat the baseline, say so plainly.

## 7. Known risks to keep in mind (mention in the README)

- Pairs can break permanently (the motivation for the factor); the stop-loss and max-hold rules limit but don't eliminate this.
- 4 quarterly observations per pair is thin; the pooled regression across pairs is what makes it estimable, and it assumes the same β applies to all pairs.
- Lexical density is a text-style metric, and the relationship between it and relative stock performance is an empirical claim (the Quantpedia strategy sorts stocks monthly on it in a long-short decile portfolio); we are testing a different usage of it, so a weak or unstable β is a plausible outcome.
- Short-selling costs/borrow availability are not fully modeled in backtests.
