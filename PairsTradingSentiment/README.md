# Factor Pairs Trading

A distance-method pairs strategy (Gatev et al. 2006). It tests whether a per-stock factor, by default lexical density from company filings, separates justified price gaps from gaps that revert. Definitions are in the `main.py` docstring; the full spec is in `instructions.md`.

## Files
| File | Role |
|---|---|
| `main.py` | The strategy: universe, pairs, signals, the factor regression, the 8-K filter and re-anchoring, trading. Doesn't depend on which factor is used. |
| `factor.py` | The factor. Currently filing text (lexical density or similarity). Replace it to test a different factor. |
| `config.json` | Marks the folder as a Lean project for the CLI. |

`research/filing_stability_edgar.py` (repo root) is the earlier standalone EDGAR/Brain script; it isn't used by the backtest.

### Writing a different factor
`factor.py` must define a class `Factor` with:
- `__init__(self, algorithm, param)`: `algorithm` is the running algorithm (for `history`, `add_data`, `time`, `log`), and `param(name, default)` reads a project parameter typed like the default.
- `name`: a short label for the `SUMMARY` log.
- `series(self, symbol)`: `[(usable_from, value), ...]` sorted by time, where `usable_from` is the first moment the value may be used (no lookahead). It's called often, so cache and fetch incrementally. An empty list means no data, so the stock can't join a pair while `REQUIRE_FACTOR` is on.
- `summary(self)`: a short string of counts for the `SUMMARY` log.

`main.py` uses F_i − F_j as the pair's factor difference, standardizes F across stocks in active pairs if `STANDARDIZE_FACTOR` is on, and regresses AG on it each quarter. Only the `factor` run (`SIGNAL_MODE=factor`) uses it, apart from the `REQUIRE_FACTOR` pair check.

## How to run
CLI: `lean cloud push --project PairsTrading`, then `lean cloud backtest PairsTrading --parameter NAME VALUE ...`.

Web IDE:
1. Create a Python project at quantconnect.com, replace its `main.py` with this one, and add `factor.py` as a second file.
2. No data subscription needed: the default factor is computed from QuantConnect's free US SEC Filings dataset.
3. Backtest three times, with `SIGNAL_MODE` set to `g`, then `ag`, then `factor`. Change it in `DEFAULTS` or as a project parameter.
4. Copy the `UNIVERSE`, `FORMATION`, `REFIT` and `SUMMARY` lines from the logs.

| Run | `SIGNAL_MODE` | Signal |
|---|---|---|
| A | `g` | z-score of the raw gap G (Gatev baseline) |
| B | `ag` | z-score of the adjusted gap AG, no factor |
| C | `factor` | z-score of the AG regression residual when \|t(β)\| ≥ `MIN_ABS_T_STAT`, else the same as B |

B vs A measures the switch from G to AG. C vs B measures what lexical density adds. A vs C alone mixes the two effects, because AG and G trigger trades differently even when β = 0: AG divides by V_i + V_j, which drifts with the market.

For a quick plumbing test, set `REQUIRE_FACTOR=False`, `SIGNAL_MODE=g` and a short `END_DATE`; this skips the filing downloads.

## Default factor: lexical density from SEC filings
Settings for this factor (`TEXT_FACTOR`, `LD_VARIANT`, `FILING_LOOKBACK_DAYS`, `MIN_FILING_WORDS`) are in `DEFAULTS` in `factor.py`.

`LD = content words / all words` in the main document of each 10-K or 10-Q (exhibits are ignored), from `SECReport10Q` / `SECReport10K`.
- **Text:** the document is inline-XBRL HTML. The hidden XBRL header, scripts, styles and tags are stripped and HTML entities decoded. Words are runs of letters, so numbers in tables don't count.
- **Content words:** every word not in `FUNCTION_WORDS` (about 210 determiners, pronouns, prepositions, conjunctions, auxiliaries, modals and particles). This approximates a part-of-speech count without an NLP library, and treats every filing the same way.
- **Form check:** some records in the dataset are not what their class says (for AEP, every "10-Q" and "10-K" record was an 8-K). Only filings whose own form type is 10-K, 10-K405, 10-KT, 10-Q or 10-QT are used; amendments are skipped.
- **Timing:** records are stamped at midnight of the filing date, but filings often arrive after the close, so each one is usable from the next day.
- **Cost:** each filing is parsed once and cached; later calls only fetch filings newer than the last fetch. Main documents run 1 to 13 MB.
- **Coverage:** companies whose 10-Ks and 10-Qs are missing from the dataset (such as combined utility filings) get no LD, so `REQUIRE_FACTOR` excludes them from pairs. The `SUMMARY` line reports how many filings were used and skipped.

## Using filings to judge gaps
Two independent options, each off or at its old behavior by default so they can be tested separately:

**Entry event filter (`ENTRY_EVENT_DAYS`, default 0 = off).** A pair doesn't open if either stock filed an 8-K with a material item in the last `ENTRY_EVENT_DAYS` calendar days, on the idea that a gap opened by news is likely justified. `EVENT_ITEMS` lists the items that count:

| Item | Event |
|---|---|
| 1.01, 1.02 | Material agreement entered / terminated |
| 1.03 | Bankruptcy |
| 2.01 | Acquisition or disposal completed |
| 2.02 | Quarterly results (earnings) |
| 2.05, 2.06 | Restructuring costs, impairments |
| 3.01 | Delisting notice |
| 4.01, 4.02 | Auditor change, restated financials |
| 5.01, 5.02 | Change of control, executive or director change |

Routine items (7.01 investor presentations, 8.01 other events such as dividends, 9.01 exhibits, 5.07 shareholder votes) don't count. Each 8-K counts from the day after filing. `ENTRY_EVENT_DAYS` is a lookback at entry time, not a holding period: exits are unchanged.

8-Ks are read with short history requests, one stock at a time, adding the dataset only for the request. A live `SECReport8K` subscription for every pair stock ran out of memory within two days of the first formation, so it isn't used. The entry check fetches up to the moment of the decision, so it always sees every filing to date.

**Re-anchoring (`EVENT_REANCHOR`, default off; needs `ENTRY_EVENT_DAYS` > 0).** Instead of only waiting out the news, measure what it did to the gap. A material 8-K for either stock opens a window of `ENTRY_EVENT_DAYS`, and the gap from the last day before the filing is recorded; a further 8-K inside the window extends it. When the window closes, the gap's move over it is treated as justified by the news and added to the pair's normal level, so later z-scores measure only what the news doesn't explain. Moves smaller than `REANCHOR_MIN_SIGMA` formation stds (default 1) are treated as noise and ignored. Shifts accumulate, apply to open positions too, and reset at re-formation. To keep lookups manageable, re-anchoring refreshes each pair's 8-Ks every `EVENT_REFRESH_DAYS` trading days (default 5), so a filing can be noticed up to that late; the window and pre-filing gap still come from the filing date and stored daily gaps, so only a window shorter than the refresh interval would be affected.

**Text factor (`TEXT_FACTOR`, in `factor.py`).** Chooses what the regression's ΔLD uses:
- `ld` (default): lexical density, as before.
- `sim`: cosine similarity of the filing's word counts to the company's previous filing of the same type (10-Q vs the prior 10-Q, 10-K vs the prior 10-K), as in Cohen, Malloy & Nguyen's "Lazy Prices". Near 1 means the filing mostly repeats the last one; lower means it was rewritten, a sign something changed. Word counts are hashed into a 16,384-slot vector to bound memory; collisions barely move the similarity. A stock's first filing in the lookback has nothing to compare against and is skipped.

## Config
Strategy tunables are in `DEFAULTS` at the top of `main.py`, and the factor's in `DEFAULTS` at the top of `factor.py`. A project parameter of the same name overrides either.

## Decisions where the spec was open
| Topic | Choice |
|---|---|
| Universe | The filter keeps the top 1.5 × `N_UNIVERSE` by one-day dollar volume (`PREFILTER_MULT`), and formation then ranks by 63-day average. |
| Subscriptions | The filter subscribes to nothing. Only stocks in selected pairs are added, at minute resolution. |
| Factor requirement | `REQUIRE_FACTOR`: a stock needs a usable factor value (by default a 10-K/10-Q LD) to join a pair. Checked lazily while picking pairs, not across all 1,000 stocks. All three runs apply it, so they trade the same pairs. |
| Run modes | `SIGNAL_MODE` replaces the spec's `USE_FACTOR` flag, adding the middle `ag` run. |
| Gate fallback | When the t-stat gate fails, `factor` falls back to the AG z-score, not the raw z_ij the spec names, so the `factor` run only ever uses AG. |
| `LD_VARIANT` | Defaults to `all` (10-K and 10-Q); `10k` uses 10-Ks only. LD is computed from the whole main document (see above). |
| Factor for quarter q | Latest value known at q-end + `FILING_LAG_DAYS` (45), capped at the refit date. |
| Standardizing the factor (`STANDARDIZE_FACTOR`) | Across stocks in active pairs, not the full universe, to avoid quarterly history calls for 1,000 stocks. |
| Refits | First trading day of Feb, May, Aug and Nov, plus each re-formation so new pairs have a model. |
| dF between refits | Frozen at the latest known value. It's a step function when computing formation residual stats. |
| Sizing | `N_active_pairs` counts selected, non-blocked, non-broken pairs. Weights are set at entry and not rebalanced. |
| `ONE_PAIR_PER_STOCK` | Defaults to `True`. With `False`, a stock can join several pairs: each pair keeps its own weight and orders use the net weight per stock. |
| Trade P&L | Estimated per pair from entry and exit prices before costs, so win rate is gross of fees. |
| Exit | Also closes if z jumps through zero to the other side. |
| Stop level | `STOP_Z` defaults to 6, not the spec's 4. In 2018–2026 tests (`ag` mode, same industry, blocking on), stop 4 lost 15.3% with a 23.6% drawdown; stop 6 lost 7.6% with a 14.6% drawdown and won 60% of trades. |
| Blocking | `BLOCK_AFTER_STOP` (default `True`): a stop-out or max-hold exit blocks the pair until the next re-formation. With `False`, the pair can re-enter once \|z\| is back in the entry band (≥ `ENTRY_Z`, < `STOP_Z`). |
| Same industry | `SAME_INDUSTRY_ONLY` defaults to `True` (the spec had it off): both stocks must share a Morningstar industry group. Groups are broad, e.g. all REITs or all banks form one group, and SPACs share the shell-company group. |
| Re-selected pairs | If the same (i, j) is chosen again, the position and hold count carry over, with new normalization. |

## Known limitations
- **Gross will sit below 2.0.** Capital is reserved for flat pairs, so the spec's "near target" check only applies if every pair is open.
- **The regression may understate the factor.** Pairs are picked for small formation gaps, and the 4-quarter sample mostly overlaps that window, which may bias β toward zero.
- **Thin sample.** With 4 quarters per pair, the pooled regression assumes one β fits all pairs.
- **Pairs can break permanently.** Stops and the max hold limit this but don't eliminate it.
- **Lexical density is unproven here.** This usage differs from the Quantpedia long-short sort, so a weak or unstable β is a plausible outcome.
- **LD is self-computed.** The function-word list approximates a part-of-speech tagger, so values won't match Brain's `lexical_density` and absolute levels aren't comparable to the paper. Standardizing across stocks (`STANDARDIZE_FACTOR`) reduces this, since only differences between stocks enter the regression.
- **Costs are incomplete.** Borrow costs and short availability aren't fully modeled.
- **Leverage is set above Reg T.** Security leverage is 4x so a 2.0 gross target isn't rejected at the margin limit.
- **Shared stocks concentrate risk.** With `ONE_PAIR_PER_STOCK=False`, one stock can sit in many pairs, and opposite legs can net out, lowering gross exposure.
- **Filtering gaps.** There's no limited-partnership field on `SecurityReference`, so some may pass the universe filter.
- **Adjusted prices.** V(t) assumes history and live adjusted prices share a basis. Fills also happen at adjusted prices, a known LEAN tradeoff.
