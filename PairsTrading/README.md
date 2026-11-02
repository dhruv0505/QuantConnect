# Lexical-Density Pairs Trading

A distance-method pairs strategy (Gatev et al. 2006). It tests whether lexical density from company filings separates justified price gaps from gaps that revert. Definitions are in the `main.py` docstring; the full spec is in `instructions.md`.

**Status: not yet run on QuantConnect.** The math is unit-tested locally, but nothing has run on LEAN, so there are no results yet.

## How to run (web IDE, no API token)
1. Create a Python project at quantconnect.com and replace its `main.py` with this one.
2. Subscribe to Brain Language Metrics on Company Filings (Cloud Access, listed at $25/mo).
3. Backtest with `USE_FACTOR` set to `False`, then again with `True`. Change it in `DEFAULTS` or as a project parameter.
4. Copy the `UNIVERSE`, `FORMATION`, `REFIT` and `SUMMARY` lines from the logs.

To test the plumbing without the subscription, set `REQUIRE_BRAIN=False`, `USE_FACTOR=False` and a short `END_DATE`.

## Config
Every tunable is in `DEFAULTS` at the top of `main.py`, and a project parameter of the same name overrides it.

## Decisions where the spec was open
| Topic | Choice |
|---|---|
| Universe | The filter keeps the top 1.5 × `N_UNIVERSE` by one-day dollar volume (`PREFILTER_MULT`), and formation then ranks by 63-day average. |
| Subscriptions | The filter subscribes to nothing. Only stocks in selected pairs are added, at minute resolution. |
| Brain requirement | Checked lazily while picking pairs, not across all 1,000 stocks. Both runs apply it, so they trade the same pairs. |
| `LD_VARIANT` | Defaults to `all` (10-K and 10-Q). Uses `report_sentiment.lexical_density` for the whole report. |
| LD for quarter q | Latest value known at q-end + `FILING_LAG_DAYS` (45), capped at the refit date. |
| Standardizing LD | Across stocks in active pairs, not the full universe, to avoid quarterly history calls for 1,000 stocks. |
| Refits | First trading day of Feb, May, Aug and Nov, plus each re-formation so new pairs have a model. |
| dLD between refits | Frozen at the latest known value. It's a step function when computing formation residual stats. |
| Sizing | `N_active_pairs` counts selected, non-blocked, non-broken pairs. Weights are set at entry and not rebalanced. |
| `ONE_PAIR_PER_STOCK` | Defaults to `True`. With `False`, a stock can join several pairs: each pair keeps its own weight and orders use the net weight per stock. |
| Trade P&L | Estimated per pair from entry and exit prices before costs, so win rate is gross of fees. |
| Exit | Also closes if z jumps through zero to the other side. |
| Blocking | A stop-out or max-hold exit blocks the pair until the next re-formation. |
| Re-selected pairs | If the same (i, j) is chosen again, the position and hold count carry over, with new normalization. |

## Known limitations
- **Gross will sit below 2.0.** Capital is reserved for flat pairs, so the spec's "near target" check only applies if every pair is open.
- **The regression may understate the factor.** Pairs are picked for small formation gaps, and the 4-quarter sample mostly overlaps that window, which may bias β toward zero.
- **Thin sample.** With 4 quarters per pair, the pooled regression assumes one β fits all pairs.
- **Pairs can break permanently.** Stops and the max hold limit this but don't eliminate it.
- **Lexical density is unproven here.** This usage differs from the Quantpedia long-short sort, so a weak or unstable β is a plausible outcome.
- **Costs are incomplete.** Borrow costs and short availability aren't fully modeled.
- **Leverage is set above Reg T.** Security leverage is 4x so a 2.0 gross target isn't rejected at the margin limit.
- **Shared stocks concentrate risk.** With `ONE_PAIR_PER_STOCK=False`, one stock can sit in many pairs, and opposite legs can net out, lowering gross exposure.
- **Filtering gaps.** There's no limited-partnership field on `SecurityReference`, so some may pass the universe filter.
- **Adjusted prices.** V(t) assumes history and live adjusted prices share a basis. Fills also happen at adjusted prices, a known LEAN tradeoff.
