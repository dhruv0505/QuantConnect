"""
Filing language stability score.

Idea (Cohen, Malloy & Nguyen, "Lazy Prices", 2020; QuantConnect research post
"Filing Language Stability as a Selection Signal"): firms that barely change the
language of their 10-K / 10-Q from one filing to the next tend to earn higher
subsequent returns than firms that rewrite it.

The QuantConnect post reads its score straight from the Brain Language Metrics
dataset (a cosine-similarity score between 0 and 1, preferring the risk-factors
section and falling back to the full report). That dataset is only available
inside QuantConnect, so this file gives you two ways to get the same kind of score:

1. filing_stability_score(ticker)     -> runs anywhere. Pulls the last two filings
                                         from SEC EDGAR and computes the cosine
                                         similarity of their text yourself.
2. brain_stability_score(algorithm, symbol)
                                       -> runs inside a QuantConnect algorithm and
                                         reads the Brain score the way the post does.

Score: ~1.0 = filing closely repeats its predecessor (stable), lower = heavily rewritten.

Requires: requests, beautifulsoup4   (pip install requests beautifulsoup4)
SEC asks for a descriptive User-Agent with contact info; set the SEC_USER_AGENT
environment variable (only needed for the EDGAR version).
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from datetime import timedelta

import requests
from bs4 import BeautifulSoup

SEC_USER_AGENT_ENV = "SEC_USER_AGENT"


def _headers() -> dict[str, str]:
    """
    HTTP headers for SEC requests. The User-Agent comes from the SEC_USER_AGENT
    environment variable, e.g. "Jane Doe jane@example.com". It is read on each
    call (not at import), so the QuantConnect function below doesn't need it.
    """
    agent = os.environ.get(SEC_USER_AGENT_ENV, "").strip()
    if not agent:
        raise RuntimeError(
            f"Set the {SEC_USER_AGENT_ENV} environment variable to your name and "
            f'contact email, e.g. "Jane Doe jane@example.com". SEC EDGAR rejects '
            f"requests without a descriptive User-Agent."
        )
    return {"User-Agent": agent}

_TOKEN_RE = re.compile(r"[a-z]{2,}")


# --------------------------------------------------------------------------- #
# 1. Standalone version (SEC EDGAR)
# --------------------------------------------------------------------------- #

def _get_cik(ticker: str) -> str:
    """Map a ticker to a zero-padded 10-digit CIK."""
    data = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=_headers(), timeout=30
    ).json()
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Ticker not found on EDGAR: {ticker}")


def _recent_filings(cik: str, form: str, n: int = 2) -> list[str]:
    """URLs of the n most recent filings of the given form type, newest first."""
    sub = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json", headers=_headers(), timeout=30
    ).json()
    recent = sub["filings"]["recent"]
    urls = []
    for f, acc, doc in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if f == form:
            urls.append(
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{acc.replace('-', '')}/{doc}"
            )
            if len(urls) == n:
                break
    if len(urls) < n:
        raise ValueError(f"Fewer than {n} {form} filings found for CIK {cik}")
    return urls


def _filing_text(url: str) -> str:
    html = requests.get(url, headers=_headers(), timeout=60).text
    return BeautifulSoup(html, "html.parser").get_text(" ")


def _risk_factors(text: str) -> str | None:
    """
    Best-effort extraction of Item 1A (Risk Factors). The table of contents also
    contains "Item 1A", so take the longest span between an "Item 1A" and the next
    "Item 1B" / "Item 2" and treat that as the real section.
    """
    starts = [m.start() for m in re.finditer(r"item\s*1a\.?", text, re.I)]
    ends = [m.start() for m in re.finditer(r"item\s*(1b|2)\.?", text, re.I)]
    best = ""
    for s in starts:
        e = next((x for x in ends if x > s), None)
        if e and e - s > len(best):
            best = text[s:e]
    return best if len(best) > 5000 else None  # too short -> extraction failed


def _cosine(a: str, b: str) -> float:
    ca, cb = Counter(_TOKEN_RE.findall(a.lower())), Counter(_TOKEN_RE.findall(b.lower()))
    dot = sum(ca[w] * cb[w] for w in ca.keys() & cb.keys())
    norm = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(
        sum(v * v for v in cb.values())
    )
    return dot / norm if norm else 0.0


def filing_stability_score(ticker: str, form: str = "10-K") -> float:
    """
    Stability score in [0, 1] for a stock: cosine similarity between the language
    of its latest `form` filing ("10-K" or "10-Q") and the one before it.

    Prefers the Risk Factors section (what the QuantConnect post and the Lazy
    Prices paper find most informative); falls back to the full filing text.
    """
    new_url, old_url = _recent_filings(_get_cik(ticker), form, n=2)
    new_text, old_text = _filing_text(new_url), _filing_text(old_url)

    new_rf, old_rf = _risk_factors(new_text), _risk_factors(old_text)
    if new_rf and old_rf:
        return _cosine(new_rf, old_rf)
    return _cosine(new_text, old_text)


# --------------------------------------------------------------------------- #
# 2. QuantConnect version (Brain Language Metrics on Company Filings)
# --------------------------------------------------------------------------- #

def brain_stability_score(algorithm, symbol, lookback_days: int = 180):
    """
    Use inside a QCAlgorithm (e.g. in Initialize / a scheduled event), with the
    equity already added. Mirrors the post: scan ~6 months of 10-K/10-Q filings and
    return the most recent similarity, preferring the risk-factors score and
    falling back to the full-report score. Returns None if no filing is found.
    """
    from QuantConnect.DataSource import BrainCompanyFilingLanguageMetricsAll

    dataset = algorithm.add_data(BrainCompanyFilingLanguageMetricsAll, symbol).symbol
    history = algorithm.history(
        BrainCompanyFilingLanguageMetricsAll, dataset, timedelta(days=lookback_days),
        Resolution.DAILY,
    )
    score = None
    for filing in history.itertuples() if hasattr(history, "itertuples") else history:
        # field names follow the QC docs / the post's usage
        rf = getattr(filing, "risk_factors_statement_sentiment", None)
        rf_sim = rf.similarity.all if rf else None
        report = getattr(filing, "report_sentiment", None)
        report_sim = report.similarity.all if report else None
        sim = rf_sim or report_sim
        if sim:
            score = sim
    return score


if __name__ == "__main__":
    import sys

    tkr = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"{tkr} 10-K stability score: {filing_stability_score(tkr):.4f}")
