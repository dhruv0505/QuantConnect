"""
Probe for the free US SEC Filings dataset, round 2: what SECReport8K records contain.
Checks whether 8-Ks carry their item codes (e.g. 2.02 results, 1.01 agreements), how often
companies file them, and how big they are.
"""
from AlgorithmImports import *
from collections import Counter
from datetime import datetime

TICKERS = ["AAPL", "JPM", "BUSE", "WAFD", "KO"]


class SECFilingsProbe(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2026, 10, 1)
        self.set_end_date(2026, 10, 3)
        self.set_cash(100_000)
        for ticker in TICKERS:
            equity = self.add_equity(ticker, Resolution.DAILY).symbol
            dataset = self.add_data(SECReport8K, equity).symbol
            rows = list(self.history[SECReport8K](dataset, datetime(2024, 1, 1), self.time, Resolution.DAILY))
            forms = Counter(str(r.report.form_type) for r in rows if r.report is not None)
            items = Counter()
            for r in rows:
                for item in (getattr(r.report, "items", None) or []):
                    items[str(item)] += 1
            self.log(f"{ticker} 8K: {len(rows)} records since 2024, forms {dict(forms)}, items {dict(items.most_common(12))}")
            for r in rows[-4:]:
                rep = r.report
                docs = list(rep.documents or [])
                self.log(f"  time={r.end_time:%Y-%m-%d} filing_date={rep.filing_date} form={rep.form_type} "
                         f"items={list(getattr(rep, 'items', None) or [])} docs={len(docs)} "
                         f"doc_types={[d.form_type for d in docs][:5]} sizes={[len(d.text or '') for d in docs][:5]}")
            self.remove_security(dataset)
