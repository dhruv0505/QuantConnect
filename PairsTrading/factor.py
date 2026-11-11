"""
The pluggable factor for main.py. Replace this file to test a different factor;
main.py doesn't need to change.

CONTRACT (all main.py relies on)
  class Factor:
      name: str                          short label for the SUMMARY log
      __init__(self, algorithm, param)   algorithm = the running QCAlgorithm;
                                         param(name, default) reads a project parameter,
                                         typed like the default (use it for your settings)
      series(self, symbol)               [(usable_from, value), ...] sorted by time.
                                         usable_from is the first algorithm time the value
                                         may be used (no lookahead). Empty = no data, so
                                         the stock can't join a pair when REQUIRE_FACTOR
                                         is on. Called often, so cache and fetch
                                         incrementally.
      summary(self)                      short string of counts for the SUMMARY log

main.py takes F_i - F_j as the pair's factor difference, optionally standardizes F across
the stocks in active pairs (STANDARDIZE_FACTOR), and regresses AG on it each quarter.

THIS FACTOR: text of each stock's latest 10-K/10-Q from QuantConnect's free US SEC Filings
dataset (SECReport10K / SECReport10Q), chosen by TEXT_FACTOR:
  "ld"  = lexical density: content words / all words. Content words are every word not
          in FUNCTION_WORDS, an approximation of a part-of-speech count.
  "sim" = cosine similarity of word counts to the company's previous filing of the same
          type (10-Q vs prior 10-Q, 10-K vs prior 10-K; Cohen, Malloy & Nguyen "Lazy Prices").
Records are stamped at midnight of the filing date, but filings often land after the
close, so each value is usable from the next day.
"""
from AlgorithmImports import *
from datetime import timedelta
from collections import Counter
import html
import re
import zlib
import numpy as np

# This factor's settings. Each can be overridden by a project parameter of the same name.
DEFAULTS = {
    "TEXT_FACTOR": "ld",           # "ld" = lexical density, "sim" = similarity to previous filing
    "LD_VARIANT": "all",           # "all" = 10-K and 10-Q, "10k" = 10-K only
    "FILING_LOOKBACK_DAYS": 600,   # filings older than this aren't fetched
    "MIN_FILING_WORDS": 2000,      # shorter main documents are treated as failed extractions
}
TEXT_FACTORS = ("ld", "sim")
HASH_DIM = 1 << 14                  # word-count vector size for similarity (memory bound)
FORMS_10K = ("10-K", "10-K405", "10-KT")       # amendments (10-K/A) are partial, so excluded
FORMS_10Q = ("10-Q", "10-QT")

# Function words: everything else counts as a content (lexical) word.
FUNCTION_WORDS = frozenset("""
a an the this that these those each every either neither some any no all both half
another such what which whose whatever whichever
i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his
himself she her hers herself it its itself they them their theirs themselves one ones
who whom whoever someone somebody something anyone anybody anything everyone everybody
everything nobody nothing none
about above across after against along amid among amongst around as at before behind below
beneath beside besides between beyond by despite down during except for from in inside into
like near of off on onto out outside over past per since than through throughout till to
toward towards under underneath unlike until up upon via with within without
and or but nor so yet because although though while whereas if unless whether once whenever
wherever where when whereby wherein whereof thereof therein thereby hereby herein hereof
hereunder thereunder
be am is are was were been being have has had having do does did doing done
will would shall should can could may might must ought
not n't to there here then thus hence therefore however also too very just only even
s t ll d ve re m
""".split())

_IX_HEADER_RE = re.compile(r"<ix:header>.*?</ix:header>", re.S | re.I)
_HIDDEN_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"[a-z]+")


# ---------- Pure text functions (no LEAN dependency) ----------

def filing_text(raw):
    """Plain text of an inline-XBRL/HTML filing document: drop the hidden XBRL header,
    scripts and styles, then all tags, then decode entities such as &#174; and &nbsp;."""
    text = _IX_HEADER_RE.sub(" ", raw)
    text = _HIDDEN_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


def words_of(text):
    """Lower-cased words. Words are runs of letters, so numbers in tables are ignored;
    apostrophes split words ("company's" -> company, s)."""
    return _WORD_RE.findall(text.lower())


def lexical_density(words, min_words=0):
    """Share of words that are content words, or None if there are fewer than min_words."""
    if not words or len(words) < min_words:
        return None
    return sum(1 for w in words if w not in FUNCTION_WORDS) / len(words)


def count_vector(words, dim=HASH_DIM):
    """Word counts hashed into a fixed-size vector (crc32, so identical across runs).
    Collisions are rare enough at 16k slots to barely move a cosine similarity."""
    vec = np.zeros(dim, dtype=np.float32)
    for word, n in Counter(words).items():
        vec[zlib.crc32(word.encode()) % dim] += n
    return vec


def cosine_similarity(u, v):
    """Cosine of two count vectors: 1.0 = same word mix, lower = rewritten."""
    norm = float(np.linalg.norm(u)) * float(np.linalg.norm(v))
    return float(u @ v) / norm if norm > 0 else None


# ---------- Factor ----------

class Factor:
    """Filing-text factor (lexical density or similarity) per stock."""

    def __init__(self, algorithm, param):
        self.algo = algorithm
        for name, default in DEFAULTS.items():
            setattr(self, name, param(name, default))
        variants = {"all": ((SECReport10Q, SECReport10K), FORMS_10Q + FORMS_10K),
                    "10k": ((SECReport10K,), FORMS_10K)}
        if self.LD_VARIANT not in variants:
            raise ValueError(f"LD_VARIANT must be one of {tuple(variants)}, got {self.LD_VARIANT!r}")
        if self.TEXT_FACTOR not in TEXT_FACTORS:
            raise ValueError(f"TEXT_FACTOR must be one of {TEXT_FACTORS}, got {self.TEXT_FACTOR!r}")
        self._filing_types, self._forms = variants[self.LD_VARIANT]
        self.name = f"{self.TEXT_FACTOR}/{self.LD_VARIANT}"
        # Filled incrementally so each filing is parsed once: symbol -> [(usable_from, value)],
        # symbol -> fetched-up-to time, filing keys seen.
        self._series, self._fetched, self._seen = {}, {}, set()
        self._prev_vec = {}            # (symbol, "10-K"/"10-Q") -> count vector of last filing
        self._stats = {"used": 0, "other_form": 0, "too_short": 0, "no_previous": 0, "errors": 0}

    def series(self, symbol):
        """[(usable_from, value)] for the stock's 10-K/10-Q filings, sorted. Only filings
        published since the last call are fetched and parsed."""
        now = self.algo.time
        start = self._fetched.get(symbol, now - timedelta(days=self.FILING_LOOKBACK_DAYS))
        if start < now:
            series = self._series.setdefault(symbol, [])
            for cls in self._filing_types:
                self._read_filings(cls, symbol, start, now, series)
            series.sort(key=lambda item: item[0])
            self._fetched[symbol] = now
        return self._series.get(symbol, [])

    def summary(self):
        return f"filings={self._stats} filing_stocks={len(self._series)}"

    def _read_filings(self, cls, symbol, start, end, series):
        """Append (usable_from, value) for each 10-K/10-Q filing of `cls` in [start, end].
        History is chronological, so for "sim" each filing is compared with the previous
        filing of the same family seen for this stock, including across fetch calls."""
        algo = self.algo
        dataset = algo.add_data(cls, symbol, Resolution.DAILY).symbol
        try:
            for row in algo.history[cls](dataset, start, end, Resolution.DAILY):
                report = row.report
                form = str(report.form_type or "").upper() if report is not None else ""
                if form not in self._forms:             # e.g. 8-K press releases mislabeled
                    self._stats["other_form"] += 1
                    continue
                key = (symbol, form, row.end_time)
                if key in self._seen:                    # fetch windows overlap at the edges
                    continue
                self._seen.add(key)
                docs = list(report.documents or [])
                main = next((d for d in docs if str(d.form_type or "").upper() == form),
                            docs[0] if docs else None)  # the filing itself, not its exhibits
                words = words_of(filing_text(main.text or "")) if main is not None else []
                if len(words) < max(1, self.MIN_FILING_WORDS):
                    self._stats["too_short"] += 1
                    continue
                if self.TEXT_FACTOR == "ld":
                    value = lexical_density(words)
                else:
                    family = "10-K" if form in FORMS_10K else "10-Q"
                    vec = count_vector(words)
                    previous = self._prev_vec.get((symbol, family))
                    self._prev_vec[(symbol, family)] = vec
                    value = cosine_similarity(previous, vec) if previous is not None else None
                    if value is None:                   # first filing seen: nothing to compare
                        self._stats["no_previous"] += 1
                        continue
                # Stamped at midnight of the filing date; usable the next day (no lookahead).
                series.append((row.end_time + timedelta(days=1), value))
                self._stats["used"] += 1
        except Exception as error:
            self._stats["errors"] += 1
            if self._stats["errors"] <= 5:
                algo.log(f"FACTOR {symbol.value} {cls.__name__}: {error}")
        finally:
            algo.remove_security(dataset)
