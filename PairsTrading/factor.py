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
  "ld"       = lexical density: content words / all words. Content words are every word not
               in FUNCTION_WORDS, an approximation of a part-of-speech count.
  "sim"      = cosine similarity of word counts to the company's previous filing of the same
               type (10-Q vs prior 10-Q, 10-K vs prior 10-K; Cohen, Malloy & Nguyen "Lazy Prices").
  "tone"     = net tone from Loughran-McDonald word lists: (pos - neg) / (pos + neg), where
               pos/neg count words from the positive/negative lists. Range -1 to +1.
  "tone_chg" = tone of this filing minus tone of the company's previous filing of the same
               type (same pairing as "sim"). Less sensitive to boilerplate than the level.
Records are stamped at midnight of the filing date, but filings often land after the
close, so each value is usable from the next day.

TONE WORD LISTS: for any real result, upload the official Loughran-McDonald master
dictionary CSV (Notre Dame SRAF; check its license terms) to the Object Store under
TONE_WORDS_KEY. Its "Positive" and "Negative" columns are nonzero for words in each list.
If the key is missing, the factor falls back to a SMALL built-in seed list and says so in
the log and the SUMMARY line. The seed list is for plumbing tests only.
Not handled: negation ("not improved"), and section choice (the whole main document counts).
"""
from AlgorithmImports import *
from datetime import timedelta
from collections import Counter
import csv
import html
import io
import re
import zlib
import numpy as np

# This factor's settings. Each can be overridden by a project parameter of the same name.
DEFAULTS = {
    "TEXT_FACTOR": "ld",           # "ld", "sim", "tone" or "tone_chg" (see the docstring)
    "LD_VARIANT": "all",           # "all" = 10-K and 10-Q, "10k" = 10-K only
    "FILING_LOOKBACK_DAYS": 600,   # filings older than this aren't fetched
    "MIN_FILING_WORDS": 2000,      # shorter main documents are treated as failed extractions
    "TONE_WORDS_KEY": "lm_master_dictionary.csv",  # Object Store key of the LM master dictionary CSV
}
TEXT_FACTORS = ("ld", "sim", "tone", "tone_chg")
TONE_FACTORS = ("tone", "tone_chg")
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

# Abbreviated stand-in for the Loughran-McDonald lists, used only when the official CSV is
# not in the Object Store. Far smaller than the real lists; do not trust results from it.
SEED_NEGATIVE = frozenset("""
adverse adversely bankrupt bankruptcy breach breaches breached closure concern concerns
decline declined declines declining default defaults defaulted deficiency deficiencies
deficit delinquent delinquency deteriorate deteriorated deterioration difficult
difficulties difficulty disrupt disruption disruptions downturn fail failed failing fails
failure failures fraud fraudulent harm harmed harmful impair impaired impairment
impairments insolvency loss losses misstatement misstatements negative negatively
penalties penalty recession restated restatement restructuring serious severe severely
shortfall shortfalls terminate terminated termination unable unfavorable unfavorably
violation violations weak weaken weakened weakness weaknesses worse worsen worsened
worsening
""".split())
SEED_POSITIVE = frozenset("""
achieve achieved achievement achievements beneficial benefit benefited benefits best better
boost confident effective efficiencies efficiency efficient enable enabled enhance enhanced
enhancement enhances favorable favorably gain gained gains improve improved improvement
improvements improves improving innovative leadership opportunities opportunity optimistic
outperform outperformed positive positively profitability profitable progress reliable
stability stable strength strengthen strengthened strengths strong stronger strongest
succeed succeeded success successes successful successfully superior surpass surpassed
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


def _in_list(value):
    """True if a master-dictionary cell is nonzero (the cell holds the year the word was added)."""
    try:
        return int(float(value)) != 0
    except (TypeError, ValueError):
        return False


def load_tone_words(csv_text):
    """(positive, negative) lower-case word sets from the Loughran-McDonald master dictionary
    CSV text. Uses its "Word", "Positive" and "Negative" columns. Returns two empty sets if
    those columns aren't there."""
    positive, negative = set(), set()
    for row in csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff"))):
        word = (row.get("Word") or "").strip().lower()
        if not word:
            continue
        if _in_list(row.get("Positive")):
            positive.add(word)
        if _in_list(row.get("Negative")):
            negative.add(word)
    return frozenset(positive), frozenset(negative)


def net_tone(words, positive, negative):
    """(pos - neg) / (pos + neg) over the words, from -1 (all negative) to +1 (all positive),
    or None if no word is in either list."""
    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    return (pos - neg) / (pos + neg) if pos + neg else None


# ---------- Factor ----------

class Factor:
    """Filing-text factor (lexical density, similarity or tone) per stock."""

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
        self._prev_tone = {}           # (symbol, "10-K"/"10-Q") -> tone of last filing
        self._stats = {"used": 0, "other_form": 0, "too_short": 0, "no_previous": 0,
                       "no_tone_words": 0, "errors": 0}
        self._positive, self._negative, self._tone_source = frozenset(), frozenset(), None
        if self.TEXT_FACTOR in TONE_FACTORS:
            self._positive, self._negative, self._tone_source = self._load_tone_lists()

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
        text = f"filings={self._stats} filing_stocks={len(self._series)}"
        if self._tone_source is not None:
            text += (f" tone_words={self._tone_source}"
                     f"(pos={len(self._positive)},neg={len(self._negative)})")
        return text

    def _load_tone_lists(self):
        """(positive, negative, source) from the Object Store CSV, else the built-in seed lists."""
        algo, key = self.algo, self.TONE_WORDS_KEY
        try:
            if key and algo.object_store.contains_key(key):
                positive, negative = load_tone_words(algo.object_store.read(key))
                if positive and negative:
                    return positive, negative, f"object_store:{key}"
                algo.log(f"FACTOR {key} has no usable Positive/Negative columns")
            else:
                algo.log(f"FACTOR Object Store key {key!r} not found")
        except Exception as error:
            algo.log(f"FACTOR could not read {key!r}: {error}")
        algo.log("FACTOR WARNING: using the abbreviated built-in tone lists, "
                 "not the official Loughran-McDonald dictionary")
        return SEED_POSITIVE, SEED_NEGATIVE, "builtin-seed"

    def _read_filings(self, cls, symbol, start, end, series):
        """Append (usable_from, value) for each 10-K/10-Q filing of `cls` in [start, end].
        History is chronological, so for "sim" and "tone_chg" each filing is compared with the
        previous filing of the same family seen for this stock, including across fetch calls."""
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
                family = "10-K" if form in FORMS_10K else "10-Q"
                if self.TEXT_FACTOR == "ld":
                    value = lexical_density(words)
                elif self.TEXT_FACTOR == "sim":
                    vec = count_vector(words)
                    previous = self._prev_vec.get((symbol, family))
                    self._prev_vec[(symbol, family)] = vec
                    value = cosine_similarity(previous, vec) if previous is not None else None
                    if value is None:                   # first filing seen: nothing to compare
                        self._stats["no_previous"] += 1
                        continue
                else:                                   # "tone" or "tone_chg"
                    tone = net_tone(words, self._positive, self._negative)
                    if tone is None:
                        self._stats["no_tone_words"] += 1
                        continue
                    if self.TEXT_FACTOR == "tone":
                        value = tone
                    else:
                        previous = self._prev_tone.get((symbol, family))
                        self._prev_tone[(symbol, family)] = tone
                        if previous is None:            # first filing seen: nothing to compare
                            self._stats["no_previous"] += 1
                            continue
                        value = tone - previous
                # Stamped at midnight of the filing date; usable the next day (no lookahead).
                series.append((row.end_time + timedelta(days=1), value))
                self._stats["used"] += 1
        except Exception as error:
            self._stats["errors"] += 1
            if self._stats["errors"] <= 5:
                algo.log(f"FACTOR {symbol.value} {cls.__name__}: {error}")
        finally:
            algo.remove_security(dataset)