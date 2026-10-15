"""Is a value actually supported by a piece of text? Deterministically, with no LLM.

This is the highest-value and the highest-false-positive-risk component in the project.
Naive substring matching would report a hallucination every time the model reformats a
correct value - ``"14 March 2024"`` extracted as ``2024-03-14``, ``"$4,500"`` as
``4500.00``. So matching climbs a ladder, cheapest and strictest first, and reports the
*degree* of grounding rather than a boolean:

=========== ===== ==============================================================
level       score meaning
=========== ===== ==============================================================
exact       1.00  literal substring
casefold    0.98  differs only in case
whitespace  0.95  differs only in spacing / punctuation at the edges
date        0.95  same calendar date, different format ("14 March 2024")
amount      0.95  same number, different format ("$4,500", "4.500,00")
ocr         0.85  identifier equal modulo OCR confusables (O/0, l/1, S/5, B/8)
fuzzy       <=0.8 near-match; free text only, never identifiers
none        0.00  not supported
=========== ===== ==============================================================

Identifiers never get fuzzy matching. ``POL-44821`` and ``POL-44812`` score ~89% on
fuzzy similarity; letting that count as "grounded" would wave through precisely the
transposition errors this tool exists to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional

from rapidfuzz import fuzz


class FieldKind(str, Enum):
    TEXT = "text"
    IDENTIFIER = "identifier"
    DATE = "date"
    AMOUNT = "amount"


class GroundingLevel(str, Enum):
    EXACT = "exact"
    CASEFOLD = "casefold"
    WHITESPACE = "whitespace"
    DATE = "date"
    AMOUNT = "amount"
    OCR = "ocr"
    FUZZY = "fuzzy"
    NONE = "none"


LEVEL_SCORE = {
    GroundingLevel.EXACT: 1.0,
    GroundingLevel.CASEFOLD: 0.98,
    GroundingLevel.WHITESPACE: 0.95,
    GroundingLevel.DATE: 0.95,
    GroundingLevel.AMOUNT: 0.95,
    GroundingLevel.OCR: 0.85,
    GroundingLevel.FUZZY: 0.8,
    GroundingLevel.NONE: 0.0,
}

FIELD_KINDS = {
    "claimant_name": FieldKind.TEXT,
    "policy_number": FieldKind.IDENTIFIER,
    "claim_number": FieldKind.IDENTIFIER,
    "date_of_loss": FieldKind.DATE,
    "reported_amount": FieldKind.AMOUNT,
    "loss_location": FieldKind.TEXT,
}

FUZZY_THRESHOLD = 90


@dataclass(frozen=True)
class GroundingResult:
    level: GroundingLevel
    score: float
    matched: str = ""
    detail: str = ""

    @property
    def grounded(self) -> bool:
        return self.level != GroundingLevel.NONE


NOT_GROUNDED = GroundingResult(GroundingLevel.NONE, 0.0)


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # Spanish - the corpus contains mixed-language documents.
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_DATE_PATTERNS = [
    # 2024-03-14
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    # 14/03/2024 or 14.03.2024 - day-first (UK/EU), month-first tried as a fallback
    (re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b"), "dmy_numeric"),
    # 14 March 2024, 14th Mar, 2024
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.I), "d_month_y"),
    # March 14, 2024
    (re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I), "month_d_y"),
    # 14 de marzo de 2024
    (re.compile(rf"\b(\d{{1,2}})\s+de\s+({_MONTH_ALT})\s+(?:de|del)\s+(\d{{4}})\b", re.I), "d_month_y"),
]


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def find_dates(text: str) -> list[tuple[date, str]]:
    """Every calendar date mentioned in ``text``, with the literal span it came from."""
    found: list[tuple[date, str]] = []
    for pattern, style in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            a, b, c = m.group(1), m.group(2), m.group(3)
            candidates: list[Optional[date]] = []
            if style == "ymd":
                candidates.append(_safe_date(int(a), int(b), int(c)))
            elif style == "dmy_numeric":
                candidates.append(_safe_date(int(c), int(b), int(a)))
                if int(a) <= 12 and int(b) <= 12 and a != b:
                    candidates.append(_safe_date(int(c), int(a), int(b)))
            elif style == "d_month_y":
                candidates.append(_safe_date(int(c), _MONTHS[b.lower()], int(a)))
            elif style == "month_d_y":
                candidates.append(_safe_date(int(c), _MONTHS[a.lower()], int(b)))
            for cand in candidates:
                if cand is not None:
                    found.append((cand, m.group(0)))
    return found


def parse_date(value: str) -> Optional[date]:
    value = value.strip()
    if not value:
        return None
    dates = find_dates(value)
    return dates[0][0] if dates else None


# --------------------------------------------------------------------------
# Amounts
# --------------------------------------------------------------------------

_NUMBER = re.compile(r"\d{1,3}(?:[.,  ]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?")
_CURRENCY_BEFORE = re.compile(r"(?:US\$|USD|EUR|GBP|\$|£|€)\s?$", re.I)
_CURRENCY_AFTER = re.compile(r"^\s?(?:USD|EUR|GBP|dollars?|d[oó]lares|euros?|pounds?)\b", re.I)


def parse_amount(text: str) -> Optional[Decimal]:
    """Parse a human-formatted number, US or European style.

    Decimal-separator heuristics: if both ``,`` and ``.`` appear, the last one is the
    decimal point. A lone separator followed by exactly two digits at the end is a
    decimal point; followed by exactly three digits it is a thousands separator
    (so "4.050" is four thousand and fifty, as a European writer means it).
    """
    s = re.sub(r"[^\d.,]", "", text.replace(" ", "").replace(" ", ""))
    if not s or not any(ch.isdigit() for ch in s):
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thousands = "." if dec == "," else ","
        s = s.replace(thousands, "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        head, _, tail = s.rpartition(sep)
        if len(tail) == 3 and head:
            s = s.replace(sep, "")
        elif len(tail) in (1, 2) and s.count(sep) == 1:
            s = f"{head}.{tail}"
        else:
            s = s.replace(sep, "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def find_amounts(text: str, *, money_only: bool = False) -> list[tuple[Decimal, str]]:
    """Numbers in ``text``. With ``money_only``, only those marked as currency."""
    found: list[tuple[Decimal, str]] = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0).strip()
        if money_only:
            before = text[max(0, m.start() - 5) : m.start()]
            after = text[m.end() : m.end() + 10]
            if not (_CURRENCY_BEFORE.search(before) or _CURRENCY_AFTER.search(after)):
                continue
        value = parse_amount(raw)
        if value is not None:
            found.append((value, raw))
    return found


# --------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------

_IDENTIFIER = re.compile(r"\b[A-Za-z0-9]{2,5}-[A-Za-z0-9]{3,}\b")
_OCR_CANON = str.maketrans({"O": "0", "D": "0", "I": "1", "L": "1", "|": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})


def ocr_canonical(identifier: str) -> str:
    return identifier.upper().translate(_OCR_CANON)


_OCR_TEXT_MULTI = (("rn", "m"), ("vv", "w"))
_OCR_TEXT_SINGLE = str.maketrans({"0": "o", "1": "l", "|": "l", "5": "s", "8": "b"})


def ocr_text_canonical(text: str) -> str:
    """Free-text form with common OCR confusions undone, for name/address matching."""
    s = _WS.sub(" ", text.casefold()).strip()
    for wrong, right in _OCR_TEXT_MULTI:
        s = s.replace(wrong, right)
    return s.translate(_OCR_TEXT_SINGLE)


def find_identifiers(text: str) -> list[str]:
    out = []
    for m in _IDENTIFIER.finditer(text):
        token = m.group(0)
        # Require a numeric tail after OCR normalisation, so words like "e-mail" don't count.
        tail = ocr_canonical(token).split("-", 1)[1]
        if sum(ch.isdigit() for ch in tail) >= 3:
            out.append(token)
    return out


def identifier_prefix(identifier: str) -> str:
    return ocr_canonical(identifier).split("-", 1)[0]


# --------------------------------------------------------------------------
# Proper names
#
# Added after the first real run: intake, told to strip boilerplate, removed the email
# subject line and sign-off - the only two places the claimant's name appeared. With
# only identifiers, dates and amounts treated as salient, that loss was invisible, and
# the downstream "John Doe" hallucination it caused was blamed on extraction alone.
# --------------------------------------------------------------------------

_NAME = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b")
_NAME_STOPWORDS = {
    "Dear", "Claims", "Claim", "Team", "Subject", "New", "Kind", "Kindest", "Regards", "Best",
    "Hello", "Hi", "Thanks", "Thank", "Online", "Submission", "Policy", "No", "Ref", "Reference",
    "Date", "Address", "Description", "Amount", "Incident", "Details", "Claimant", "To", "Whom",
    "It", "May", "Concern", "Please", "Note", "The", "My", "On", "Your", "Our", "This", "That",
    "Insured", "Fax", "Received", "Auto", "Notification", "Another", "Unfortunately", "Sincerely",
    "Many", "Me", "Mi", "El", "Reclamo", "Hola", "Gracias", "Sorry", "Yours", "Faithfully",
    "January", "February", "March", "April", "June", "July", "August", "September", "October",
    "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "Terms", "Conditions", "Customer", "Service",
}


def find_names(text: str) -> list[str]:
    """Multi-word proper names (people, streets, places). Single words are ignored -
    too many false positives - and greeting/form vocabulary is trimmed off the ends."""
    out: list[str] = []
    for m in _NAME.finditer(text):
        words = m.group(0).split()
        while words and words[0] in _NAME_STOPWORDS:
            words.pop(0)
        while words and words[-1] in _NAME_STOPWORDS:
            words.pop()
        if len(words) >= 2 and not any(w in _NAME_STOPWORDS for w in words):
            name = " ".join(words)
            if name not in out:
                out.append(name)
    return out


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _squash(s: str) -> str:
    return _WS.sub(" ", s).strip().strip(".,;:!?\"'()[]").casefold()


def ground(value: str, haystack: str, kind: FieldKind = FieldKind.TEXT) -> GroundingResult:
    """How well is ``value`` supported by ``haystack``?"""
    value = (value or "").strip()
    if not value or not haystack:
        return NOT_GROUNDED

    if value in haystack:
        return GroundingResult(GroundingLevel.EXACT, LEVEL_SCORE[GroundingLevel.EXACT], value)
    if value.casefold() in haystack.casefold():
        return GroundingResult(GroundingLevel.CASEFOLD, LEVEL_SCORE[GroundingLevel.CASEFOLD], value)
    squashed = _squash(value)
    if squashed and squashed in _WS.sub(" ", haystack).casefold():
        return GroundingResult(GroundingLevel.WHITESPACE, LEVEL_SCORE[GroundingLevel.WHITESPACE], value)

    if kind == FieldKind.DATE:
        target = parse_date(value)
        if target is None:
            return GroundingResult(GroundingLevel.NONE, 0.0, detail=f"unparseable date {value!r}")
        for found, span in find_dates(haystack):
            if found == target:
                return GroundingResult(GroundingLevel.DATE, LEVEL_SCORE[GroundingLevel.DATE], span)
        return GroundingResult(GroundingLevel.NONE, 0.0, detail=f"date {target.isoformat()} not in text")

    if kind == FieldKind.AMOUNT:
        target = parse_amount(value)
        if target is None:
            return GroundingResult(GroundingLevel.NONE, 0.0, detail=f"unparseable amount {value!r}")
        for found, span in find_amounts(haystack):
            if abs(found - target) < Decimal("0.005"):
                return GroundingResult(GroundingLevel.AMOUNT, LEVEL_SCORE[GroundingLevel.AMOUNT], span)
        return GroundingResult(GroundingLevel.NONE, 0.0, detail=f"amount {target} not in text")

    if kind == FieldKind.IDENTIFIER:
        canon = ocr_canonical(value)
        for token in find_identifiers(haystack):
            if ocr_canonical(token) == canon:
                return GroundingResult(GroundingLevel.OCR, LEVEL_SCORE[GroundingLevel.OCR], token)
        return GroundingResult(GroundingLevel.NONE, 0.0, detail="identifier not in text (no fuzzy matching for identifiers)")

    # Free text: undo OCR confusables before matching. "Adeyemi" read as "Adeyerni"
    # scores 84.6 raw but 100 once "rn" -> "m" is undone, while a genuinely different
    # surname scores ~77 - so canonicalising beats lowering the fuzzy threshold, which
    # would squeeze the gap between "OCR noise" and "wrong person" to a few points.
    canon_value, canon_hay = ocr_text_canonical(value), ocr_text_canonical(haystack)
    if canon_value and canon_value in canon_hay:
        return GroundingResult(GroundingLevel.OCR, LEVEL_SCORE[GroundingLevel.OCR], detail="equal modulo OCR confusables")

    ratio = fuzz.partial_ratio(canon_value, canon_hay)
    if ratio >= FUZZY_THRESHOLD:
        score = round(0.5 + 0.3 * (ratio - FUZZY_THRESHOLD) / (100 - FUZZY_THRESHOLD), 3)
        return GroundingResult(GroundingLevel.FUZZY, score, detail=f"partial_ratio={ratio:.0f}")
    return GroundingResult(GroundingLevel.NONE, 0.0, detail=f"best partial_ratio={ratio:.0f}")


def quote_grounding(quote: str, haystack: str) -> GroundingResult:
    """Is a supposedly-verbatim quote actually in the text?

    Quotes are long, so fuzzy matching is meaningful here, but a quote is *supposed* to
    be exact - near-matches score lower and a paraphrase does not count.
    """
    quote = (quote or "").strip()
    if not quote:
        return NOT_GROUNDED
    exact = ground(quote, haystack, FieldKind.TEXT)
    if exact.level in (GroundingLevel.EXACT, GroundingLevel.CASEFOLD, GroundingLevel.WHITESPACE):
        return exact
    ratio = fuzz.partial_ratio(_squash(quote), _WS.sub(" ", haystack).casefold())
    if ratio >= 95:
        return GroundingResult(GroundingLevel.FUZZY, 0.8, detail=f"near-verbatim, partial_ratio={ratio:.0f}")
    if ratio >= 85:
        return GroundingResult(GroundingLevel.FUZZY, 0.4, detail=f"paraphrased, partial_ratio={ratio:.0f}")
    return GroundingResult(GroundingLevel.NONE, 0.0, detail=f"not in text, partial_ratio={ratio:.0f}")


def values_equal(a: str, b: str, kind: FieldKind) -> bool:
    """Semantic equality used when comparing two extracted values (or value vs truth)."""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return a == b
    if kind == FieldKind.DATE:
        da, db = parse_date(a), parse_date(b)
        return da is not None and da == db
    if kind == FieldKind.AMOUNT:
        pa, pb = parse_amount(a), parse_amount(b)
        return pa is not None and pb is not None and abs(pa - pb) < Decimal("0.005")
    if kind == FieldKind.IDENTIFIER:
        return ocr_canonical(a) == ocr_canonical(b)
    return _squash(a) == _squash(b) or fuzz.ratio(_squash(a), _squash(b)) >= 92


def candidates(field: str, text: str, like: str = "") -> list[str]:
    """Distinct values in ``text`` that *could* fill ``field``.

    More than one candidate means the extraction step had a genuine choice to make -
    exactly where decoys, conflicting dates and multiple amounts cause failures, and
    where the LLM judge is asked to adjudicate.
    """
    kind = FIELD_KINDS.get(field)
    if kind == FieldKind.DATE:
        seen: dict[date, str] = {}
        for d, span in find_dates(text):
            seen.setdefault(d, span)
        return [d.isoformat() for d in seen]
    if kind == FieldKind.AMOUNT:
        seen_amt: dict[Decimal, str] = {}
        for a, span in find_amounts(text, money_only=True):
            seen_amt.setdefault(a.quantize(Decimal("0.01")), span)
        return [f"{a:.2f}" for a in seen_amt]
    if kind == FieldKind.IDENTIFIER:
        wanted = {"policy_number": "P01", "claim_number": "C1M"}.get(field)
        prefix = identifier_prefix(like) if like else wanted
        out: dict[str, str] = {}
        for token in find_identifiers(text):
            if prefix and identifier_prefix(token) != prefix:
                continue
            out.setdefault(ocr_canonical(token), token)
        return list(out.values())
    return []
