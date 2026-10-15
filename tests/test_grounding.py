"""The grounding ladder is the main false-positive risk in the project, so it is tested
against the reformatting cases real model output produces - several taken verbatim from
llama3.2:3b's first extraction on the probe document."""

from decimal import Decimal

import pytest

from ffx.analyzer.grounding import (
    FieldKind,
    GroundingLevel,
    candidates,
    find_amounts,
    ground,
    parse_amount,
    parse_date,
    quote_grounding,
    values_equal,
)

PROBE_DOC = (
    "Insured: Maria Okafor. Policy POL-44821 covers the property at 14 Ash Lane, Bristol.\n"
    "She previously held policy POL-99881 which lapsed in 2019.\n"
    "On 2024-03-14 a kitchen fire caused smoke damage. She is claiming $4,500.00.\n"
    "Reference CLM-7781."
)


# ---------------------------------------------------------------- must PASS
# Correct extractions that a naive substring check would wrongly call hallucinations.


def test_exact_match():
    r = ground("POL-44821", PROBE_DOC, FieldKind.IDENTIFIER)
    assert r.level == GroundingLevel.EXACT and r.score == 1.0


def test_amount_reformatted_by_model_is_grounded():
    # Observed: model emitted "4500.00" for "$4,500.00" in the source.
    r = ground("4500.00", "She is claiming $4,500.00.", FieldKind.AMOUNT)
    assert r.level == GroundingLevel.AMOUNT
    assert r.grounded


@pytest.mark.parametrize(
    "source",
    ["On 14 March 2024 a fire", "On March 14, 2024 a fire", "On 14/03/2024 a fire", "El 14 de marzo de 2024 hubo"],
)
def test_date_in_any_human_format_is_grounded(source):
    r = ground("2024-03-14", source, FieldKind.DATE)
    assert r.grounded, source


def test_ocr_garbled_identifier_is_grounded_at_reduced_score():
    r = ground("POL-44821", "Policy No: P0L-4482l", FieldKind.IDENTIFIER)
    assert r.level == GroundingLevel.OCR
    assert r.score < 1.0


def test_casefold_and_whitespace():
    assert ground("maria okafor", PROBE_DOC).level == GroundingLevel.CASEFOLD
    assert ground("14  Ash Lane,  Bristol", PROBE_DOC).level == GroundingLevel.WHITESPACE


def test_name_with_ocr_rn_for_m_is_grounded():
    # "m" scanned as "rn": 84.6 raw fuzzy, identical after OCR canonicalisation.
    r = ground("Priya Adeyemi", "Cloimant: Priya Adeyerni", FieldKind.TEXT)
    assert r.level == GroundingLevel.OCR


def test_different_surname_is_not_grounded():
    # The control for the OCR rule above: must not be absorbed by it.
    assert not ground("Priya Adeyemi", "Claimant: Priya Adebayo", FieldKind.TEXT).grounded


# ---------------------------------------------------------------- must FAIL
# Hallucinations and wrong values that must never be waved through.


def test_transposed_identifier_is_not_grounded():
    # ~89% fuzzy-similar. Fuzzy matching on identifiers would let this through.
    r = ground("POL-44812", PROBE_DOC, FieldKind.IDENTIFIER)
    assert not r.grounded


def test_invented_policy_number_is_not_grounded():
    assert not ground("POL-12345", PROBE_DOC, FieldKind.IDENTIFIER).grounded


def test_wrong_amount_is_not_grounded():
    assert not ground("45000.00", PROBE_DOC, FieldKind.AMOUNT).grounded


def test_wrong_date_is_not_grounded():
    assert not ground("2024-03-15", PROBE_DOC, FieldKind.DATE).grounded


def test_peril_value_unsupported_by_its_own_fabricated_quote():
    # Observed on the probe: peril="liability" backed by the quote "lapse in 2019."
    # (the document says "lapsed in 2019"). The value is not in its own quote.
    assert not ground("liability", "lapse in 2019.").grounded


def test_unparseable_date_is_reported_not_crashed():
    r = ground("last Tuesday", PROBE_DOC, FieldKind.DATE)
    assert not r.grounded and "unparseable" in r.detail


# ---------------------------------------------------------------- quotes


def test_verbatim_quote():
    assert quote_grounding("Reference CLM-7781.", PROBE_DOC).score == 1.0


def test_fabricated_quote():
    assert not quote_grounding("The insured admitted liability for the fire.", PROBE_DOC).grounded


def test_near_verbatim_quote_scores_below_exact():
    r = quote_grounding("She previously held policy POL-99881 which lapsed in 2019", PROBE_DOC)
    assert r.grounded and r.score >= 0.95  # only trailing punctuation differs


# ---------------------------------------------------------------- parsers


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$4,500.00", "4500.00"),
        ("$4,500", "4500"),
        ("USD 12,345.67", "12345.67"),
        ("4.050,00", "4050.00"),  # European
        ("3.200", "3200"),  # European thousands, no decimals
        ("4.05", "4.05"),
        ("94750", "94750"),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == Decimal(expected)


def test_money_only_ignores_years_and_identifiers():
    text = "Policy POL-44821, cancelled in 2019. Claiming $4,500.00 for damage on 14 March 2024."
    assert [str(v) for v, _ in find_amounts(text, money_only=True)] == ["4500.00"]


def test_parse_date_formats():
    assert parse_date("2024-03-14").isoformat() == "2024-03-14"
    assert parse_date("14 March 2024").isoformat() == "2024-03-14"
    assert parse_date("March 14, 2024").isoformat() == "2024-03-14"
    assert parse_date("nonsense") is None


# ---------------------------------------------------------------- candidates


def test_decoy_document_has_two_policy_candidates():
    assert candidates("policy_number", PROBE_DOC) == ["POL-44821", "POL-99881"]


def test_conflicting_dates_are_both_candidates():
    text = "Reporting this on 2 April 2024. The damage happened on 20 March 2024."
    assert candidates("date_of_loss", text) == ["2024-04-02", "2024-03-20"]


def test_amount_candidates_include_decoy_estimate():
    text = "The contractor quoted $31,200 for a full renovation. I am only claiming $4,500.00."
    assert candidates("reported_amount", text) == ["31200.00", "4500.00"]


def test_find_names_skips_greetings_and_form_labels():
    from ffx.analyzer.grounding import find_names

    text = (
        "Subject: New claim - Nadia Osei\n\nDear Claims Team,\n\nA loss at 5 Beacon Hill, Exeter. "
        "Claimant: Leila Petrov\n\nKind regards,\nNadia Osei"
    )
    assert find_names(text) == ["Nadia Osei", "Beacon Hill", "Leila Petrov"]


def test_values_equal_is_semantic():
    assert values_equal("2024-03-14", "14 March 2024", FieldKind.DATE)
    assert values_equal("4500", "$4,500.00", FieldKind.AMOUNT)
    assert values_equal("POL-44821", "P0L-4482l", FieldKind.IDENTIFIER)
    assert not values_equal("POL-44821", "POL-44812", FieldKind.IDENTIFIER)
