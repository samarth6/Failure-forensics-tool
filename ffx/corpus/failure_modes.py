"""Renderers for the 14 corpus conditions.

Each renderer takes the ground-truth facts of a claim and writes a document that
exhibits one realistic difficulty. The renderers describe *documents*, never outcomes:
nothing here scripts how the model responds. Whether a given document actually trips
the model is measured at run time.

Dates and amounts are deliberately written in varied human formats ("14 March 2024",
"$4,500", "3.200,00") while the answer key stores canonical forms. That gap is what
exercises the grounding normalisation ladder - a correct extraction must not be flagged
as a hallucination just because the model reformatted a date.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Callable

from ffx.models import (
    AnswerKey,
    Channel,
    ClaimCategory,
    DocType,
    FailureCategory,
    FailureMode,
    Peril,
    RoutingQueue,
    Severity,
    StepName,
)

# --------------------------------------------------------------------------
# Fact pools
# --------------------------------------------------------------------------

FIRST_NAMES = [
    "Maria", "James", "Aisha", "Tom", "Priya", "Daniel", "Chloe", "Samuel", "Fatima", "Oliver",
    "Grace", "Hiro", "Leila", "Marcus", "Nadia", "Owen", "Rosa", "Tariq", "Uma", "Victor",
]
LAST_NAMES = [
    "Okafor", "Whitfield", "Rahman", "Brennan", "Sharma", "Kowalski", "Martin", "Adeyemi",
    "Haddad", "Lindqvist", "Nakamura", "Osei", "Petrov", "Quinn", "Reyes", "Silva",
]
LOCATIONS = [
    "14 Ash Lane, Bristol", "22 Mill Road, Leeds", "7 Harbour View, Plymouth",
    "Flat 3, 118 Queen Street, Glasgow", "41 Orchard Close, Norwich", "9 Station Parade, Reading",
    "63 Kingsway, Cardiff", "5 Beacon Hill, Exeter", "Unit 12, Riverside Estate, Derby",
    "30 Elm Grove, Brighton", "88 High Street, Oxford", "2 Linden Court, York",
]

# (peril, category, narrative templates, amount range)
PERIL_PROFILES = {
    Peril.FIRE: (
        ClaimCategory.PROPERTY,
        [
            "a kitchen fire started by an unattended pan caused smoke and heat damage to the kitchen and hallway",
            "an electrical fault in the utility room started a fire that destroyed the washing machine and scorched the walls",
        ],
        (2_000, 60_000),
    ),
    Peril.WATER: (
        ClaimCategory.PROPERTY,
        [
            "a burst pipe under the bathroom sink flooded the ground floor overnight",
            "a failed washing machine hose leaked for several hours and ruined the kitchen flooring",
        ],
        (1_500, 30_000),
    ),
    Peril.THEFT: (
        ClaimCategory.PROPERTY,
        [
            "burglars forced the back door and stole jewellery, a laptop and two bicycles",
            "the garage was broken into and power tools and a lawnmower were stolen",
        ],
        (800, 15_000),
    ),
    Peril.COLLISION: (
        ClaimCategory.AUTO,
        [
            "the insured vehicle was rear-ended while stationary at a traffic light",
            "the insured vehicle collided with a van that pulled out of a side road",
        ],
        (1_200, 25_000),
    ),
    Peril.WIND: (
        ClaimCategory.PROPERTY,
        [
            "a storm tore tiles off the roof and flattened the garden fence",
            "high winds brought down a chimney stack onto the conservatory roof",
        ],
        (1_000, 40_000),
    ),
    Peril.LIABILITY: (
        ClaimCategory.LIABILITY,
        [
            "a delivery driver slipped on the front steps and fractured a wrist",
            "a customer tripped over a loose floor mat in the shop and injured a knee",
        ],
        (3_000, 120_000),
    ),
}


@dataclass(frozen=True)
class ClaimFacts:
    claimant_name: str
    policy_number: str
    claim_number: str
    date_of_loss: date
    amount: float
    location: str
    peril: Peril
    narrative: str
    category: ClaimCategory

    @property
    def severity(self) -> Severity:
        if self.amount < 5_000:
            return Severity.LOW
        if self.amount < 25_000:
            return Severity.MEDIUM
        if self.amount < 100_000:
            return Severity.HIGH
        return Severity.CRITICAL

    @property
    def routing(self) -> RoutingQueue:
        return {
            Severity.LOW: RoutingQueue.FAST_TRACK,
            Severity.MEDIUM: RoutingQueue.STANDARD,
            Severity.HIGH: RoutingQueue.COMPLEX,
            Severity.CRITICAL: RoutingQueue.COMPLEX,
        }[self.severity]


def random_policy(rng: random.Random) -> str:
    return f"POL-{rng.randint(10000, 99999)}"


def random_claim_no(rng: random.Random) -> str:
    return f"CLM-{rng.randint(1000, 99999)}"


def sample_facts(rng: random.Random, peril: Peril | None = None) -> ClaimFacts:
    peril = peril or rng.choice(list(PERIL_PROFILES))
    category, narratives, (lo, hi) = PERIL_PROFILES[peril]
    amount = round(rng.uniform(lo, hi) / 50) * 50  # humans claim round-ish numbers
    return ClaimFacts(
        claimant_name=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        policy_number=random_policy(rng),
        claim_number=random_claim_no(rng),
        date_of_loss=date(2024, 1, 1) + timedelta(days=rng.randint(0, 400)),
        amount=float(amount),
        location=rng.choice(LOCATIONS),
        peril=peril,
        narrative=rng.choice(narratives),
        category=category,
    )


# --------------------------------------------------------------------------
# Human formatting
# --------------------------------------------------------------------------


def fmt_date(d: date, rng: random.Random) -> str:
    style = rng.randint(0, 3)
    if style == 0:
        return d.strftime("%d %B %Y").lstrip("0")
    if style == 1:
        return d.strftime("%B %d, %Y").replace(" 0", " ")
    if style == 2:
        return d.isoformat()
    return d.strftime("%d/%m/%Y")


def fmt_amount(x: float, rng: random.Random) -> str:
    style = rng.randint(0, 2)
    if style == 0:
        return f"${x:,.2f}"
    if style == 1:
        return f"${x:,.0f}"
    return f"USD {x:,.2f}"


def answer_key_for(f: ClaimFacts, **overrides) -> AnswerKey:
    base = dict(
        claimant_name=f.claimant_name,
        policy_number=f.policy_number,
        claim_number=f.claim_number,
        date_of_loss=f.date_of_loss.isoformat(),
        reported_amount=f"{f.amount:.2f}",
        loss_location=f.location,
        peril=f.peril,
        claim_category=f.category,
        severity=f.severity,
        routing_queue=f.routing,
        doc_type=DocType.CLAIM_NOTICE,
    )
    base.update(overrides)
    return AnswerKey(**base)


# --------------------------------------------------------------------------
# Document bodies
# --------------------------------------------------------------------------


def _email(f: ClaimFacts, rng: random.Random, *, policy_line: bool = True, extra: str = "") -> str:
    greeting = rng.choice(["Hello,", "Dear Claims Team,", "Hi there,"])
    policy = f"My policy number is {f.policy_number}. " if policy_line else ""
    return (
        f"Subject: New claim - {f.claimant_name}\n\n"
        f"{greeting}\n\n"
        f"I am writing to report a loss at {f.location}. {policy}"
        f"On {fmt_date(f.date_of_loss, rng)} {f.narrative}. "
        f"{extra}"
        f"I would like to claim {fmt_amount(f.amount, rng)} to cover the damage. "
        f"Your reference for this is {f.claim_number}.\n\n"
        f"Kind regards,\n{f.claimant_name}\n"
    )


def _web_form(f: ClaimFacts, rng: random.Random) -> str:
    return (
        "ONLINE CLAIM SUBMISSION\n"
        f"Claimant: {f.claimant_name}\n"
        f"Policy No: {f.policy_number}\n"
        f"Claim Ref: {f.claim_number}\n"
        f"Date of incident: {fmt_date(f.date_of_loss, rng)}\n"
        f"Address of loss: {f.location}\n"
        f"Description: {f.narrative[0].upper() + f.narrative[1:]}.\n"
        f"Amount claimed: {fmt_amount(f.amount, rng)}\n"
    )


def render_clean(f: ClaimFacts, rng: random.Random):
    if rng.random() < 0.5:
        return _email(f, rng), Channel.EMAIL, answer_key_for(f)
    return _web_form(f, rng), Channel.WEB_FORM, answer_key_for(f)


def render_missing_policy(f: ClaimFacts, rng: random.Random):
    text = _email(f, rng, policy_line=False, extra="I could not find my policy documents but I have been insured with you for years. ")
    key = answer_key_for(
        f,
        policy_number="",
        expected_failure_step=StepName.EXTRACTION,
        expected_category=FailureCategory.EXTRACTION_HALLUCINATION,
    )
    return text, Channel.EMAIL, key


def render_decoy_policy(f: ClaimFacts, rng: random.Random):
    decoy = random_policy(rng)
    extra = (
        f"Please note I previously held policy {decoy} with another household, "
        f"which was cancelled in {rng.randint(2015, 2021)}. "
    )
    text = _email(f, rng, extra=extra)
    key = answer_key_for(
        f,
        decoy_values=[decoy],
        expected_failure_step=StepName.EXTRACTION,
        expected_category=FailureCategory.EXTRACTION_HALLUCINATION,
    )
    return text, Channel.EMAIL, key


def render_conflicting_dates(f: ClaimFacts, rng: random.Random):
    reported = f.date_of_loss + timedelta(days=rng.randint(3, 20))
    text = (
        f"Subject: Claim {f.claim_number}\n\n"
        f"Dear Claims Team,\n\n"
        f"I am reporting this on {fmt_date(reported, rng)}, having returned from holiday. "
        f"The damage at {f.location} happened while I was away, on {fmt_date(f.date_of_loss, rng)}, "
        f"when {f.narrative}. My policy is {f.policy_number}. "
        f"The repair quote is {fmt_amount(f.amount, rng)}.\n\n{f.claimant_name}\n"
    )
    key = answer_key_for(
        f,
        decoy_values=[reported.isoformat()],
        expected_failure_step=StepName.EXTRACTION,
        expected_category=FailureCategory.EXTRACTION_HALLUCINATION,
    )
    return text, Channel.EMAIL, key


_OCR_SWAPS = {"O": "0", "l": "1", "S": "5", "I": "l", "B": "8", "e": "c", "a": "o", "m": "rn"}


def _garble(text: str, rng: random.Random, rate: float = 0.08) -> str:
    out = []
    for ch in text:
        if ch in _OCR_SWAPS and rng.random() < rate:
            out.append(_OCR_SWAPS[ch])
        elif ch == " " and rng.random() < 0.02:
            out.append("")
        else:
            out.append(ch)
    return "".join(out)


def render_ocr_garble(f: ClaimFacts, rng: random.Random):
    text = "[FAX RECEIVED - AUTO OCR]\n" + _garble(_web_form(f, rng), rng)
    key = answer_key_for(
        f,
        expected_failure_step=StepName.INTAKE,
        expected_category=FailureCategory.CONTEXT_LOSS,
    )
    return text, Channel.FAX_OCR, key


def render_truncated(f: ClaimFacts, rng: random.Random):
    full = _email(f, rng)
    cut = full.index("I would like to claim")
    text = full[:cut].rstrip() + " I would like to cl"
    key = answer_key_for(
        f,
        reported_amount="",
        claim_number="",
        expected_failure_step=StepName.EXTRACTION,
        expected_category=FailureCategory.EXTRACTION_HALLUCINATION,
    )
    return text, Channel.PDF_TEXT, key


def render_off_domain(f: ClaimFacts, rng: random.Random):
    text = rng.choice(
        [
            (
                "THE OLIVE BRANCH TRATTORIA\n12 Market Square\n\n"
                "Table 7   Covers: 4\n2 x Margherita ........ 22.00\n1 x Lasagne ........... 14.50\n"
                "3 x House Red ......... 21.00\nService (12.5%) ........ 7.18\n"
                "TOTAL ................ 64.68\n\nThank you for dining with us!"
            ),
            (
                "Subject: Newsletter sign-up\n\nHi, please add me to your mailing list for "
                "home-improvement tips. I saw your stand at the county show last weekend and "
                "loved the leaflet about loft insulation. Thanks, "
                f"{f.claimant_name}"
            ),
        ]
    )
    key = AnswerKey(
        claimant_name="",
        doc_type=DocType.CORRESPONDENCE,
        claim_category=ClaimCategory.UNKNOWN,
        severity=Severity.LOW,
        routing_queue=RoutingQueue.STANDARD,
        expected_failure_step=StepName.INTAKE,
        expected_category=FailureCategory.MISCLASSIFICATION,
    )
    return text, Channel.EMAIL, key


def render_multi_claim(f: ClaimFacts, rng: random.Random):
    old = sample_facts(rng)
    text = (
        f"Subject: Another claim - {f.claimant_name}\n\n"
        f"Dear Claims Team,\n\n"
        f"You may remember my earlier claim {old.claim_number}: on {fmt_date(old.date_of_loss, rng)} "
        f"{old.narrative}, and you paid {fmt_amount(old.amount, rng)}. That one is closed.\n\n"
        f"Unfortunately I need to report a new loss. On {fmt_date(f.date_of_loss, rng)} {f.narrative} "
        f"at {f.location}. My policy number is {f.policy_number} and I am claiming "
        f"{fmt_amount(f.amount, rng)}. Please open a new reference, {f.claim_number}.\n\n"
        f"Thanks,\n{f.claimant_name}\n"
    )
    key = answer_key_for(
        f,
        decoy_values=[old.claim_number, old.date_of_loss.isoformat(), f"{old.amount:.2f}"],
        expected_failure_step=StepName.EXTRACTION,
        expected_category=FailureCategory.EXTRACTION_HALLUCINATION,
    )
    return text, Channel.EMAIL, key


def render_ambiguous_peril(f: ClaimFacts, rng: random.Random):
    # A liability claim whose circumstances are described in water/fire vocabulary.
    # The injury to a third party is what makes it liability, not the puddle.
    f = replace(f, peril=Peril.LIABILITY, category=ClaimCategory.LIABILITY)
    circumstance = rng.choice(
        [
            "a leaking roof left a pool of water by the shop entrance, and a customer slipped on it and broke an ankle",
            "the fire sprinkler system discharged by mistake, soaking the floor, and a visitor fell and injured their back",
        ]
    )
    text = (
        f"Subject: Incident at {f.location}\n\n"
        f"To whom it may concern,\n\n"
        f"On {fmt_date(f.date_of_loss, rng)} {circumstance}. The injured person's solicitor has "
        f"written to us seeking compensation of {fmt_amount(f.amount, rng)}. "
        f"Our policy is {f.policy_number}, reference {f.claim_number}.\n\n{f.claimant_name}\n"
    )
    key = answer_key_for(
        f,
        expected_failure_step=StepName.EXTRACTION,
        expected_category=FailureCategory.MISCLASSIFICATION,
    )
    return text, Channel.EMAIL, key


_BOILERPLATE = (
    "This communication and any attachments are confidential and intended solely for the "
    "addressee. If you have received it in error please notify the sender and delete it. "
    "Nothing in this message constitutes an admission of liability or a waiver of any right "
    "under the policy. The insurer reserves all rights under the terms and conditions of cover, "
    "including but not limited to the right to decline indemnity where a condition precedent "
    "has not been complied with, and the right to appoint loss adjusters, surveyors or other "
    "experts at its sole discretion. Policyholders are reminded of their duty to take all "
    "reasonable precautions to prevent further loss or damage and to retain any damaged items "
    "for inspection. "
)


def render_verbose_padding(f: ClaimFacts, rng: random.Random):
    # Key identifiers up front, the incident at the very end, and enough boilerplate in
    # between to exceed the model context window. When the prompt overflows, the start
    # of the document is what gets dropped - taking the identifiers with it.
    header = (
        f"CLAIM NOTIFICATION\nClaimant: {f.claimant_name}\nPolicy: {f.policy_number}\n"
        f"Reference: {f.claim_number}\n\n"
    )
    padding = "\n\n".join(_BOILERPLATE for _ in range(26))
    tail = (
        f"\n\nINCIDENT DETAILS\nOn {fmt_date(f.date_of_loss, rng)} {f.narrative} at {f.location}. "
        f"Amount claimed: {fmt_amount(f.amount, rng)}.\n"
    )
    key = answer_key_for(
        f,
        expected_failure_step=StepName.INTAKE,
        expected_category=FailureCategory.CONTEXT_LOSS,
    )
    return header + padding + tail, Channel.PDF_TEXT, key


def render_non_english(f: ClaimFacts, rng: random.Random):
    euro_amount = f"{f.amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    d = f.date_of_loss
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre"]
    text = (
        f"Subject: Claim {f.claim_number}\n\n"
        f"Hello, I am writing partly in Spanish, sorry.\n\n"
        f"Me llamo {f.claimant_name}. Mi numero de poliza es {f.policy_number}. "
        f"El {d.day} de {meses[d.month - 1]} de {d.year} tuvimos un siniestro en {f.location}: "
        f"{f.narrative}. Reclamo {euro_amount} dolares por los danos.\n\n"
        f"Thank you.\n"
    )
    key = answer_key_for(
        f,
        expected_failure_step=StepName.EXTRACTION,
        expected_category=FailureCategory.EXTRACTION_HALLUCINATION,
    )
    return text, Channel.EMAIL, key


def render_contradictory_amount(f: ClaimFacts, rng: random.Random):
    estimate = round(f.amount * rng.uniform(4, 9) / 100) * 100
    text = (
        f"Subject: Claim for {f.claimant_name}\n\n"
        f"Hello,\n\nPolicy {f.policy_number}, reference {f.claim_number}. "
        f"On {fmt_date(f.date_of_loss, rng)} {f.narrative} at {f.location}.\n\n"
        f"The contractor quoted {fmt_amount(estimate, rng)} for a full renovation of the area, "
        f"but that includes upgrades I am paying for myself. I am only claiming "
        f"{fmt_amount(f.amount, rng)}, which is the cost of putting it back as it was.\n\n"
        f"Regards,\n{f.claimant_name}\n"
    )
    key = answer_key_for(
        f,
        decoy_values=[f"{estimate:.2f}"],
        expected_failure_step=StepName.EXTRACTION,
        expected_category=FailureCategory.EXTRACTION_HALLUCINATION,
    )
    return text, Channel.EMAIL, key


def render_severity_boundary(f: ClaimFacts, rng: random.Random):
    """A clean claim whose amount sits on, or one cent beside, a severity boundary.

    Extraction should be straightforward; the intended pressure point is whether
    classification applies the documented threshold exactly rather than approximately.
    """
    amount = rng.choice([4_999.99, 5_000.00, 24_999.99, 25_000.00, 99_999.99, 100_000.00])
    f = replace(f, amount=amount)
    text = _web_form(f, rng)
    key = answer_key_for(
        f,
        expected_failure_step=StepName.CLASSIFICATION,
        expected_category=FailureCategory.MISCLASSIFICATION,
    )
    return text, Channel.WEB_FORM, key


def render_sparse_claim(f: ClaimFacts, rng: random.Random):
    """A valid notice with several legitimately unknown facts.

    A correct extraction leaves the unavailable fields empty. The summary must resist
    filling those gaps and should surface them as open questions, making summarization
    (rather than extraction) the intended pressure point.
    """
    text = (
        "Subject: Initial loss notification - details to follow\n\n"
        "Hello,\n\n"
        f"My name is {f.claimant_name}. On {fmt_date(f.date_of_loss, rng)}, {f.narrative}. "
        "I do not have the policy number, claim reference, exact loss address, or a repair "
        "estimate available yet. Please register this initial notification and tell me what "
        "information you need next.\n\n"
        f"Regards,\n{f.claimant_name}\n"
    )
    key = answer_key_for(
        f,
        policy_number="",
        claim_number="",
        reported_amount="",
        loss_location="",
        severity=Severity.MEDIUM,
        routing_queue=RoutingQueue.STANDARD,
        expected_failure_step=StepName.SUMMARIZATION,
        expected_category=FailureCategory.SUMMARY_HALLUCINATION,
    )
    return text, Channel.EMAIL, key


RENDERERS: dict[FailureMode, Callable] = {
    FailureMode.CLEAN: render_clean,
    FailureMode.MISSING_POLICY_NUMBER: render_missing_policy,
    FailureMode.DECOY_POLICY_NUMBER: render_decoy_policy,
    FailureMode.CONFLICTING_DATES: render_conflicting_dates,
    FailureMode.OCR_GARBLE: render_ocr_garble,
    FailureMode.TRUNCATED: render_truncated,
    FailureMode.OFF_DOMAIN: render_off_domain,
    FailureMode.MULTI_CLAIM: render_multi_claim,
    FailureMode.AMBIGUOUS_PERIL: render_ambiguous_peril,
    FailureMode.VERBOSE_PADDING: render_verbose_padding,
    FailureMode.NON_ENGLISH_FRAGMENT: render_non_english,
    FailureMode.CONTRADICTORY_AMOUNT: render_contradictory_amount,
    FailureMode.SEVERITY_BOUNDARY: render_severity_boundary,
    FailureMode.SPARSE_CLAIM: render_sparse_claim,
}
