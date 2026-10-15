"""Typed domain schemas for the 4-step insurance-claim pipeline.

Design note - why every LLM-facing field is a plain string
----------------------------------------------------------
The pipeline runs on a 3B local model with grammar-constrained decoding. Nullable and
richly-typed fields (``date``, ``Decimal``, ``Optional[...]``) make the generated grammar
markedly harder for a small model to satisfy, and a schema violation surfaces as an
opaque parse error rather than as data.

So the model emits strings and uses ``""`` to mean absent. Normalisation into real types
happens in code (``ffx.analyzer.grounding``), where a failure to parse becomes an
observable *signal* instead of a swallowed exception. This also keeps the model's raw
text intact, which is exactly what the grounding checks need.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware UTC now. (``datetime.UTC`` is 3.11+; this repo targets 3.10.)"""
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Enumerations
#
# Every enum carries an explicit escape value ("unknown" / "other"). Without one,
# grammar-constrained decoding forces the model to pick *some* listed value, which
# manufactures confident wrong answers and destroys the signal we are trying to read.
# --------------------------------------------------------------------------


class Channel(str, Enum):
    EMAIL = "email"
    FAX_OCR = "fax_ocr"
    WEB_FORM = "web_form"
    PDF_TEXT = "pdf_text"


class DocType(str, Enum):
    CLAIM_NOTICE = "claim_notice"
    POLICY_QUERY = "policy_query"
    MEDICAL_BILL = "medical_bill"
    CORRESPONDENCE = "correspondence"
    UNKNOWN = "unknown"


class QualityFlag(str, Enum):
    OCR_NOISE = "ocr_noise"
    TRUNCATED = "truncated"
    MULTI_DOCUMENT = "multi_document"
    NON_ENGLISH = "non_english"
    VERY_SHORT = "very_short"
    VERBOSE = "verbose"


class Peril(str, Enum):
    FIRE = "fire"
    WATER = "water"
    THEFT = "theft"
    COLLISION = "collision"
    WIND = "wind"
    LIABILITY = "liability"
    OTHER = "other"
    UNKNOWN = "unknown"


class ClaimCategory(str, Enum):
    AUTO = "auto"
    PROPERTY = "property"
    LIABILITY = "liability"
    HEALTH = "health"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RoutingQueue(str, Enum):
    FAST_TRACK = "fast_track"
    STANDARD = "standard"
    COMPLEX = "complex"
    SIU_REFERRAL = "siu_referral"


class ClaimFieldName(str, Enum):
    """The only field names evidence may refer to.

    Left as a free string, a 3B model invents its own names (observed: ``lapse_date``,
    ``loss_amount``, ``reference_number``), which silently decouples the evidence list
    from the fields it is supposed to substantiate. Constraining it to an enum forces
    the two halves of :class:`ExtractionResult` to line up.
    """

    CLAIMANT_NAME = "claimant_name"
    POLICY_NUMBER = "policy_number"
    CLAIM_NUMBER = "claim_number"
    DATE_OF_LOSS = "date_of_loss"
    REPORTED_AMOUNT = "reported_amount"
    LOSS_LOCATION = "loss_location"
    PERIL = "peril"


class StepName(str, Enum):
    INTAKE = "intake"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"


STEP_ORDER = [
    StepName.INTAKE,
    StepName.EXTRACTION,
    StepName.CLASSIFICATION,
    StepName.SUMMARIZATION,
]


class FailureMode(str, Enum):
    """Conditions injected into the synthetic corpus.

    These describe the *document*, not the outcome. Whether the model actually fails on
    a given document is observed at run time, never assumed.
    """

    CLEAN = "clean"
    MISSING_POLICY_NUMBER = "missing_policy_number"
    DECOY_POLICY_NUMBER = "decoy_policy_number"
    CONFLICTING_DATES = "conflicting_dates"
    OCR_GARBLE = "ocr_garble"
    TRUNCATED = "truncated"
    OFF_DOMAIN = "off_domain"
    MULTI_CLAIM = "multi_claim"
    AMBIGUOUS_PERIL = "ambiguous_peril"
    VERBOSE_PADDING = "verbose_padding"
    NON_ENGLISH_FRAGMENT = "non_english_fragment"
    CONTRADICTORY_AMOUNT = "contradictory_amount"
    SEVERITY_BOUNDARY = "severity_boundary"
    SPARSE_CLAIM = "sparse_claim"


class FailureCategory(str, Enum):
    """The diagnosis taxonomy (brief Phase 3).

    The brief's five categories plus one: ``SUMMARY_HALLUCINATION``. None of the original
    five describes a summarizer asserting a fact its input does not contain, and filing
    that under "Extraction Hallucination" would name the wrong step - the one thing a
    forensics tool must not do.
    """

    EXTRACTION_HALLUCINATION = "extraction_hallucination"
    MISCLASSIFICATION = "misclassification"
    PROPAGATION_ERROR = "propagation_error"
    PROMPT_FAILURE = "prompt_failure"
    CONTEXT_LOSS = "context_loss"
    SUMMARY_HALLUCINATION = "summary_hallucination"
    # Not a pipeline failure at all: the model server crashed, ran out of memory, or a
    # provider was down. Kept separate so hardware trouble is never blamed on a prompt.
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    NO_FAILURE = "no_failure"


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


class AnswerKey(BaseModel):
    """Ground truth for a synthetic document.

    Powers the "should-have-produced" pane of the diff view and the counterfactual
    re-run. Fields use ``""`` for "legitimately absent from the document" - which is
    itself the correct answer for modes like ``missing_policy_number``.
    """

    claimant_name: str = ""
    policy_number: str = ""
    claim_number: str = ""
    date_of_loss: str = ""
    reported_amount: str = ""
    loss_location: str = ""
    peril: Peril = Peril.UNKNOWN

    claim_category: ClaimCategory = ClaimCategory.UNKNOWN
    severity: Severity = Severity.MEDIUM
    routing_queue: RoutingQueue = RoutingQueue.STANDARD
    doc_type: DocType = DocType.CLAIM_NOTICE

    # Values that appear in the text as bait but must NOT be extracted.
    decoy_values: list[str] = Field(default_factory=list)
    # Which step we expect to break, if any. None means the document should pass.
    expected_failure_step: Optional[StepName] = None
    expected_category: FailureCategory = FailureCategory.NO_FAILURE


class RawDocument(BaseModel):
    doc_id: str
    source_text: str
    channel: Channel = Channel.EMAIL
    received_at: datetime = Field(default_factory=utcnow)


class CorpusDocument(BaseModel):
    document: RawDocument
    failure_mode: FailureMode
    answer_key: AnswerKey


# --------------------------------------------------------------------------
# Step 1 - Intake
# --------------------------------------------------------------------------


class IntakeResult(BaseModel):
    """Normalise, segment and triage the raw document."""

    normalized_text: str = Field(description="Cleaned document text, verbatim content preserved.")
    detected_doc_type: DocType = Field(description="What kind of document this is.")
    quality_flags: list[QualityFlag] = Field(
        default_factory=list, description="Any quality problems observed in the document."
    )
    confidence: int = Field(ge=1, le=5, description="Self-reported confidence, 1 (lowest) to 5 (highest).")


# --------------------------------------------------------------------------
# Step 2 - Extraction
# --------------------------------------------------------------------------


class FieldEvidence(BaseModel):
    """Per-field provenance. The linchpin of deterministic hallucination detection.

    ``source_quote`` must be text copied verbatim from the input. If the quote is not
    found in the input the model fabricated its own evidence; if the quote is present but
    does not support ``value``, the value is ungrounded. Both checks are pure string
    operations requiring no LLM - see ``ffx.analyzer.grounding``.
    """

    field: ClaimFieldName = Field(description="Which claim field this evidence supports.")
    value: str = Field(default="", description="The extracted value, or empty string if absent.")
    source_quote: str = Field(
        default="",
        description="Text copied EXACTLY from the document that this value came from. Empty if absent.",
    )


class ClaimFields(BaseModel):
    """All strings by design - see the module docstring."""

    claimant_name: str = Field(default="", description="Full name of the claimant, or empty string.")
    policy_number: str = Field(default="", description="Policy number for THIS claim, or empty string.")
    claim_number: str = Field(default="", description="Claim/reference number, or empty string.")
    date_of_loss: str = Field(default="", description="Date the loss occurred as YYYY-MM-DD, or empty string.")
    reported_amount: str = Field(default="", description="Amount claimed, digits only e.g. 4500.00, or empty.")
    loss_location: str = Field(default="", description="Where the loss occurred, or empty string.")
    peril: Peril = Field(default=Peril.UNKNOWN, description="The cause of loss.")


class ExtractionResult(BaseModel):
    fields: ClaimFields
    evidence: list[FieldEvidence] = Field(
        default_factory=list, description="One entry per field you filled in, with the exact supporting quote."
    )
    missing_fields: list[str] = Field(
        default_factory=list, description="Names of fields you could not find in the document."
    )
    confidence: int = Field(ge=1, le=5, description="Self-reported confidence, 1 (lowest) to 5 (highest).")


# --------------------------------------------------------------------------
# Step 3 - Classification
# --------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    claim_category: ClaimCategory = Field(description="Line of business for this claim.")
    severity: Severity = Field(description="How severe this claim is.")
    routing_queue: RoutingQueue = Field(description="Which handling queue this should go to.")
    rationale: str = Field(default="", description="One sentence explaining the routing decision.")
    confidence: int = Field(ge=1, le=5, description="Self-reported confidence, 1 (lowest) to 5 (highest).")


# --------------------------------------------------------------------------
# Step 4 - Summarization
# --------------------------------------------------------------------------


class SummaryResult(BaseModel):
    headline: str = Field(description="One-line summary of the claim.")
    body: str = Field(description="Three to five sentences summarising the claim for an adjuster.")
    key_facts: list[str] = Field(
        default_factory=list, description="Bullet facts, each traceable to an extracted field."
    )
    open_questions: list[str] = Field(
        default_factory=list, description="What is still unknown and must be chased up."
    )
    confidence: int = Field(ge=1, le=5, description="Self-reported confidence, 1 (lowest) to 5 (highest).")


STEP_OUTPUT_MODELS = {
    StepName.INTAKE: IntakeResult,
    StepName.EXTRACTION: ExtractionResult,
    StepName.CLASSIFICATION: ClassificationResult,
    StepName.SUMMARIZATION: SummaryResult,
}

CLAIM_FIELD_NAMES = list(ClaimFields.model_fields.keys())


def normalized_field_map(extraction: ExtractionResult) -> dict:
    """Flatten extracted claim fields to ``{field_name: value}`` for retention tracking."""
    out = {}
    for key, value in extraction.fields.model_dump().items():
        out[key] = value.value if isinstance(value, Enum) else str(value)
    return out


def dump(obj: Any) -> Any:
    """JSON-safe dump used by the tracing store."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dump(v) for v in obj]
    return obj
