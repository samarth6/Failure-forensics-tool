"""The four pipeline steps: Intake -> Extraction -> Classification -> Summarization.

Each step is declared, not hand-wired: an input model, a prompt builder, an output model
and a token budget. One generic executor (:mod:`ffx.pipeline.runner`) runs any of them
under tracing. That separation is what lets the counterfactual analyzer re-run *any*
downstream step from a spliced input without special cases.

Information flow is deliberately narrow. Classification sees extracted fields, not the
raw document; summarization sees fields and classification, not the document. So if a
fact is wrong at the end, it was wrong in a structured payload somewhere upstream - which
is exactly what makes the backward walk able to locate it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field

from ffx.llm.schema import describe, strict_schema
from ffx.models import (
    ClaimFields,
    ClassificationResult,
    DocType,
    ExtractionResult,
    IntakeResult,
    QualityFlag,
    RawDocument,
    StepName,
    SummaryResult,
)

SYSTEM_PROMPT = (
    "You are one step in an automated insurance claims processing pipeline. "
    "Follow the instructions exactly and respond with JSON only."
)


# --------------------------------------------------------------------------
# Composite inputs for the later steps
# --------------------------------------------------------------------------


class ClassificationInput(BaseModel):
    doc_type: DocType
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    fields: ClaimFields


class SummarizationInput(BaseModel):
    fields: ClaimFields
    missing_fields: list[str] = Field(default_factory=list)
    classification: ClassificationResult


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def intake_prompt(doc: RawDocument) -> str:
    return (
        "TASK: You are the INTAKE step. Prepare the document below for data extraction.\n"
        "1. normalized_text: rewrite the document with obvious OCR errors fixed and legal "
        "boilerplate or email disclaimers removed. Keep EVERY claim-relevant sentence, "
        "including names, policy numbers, reference numbers, dates, amounts, addresses and "
        "the description of what happened. Do not summarise or invent anything.\n"
        "2. detected_doc_type: claim_notice if someone is reporting an insurance loss, "
        "otherwise the closest other type, or unknown.\n"
        "3. quality_flags: any problems you notice with the document.\n"
        "4. confidence: 1-5.\n\n"
        f"OUTPUT FIELDS:\n{describe(strict_schema(IntakeResult))}\n\n"
        f"DOCUMENT (received via {doc.channel.value}):\n<<<\n{doc.source_text}\n>>>\n"
    )


def extraction_prompt(intake: IntakeResult) -> str:
    return (
        "TASK: You are the EXTRACTION step. Extract the insurance claim fields from the "
        "document below.\n"
        "Rules:\n"
        "- Use an empty string for any field the document does not state. Never guess.\n"
        "- policy_number is the policy covering THIS loss.\n"
        "- date_of_loss is when the loss happened, formatted YYYY-MM-DD.\n"
        "- reported_amount is the amount being claimed, digits only with two decimals, e.g. 4500.00.\n"
        "- For every field you fill in, add an evidence entry with the same value and a "
        "source_quote copied EXACTLY, character for character, from the document.\n"
        "- List any field you could not find in missing_fields.\n\n"
        f"OUTPUT FIELDS:\n{describe(strict_schema(ExtractionResult))}\n\n"
        f"DOCUMENT:\n<<<\n{intake.normalized_text}\n>>>\n"
    )


def classification_prompt(inp: ClassificationInput) -> str:
    return (
        "TASK: You are the CLASSIFICATION step. Decide how this claim should be handled, "
        "using ONLY the extracted data below.\n"
        "claim_category: auto for vehicle damage; property for damage to buildings or "
        "belongings; liability when a third party was injured or suffered loss and is "
        "claiming against the insured; health for medical treatment; unknown if this is "
        "not an insurance claim.\n"
        "severity by amount claimed: low under 5,000; medium 5,000 to 25,000; high 25,000 "
        "to 100,000; critical over 100,000.\n"
        "routing_queue: fast_track for low severity with complete information; standard "
        "for typical claims; complex for high or critical severity; siu_referral if the "
        "information is inconsistent or suspicious.\n\n"
        f"OUTPUT FIELDS:\n{describe(strict_schema(ClassificationResult))}\n\n"
        f"EXTRACTED DATA:\n{inp.model_dump_json(indent=2)}\n"
    )


def summarization_prompt(inp: SummarizationInput) -> str:
    return (
        "TASK: You are the SUMMARIZATION step. Write a brief for the claims adjuster using "
        "ONLY the data below. Do not add facts that are not present. Mention the claimant, "
        "policy number, date of loss, amount and cause where known. Put anything missing "
        "into open_questions.\n\n"
        f"OUTPUT FIELDS:\n{describe(strict_schema(SummaryResult))}\n\n"
        f"CLAIM DATA:\n{inp.model_dump_json(indent=2)}\n"
    )


# --------------------------------------------------------------------------
# Step registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSpec:
    name: StepName
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    build_prompt: Callable[[BaseModel], str]
    max_tokens: int

    @property
    def schema(self) -> dict:
        return strict_schema(self.output_model)


STEPS: dict[StepName, StepSpec] = {
    StepName.INTAKE: StepSpec(StepName.INTAKE, RawDocument, IntakeResult, intake_prompt, 1400),
    StepName.EXTRACTION: StepSpec(StepName.EXTRACTION, IntakeResult, ExtractionResult, extraction_prompt, 1000),
    StepName.CLASSIFICATION: StepSpec(
        StepName.CLASSIFICATION, ClassificationInput, ClassificationResult, classification_prompt, 300
    ),
    StepName.SUMMARIZATION: StepSpec(
        StepName.SUMMARIZATION, SummarizationInput, SummaryResult, summarization_prompt, 700
    ),
}


def build_input(step: StepName, doc: RawDocument, outputs: dict[StepName, BaseModel]) -> BaseModel:
    """Assemble a step's input from the document and upstream outputs."""
    if step == StepName.INTAKE:
        return doc
    if step == StepName.EXTRACTION:
        return outputs[StepName.INTAKE]
    if step == StepName.CLASSIFICATION:
        intake: IntakeResult = outputs[StepName.INTAKE]  # type: ignore[assignment]
        extraction: ExtractionResult = outputs[StepName.EXTRACTION]  # type: ignore[assignment]
        return ClassificationInput(
            doc_type=intake.detected_doc_type,
            quality_flags=intake.quality_flags,
            fields=extraction.fields,
        )
    if step == StepName.SUMMARIZATION:
        extraction = outputs[StepName.EXTRACTION]  # type: ignore[assignment]
        return SummarizationInput(
            fields=extraction.fields,  # type: ignore[attr-defined]
            missing_fields=extraction.missing_fields,  # type: ignore[attr-defined]
            classification=outputs[StepName.CLASSIFICATION],  # type: ignore[arg-type]
        )
    raise ValueError(step)


def pipeline_version() -> str:
    """Fingerprint of every prompt template and schema.

    Stored on each trace so eval regression runs can tell whether a pass/fail change came
    from a prompt edit or from something else.
    """
    probe_doc = RawDocument(doc_id="probe", source_text="{DOC}")
    probe_intake = IntakeResult(normalized_text="{DOC}", detected_doc_type=DocType.UNKNOWN, confidence=3)
    probe_fields = ClaimFields()
    probe_cls = ClassificationResult(
        claim_category="unknown", severity="low", routing_queue="standard", confidence=3
    )
    material = "|".join(
        [
            SYSTEM_PROMPT,
            intake_prompt(probe_doc),
            extraction_prompt(probe_intake),
            classification_prompt(ClassificationInput(doc_type=DocType.UNKNOWN, fields=probe_fields)),
            summarization_prompt(SummarizationInput(fields=probe_fields, classification=probe_cls)),
            *[str(spec.max_tokens) for spec in STEPS.values()],
        ]
    )
    return "pv_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:10]
