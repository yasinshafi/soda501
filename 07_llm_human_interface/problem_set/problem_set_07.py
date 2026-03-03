###############################################################################
# LLM-Assisted Data Extraction (Human-in-the-Loop)
# Submission Script — Questions 4 and 5
# Submitted by: Yasin Shafi
###############################################################################

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
import os
os.environ["HF_HOME"] = "/storage/work/fxs5261/huggingface_cache"
import json
import re

import numpy as np
import pandas as pd

import torch

from datetime import date
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

np.random.seed(123)
set_seed(123)

# ---------------------------------------------------------------------
# Part 1: Define a Schema
# ---------------------------------------------------------------------
EventType = Literal[
    "protest",
    "election",
    "policy_change",
    "violence",
    "disaster",
    "other"
]

GeoPrecision = Literal[
    "country_only",
    "admin1_or_state",
    "city_or_local",
    "unknown"
]

class EvidenceSpan(BaseModel):
    field: Literal["event_type", "date", "location", "actors", "outcome"]
    quote: str

class EventExtraction(BaseModel):
    doc_id: str
    event_type: EventType = "other"
    event_date_iso: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD if available; otherwise null.")
    date_is_approximate: bool = Field(default=True, description="True if the date is estimated/inferred.")
    country: Optional[str] = None
    admin1_or_state: Optional[str] = None
    city_or_local: Optional[str] = None
    geo_precision: GeoPrecision = "unknown"
    actors: List[str] = Field(default_factory=list, description="Key actors mentioned.")
    outcome_summary: Optional[str] = Field(default=None, description="One-sentence outcome summary.")
    extraction_confidence: float = Field(default=0.2, ge=0.0, le=1.0, description="Model self-rated confidence (0 to 1).")
    uncertainty_flags: List[str] = Field(default_factory=list, description="Issues that make extraction uncertain.")
    evidence: List[EvidenceSpan] = Field(default_factory=list, description="Short quotes supporting each extracted field.")

# ---------------------------------------------------------------------
# Part 2: Mini Corpus
# ---------------------------------------------------------------------
docs = [
    {"doc_id": "doc_001", "text": "Breaking: Thousands rallied in Santiago on 2026-03-14 demanding pension reform. Police reported minor clashes; 12 were arrested."},
    {"doc_id": "doc_002", "text": "On March 2nd, lawmakers passed the 'Clean Air Act' amendment in the national assembly. Environmental groups praised the vote."},
    {"doc_id": "doc_003", "text": "Election officials said voting will take place next Sunday. Turnout is expected to be high in the capital."},
    {"doc_id": "doc_004", "text": "A 6.2 magnitude earthquake struck near the coastal city overnight, damaging dozens of homes and cutting power to 40,000 residents."},
    {"doc_id": "doc_005", "text": "Witnesses described gunfire outside a nightclub late Friday; at least two people were injured, but details remain unclear."},
    {"doc_id": "doc_006", "text": "The governor announced a new curfew order effective immediately. Critics called it an overreach."},
    {"doc_id": "doc_007", "text": "Early April saw renewed demonstrations in the northern province after fuel prices rose again."},
    {"doc_id": "doc_008", "text": "Floodwaters inundated low-lying neighborhoods; emergency shelters opened at local schools, officials said."},
    {"doc_id": "doc_009", "text": "Opposition leaders met with international observers in Brussels to discuss election monitoring."},
    {"doc_id": "doc_010", "text": "Police said the suspect was arrested after a stabbing in downtown; the mayor urged calm."},
    {"doc_id": "doc_011", "text": "Parliament reversed the prior ban on rideshare apps, citing labor market flexibility."},
    {"doc_id": "doc_012", "text": "A protest was planned for tomorrow, but organizers postponed it due to severe weather warnings."},
    {"doc_id": "doc_013", "text": "Following a landslide, the ministry declared a state of emergency in two districts."},
    {"doc_id": "doc_014", "text": "The court ruling sparked demonstrations across the city center; human rights groups condemned the decision."},
    {"doc_id": "doc_015", "text": "The article mentions reforms and elections in passing but gives no clear time or place."},
]

docs_df = pd.DataFrame(docs)
print("\ndocs_df shape:", docs_df.shape)

# ---------------------------------------------------------------------
# Part 3: Prompt Design
# ---------------------------------------------------------------------

# Q4 REPORTING: Exact prompt used
system_instructions = (
    "Task: Extract ONE event record from the text.\n"
    "Return EXACTLY the following 9 lines, one per line, in the format key: value\n"
    "Use empty value if unknown.\n"
    "\n"
    "event_type: protest|election|policy_change|violence|disaster|other\n"
    "event_date_iso: YYYY-MM-DD\n"
    "date_is_approximate: true|false\n"
    "country:\n"
    "admin1_or_state:\n"
    "city_or_local:\n"
    "geo_precision: country_only|admin1_or_state|city_or_local|unknown\n"
    "actors: comma-separated list\n"
    "outcome_summary: one sentence\n"
    "\n"
    "Do not output anything else.\n"
)

print("\n--- Q4: Exact prompt used ---")
print(system_instructions)

# ---------------------------------------------------------------------
# Part 4: Load Model
# ---------------------------------------------------------------------

# Q4 REPORTING: Model name
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"\n--- Q4: Model used: {model_name} ---")

print("\nLoading tokenizer + model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32  # CPU-safe
)

device = torch.device("cpu")
model = model.to(device)
print(f"Device: {device}")

# ---------------------------------------------------------------------
# Part 4: Extraction Loop
# ---------------------------------------------------------------------
extractions = []
parse_failure_count = 0

print("\nRunning extraction...")

for i in range(len(docs_df)):
    doc_id = docs_df.loc[i, "doc_id"]
    text = docs_df.loc[i, "text"]

    prompt = (
        f"{system_instructions}\n"
        f"Document ID: {doc_id}\n"
        f"Text: {text}\n"
    )

    # 1) Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2) Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 3) Decode — strip the prompt from the output (causal LM repeats it)
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Only keep text after the prompt
    out_text = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text

    # 4) Parse key:value lines
    parse_ok = True
    parse_error = ""
    parse_flags: List[str] = []

    allowed_keys = {
        "event_type", "event_date_iso", "date_is_approximate",
        "country", "admin1_or_state", "city_or_local",
        "geo_precision", "actors", "outcome_summary"
    }

    lines = [ln.strip() for ln in out_text.splitlines() if ":" in ln]
    kv = {}
    for ln in lines:
        k, v = ln.split(":", 1)
        k_norm = k.strip().lower()
        if k_norm in allowed_keys:
            kv[k_norm] = v.strip()

    # Q4: Log parse failures
    if len(kv) == 0:
        parse_ok = False
        parse_error = "no_key_value_lines_found"
        parse_flags.append("parse_failed_local_model_output")
        parse_failure_count += 1

    event_type = (kv.get("event_type", "other") or "other").strip().lower()
    if event_type not in {"protest", "election", "policy_change", "violence", "disaster", "other"}:
        parse_flags.append("invalid_event_type_from_model")
        event_type = "other"

    event_date_iso = (kv.get("event_date_iso", "") or "").strip() or None

    date_is_approx_raw = (kv.get("date_is_approximate", "") or "").strip().lower()
    if date_is_approx_raw in {"true", "false"}:
        date_is_approximate = (date_is_approx_raw == "true")
    else:
        parse_flags.append("date_is_approximate_missing_or_invalid")
        date_is_approximate = True

    country = (kv.get("country", "") or "").strip() or None
    admin1_or_state = (kv.get("admin1_or_state", "") or "").strip() or None
    city_or_local = (kv.get("city_or_local", "") or "").strip() or None

    geo_precision = (kv.get("geo_precision", "unknown") or "unknown").strip().lower()
    if geo_precision not in {"country_only", "admin1_or_state", "city_or_local", "unknown"}:
        parse_flags.append("invalid_geo_precision_from_model")
        geo_precision = "unknown"

    actors_raw = (kv.get("actors", "") or "").strip()
    actors = [a.strip() for a in actors_raw.split(",") if a.strip()] if actors_raw else []

    outcome_summary = (kv.get("outcome_summary", "") or "").strip() or None

    extracted_obj = EventExtraction(
        doc_id=doc_id,
        event_type=event_type,
        event_date_iso=event_date_iso,
        date_is_approximate=date_is_approximate,
        country=country,
        admin1_or_state=admin1_or_state,
        city_or_local=city_or_local,
        geo_precision=geo_precision,
        actors=actors,
        outcome_summary=outcome_summary,
        extraction_confidence=0.35 if parse_ok else 0.2,
        uncertainty_flags=parse_flags,
        evidence=[]
    )

    extra_dict = extracted_obj.model_dump()
    extra_dict["raw_text"] = text
    extra_dict["local_model_raw_output"] = out_text  # Q4: raw output logged here
    extra_dict["parse_ok"] = parse_ok
    extra_dict["parse_error"] = parse_error

    extra_dict["evidence_json"] = json.dumps(extra_dict["evidence"], ensure_ascii=False)
    extra_dict["uncertainty_flags_json"] = json.dumps(extra_dict["uncertainty_flags"], ensure_ascii=False)
    extra_dict.pop("evidence")
    extra_dict.pop("uncertainty_flags")

    extractions.append(extra_dict)
    print(f"  [{i+1:02d}/15] {doc_id} | event_type={event_type} | parse_ok={parse_ok}")

extractions_df = pd.DataFrame(extractions)
os.makedirs("outputs", exist_ok=True)
extractions_df.to_csv("outputs/extractions_raw.csv", index=False)

# ---------------------------------------------------------------------
# Q4: Parse Failure Report
# ---------------------------------------------------------------------
total_docs = len(extractions_df)
n_failures = parse_failure_count
share_failures = n_failures / total_docs

print("\n--- Q4: Parse Failure Report ---")
print(f"Total documents:   {total_docs}")
print(f"Parse failures:    {n_failures}  ({share_failures:.1%})")
print(
    "Handling strategy: When no valid key:value lines were found in the model output, "
    "the pipeline fell back to safe schema defaults (event_type='other', all fields null/empty, "
    "extraction_confidence=0.2) and appended 'parse_failed_local_model_output' to uncertainty_flags. "
    "The raw model output was logged in the 'local_model_raw_output' column for manual inspection."
)

# ---------------------------------------------------------------------
# Part 5 (Q5): Uncertainty Flags
# ---------------------------------------------------------------------
extractions_df["extraction_confidence"] = pd.to_numeric(extractions_df["extraction_confidence"], errors="coerce")

extractions_df["flag_parse_failed"]    = ~extractions_df["parse_ok"]
extractions_df["flag_low_confidence"]  = extractions_df["extraction_confidence"] < 0.70
extractions_df["flag_missing_date"]    = extractions_df["event_date_iso"].isna()
extractions_df["flag_missing_country"] = extractions_df["country"].isna()
extractions_df["flag_geo_unknown"]     = extractions_df["geo_precision"].isin(["unknown", "country_only"])

flag_cols = [
    "flag_parse_failed",
    "flag_low_confidence",
    "flag_missing_date",
    "flag_missing_country",
    "flag_geo_unknown"
]
extractions_df["needs_human_review"] = extractions_df[flag_cols].any(axis=1)

print("\n--- Q5: Review Flag Counts ---")
flag_summary = extractions_df[flag_cols + ["needs_human_review"]].sum(numeric_only=True)
print(flag_summary)
print(f"\nDocuments needing human review: {extractions_df['needs_human_review'].sum()} / {total_docs}")

extractions_df.to_csv("outputs/extractions_with_flags.csv", index=False)

# ---------------------------------------------------------------------
# Part 6 (Q5): Audit Sheet
# ---------------------------------------------------------------------
audit_random_n = 5
audit_random  = extractions_df.sample(n=audit_random_n, random_state=123)
audit_flagged = extractions_df[extractions_df["needs_human_review"]].copy()

audit_sheet = pd.concat([audit_random, audit_flagged], ignore_index=True).drop_duplicates(subset=["doc_id"])
audit_sheet  = audit_sheet.sort_values("doc_id").reset_index(drop=True)

# Blank columns for human review
audit_sheet["human_is_correct"]        = ""
audit_sheet["human_correct_event_type"] = ""
audit_sheet["human_correct_date_iso"]  = ""
audit_sheet["human_correct_location"]  = ""
audit_sheet["failure_mode"]            = ""
audit_sheet["reviewer_notes"]          = ""

audit_sheet.to_csv("outputs/human_audit_sheet.csv", index=False)
print(f"\n--- Q5: Audit sheet written ({len(audit_sheet)} rows) ---")
print("File: outputs/human_audit_sheet.csv")
print("Fill in 'human_is_correct', 'failure_mode', and 'reviewer_notes' columns manually.")

# ---------------------------------------------------------------------
# Q5: Audit Statistics (fill these in after manual review)
# ---------------------------------------------------------------------
print("\n--- Q5: Audit Statistics (to be filled after manual review) ---")
print("After filling the audit sheet, run the following to compute statistics:")
print("""
  audit_filled = pd.read_csv('outputs/human_audit_sheet.csv')
  n_audited    = audit_filled['human_is_correct'].notna().sum()
  n_correct    = (audit_filled['human_is_correct'] == 'yes').sum()
  share_correct = n_correct / n_audited if n_audited > 0 else 0
  print(f'Share correct: {share_correct:.1%}  ({n_correct}/{n_audited})')
  print('\\nFailure mode frequency:')
  print(audit_filled['failure_mode'].value_counts())
""")

print("\nDone. Outputs saved to outputs/")