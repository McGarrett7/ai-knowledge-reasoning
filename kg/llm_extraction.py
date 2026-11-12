"""Extraction operator L: D -> P(E x R x E x [0,1])  (Sec. 4.2).

Raster samples are verbalised as short Vietnamese/English observation notes
(the corpus D). An extractor turns each note into weighted triples
(h, r, t, p); p becomes the extraction weight w_ext used as the prior belief
mu(t) by the reasoning algorithms.

  * RuleExtractor - deterministic regular expressions, p = 1. Used when no LLM
                    is configured; it reproduces the template KG exactly.
  * LLMExtractor  - any backend from kg/llm.py, prompts in prompts/extraction_*.txt.

`evaluate_extraction` scores an extractor against the template triples, which
gives the extraction precision/recall asked for in review (R2-6).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import REL_POTENTIAL, REL_PRODUCES, SOLAR_ENERGY
from kg.llm import LLMBackend, load_prompt
from kg.ontology import normalize_value

TEMPLATES = [
    "Ô lưới tại hàng {x}, cột {y} trên bản đồ tiềm năng điện mặt trời Việt Nam có chỉ số PVOUT là {v}; "
    "khu vực này có thể sản xuất năng lượng mặt trời.",
    "Trạm quan trắc ở vị trí (hàng {x}, cột {y}) ghi nhận tiềm năng quang điện {v} và được xếp vào nhóm nguồn điện mặt trời.",
    "Grid cell row {x}, column {y} of the Vietnam PV potential map reports a PVOUT value of {v}; "
    "the site is suitable for producing solar energy.",
    "Survey note: at raster position ({x}, {y}) the photovoltaic power potential is {v}. Energy source: solar.",
]

_PATTERNS = [
    re.compile(r"hàng (\d+), cột (\d+).*?(?:chỉ số PVOUT là|tiềm năng quang điện) ([\d.]+)"),
    re.compile(r"row (\d+), column (\d+).*?PVOUT value of ([\d.]+)"),
    re.compile(r"raster position \((\d+), (\d+)\).*?potential is ([\d.]+)"),
]
_SOLAR = re.compile(r"mặt trời|solar", re.IGNORECASE)

TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "integer"},
                    "head": {"type": "string"},
                    "relation": {"type": "string", "enum": [REL_POTENTIAL, REL_PRODUCES]},
                    "tail": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["doc_id", "head", "relation", "tail", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}


def make_documents(samples: pd.DataFrame, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    docs = []
    for i, (x, y, v) in enumerate(samples[["x", "y", "pv_potential"]].itertuples(index=False)):
        tpl = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
        docs.append({"id": i, "text": tpl.format(x=int(x), y=int(y), v=f"{float(v):g}")})
    return docs


def gold_triples(samples: pd.DataFrame) -> set:
    out = set()
    for x, y, v in samples[["x", "y", "pv_potential"]].itertuples(index=False):
        loc = f"location_{int(x)}_{int(y)}"
        out.add((loc, REL_POTENTIAL, normalize_value(v)))
        out.add((loc, REL_PRODUCES, SOLAR_ENERGY))
    return out


class RuleExtractor:
    name = "rule"

    def extract(self, docs: list[dict]) -> pd.DataFrame:
        rows = []
        for d in docs:
            for pat in _PATTERNS:
                m = pat.search(d["text"])
                if m:
                    loc = f"location_{m.group(1)}_{m.group(2)}"
                    rows.append((d["id"], loc, REL_POTENTIAL, normalize_value(m.group(3)), 1.0))
                    if _SOLAR.search(d["text"]):
                        rows.append((d["id"], loc, REL_PRODUCES, SOLAR_ENERGY, 1.0))
                    break
        return pd.DataFrame(rows, columns=["doc_id", "head", "relation", "tail", "weight"])


class LLMExtractor:
    def __init__(self, backend: LLMBackend, batch_size: int = 25):
        self.backend = backend
        self.batch_size = batch_size
        self.name = f"llm:{backend.model}"
        self.system = load_prompt("extraction_system.txt")
        self.user = load_prompt("extraction_user.txt")

    def extract(self, docs: list[dict]) -> pd.DataFrame:
        rows = []
        for start in range(0, len(docs), self.batch_size):
            batch = docs[start:start + self.batch_size]
            text = "\n".join(f"[{d['id']}] {d['text']}" for d in batch)
            out = self.backend.complete_json(self.system, self.user.format(documents=text), TRIPLE_SCHEMA)
            for t in out.get("triples", []):
                rows.append((t["doc_id"], t["head"], t["relation"], normalize_value(t["tail"]),
                             float(min(max(t["confidence"], 0.0), 1.0))))
        return pd.DataFrame(rows, columns=["doc_id", "head", "relation", "tail", "weight"])


def evaluate_extraction(pred: pd.DataFrame, gold: set) -> dict:
    got = set(map(tuple, pred[["head", "relation", "tail"]].values))
    tp = len(got & gold)
    precision = tp / len(got) if got else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"extracted": len(got), "gold": len(gold), "precision": precision, "recall": recall, "f1": f1}
