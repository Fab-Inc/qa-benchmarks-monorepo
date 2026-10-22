"""Section-granular resume for judge runs.

Judgment rows carry ``section_shas`` — a content hash per rubric section — so
a rerun after editing only some rubric sections reuses the cached results of
unchanged sections instead of re-judging the whole rubric.

Legacy rows (no ``section_shas``) get a conservative fallback: a section is
reusable iff the stored criterion ids, question texts and types match the
current section exactly, and the section has no ``evidence: true`` criteria
(old rows cannot contain evidence answers).
"""

from __future__ import annotations

import hashlib
from typing import Any

from decodable_text_generation.judge.rubric import Section


def section_sha(section: Section) -> str:
    """Content hash of one rubric section's definition.

    ``exclude_none`` keeps unset optional fields (e.g. newly added ones like
    ``evidence_fields``) out of the hash so they don't invalidate reuse.
    """
    return hashlib.sha256(
        section.model_dump_json(exclude_none=True).encode()
    ).hexdigest()[:16]


def rubric_section_shas(rubric: dict[str, Section]) -> dict[str, str]:
    return {name: section_sha(s) for name, s in rubric.items()}


def partition_sections(
    row: dict[str, Any], rubric: dict[str, Section]
) -> tuple[set[str], set[str]]:
    """Split rubric sections into (reusable, stale) for a cached judgment."""
    stored_shas = row.get("section_shas")
    reusable, stale = set(), set()
    for name, section in rubric.items():
        if stored_shas is not None:
            (reusable if stored_shas.get(name) == section_sha(section) else stale).add(
                name
            )
            continue
        old_crits = {
            c["criterion_id"]: (c["question"], c["criterion_type"])
            for c in row.get("sections", {}).get(name, [])
        }
        new_crits = {
            c.id: (c.question, c.criterion_type.value) for c in section.criteria
        }
        if old_crits == new_crits and not any(c.evidence for c in section.criteria):
            reusable.add(name)
        else:
            stale.add(name)
    return reusable, stale


def merge_judgment(
    cached_row: dict[str, Any],
    fresh: dict[str, Any],
    rubric: dict[str, Section],
    reusable: set[str],
) -> dict[str, Any]:
    """Combine cached section results with freshly judged ones.

    ``fresh`` is the evaluate_text result over the stale-section subset.
    """
    merged = dict(fresh)
    merged["sections"] = {
        **{name: cached_row["sections"][name] for name in reusable},
        **fresh["sections"],
    }
    merged["section_weights"] = {name: s.weight for name, s in rubric.items()}
    merged["reused_sections"] = sorted(reusable)
    return merged
