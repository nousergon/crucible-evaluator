"""
citations.py — typed, code-checked citations for the Director's two LLM calls
(alpha-engine-config-I12312; the citation-identity half of the determinism
epic alpha-engine-config-I12101).

The problem this replaces. ``ActionItem.evidence`` was documented as
"MetricRecord names / tile names / artifact paths" and was, in practice, prose
("predictor tile momentum_l1_ic"). Identity was recovered after the fact by
regex-tokenising that prose (``loop_verification.resolve_cited_metrics``): 26 of
28 ledger rows cited prose on 2026-08-22 (I8178), and nothing could say whether
a cited name was in the input the model actually saw.

The contract here is deterministic end to end:

1. **Offer.** Every piece of input context the model may cite is registered in
   a :class:`SourceRegistry` WHILE the prompt is rendered, under a stable id,
   and the rendered line carries that id verbatim as a ``{cite=<id>}`` tag. The
   offered set is exactly what the prompt contained — nothing is citable that
   was not rendered, and nothing rendered with a tag is missing from the set.
2. **Cite.** The structured output carries ``citations: [{id, kind, quote?}]``
   (``director.schema.Citation``); free-text ``evidence``/``rationale`` stay
   separate and carry no identity.
3. **Check.** After the call, :func:`validate_citations` checks every citation
   against the registry — exact id match, kind match, and (when given) the
   quote appears in the cited line. Each citation is stamped ``check`` /
   ``check_reason``; an unknown id is RECORDED as invalid, never dropped and
   never repaired to a near match. An item with zero valid citations is
   stamped ``ungrounded``.

Id grammar (``<kind>:<key>``):

  ``metric:<component name>``     a Report Card component, exact name as on the card
  ``tile:<tile key>``             a Report Card tile header
  ``card:<block>``                a card-level block (attestation, pipeline_gates,
                                  run_scope, census, staleness)
  ``carryover:<ledger item id>``  a carry-over ledger row
  ``issue:<repo>-I<N>``           a backlog / recently-resolved issue line
  ``artifact:<s3 key or uri>``    an artifact the input names as a source
  ``plan_item:<action item id>``  a prior plan's action item (retro judge only)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

#: Bumped whenever the id grammar, the tag rendering or the check semantics
#: change, so an archived plan states which contract its ``citation_check`` was
#: computed under.
CITATION_CONTRACT_VERSION = "director-citations/v1"

#: Plan / retro artifact keys (the models are ``extra="allow"``).
CITATION_CHECK_KEY = "citation_check"
CITATION_SOURCES_KEY = "citation_sources"

KINDS = ("metric", "tile", "card", "carryover", "issue", "artifact", "plan_item")

VALID = "valid"
UNKNOWN_ID = "unknown_id"
KIND_MISMATCH = "kind_mismatch"
QUOTE_NOT_FOUND = "quote_not_found"
CHECK_STATUSES = (VALID, UNKNOWN_ID, KIND_MISMATCH, QUOTE_NOT_FOUND)

GROUNDED = "grounded"
UNGROUNDED = "ungrounded"

_WS = re.compile(r"\s+")


def source_id(kind: str, key) -> str:
    """``<kind>:<key>`` — the one way an id is spelled."""
    if kind not in KINDS:
        raise ValueError(f"unknown citation kind {kind!r}")
    key = str(key).strip()
    if not key:
        raise ValueError(f"empty key for citation kind {kind!r}")
    return f"{kind}:{key}"


def cite_tag(sid: str) -> str:
    """The verbatim marker rendered into the prompt next to a citable line."""
    return "{cite=" + sid + "}"


def _norm_ws(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


@dataclass
class SourceEntry:
    id: str
    kind: str
    #: Every rendered line this id tagged (a component name shared by two
    #: tiles is one id with two lines — the same collapse
    #: ``loop_verification.component_status_map`` makes).
    texts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.texts)


class SourceRegistry:
    """The exact set of ids a prompt offered the model, with the line(s) each
    tagged. Built while rendering; read after the call to validate."""

    def __init__(self) -> None:
        self._entries: dict[str, SourceEntry] = {}

    def offer(self, kind: str, key, text: str) -> str:
        """Register ``text`` as citable under ``kind:key`` and return the
        ``{cite=...}`` tag to render beside it."""
        sid = source_id(kind, key)
        entry = self._entries.get(sid)
        if entry is None:
            entry = self._entries[sid] = SourceEntry(id=sid, kind=kind)
        elif entry.kind != kind:  # pragma: no cover — the id carries its kind
            raise ValueError(f"id {sid!r} offered as both {entry.kind} and {kind}")
        entry.texts.append(str(text))
        return cite_tag(sid)

    def tagged(self, kind: str, key, text: str) -> str:
        """``text`` with its tag appended — the common render form."""
        return f"{text} {self.offer(kind, key, text)}"

    def get(self, sid: str) -> SourceEntry | None:
        return self._entries.get(sid)

    def __contains__(self, sid) -> bool:
        return sid in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def ids(self) -> list[str]:
        return sorted(self._entries)

    def ids_of_kind(self, kind: str) -> list[str]:
        return sorted(i for i, e in self._entries.items() if e.kind == kind)

    def manifest(self) -> dict:
        """The offered set, persisted beside the output so an archived plan can
        be re-checked without re-rendering its prompt. ``sha256`` is over the
        canonical JSON of ``sources`` — two runs that offered the same context
        carry the same digest."""
        sources = {
            sid: {"kind": e.kind, "text": e.text}
            for sid, e in sorted(self._entries.items())
        }
        canonical = json.dumps(sources, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {
            "contract": CITATION_CONTRACT_VERSION,
            "n_sources": len(sources),
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "sources": sources,
        }

    @classmethod
    def from_manifest(cls, manifest: dict | None) -> "SourceRegistry":
        reg = cls()
        for sid, row in ((manifest or {}).get("sources") or {}).items():
            kind = (row or {}).get("kind")
            reg._entries[sid] = SourceEntry(
                id=sid, kind=str(kind), texts=[str((row or {}).get("text") or "")],
            )
        return reg


def check_citation(citation, registry: SourceRegistry) -> tuple[str, str | None]:
    """``(status, reason)`` for one citation against the offered set. Exact
    match only: whitespace inside a quote is collapsed for comparison, but the
    id is never normalised, case-folded or fuzzy-matched — a near miss is an
    ``unknown_id`` that names what was cited."""
    cid = getattr(citation, "id", None) if not isinstance(citation, dict) else citation.get("id")
    kind = getattr(citation, "kind", None) if not isinstance(citation, dict) else citation.get("kind")
    quote = getattr(citation, "quote", None) if not isinstance(citation, dict) else citation.get("quote")
    entry = registry.get(cid) if isinstance(cid, str) else None
    if entry is None:
        return UNKNOWN_ID, f"{cid!r} was not offered in the input"
    if kind != entry.kind:
        return KIND_MISMATCH, f"cited as kind {kind!r}; the input offered {cid!r} as {entry.kind!r}"
    if quote is not None and _norm_ws(quote):
        if _norm_ws(quote) not in _norm_ws(entry.text):
            return QUOTE_NOT_FOUND, f"quote {quote!r} does not appear in the line tagged {cid!r}"
    return VALID, None


def _stamp(citations, registry: SourceRegistry) -> list[tuple[object, str, str | None]]:
    out = []
    for c in citations or []:
        status, reason = check_citation(c, registry)
        if isinstance(c, dict):
            c["check"], c["check_reason"] = status, reason
        else:
            c.check, c.check_reason = status, reason
        out.append((c, status, reason))
    return out


def _cid(c) -> str | None:
    return c.get("id") if isinstance(c, dict) else getattr(c, "id", None)


def _ckind(c) -> str | None:
    return c.get("kind") if isinstance(c, dict) else getattr(c, "kind", None)


def validate_citations(items, registry: SourceRegistry) -> dict:
    """Validate every action item's ``citations`` in place and return the
    plan-level ``citation_check`` block.

    Mutates: each citation's ``check``/``check_reason`` and each item's
    ``citation_grounding``. Works on pydantic ``ActionItem``s and on plain
    dicts (a ledger row / an archived plan re-checked from its manifest)."""
    counts = {s: 0 for s in CHECK_STATUSES}
    invalid: list[dict] = []
    ungrounded: list[str] = []
    per_item: dict[str, dict] = {}
    n_items = 0
    for item in items or []:
        n_items += 1
        is_dict = isinstance(item, dict)
        item_id = str(item.get("id") if is_dict else getattr(item, "id", ""))
        cits = item.get("citations") if is_dict else getattr(item, "citations", None)
        stamped = _stamp(cits, registry)
        n_valid = 0
        for c, status, reason in stamped:
            counts[status] += 1
            if status == VALID:
                n_valid += 1
            else:
                invalid.append({"item_id": item_id, "id": _cid(c), "kind": _ckind(c),
                                "status": status, "reason": reason})
        grounding = GROUNDED if n_valid else UNGROUNDED
        if is_dict:
            item["citation_grounding"] = grounding
        else:
            item.citation_grounding = grounding
        if not n_valid:
            ungrounded.append(item_id)
        per_item[item_id] = {"n_citations": len(stamped), "n_valid": n_valid, "grounding": grounding}
    return {
        "contract": CITATION_CONTRACT_VERSION,
        "n_offered": len(registry),
        "offered_sha256": registry.manifest()["sha256"],
        "n_items": n_items,
        "grounded_items": n_items - len(ungrounded),
        "ungrounded_item_ids": ungrounded,
        "n_citations": sum(counts.values()),
        "counts": counts,
        "invalid": invalid,
        "items": per_item,
    }


def validate_grade_citations(grade, registry: SourceRegistry) -> dict:
    """Validate a retro ``RetroGrade.citations`` in place → its check block.
    Same semantics as :func:`validate_citations`, over one flat list."""
    cits = grade.get("citations") if isinstance(grade, dict) else getattr(grade, "citations", None)
    stamped = _stamp(cits, registry)
    counts = {s: 0 for s in CHECK_STATUSES}
    invalid = []
    by_dimension: dict[str, int] = {}
    for c, status, reason in stamped:
        counts[status] += 1
        if status == VALID:
            dim = (c.get("dimension") if isinstance(c, dict) else getattr(c, "dimension", None)) or "unspecified"
            by_dimension[dim] = by_dimension.get(dim, 0) + 1
        else:
            invalid.append({"id": _cid(c), "kind": _ckind(c), "status": status, "reason": reason})
    return {
        "contract": CITATION_CONTRACT_VERSION,
        "n_offered": len(registry),
        "offered_sha256": registry.manifest()["sha256"],
        "n_citations": len(stamped),
        "counts": counts,
        "grounding": GROUNDED if counts[VALID] else UNGROUNDED,
        "valid_by_dimension": by_dimension,
        "invalid": invalid,
    }


def valid_citation_ids(item, kind: str | None = None) -> list[str] | None:
    """The ids of ``item``'s citations that passed the check, optionally of one
    kind. ``None`` when the item carries no typed citations at all — the
    caller's signal to take the legacy prose path for a plan written before
    this contract. A citation that was never checked (``check`` absent) does
    not count: unverified identity is not identity."""
    cits = item.get("citations") if isinstance(item, dict) else getattr(item, "citations", None)
    if not cits:
        return None
    out = []
    for c in cits:
        check = c.get("check") if isinstance(c, dict) else getattr(c, "check", None)
        if check != VALID:
            continue
        if kind is not None and _ckind(c) != kind:
            continue
        cid = _cid(c)
        if isinstance(cid, str):
            out.append(cid)
    return out


def metric_name(sid: str) -> str | None:
    """``metric:<name>`` → ``<name>``; ``None`` for any other kind."""
    if isinstance(sid, str) and sid.startswith("metric:"):
        return sid[len("metric:"):]
    return None


#: Rendered into the human message of both calls, after the input, so the
#: instruction travels with the code that builds the tags regardless of which
#: system prompt (tuned or example) is loaded.
PLAN_CITATION_INSTRUCTIONS = (
    "CITATIONS (required, code-checked): every line above that you may rely on ends with a "
    "{cite=...} tag. For each action item, list in `citations` every source it rests on as "
    "{id, kind, quote?}: `id` copied VERBATIM from a tag (e.g. metric:momentum_l1_ic, "
    "tile:predictor, carryover:<slug>, issue:alpha-engine-config-I123), `kind` = the id's "
    "prefix, and optionally `quote` = a short fragment copied verbatim from that tagged line "
    "(e.g. the value or status). Code checks every id against the tags in this input: an id "
    "that does not appear here is recorded as INVALID, and an item with no valid citation is "
    "recorded as UNGROUNDED. Do not invent ids, do not cite a metric that is not tagged here, "
    "and keep prose in `rationale`/`evidence` — those fields carry no citation identity."
)

RETRO_CITATION_INSTRUCTIONS = (
    "CITATIONS (required, code-checked): every citable line above ends with a {cite=...} tag "
    "— the prior plan's items carry plan_item ids and the current card's lines carry metric / "
    "tile / card / artifact ids. In `citations`, list every "
    "prior-plan item and current-card source your grade relied on as {id, kind, quote?, "
    "dimension}: `id` copied VERBATIM from a tag, `kind` = the id's prefix, `dimension` = the "
    "score it supports (grounding / calibration / actionability). Code checks every id against "
    "the tags in this input; an id that does not appear here is recorded as INVALID."
)


# ── llm_trace.v1 sources (alpha-engine-config-I12312, audit §6.3 item 4a) ──
#
# The check above runs AFTER the call, on the plan. The trace record krepis
# writes for the call itself (``krepis.llm_trace``, ``llm_trace.v1``) carries
# ``citations[]`` — every source injected into the prompt, with a stable source
# id, the sha256 of the exact injected text and its offsets — but only for
# sources the caller declares. Until now the Director declared none, so a trace
# of a Director call could not say which metric or issue lines it was offered.
#
# :func:`trace_sources` declares them, from the ``{cite=...}`` tags the prompt
# already carries. Three things are deliberate:
#
# - **The prompt is not changed.** The result is a ``RenderedSources`` whose
#   text IS the user content, so krepis locates it (exactly once, offset 0)
#   instead of prepending a second rendering.
# - **The typed id is the ``source_id``**, the join key evals use. krepis'
#   ``cid`` grammar (``[A-Za-z0-9_.-]{1,64}``) cannot spell ``metric:<name>``,
#   so each id gets a positional cid (``s1``, ``s2``; a repeated tag ``s1.1``).
# - **``cited_in_output`` reads false on these rows by construction.** krepis
#   resolves it from ``[cid]`` markers; the Director's model cites in the
#   structured ``citations`` field instead. The authoritative linkage is the
#   artifact's ``citation_check``, which carries the same ``trace_id``
#   (:func:`stamp_trace_ids`).

#: Director kind → ``krepis.llm_trace.SOURCE_KINDS``. The Director's own kind is
#: kept in ``meta.director_kind``; this mapping only places it in krepis' closed
#: vocabulary.
TRACE_SOURCE_KINDS = {
    "metric": "metric",
    "tile": "metric",
    "card": "artifact",
    "carryover": "artifact",
    "issue": "doc",
    "artifact": "artifact",
    "plan_item": "artifact",
}

#: Key on the plan / retro artifact naming the trace of the call that wrote it.
TRACE_IDS_KEY = "llm_trace"

_TAG_RE = re.compile(r"\{cite=((" + "|".join(KINDS) + r"):[^{}\n]+)\}")


def trace_sources(user_content: str, registry: SourceRegistry | None = None):
    """The ``krepis.llm_trace.RenderedSources`` for one Director call, or
    ``None`` when the prompt offered nothing.

    One source per ``{cite=...}`` tag in ``user_content``, in prompt order. Its
    text is the tagged span — from the start of its line (or the end of an
    earlier tag on that line) through the tag — exactly as sent, so the
    recorded sha256 and offsets are of characters the model read. With
    ``registry``, a tag whose id the registry did not offer is skipped (a tag
    quoted inside a carried-over row's own text is not an offer).
    """
    from krepis.llm_trace import RenderedSource, RenderedSources, SourceBlock, sha256_hex

    if not user_content:
        return None
    numbers: dict[str, int] = {}
    seen: dict[str, int] = {}
    rendered = []
    last_end = 0
    for m in _TAG_RE.finditer(user_content):
        sid, kind = m.group(1), m.group(2)
        line_start = user_content.rfind("\n", 0, m.start()) + 1
        start = max(line_start, last_end)
        end = last_end = m.end()
        if registry is not None and sid not in registry:
            continue
        text = user_content[start:end]
        n = numbers.setdefault(sid, len(numbers) + 1)
        k = seen.get(sid, 0)
        seen[sid] = k + 1
        cid = f"s{n}" if k == 0 else f"s{n}.{k}"
        block = SourceBlock(
            cid=cid,
            kind=TRACE_SOURCE_KINDS[kind],
            source_id=sid,
            text=text,
            meta={"director_kind": kind, "contract": CITATION_CONTRACT_VERSION, "occurrence": k},
        )
        rendered.append(RenderedSource(
            block=block, content_sha256=sha256_hex(text), char_start=start, char_end=end,
        ))
    if not rendered:
        return None
    return RenderedSources(text=user_content, sources=tuple(rendered))


def trace_sources_or_none(user_content: str, registry: SourceRegistry | None = None):
    """:func:`trace_sources`, never raising. Trace sources are telemetry: a
    fault building them is logged and the call goes ahead without them."""
    try:
        return trace_sources(user_content, registry)
    except Exception as exc:  # noqa: BLE001 — telemetry never blocks the call
        import logging

        logging.getLogger(__name__).warning(
            "director: trace sources unavailable, call proceeds without them: %s", exc,
        )
        return None


def stamp_trace_ids(obj, result) -> None:
    """Put the call's ``trace_id`` / ``span_id`` on the plan or grade it wrote,
    so the archived ``citation_check`` joins the ``llm_trace.v1`` row. A result
    without them (a test double, a krepis predating the trace) stamps
    nothing."""
    trace_id = getattr(result, "trace_id", None)
    if not isinstance(trace_id, str) or not trace_id:
        return
    try:
        setattr(obj, TRACE_IDS_KEY, {
            "trace_id": trace_id,
            "span_id": getattr(result, "span_id", None),
        })
    except Exception:  # noqa: BLE001 — a model without extra fields
        pass
