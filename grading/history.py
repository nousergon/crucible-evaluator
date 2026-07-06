"""
history.py — cross-cycle trend history from PRIOR weekly report cards (config#1836).

``build_metric`` has always accepted ``trend_4w`` / ``trend_13w`` (and
``derive_trend_decoration`` renders the ↑↑/↑/→/↓/↓↓ glyph from them), but no
tile ever passed them — every weekly card was a single-cycle snapshot. This
module closes that: it loads the prior N weekly
``evaluator/{date}/report_card.json`` artifacts and exposes per-component value
series keyed by the stable ``(tile, component_name)`` pair, so tile builders can
thread real cross-cycle trends into the components they grade.

Binding constraints (config#1836):
  * **Prior CARDS are the SSOT for graded values** — we never re-derive a prior
    week's value from raw upstream artifacts (a re-derivation path is the
    rebuild-writer bug class; the graded card is the producer-owned fact).
  * **N/A weeks are skipped, never zero-filled** — a component that graded
    ``N/A-*`` (or carried no value) in a prior week simply contributes no point
    to the series, so the trend reflects only weeks with an honest reading.
  * **Missing/short history is fail-soft ONLY with a WARN** naming how many
    prior cards were found (no-silent-swallow: this is secondary observability
    hung off the primary card build, and the shortfall is recorded loudly).
    Any real S3 error (auth / throttle / wrong bucket) still RAISES — same
    posture as ``grading/artifacts.py``.
  * **N is bounded at 13** (one quarter of weekly cards): the grader Lambda has
    a finite timeout, and 13 small sequential GETs is an acceptable budget.
"""

from __future__ import annotations

import json
import logging
import re

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# The evaluator report-card namespace. The literal prefix is deliberate: the
# IAM contract test (tests/test_iam_s3_contract.py mechanism 2) resolves it to
# the granted ``evaluator`` prefix. Must stay consistent with
# ``grading.aggregate.report_card_key`` — guarded by _CARD_KEY_RE's unit test.
_CARD_PREFIX = "evaluator/"
_CARD_KEY_RE = re.compile(r"^evaluator/(\d{4}-\d{2}-\d{2})/report_card\.json$")

# Hard bound on how many prior cards a history load may fetch (config#1836:
# trend_13w is the longest window any consumer renders).
MAX_HISTORY_CARDS = 13

_NA_PREFIX = "N/A"


class CardHistory:
    """Per-component value series extracted from prior weekly report cards.

    ``series`` maps the stable ``(tile, component_name)`` key to that
    component's prior values in CHRONOLOGICAL order (oldest → newest) — exactly
    the ordering ``krepis.metrics.derive_trend_decoration`` expects. Weeks
    where the component was N/A (or absent — unwired producers appear and
    disappear across weeks) contribute no point.
    """

    def __init__(self, series: dict[tuple[str, str], list[float]], n_cards_found: int):
        self._series = series
        self.n_cards_found = n_cards_found

    @classmethod
    def empty(cls) -> "CardHistory":
        return cls({}, 0)

    def prior_values(self, tile: str, name: str) -> list[float]:
        """Prior value series (oldest → newest) for one component; [] if none."""
        return list(self._series.get((tile, name), ()))

    def trends_for(self, tile: str, name: str, current_value: float | None) -> dict:
        """``build_metric`` kwargs (``trend_4w`` / ``trend_13w``) for one component.

        The series is the prior cards' values with THIS cycle's value appended
        (when value-bearing), so the trend glyph's most recent step is this
        week's change. Returns ``{}`` when there is no prior history — the
        record then keeps its default (None trends, "→"), indistinguishable
        from the pre-config#1836 behavior.
        """
        prior = self._series.get((tile, name))
        if not prior:
            return {}
        full = prior + ([float(current_value)] if current_value is not None else [])
        return {"trend_4w": full[-4:], "trend_13w": full[-13:]}


def _extract_series(cards: list[dict]) -> dict[tuple[str, str], list[float]]:
    """Pull value-bearing component readings out of parsed cards (oldest first)."""
    series: dict[tuple[str, str], list[float]] = {}
    for card in cards:
        tiles = card.get("tiles")
        if not isinstance(tiles, dict):
            continue  # pre-RC-v2 card without MetricRecord tiles — nothing to extract
        for tile_name, tile in tiles.items():
            comps = (tile or {}).get("components")
            if not isinstance(comps, list):
                continue
            for comp in comps:
                if not isinstance(comp, dict):
                    continue
                name = comp.get("name")
                value = comp.get("value")
                status = str(comp.get("status", ""))
                # Skip N/A weeks — no zero-filling (config#1836).
                if not name or value is None or status.startswith(_NA_PREFIX):
                    continue
                try:
                    series.setdefault((tile_name, name), []).append(float(value))
                except (TypeError, ValueError):
                    logger.warning(
                        "Non-numeric prior value for (%s, %s): %r — skipped",
                        tile_name, name, value,
                    )
    return series


def load_card_history(
    bucket: str,
    run_date: str,
    s3_client=None,
    *,
    n_cards: int = MAX_HISTORY_CARDS,
) -> CardHistory:
    """Load the prior ``n_cards`` weekly report cards strictly BEFORE ``run_date``.

    Lists ``evaluator/`` for dated ``report_card.json`` keys, takes the latest
    ``n_cards`` (bounded at ``MAX_HISTORY_CARDS``), and reads them oldest-first.
    Short/absent history WARNs with the found-card count and degrades to an
    empty/partial ``CardHistory`` (trends simply stay unpopulated); a corrupt
    card (half-written from a crashed run) is skipped with a WARN, mirroring
    ``get_json_windowed``. Real S3 errors raise — fail-loud.
    """
    s3 = s3_client or boto3.client("s3")
    n_cards = max(1, min(n_cards, MAX_HISTORY_CARDS))

    dated_keys: list[tuple[str, str]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=_CARD_PREFIX):
        for obj in page.get("Contents", []):
            m = _CARD_KEY_RE.match(obj["Key"])
            if m and m.group(1) < run_date:
                dated_keys.append((m.group(1), obj["Key"]))
    dated_keys = sorted(dated_keys)[-n_cards:]  # latest N priors, oldest → newest

    cards: list[dict] = []
    for date_s, key in dated_keys:
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            cards.append(json.loads(resp["Body"].read()))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                # Deleted between list and get — a legitimate skip, recorded.
                logger.warning("Prior card vanished between list and read: s3://%s/%s", bucket, key)
                continue
            logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
            raise
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Skipping corrupt prior card s3://%s/%s: %s", bucket, key, e)
            continue

    n_found = len(cards)
    if n_found < n_cards:
        # Fail-soft ONLY with the found-card count on record (config#1836).
        logger.warning(
            "Cross-cycle trend history short: found %d prior report card(s) before %s "
            "under s3://%s/%s (wanted %d) — trend_4w/trend_13w will be partial or absent.",
            n_found, run_date, bucket, _CARD_PREFIX, n_cards,
        )
    else:
        logger.info(
            "Loaded %d prior report cards before %s for cross-cycle trends.",
            n_found, run_date,
        )
    return CardHistory(_extract_series(cards), n_found)
