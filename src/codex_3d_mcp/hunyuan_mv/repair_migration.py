"""Data-only migration: old approvals never certify the new repair policy."""

from __future__ import annotations

from typing import Any


BATCH_SCHEMA_VERSION = 2


def migrate_repair_stage(batch: dict[str, Any]) -> bool:
    """Add missing records while preserving artifacts, reviews and paused jobs."""
    changed = False
    missing_certificate = False
    for asset in batch.get("assets", []):
        asset.setdefault("repair_idempotency", {})
        if "shape_repair" not in asset["stages"]:
            asset["stages"]["shape_repair"] = {
                "state": "pending", "attempt": 0, "approved": False,
                "gate": None, "error": None, "artifacts": [], "previews": {},
                "history": [], "migration_note": "New master certificate required",
            }
            changed = True
        if not asset.get("accepted_master"):
            missing_certificate = True
    if batch.get("schema_version") != BATCH_SCHEMA_VERSION:
        batch["schema_version"] = BATCH_SCHEMA_VERSION
        batch["repair_policy_migration"] = {
            "old_approvals_retained": True,
            "old_outputs_certified": False,
            "new_master_validation_required": missing_certificate,
        }
        changed = True
        if batch.get("state") not in {"completed", "cancelled"}:
            batch["paused"] = True
            batch["state"] = "paused"
            if missing_certificate and batch.get("stage") in {"remesh", "paint", "finish"}:
                batch["stage_before_repair_migration"] = batch["stage"]
                batch["stage"] = "shape_repair"
    return changed
