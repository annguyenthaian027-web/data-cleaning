"""Audit log: ghi lại mọi thay đổi ở mức từng ô / từng dòng."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

COLUMNS = ["timestamp", "version", "rule_id", "rule_name", "action", "row", "column", "old_value", "new_value"]


def _s(v):
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


class AuditLog:
    def __init__(self):
        self.entries: list[dict] = []

    def add_changes(self, version: int, rule, changes: list[tuple]) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for row, col, old, new in changes:
            self.entries.append({
                "timestamp": ts, "version": version, "rule_id": rule.id,
                "rule_name": rule.name or rule.action, "action": rule.action,
                "row": int(row), "column": col, "old_value": _s(old), "new_value": _s(new)})

    def add_event(self, version: int, action: str, note: str = "") -> None:
        self.entries.append({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "version": version, "rule_id": None, "rule_name": note, "action": action,
            "row": None, "column": None, "old_value": None, "new_value": None})

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries, columns=COLUMNS)

    def to_json(self) -> str:
        return json.dumps(self.entries, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.entries)
