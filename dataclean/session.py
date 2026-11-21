"""Phiên làm sạch: preview trước/sau, áp dụng rule, UNDO/REDO, lưu phiên bản + audit log."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .audit import AuditLog
from .quality import compute_quality
from .rules import Rule, RuleEngine


@dataclass
class Version:
    id: int
    label: str
    df: pd.DataFrame
    rules: list


class CleaningSession:
    def __init__(self, df: pd.DataFrame, filename: str = "data"):
        self.filename = filename
        self.versions: list[Version] = [Version(0, "Dữ liệu gốc", df.copy(), [])]
        self.cursor = 0
        self.audit = AuditLog()

    # ---- trạng thái
    @property
    def current(self) -> pd.DataFrame:
        return self.versions[self.cursor].df

    @property
    def original(self) -> pd.DataFrame:
        return self.versions[0].df

    @property
    def can_undo(self) -> bool:
        return self.cursor > 0

    @property
    def can_redo(self) -> bool:
        return self.cursor < len(self.versions) - 1

    def history(self) -> list[dict]:
        return [{"version": v.id, "label": v.label, "rows": len(v.df),
                 "rules": [r["name"] or r["action"] for r in v.rules], "is_current": i == self.cursor}
                for i, v in enumerate(self.versions)]

    # ---- preview (không thay đổi phiên)
    def preview(self, rules: Iterable, sample: int = 20) -> dict:
        res = RuleEngine(list(rules)).apply(self.current)
        changes = [{"rule": r.name or r.action, "row": row, "column": col, "old": old, "new": new}
                   for r, ch in res.changes for row, col, old, new in ch]
        return {
            "stats": res.stats,
            "changes_sample": changes[:sample],
            "quality_before": compute_quality(self.current)["overall"],
            "quality_after": compute_quality(res.df)["overall"],
            "after_head": res.df.head(sample).to_dict("records"),
        }

    # ---- áp dụng (commit phiên bản mới)
    def apply(self, rules: Iterable, label: str = "") -> Version:
        rules = [r if isinstance(r, Rule) else Rule.from_dict(r) for r in rules]
        engine = RuleEngine(rules)
        res = engine.apply(self.current)
        del self.versions[self.cursor + 1:]          # cắt nhánh redo
        ver = Version(len(self.versions), label or f"Áp dụng {len(rules)} quy tắc",
                      res.df, [r.to_dict() for r in rules])
        self.versions.append(ver)
        self.cursor = len(self.versions) - 1
        for r, ch in res.changes:
            self.audit.add_changes(ver.id, r, ch)
        return ver

    def undo(self) -> pd.DataFrame:
        if not self.can_undo:
            raise RuntimeError("Không còn bước để hoàn tác.")
        self.cursor -= 1
        self.audit.add_event(self.versions[self.cursor].id, "UNDO", "Hoàn tác về phiên bản này")
        return self.current

    def redo(self) -> pd.DataFrame:
        if not self.can_redo:
            raise RuntimeError("Không có bước để làm lại.")
        self.cursor += 1
        self.audit.add_event(self.versions[self.cursor].id, "REDO", "Làm lại phiên bản này")
        return self.current

    def restore(self, version_id: int) -> pd.DataFrame:
        if not 0 <= version_id < len(self.versions):
            raise KeyError(version_id)
        self.cursor = version_id
        self.audit.add_event(version_id, "RESTORE", f"Khôi phục phiên bản {version_id}")
        return self.current
