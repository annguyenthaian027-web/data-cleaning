"""Lớp trung gian: chuyển gợi ý (JSON) của LLM thành `Rule` mà RuleEngine thực thi được.

ĐÂY LÀ RANH GIỚI AN TOÀN giữa LLM và dữ liệu thật. Output của LLM luôn bị coi là KHÔNG TIN CẬY
(có thể sai, bịa cột, hoặc bị prompt-injection qua tên cột/giá trị) nên mỗi gợi ý phải qua:

  1. chuẩn hoá tên action (alias)  → phải nằm trong whitelist `ACTION_SPECS` (khớp core `ACTIONS`)
  2. phân giải tên cột (sửa gần đúng: khác hoa/thường/dấu) → cột phải tồn tại
  3. kiểm tra + ép kiểu từng tham số theo spec; tham số lạ bị loại
  4. kiểm tra kiểu cột hợp lý (vd. không tính median trên cột chữ)
  5. phân loại rủi ro (low/medium/high) → chỉ rủi ro thấp được BẬT sẵn; còn lại người dùng phải tự bật
  6. loại trùng / xung đột (đổi tên trùng nhau, đổi tên đè cột khác)
  7. chạy thử (dry-run) tuần tự trên mẫu dữ liệu theo đúng thứ tự pipeline → loại gợi ý lỗi/không có tác dụng
  8. sắp xếp theo thứ tự pipeline chuẩn (rename luôn cuối để các rule khác dùng tên cột gốc)

Kết quả: `TranslationResult` gồm danh sách `Rule` hợp lệ và danh sách `Proposal` (kể cả bị từ chối, kèm lý do)
để giao diện giải thích minh bạch cho người dùng.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

import numpy as np
import pandas as pd

from dataclean import ACTIONS, Rule, RuleEngine, RuleError
from dataclean.detectors import infer_column_types

# --------------------------------------------------------------------------- spec
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
DATE_OUTPUT_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y")


@dataclass(frozen=True)
class ParamSpec:
    kind: str                       # enum | bool | float | str | str_list
    choices: tuple = ()
    lo: Optional[float] = None
    hi: Optional[float] = None
    max_len: int = 200
    doc: str = ""


@dataclass(frozen=True)
class ActionSpec:
    description: str
    stage: int                                     # thứ tự trong pipeline (nhỏ chạy trước)
    risk: str = "low"                              # rủi ro cơ sở
    params: dict = field(default_factory=dict)
    required: tuple = ()
    multi_column: bool = True                      # cho phép list cột
    all_columns_ok: bool = False                   # cho phép column=null (áp dụng mọi cột)
    expects_type: Optional[str] = None             # kiểu cột suy luận mong đợi (chỉ cảnh báo)
    needs_numeric: bool = False                    # bắt buộc cột số (chỉ khi tham số kéo theo)


_ON_INVALID = ParamSpec("enum", ("keep", "null"), doc="keep=giữ nguyên giá trị sai; null=xoá thành rỗng")

ACTION_SPECS: dict[str, ActionSpec] = {
    "trim_whitespace": ActionSpec("Xoá khoảng trắng thừa, chuẩn hoá Unicode NFC", 0, "low", all_columns_ok=True),
    "normalize_missing": ActionSpec(
        "Đổi 'N/A', 'null', '-'... thành ô rỗng thật", 1, "low", all_columns_ok=True,
        params={"tokens": ParamSpec("str_list", max_len=40, doc="danh sách chuỗi coi là thiếu (tuỳ chọn)")}),
    "change_case": ActionSpec(
        "Đổi chữ hoa/thường", 2, "medium",
        params={"mode": ParamSpec("enum", ("lower", "upper", "title", "capitalize"))}),
    "normalize_date": ActionSpec(
        "Chuẩn hoá ngày về một định dạng", 3, "low", expects_type="date",
        params={"output_format": ParamSpec("enum", DATE_OUTPUT_FORMATS), "on_invalid": _ON_INVALID}),
    "normalize_phone": ActionSpec(
        "Chuẩn hoá số điện thoại VN về dạng 0xxxxxxxxx", 3, "low", expects_type="phone",
        params={"on_invalid": _ON_INVALID}),
    "validate_email": ActionSpec(
        "Chuẩn hoá (chữ thường, bỏ khoảng trắng) và kiểm tra email", 3, "low", expects_type="email",
        params={"on_invalid": _ON_INVALID}),
    "convert_numeric": ActionSpec(
        "Chuyển chuỗi số (1,000 / 15.000.000 / 15000đ) thành số sạch", 3, "low",
        params={"decimal": ParamSpec("enum", (".", ",", "auto"), doc="dấu thập phân; '.'=kiểu Mỹ, ','=kiểu VN"),
                "on_invalid": _ON_INVALID}),
    "handle_outliers": ActionSpec(
        "Xử lý giá trị ngoại lai trong cột số", 4, "high", needs_numeric=True,
        params={"method": ParamSpec("enum", ("iqr", "zscore")),
                "action": ParamSpec("enum", ("clip", "null", "drop"), doc="clip=kẹp về biên; null=xoá giá trị; drop=xoá cả dòng"),
                "k": ParamSpec("float", lo=0.5, hi=5.0), "z": ParamSpec("float", lo=1.0, hi=6.0),
                "decimal": ParamSpec("enum", (".", ",", "auto"))}),
    "fill_missing": ActionSpec(
        "Điền giá trị thiếu", 5, "medium",
        params={"strategy": ParamSpec("enum", ("mean", "median", "mode", "constant")),
                "value": ParamSpec("str", max_len=200, doc="bắt buộc khi strategy=constant")},
        required=("strategy",)),
    "drop_duplicates": ActionSpec(
        "Xoá dòng trùng lặp (giữ dòng đầu)", 6, "medium", all_columns_ok=True,
        params={"ignore_case": ParamSpec("bool")}),
    "drop_missing_rows": ActionSpec("Xoá dòng bị thiếu dữ liệu ở các cột chỉ định", 7, "high"),
    "rename_column": ActionSpec(
        "Đổi tên cột", 9, "medium", multi_column=False,
        params={"new_name": ParamSpec("str", max_len=64)}, required=("new_name",)),
}

# bảo đảm whitelist luôn khớp core: thêm action mới ở core mà quên spec -> lỗi ngay khi import
_missing = set(ACTIONS) - set(ACTION_SPECS)
_extra = set(ACTION_SPECS) - set(ACTIONS)
if _missing or _extra:  # pragma: no cover
    raise RuntimeError(f"ACTION_SPECS lệch với core ACTIONS: thiếu={sorted(_missing)}, thừa={sorted(_extra)}")

ALIASES = {
    "remove_duplicates": "drop_duplicates", "dedupe": "drop_duplicates", "deduplicate": "drop_duplicates",
    "drop_duplicate_rows": "drop_duplicates", "remove_duplicate_rows": "drop_duplicates",
    "fill_na": "fill_missing", "fillna": "fill_missing", "impute": "fill_missing", "fill_nulls": "fill_missing",
    "strip": "trim_whitespace", "trim": "trim_whitespace", "strip_whitespace": "trim_whitespace",
    "standardize_phone": "normalize_phone", "format_phone": "normalize_phone", "clean_phone": "normalize_phone",
    "standardize_date": "normalize_date", "parse_date": "normalize_date", "format_date": "normalize_date",
    "to_numeric": "convert_numeric", "parse_number": "convert_numeric", "parse_numeric": "convert_numeric",
    "clean_email": "validate_email", "normalize_email": "validate_email", "validate_emails": "validate_email",
    "rename": "rename_column", "rename_columns": "rename_column",
    "remove_outliers": "handle_outliers", "clip_outliers": "handle_outliers", "outliers": "handle_outliers",
    "standardize_missing": "normalize_missing", "normalize_nulls": "normalize_missing",
    "drop_missing": "drop_missing_rows", "drop_na": "drop_missing_rows", "dropna": "drop_missing_rows",
}


def catalog_text() -> str:
    """Mô tả action + tham số để nhúng vào prompt (một nguồn sự thật duy nhất với bộ kiểm duyệt)."""
    lines = []
    for name, sp in sorted(ACTION_SPECS.items(), key=lambda kv: kv[1].stage):
        ps = []
        for pn, p in sp.params.items():
            d = {"enum": "một trong " + "|".join(map(str, p.choices)), "bool": "true|false",
                 "float": f"số {p.lo}..{p.hi}", "str": "chuỗi", "str_list": "danh sách chuỗi"}[p.kind]
            ps.append(f"{pn} ({d}{', bắt buộc' if pn in sp.required else ''})")
        col = ("column có thể null=mọi cột" if sp.all_columns_ok
               else "column bắt buộc" + (" (1 tên)" if not sp.multi_column else " (tên hoặc danh sách tên)"))
        lines.append(f"- {name}: {sp.description}. {col}. params: {'; '.join(ps) if ps else 'không có'}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- kết quả
@dataclass
class Proposal:
    index: int
    raw: dict
    status: str                       # accepted | corrected | rejected | no_effect
    risk: str = "low"
    rule: Optional[Rule] = None
    notes: list = field(default_factory=list)
    effect: dict = field(default_factory=dict)      # kết quả dry-run: {"changes": n, "rows_removed": k}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rule"] = self.rule.to_dict() if self.rule else None
        return d


@dataclass
class TranslationResult:
    proposals: list
    rules: list                       # list[Rule] đã sắp xếp theo pipeline, enabled theo rủi ro

    @property
    def accepted(self) -> list:
        return [p for p in self.proposals if p.status in ("accepted", "corrected")]

    @property
    def rejected(self) -> list:
        return [p for p in self.proposals if p.status in ("rejected", "no_effect")]

    def to_dict(self) -> dict:
        return {"proposals": [p.to_dict() for p in self.proposals], "rules": [r.to_dict() for r in self.rules]}


# --------------------------------------------------------------------------- helpers
def _key(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^0-9a-z]+", "", s.lower())


def resolve_column(name: Any, columns: list) -> tuple[Optional[str], Optional[str]]:
    """(tên_cột_thật, ghi_chú). Khớp chính xác → khớp bỏ dấu/hoa-thường → khớp tiền tố (tên bị cắt)."""
    if not isinstance(name, str) or not name.strip():
        return None, None
    if name in columns:
        return name, None
    k = _key(name)
    hits = [c for c in columns if _key(c) == k] if k else []
    if len(hits) == 1:
        return hits[0], f"Đã sửa tên cột '{name}' → '{hits[0]}'"
    if len(name) >= 100:                    # chỉ khớp tiền tố cho tên bị cắt ngắn (privacy.clean_name cắt ở 120 ký tự)
        hits = [c for c in columns if _key(c).startswith(k)]
        if len(hits) == 1:
            return hits[0], f"Đã khớp tên cột '{name}' → '{hits[0]}'"
    return None, None


def _coerce(p: ParamSpec, v: Any):
    if p.kind == "enum":
        s = str(v).strip().lower() if not isinstance(v, bool) else str(v)
        for c in p.choices:
            if s == str(c).lower():
                return c
        raise ValueError(f"giá trị phải thuộc {list(p.choices)}")
    if p.kind == "bool":
        if isinstance(v, bool):
            return v
        if str(v).strip().lower() in {"true", "1", "yes", "có"}:
            return True
        if str(v).strip().lower() in {"false", "0", "no", "không"}:
            return False
        raise ValueError("phải là true/false")
    if p.kind == "float":
        f = float(v)
        if f != f or not (p.lo <= f <= p.hi):
            raise ValueError(f"phải trong khoảng {p.lo}..{p.hi}")
        return f
    if p.kind == "str":
        if isinstance(v, (dict, list)) or v is None:
            raise ValueError("phải là chuỗi")
        s = str(v)
        if len(s) > p.max_len or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", s):
            raise ValueError(f"chuỗi quá dài (>{p.max_len}) hoặc chứa ký tự điều khiển")
        return s
    if p.kind == "str_list":
        if not isinstance(v, list) or not v or len(v) > 50:
            raise ValueError("phải là danh sách 1–50 chuỗi")
        out = [str(x) for x in v]
        if any(len(x) > p.max_len for x in out):
            raise ValueError("phần tử quá dài")
        return out
    raise ValueError("kiểu không hỗ trợ")  # pragma: no cover


def _bump(a: str, b: str) -> str:
    return a if RISK_ORDER[a] >= RISK_ORDER[b] else b


def _safe_new_name(v: str) -> str:
    s = re.sub(r"\s+", " ", v).strip()
    if not s:
        raise ValueError("tên mới rỗng")
    if s[0] in "=+-@":
        raise ValueError("tên mới không được bắt đầu bằng = + - @")
    return s


# --------------------------------------------------------------------------- translator
class RuleTranslator:
    def __init__(self, df: pd.DataFrame, types: Optional[dict] = None, dry_run_rows: int = 5000, max_rules: int = 40):
        self.df = df
        self.columns = list(df.columns)
        self.types = types or infer_column_types(df)
        # mẫu dry-run: toàn bộ nếu nhỏ, ngược lại lấy cách đều (giữ nhãn index gốc) để không bỏ sót phần cuối file
        self.full_check = len(df) <= dry_run_rows
        self.sample = df if self.full_check else df.iloc[np.linspace(0, len(df) - 1, dry_run_rows).astype(int)]
        self.max_rules = max_rules

    # ---- một gợi ý
    def _validate_one(self, i: int, raw: Any) -> Proposal:
        pr = Proposal(index=i, raw=raw if isinstance(raw, dict) else {"_": str(raw)[:200]}, status="rejected")
        if not isinstance(raw, dict):
            pr.notes.append("Gợi ý không phải đối tượng JSON.")
            return pr
        # 1. action
        act_raw = str(raw.get("action", "")).strip()
        act = ALIASES.get(act_raw.lower(), act_raw)
        if act != act_raw:
            pr.notes.append(f"Đã đổi tên action '{act_raw}' → '{act}'")
        spec = ACTION_SPECS.get(act)
        if spec is None:
            pr.notes.append(f"Action '{act_raw}' không được hỗ trợ.")
            return pr
        risk = spec.risk
        corrected = act != act_raw

        # 2. cột
        col_raw = raw.get("column")
        cols: Optional[list] = None
        if col_raw is None or (isinstance(col_raw, str) and col_raw.strip().lower() in {"", "*", "all", "null", "none", "tất cả"}):
            if not spec.all_columns_ok:
                pr.notes.append(f"'{act}' bắt buộc chỉ rõ cột (không được áp dụng cho mọi cột).")
                return pr
            column: Any = None
        else:
            items = col_raw if isinstance(col_raw, list) else [col_raw]
            if not items or len(items) > 30 or (isinstance(col_raw, list) and not spec.multi_column) or \
                    (len(items) > 1 and not spec.multi_column):
                pr.notes.append("Số lượng cột không hợp lệ cho action này.")
                return pr
            resolved = []
            for it in items:
                c, note = resolve_column(it, self.columns)
                if c is None:
                    pr.notes.append(f"Không tìm thấy cột '{str(it)[:60]}' trong dữ liệu.")
                    return pr
                if note:
                    pr.notes.append(note)
                    corrected = True
                resolved.append(c)
            column = resolved[0] if len(resolved) == 1 and not isinstance(col_raw, list) else resolved
            if isinstance(column, list) and len(column) == 1 and not spec.multi_column:
                column = column[0]
        cols = None if column is None else ([column] if isinstance(column, str) else list(column))

        # 3. tham số
        params_raw = raw.get("params") or {}
        if not isinstance(params_raw, dict):
            pr.notes.append("`params` phải là đối tượng.")
            return pr
        params: dict = {}
        for k, v in params_raw.items():
            ps = spec.params.get(k)
            if ps is None:
                pr.notes.append(f"Bỏ tham số không hỗ trợ '{k}'.")
                corrected = True
                continue
            if v is None:
                continue
            try:
                params[k] = _coerce(ps, v)
            except (ValueError, TypeError) as e:
                pr.notes.append(f"Tham số '{k}' không hợp lệ: {e}")
                return pr
        for k in spec.required:
            if k not in params:
                pr.notes.append(f"Thiếu tham số bắt buộc '{k}'.")
                return pr
        if act == "fill_missing":
            if params["strategy"] == "constant":
                if "value" not in params:
                    pr.notes.append("fill_missing/constant cần tham số 'value'.")
                    return pr
                risk = _bump(risk, "high")          # điền hằng số = chèn dữ liệu không có thật
            if params["strategy"] in {"mean", "median"} and cols:
                bad = [c for c in cols if self.types.get(c) != "numeric"]
                if bad:
                    pr.notes.append(f"Không thể tính {params['strategy']} cho cột không phải số: {bad}.")
                    return pr
        if act == "rename_column":
            try:
                params["new_name"] = _safe_new_name(params["new_name"])
            except ValueError as e:
                pr.notes.append(f"Tên mới không hợp lệ: {e}.")
                return pr
        if act == "handle_outliers":
            bad = [c for c in cols if self.types.get(c) != "numeric"]
            if bad:
                pr.notes.append(f"Cột không phải số, không thể xử lý ngoại lai: {bad}.")
                return pr
        if params.get("on_invalid") == "null":
            risk = _bump(risk, "medium")

        # 4. kiểu cột có hợp lý không (chỉ cảnh báo + tăng rủi ro)
        if spec.expects_type and cols:
            odd = [c for c in cols if self.types.get(c) not in (spec.expects_type, "text")]
            if odd:
                pr.notes.append(f"Cột {odd} được nhận diện là {[self.types.get(c) for c in odd]}, "
                                f"không phải '{spec.expects_type}' — hãy xem trước kỹ.")
                risk = _bump(risk, "medium")
        if cols is None and act == "drop_duplicates":
            pass  # loại trùng toàn dòng là hành vi chuẩn
        elif isinstance(column, list) and len(column) > 5 and RISK_ORDER[risk] < 1:
            risk = "medium"

        reason = str(raw.get("reason") or "")[:500]
        name = str(raw.get("name") or f"{spec.description.split(',')[0]}" + (f": {', '.join(cols)}" if cols else ""))[:120]
        pr.rule = Rule(action=act, column=column, params=params, name="[AI] " + name, enabled=(risk == "low"),
                       reason=reason)
        pr.risk = risk
        pr.status = "corrected" if corrected else "accepted"
        return pr

    # ---- cả danh sách
    def translate(self, raw_rules: list) -> TranslationResult:
        proposals = [self._validate_one(i, r) for i, r in enumerate((raw_rules or [])[: self.max_rules * 2])]
        if raw_rules and len(raw_rules) > self.max_rules * 2:
            proposals.append(Proposal(len(proposals), {"_": "..."}, "rejected", notes=[
                f"Bỏ qua {len(raw_rules) - self.max_rules * 2} gợi ý vượt giới hạn."]))
        ok = [p for p in proposals if p.rule is not None]

        # 6. trùng lặp / xung đột
        seen: set = set()
        targets: dict = {}
        for p in ok:
            r = p.rule
            sig = (r.action, repr(r.column), repr(sorted(r.params.items(), key=lambda kv: kv[0])))
            if sig in seen:
                p.status, p.rule = "rejected", None
                p.notes.append("Trùng với một gợi ý trước đó.")
                continue
            seen.add(sig)
            if r.action == "rename_column":
                nn = r.params["new_name"]
                clash = (nn in self.columns and nn != r.column)
                if nn == r.column:
                    p.status, p.rule = "no_effect", None
                    p.notes.append("Tên mới giống tên cũ.")
                elif clash:
                    p.status, p.rule = "rejected", None
                    p.notes.append(f"Tên mới '{nn}' trùng một cột đang có.")
                elif nn.lower() in targets:
                    p.status, p.rule = "rejected", None
                    p.notes.append(f"Tên mới '{nn}' trùng với một gợi ý đổi tên khác.")
                else:
                    targets[nn.lower()] = r.column
            elif r.action in ("fill_missing", "convert_numeric", "handle_outliers"):
                # cùng action + cùng cột nhưng khác tham số = mâu thuẫn, chỉ giữ cái đầu
                k2 = (r.action, repr(r.column))
                if k2 in seen:
                    p.status, p.rule = "rejected", None
                    p.notes.append("Mâu thuẫn với một gợi ý cùng loại trên cùng cột (đã giữ gợi ý đầu).")
                seen.add(k2)
        ok = [p for p in proposals if p.rule is not None]

        # 8. thứ tự pipeline (ổn định theo thứ tự AI đưa ra)
        ok.sort(key=lambda p: (ACTION_SPECS[p.rule.action].stage, p.index))

        # 7. dry-run tuần tự trên mẫu, đúng thứ tự pipeline
        cur = self.sample.copy()
        for p in ok:
            try:
                # RuleEngine bỏ qua rule enabled=False -> chạy thử bằng bản sao đã bật, nếu không mọi rule rủi ro cao/trung bình đều 'không có tác dụng'
                res = RuleEngine([replace(p.rule, enabled=True)]).apply(cur)
            except (RuleError, KeyError, ValueError, TypeError) as e:
                p.status, p.rule = "rejected", None
                p.notes.append(f"Chạy thử thất bại: {e}")
                continue
            p.effect = {"changes": res.stats["changes"], "rows_removed": res.stats["rows_removed"],
                        "sample_rows": len(self.sample)}
            if res.stats["changes"] == 0 and res.stats["rows_removed"] == 0:
                if self.full_check:
                    p.status, p.rule = "no_effect", None
                    p.notes.append("Chạy thử trên toàn bộ dữ liệu: không thay đổi gì.")
                    continue
                p.notes.append(f"Mẫu {len(self.sample):,}/{len(self.df):,} dòng không bị thay đổi — chưa kiểm chứng "
                               "trên toàn bộ dữ liệu, hãy xem trước.")
            cur = res.df
        rules = [p.rule for p in ok if p.rule is not None][: self.max_rules]
        return TranslationResult(proposals=proposals, rules=rules)
