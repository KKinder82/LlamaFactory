"""LlamaFactory 训练数据格式校验模块.

支持校验 alpaca 格式的 JSON 训练数据文件，检查以下内容：
- JSON 解析合法性（含 BOM 检测）
- 顶层结构必须为列表
- 每条记录必须包含必填字段（instruction、output）
- 字段值类型和非空约束
- 可选字段（input、system、history）的格式合法性
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
from pyarrow import compute as pc


REQUIRED_FIELDS = {"instruction", "output"}
OPTIONAL_FIELDS = {"input", "system", "history"}
ALL_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

# fix_file 使用的标准字段集（精确四个字段）
FIX_FIELDS = ("instruction", "input", "output", "system")


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_records: int = 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = [
            f"总记录数: {self.total_records}",
            f"校验结果: {'通过' if self.is_valid else '失败'}",
            f"错误数量: {len(self.errors)}",
            f"警告数量: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("\n错误列表:")
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")
        if self.warnings:
            lines.append("\n警告列表:")
            for w in self.warnings:
                lines.append(f"  [WARN]  {w}")
        return "\n".join(lines)


def _check_record(record: Any, index: int, result: ValidationResult) -> None:
    """校验单条记录."""
    prefix = f"第 {index + 1} 条记录"

    if not isinstance(record, dict):
        result.add_error(f"{prefix}: 应为对象(dict)，实际为 {type(record).__name__}")
        return

    # 检查未知字段
    unknown = set(record.keys()) - ALL_FIELDS
    if unknown:
        result.add_warning(f"{prefix}: 包含未知字段 {sorted(unknown)}")

    # 检查必填字段
    for req in REQUIRED_FIELDS:
        if req not in record:
            result.add_error(f"{prefix}: 缺少必填字段 '{req}'")
        elif not isinstance(record[req], str):
            result.add_error(f"{prefix}: 字段 '{req}' 应为字符串，实际为 {type(record[req]).__name__}")
        elif not record[req].strip():
            result.add_error(f"{prefix}: 必填字段 '{req}' 不能为空")

    # 检查可选字段类型
    for opt in ("input", "system"):
        if opt in record and record[opt] is not None and not isinstance(record[opt], str):
            result.add_error(f"{prefix}: 字段 '{opt}' 应为字符串，实际为 {type(record[opt]).__name__}")

    # 检查 history 格式：应为 list[list[str, str]]
    if "history" in record:
        history = record["history"]
        if history is None or history == []:
            result.add_warning(f"{prefix}: 字段 'history' 为空，建议删除")
        elif not isinstance(history, list):
            result.add_error(f"{prefix}: 字段 'history' 应为列表，实际为 {type(history).__name__}")
        else:
            for turn_idx, turn in enumerate(history):
                if not isinstance(turn, list) or len(turn) != 2:
                    result.add_error(
                        f"{prefix}: 'history[{turn_idx}]' 应为包含 2 个字符串的列表，实际为 {repr(turn)[:80]}"
                    )
                elif not all(isinstance(s, str) for s in turn):
                    result.add_error(f"{prefix}: 'history[{turn_idx}]' 中的元素应为字符串")


def load_and_display(path: str | Path, max_records: int = 10, max_field_len: int = 51200) -> None:
    """读取 JSON 训练数据文件并打印前若干条记录.

    Args:
        path: JSON 文件路径。
        max_records: 最多显示的记录条数，默认 3。
        max_field_len: 每个字段值的最大显示字符数，超出部分截断，默认 120。
    """
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] 文件不存在: {path}")
        return

    with open(path, "rb") as f:
        encoding = "utf-8-sig" if f.read(3) == b"\xef\xbb\xbf" else "utf-8"

    try:
        with open(path, encoding=encoding) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 解析失败: {e}")
        return

    if not isinstance(data, list):
        print(f"[ERROR] 顶层结构应为列表，实际为 {type(data).__name__}")
        return

    total = len(data)
    show = min(max_records, total)
    print(f"文件读取成功，共 {total} 条记录，显示前 {show} 条：")
    print("=" * 60)

    for i in range(show):
        record = data[i]
        print(f"\n【第 {i + 1} 条】")
        for key, value in record.items():
            if isinstance(value, str):
                display = value.replace("\n", "↵ ")
                if len(display) > max_field_len:
                    display = display[:max_field_len] + f"...（共 {len(value)} 字符）"
            else:
                display = repr(value)
                if len(display) > max_field_len:
                    display = display[:max_field_len] + "..."
            print(f"  {key}: {display}")

    print("\n" + "=" * 60)


def fix_file(path: str | Path, output_path: str | Path | None = None) -> None:
    """修复 JSON 训练数据文件，使每条记录只含标准四个字段.

    修复规则：
    - 每条记录只保留 instruction / input / output / system 四个字段。
    - 缺少的字段补充为空字符串 ""。
    - 含有其他未知字段的记录视为错误，不写入输出文件并打印错误信息。
    - 修复后的文件以紧凑 UTF-8（无 BOM）格式写入。

    Args:
        path: 源 JSON 文件路径。
        output_path: 输出文件路径，默认覆盖源文件。
    """
    path = Path(path)
    output_path = Path(output_path) if output_path else path

    if not path.exists():
        print(f"[ERROR] 文件不存在: {path}")
        return

    with open(path, "rb") as f:
        encoding = "utf-8-sig" if f.read(3) == b"\xef\xbb\xbf" else "utf-8"

    try:
        with open(path, encoding=encoding) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 解析失败: {e}")
        return

    if not isinstance(data, list):
        print(f"[ERROR] 顶层结构应为列表，实际为 {type(data).__name__}")
        return

    fixed: list[dict] = []
    error_count = 0
    fixed_count = 0
    allowed = set(FIX_FIELDS)

    for i, record in enumerate(data):
        prefix = f"第 {i + 1} 条记录"
        if not isinstance(record, dict):
            print(f"[ERROR] {prefix}: 应为对象(dict)，实际为 {type(record).__name__}，已跳过")
            error_count += 1
            continue

        unknown = set(record.keys()) - allowed
        if unknown:
            print(f"[ERROR] {prefix}: 含有未知字段 {sorted(unknown)}，已跳过")
            error_count += 1
            continue

        new_record = {f: record.get(f) or "" for f in FIX_FIELDS}
        fixed.append(new_record)

        # 统计实际被补全的字段数
        patched = [f for f in FIX_FIELDS if f not in record]
        if patched:
            fixed_count += 1

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(fixed, f, ensure_ascii=False, separators=(",", ":"))

    print(
        f"修复完成：共 {len(data)} 条 → 写入 {len(fixed)} 条"
        f"（跳过 {error_count} 条错误，补全字段 {fixed_count} 条）"
    )
    print(f"已写入: {output_path}")


# 期望的 PyArrow Schema（四个字段均为 utf8 字符串）
ARROW_SCHEMA = pa.schema([
    pa.field("instruction", pa.utf8(), nullable=False),
    pa.field("input",       pa.utf8(), nullable=False),
    pa.field("output",      pa.utf8(), nullable=False),
    pa.field("system",      pa.utf8(), nullable=False),
])


def validate_with_arrow(path: str | Path) -> ValidationResult:
    """使用 PyArrow 校验 JSON 训练数据文件的 Schema 与数据质量.

    校验内容：
    - 使用 PyArrow 将 JSON 加载为 Arrow Table
    - 检查字段名是否完全匹配期望 Schema
    - 检查每列数据类型是否为 utf8 字符串
    - 检查必填字段（instruction / output）是否存在 null 或空字符串
    - 输出各列的 null 数量统计

    Args:
        path: JSON 文件路径。

    Returns:
        ValidationResult 对象，包含校验结果、错误和警告列表。
    """
    result = ValidationResult()
    path = Path(path)

    if not path.exists():
        result.add_error(f"文件不存在: {path}")
        return result

    # 处理 BOM
    with open(path, "rb") as f:
        raw_start = f.read(3)
    encoding = "utf-8-sig" if raw_start == b"\xef\xbb\xbf" else "utf-8"
    if encoding == "utf-8-sig":
        result.add_warning("文件包含 UTF-8 BOM，建议使用不带 BOM 的 UTF-8 编码")

    # 用标准 json 模块加载数组格式的 JSON，再转为 Arrow Table
    try:
        with open(path, encoding=encoding) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result.add_error(f"JSON 解析失败: {e}")
        return result

    if not isinstance(data, list):
        result.add_error(f"顶层结构应为列表(list)，实际为 {type(data).__name__}")
        return result

    result.total_records = len(data)
    if result.total_records == 0:
        result.add_warning("文件中没有任何记录")
        return result

    # 转换为 Arrow Table（自动推断 Schema）
    try:
        table: pa.Table = pa.Table.from_pylist(data)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
        result.add_error(f"Arrow Table 构建失败: {e}")
        return result

    actual_fields = set(table.schema.names)
    expected_fields = set(ARROW_SCHEMA.names)

    # 检查多余字段
    extra = actual_fields - expected_fields
    if extra:
        result.add_error(f"存在未知字段: {sorted(extra)}")

    # 检查缺失字段
    missing = expected_fields - actual_fields
    if missing:
        result.add_error(f"缺少必要字段: {sorted(missing)}")

    # 逐字段检查类型与空值
    for fname in ARROW_SCHEMA.names:
        if fname not in actual_fields:
            continue

        col: pa.ChunkedArray = table.column(fname)

        # 类型检查
        if not pa.types.is_string(col.type) and not pa.types.is_large_string(col.type):
            result.add_error(
                f"字段 '{fname}' 类型应为 utf8/string，实际推断为 {col.type}"
            )

        # null 数量
        null_count = col.null_count
        if null_count > 0:
            if fname in REQUIRED_FIELDS:
                result.add_error(f"必填字段 '{fname}' 含有 {null_count} 个 null 值")
            else:
                result.add_warning(f"字段 '{fname}' 含有 {null_count} 个 null 值")

        # 必填字段空字符串检查（转为非 null 后比较）
        if fname in REQUIRED_FIELDS and pa.types.is_string(col.type):
            combined = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
            empty_mask = pc.equal(pc.utf8_trim_whitespace(combined), pa.scalar("", pa.utf8()))
            empty_count = pc.sum(empty_mask.cast(pa.int32())).as_py() or 0
            if empty_count > 0:
                result.add_error(f"必填字段 '{fname}' 含有 {empty_count} 条空字符串")

    # 打印 Arrow Schema 信息
    print("[Arrow] 推断 Schema:")
    for f in table.schema:
        null_count = table.column(f.name).null_count
        print(f"  {f.name}: {f.type}  (nulls={null_count}, rows={table.num_rows})")

    return result


def validate_file(path: str | Path) -> ValidationResult:
    """校验指定路径的 JSON 训练数据文件.

    Args:
        path: JSON 文件路径。

    Returns:
        ValidationResult 对象，包含校验结果、错误和警告列表。
    """
    result = ValidationResult()
    path = Path(path)

    if not path.exists():
        result.add_error(f"文件不存在: {path}")
        return result

    # 检测 BOM
    with open(path, "rb") as f:
        raw = f.read(3)
    if raw[:3] == b"\xef\xbb\xbf":
        result.add_warning("文件包含 UTF-8 BOM，建议使用不带 BOM 的 UTF-8 编码")
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"

    # 解析 JSON
    try:
        with open(path, encoding=encoding) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result.add_error(f"JSON 解析失败: {e}")
        return result

    # 顶层结构必须为列表
    if not isinstance(data, list):
        result.add_error(f"顶层结构应为列表(list)，实际为 {type(data).__name__}")
        return result

    result.total_records = len(data)
    if result.total_records == 0:
        result.add_warning("文件中没有任何记录")
        return result

    # 逐条校验
    for i, record in enumerate(data):
        _check_record(record, i, result)

    return result


def main(path_json: str | None = None, method: str = "validate", output_path: str | None = None) -> None:
    """入口函数.

    Args:
        path_json: JSON 文件路径。为 None 时从命令行参数读取。
        method: 执行方式，可选 'validate' / 'show' / 'fix'，默认 'validate'。
        output_path: 仅 method='fix' 时有效，指定输出文件路径，默认覆盖源文件。
    """
    # 从命令行参数解析（当直接运行脚本时）
    if path_json is None:
        if len(sys.argv) < 2:
            print("用法: python validate.py <json文件路径> [--show | --fix [输出文件路径]]")
            sys.exit(1)
        path_json = sys.argv[1]
        if "--show" in sys.argv:
            method = "show"
        elif "--fix" in sys.argv:
            method = "fix"
            idx = sys.argv.index("--fix")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
                output_path = sys.argv[idx + 1]
        elif "--arrow" in sys.argv:
            method = "arrow"
        else:
            method = "validate"

    if method == "show":
        load_and_display(path_json)
    elif method == "fix":
        fix_file(path_json, output_path)
    elif method == "arrow":
        result = validate_with_arrow(path_json)
        print(result.summary())
        sys.exit(0 if result.is_valid else 1)
    else:
        result = validate_file(path_json)
        print(result.summary())
        sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    path_json = r"G:\02.github\LlamaFactory2\data\99.all\bori_nl_all2.json"
    method = "arrow"
    main(path_json=path_json, method=method)
