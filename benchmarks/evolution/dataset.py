"""Thirty deterministic repair tasks grouped into ten transfer families.

Each family contains one learning task and two held-out transfer tasks.  The
tasks intentionally use different domain names while preserving the underlying
repair pattern, so an evaluation can measure procedural reuse without handing
the exact patch to the Agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True)
class EvolutionTask:
    """One isolated Python repair task with evaluator-only reference material."""

    task_id: str
    family: str
    stage: str
    prompt: str
    module: str
    buggy_code: str
    fixed_code: str
    public_tests: str
    hidden_tests: str

    @property
    def filename(self) -> str:
        return f"{self.module}.py"


def _clean(text: str) -> str:
    return dedent(text).strip() + "\n"


def _task(
    task_id: str,
    family: str,
    stage: str,
    prompt: str,
    module: str,
    buggy_code: str,
    fixed_code: str,
    public_tests: str,
    hidden_tests: str,
) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        family=family,
        stage=stage,
        prompt=prompt.strip(),
        module=module,
        buggy_code=_clean(buggy_code),
        fixed_code=_clean(fixed_code),
        public_tests=_clean(public_tests),
        hidden_tests=_clean(hidden_tests),
    )


def _inclusive_window(
    task_id: str,
    stage: str,
    module: str,
    function: str,
    noun: str,
) -> EvolutionTask:
    prompt = (
        f"修复 {module}.py 中的 {function}。接口约定 start 和 stop 都是包含端点的"
        f"索引，但当前返回的{noun}窗口缺少最后一个元素。保持函数签名不变并运行测试。"
    )
    buggy = f"""
    def {function}(values, start, stop):
        return values[start:stop]
    """
    fixed = f"""
    def {function}(values, start, stop):
        return values[start:stop + 1]
    """
    public = f"""
    from {module} import {function}

    def test_includes_stop_item():
        assert {function}([10, 20, 30, 40], 1, 2) == [20, 30]
    """
    hidden = f"""
    from {module} import {function}

    def test_single_item_window():
        assert {function}(["a", "b", "c"], 1, 1) == ["b"]

    def test_whole_window():
        assert {function}([1, 2, 3], 0, 2) == [1, 2, 3]
    """
    return _task(task_id, "inclusive-boundary", stage, prompt, module, buggy, fixed, public, hidden)


def _optional_text(
    task_id: str,
    stage: str,
    module: str,
    function: str,
    fallback: str,
    noun: str,
) -> EvolutionTask:
    prompt = (
        f"修复 {module}.py 中的 {function}。{noun}可能是 None 或只包含空白，"
        f"这两种情况都应返回 {fallback!r}；有效文本需要去除首尾空白。"
    )
    buggy = f"""
    def {function}(value):
        cleaned = value.strip()
        return cleaned or {fallback!r}
    """
    fixed = f"""
    def {function}(value):
        if value is None:
            return {fallback!r}
        cleaned = value.strip()
        return cleaned or {fallback!r}
    """
    public = f"""
    from {module} import {function}

    def test_none_uses_fallback():
        assert {function}(None) == {fallback!r}
    """
    hidden = f"""
    from {module} import {function}

    def test_blank_uses_fallback():
        assert {function}("   ") == {fallback!r}

    def test_valid_text_is_trimmed():
        assert {function}("  valid value  ") == "valid value"
    """
    return _task(task_id, "optional-normalization", stage, prompt, module, buggy, fixed, public, hidden)


def _bounded_integer(
    task_id: str,
    stage: str,
    module: str,
    function: str,
    low: int,
    high: int,
    noun: str,
) -> EvolutionTask:
    prompt = (
        f"修复 {module}.py 中的 {function}。{noun}只能位于闭区间 [{low}, {high}]，"
        "越界时必须抛出 ValueError，边界值应被接受。"
    )
    buggy = f"""
    def {function}(value):
        if value < {low} and value > {high}:
            raise ValueError("out of range")
        return value
    """
    fixed = f"""
    def {function}(value):
        if value < {low} or value > {high}:
            raise ValueError("out of range")
        return value
    """
    public = f"""
    import pytest
    from {module} import {function}

    def test_rejects_below_minimum():
        with pytest.raises(ValueError):
            {function}({low - 1})
    """
    hidden = f"""
    import pytest
    from {module} import {function}

    def test_rejects_above_maximum():
        with pytest.raises(ValueError):
            {function}({high + 1})

    def test_accepts_both_boundaries():
        assert {function}({low}) == {low}
        assert {function}({high}) == {high}
    """
    return _task(task_id, "range-validation", stage, prompt, module, buggy, fixed, public, hidden)


def _stable_unique(
    task_id: str,
    stage: str,
    module: str,
    function: str,
    noun: str,
) -> EvolutionTask:
    prompt = (
        f"修复 {module}.py 中的 {function}。需要对{noun}去重，同时严格保留第一次出现的顺序，"
        "不能对结果排序。"
    )
    buggy = f"""
    def {function}(values):
        return sorted(set(values))
    """
    fixed = f"""
    def {function}(values):
        seen = set()
        result = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result
    """
    public = f"""
    from {module} import {function}

    def test_preserves_first_seen_order():
        assert {function}(["z", "a", "z", "b"]) == ["z", "a", "b"]
    """
    hidden = f"""
    from {module} import {function}

    def test_all_duplicates():
        assert {function}([3, 3, 3]) == [3]

    def test_reverse_numeric_order_is_not_sorted():
        assert {function}([3, 1, 2, 1]) == [3, 1, 2]
    """
    return _task(task_id, "stable-deduplication", stage, prompt, module, buggy, fixed, public, hidden)


def _deep_merge(
    task_id: str,
    stage: str,
    module: str,
    function: str,
    noun: str,
) -> EvolutionTask:
    prompt = (
        f"修复 {module}.py 中的 {function}。合并{noun}时，两个值都是字典的键必须递归合并，"
        "普通值和列表仍由 override 整体替换，并且不能修改输入对象。"
    )
    buggy = f"""
    def {function}(base, override):
        return {{**base, **override}}
    """
    fixed = f"""
    def {function}(base, override):
        result = dict(base)
        for key, value in override.items():
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key] = {function}(result[key], value)
            else:
                result[key] = value
        return result
    """
    public = f"""
    from {module} import {function}

    def test_nested_keys_are_preserved():
        base = {{"db": {{"host": "localhost", "port": 5432}}}}
        override = {{"db": {{"port": 6432}}}}
        assert {function}(base, override) == {{"db": {{"host": "localhost", "port": 6432}}}}
    """
    hidden = f"""
    from {module} import {function}

    def test_deep_levels_and_input_immutability():
        base = {{"a": {{"b": {{"x": 1, "y": 2}}}}, "items": [1]}}
        override = {{"a": {{"b": {{"y": 9}}}}, "items": [2]}}
        result = {function}(base, override)
        assert result == {{"a": {{"b": {{"x": 1, "y": 9}}}}, "items": [2]}}
        assert base["a"]["b"] == {{"x": 1, "y": 2}}
    """
    return _task(task_id, "recursive-merge", stage, prompt, module, buggy, fixed, public, hidden)


def _cache_invalidation(
    task_id: str,
    stage: str,
    module: str,
    class_name: str,
    getter: str,
    setter: str,
    factor: int,
    noun: str,
) -> EvolutionTask:
    prompt = (
        f"修复 {module}.py 中 {class_name} 的缓存一致性问题。调用 {setter} 更新{noun}后，"
        f"后续 {getter} 必须返回基于新值计算的结果，且不能清空其他键的有效缓存。"
    )
    buggy = f"""
    class {class_name}:
        def __init__(self, values):
            self._values = dict(values)
            self._cache = {{}}

        def {getter}(self, key):
            if key not in self._cache:
                self._cache[key] = self._values[key] * {factor}
            return self._cache[key]

        def {setter}(self, key, value):
            self._values[key] = value
    """
    fixed = f"""
    class {class_name}:
        def __init__(self, values):
            self._values = dict(values)
            self._cache = {{}}

        def {getter}(self, key):
            if key not in self._cache:
                self._cache[key] = self._values[key] * {factor}
            return self._cache[key]

        def {setter}(self, key, value):
            self._values[key] = value
            self._cache.pop(key, None)
    """
    public = f"""
    from {module} import {class_name}

    def test_update_invalidates_cached_key():
        store = {class_name}({{"a": 2}})
        assert store.{getter}("a") == {2 * factor}
        store.{setter}("a", 5)
        assert store.{getter}("a") == {5 * factor}
    """
    hidden = f"""
    from {module} import {class_name}

    def test_other_cached_keys_remain_valid():
        store = {class_name}({{"a": 2, "b": 3}})
        assert store.{getter}("a") == {2 * factor}
        assert store.{getter}("b") == {3 * factor}
        store.{setter}("a", 7)
        assert store.{getter}("a") == {7 * factor}
        assert store.{getter}("b") == {3 * factor}
    """
    return _task(task_id, "cache-invalidation", stage, prompt, module, buggy, fixed, public, hidden)


def _path_containment(
    task_id: str,
    stage: str,
    module: str,
    function: str,
    noun: str,
) -> EvolutionTask:
    prompt = (
        f"修复 {module}.py 中的 {function}。它需要判断{noun}路径是否位于根目录内部或等于根目录；"
        "不能使用容易把 root-backup 误判为 root 子目录的字符串前缀比较。"
    )
    buggy = f"""
    from pathlib import Path

    def {function}(root, candidate):
        root_path = Path(root).resolve()
        candidate_path = Path(candidate).resolve()
        return str(candidate_path).startswith(str(root_path))
    """
    fixed = f"""
    from pathlib import Path

    def {function}(root, candidate):
        root_path = Path(root).resolve()
        candidate_path = Path(candidate).resolve()
        try:
            candidate_path.relative_to(root_path)
        except ValueError:
            return False
        return True
    """
    public = f"""
    from {module} import {function}

    def test_sibling_with_same_prefix_is_rejected(tmp_path):
        root = tmp_path / "root"
        sibling = tmp_path / "root-backup" / "file.txt"
        root.mkdir()
        sibling.parent.mkdir()
        sibling.write_text("x")
        assert {function}(root, sibling) is False
    """
    hidden = f"""
    from {module} import {function}

    def test_descendant_and_root_are_accepted(tmp_path):
        root = tmp_path / "safe"
        child = root / "nested" / "file.txt"
        child.parent.mkdir(parents=True)
        child.write_text("x")
        assert {function}(root, child) is True
        assert {function}(root, root) is True

    def test_parent_escape_is_rejected(tmp_path):
        root = tmp_path / "safe"
        root.mkdir()
        outside = root / ".." / "outside.txt"
        outside.resolve().write_text("x")
        assert {function}(root, outside) is False
    """
    return _task(task_id, "structural-path-containment", stage, prompt, module, buggy, fixed, public, hidden)


def _retry_budget(
    task_id: str,
    stage: str,
    module: str,
    function: str,
    noun: str,
) -> EvolutionTask:
    prompt = (
        f"修复 {module}.py 中的 {function}。max_attempts 表示包括首次调用在内的总{noun}次数；"
        "RuntimeError 可以重试，耗尽次数后必须重新抛出最后一次异常。"
    )
    buggy = f"""
    def {function}(operation, max_attempts=3):
        last_error = None
        for _ in range(max_attempts - 1):
            try:
                return operation()
            except RuntimeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("operation was not attempted")
    """
    fixed = f"""
    def {function}(operation, max_attempts=3):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        last_error = None
        for _ in range(max_attempts):
            try:
                return operation()
            except RuntimeError as exc:
                last_error = exc
        raise last_error
    """
    public = f"""
    from {module} import {function}

    def test_can_succeed_on_last_attempt():
        calls = []
        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("temporary")
            return "ok"
        assert {function}(operation, max_attempts=3) == "ok"
        assert len(calls) == 3
    """
    hidden = f"""
    import pytest
    from {module} import {function}

    def test_one_attempt_really_calls_once():
        calls = []
        def operation():
            calls.append(1)
            raise RuntimeError("last")
        with pytest.raises(RuntimeError, match="last"):
            {function}(operation, max_attempts=1)
        assert len(calls) == 1

    def test_zero_attempts_is_invalid():
        with pytest.raises(ValueError):
            {function}(lambda: None, max_attempts=0)
    """
    return _task(task_id, "retry-budget", stage, prompt, module, buggy, fixed, public, hidden)


def _omit_none(
    task_id: str,
    stage: str,
    module: str,
    function: str,
    noun: str,
) -> EvolutionTask:
    prompt = (
        f"修复 {module}.py 中的 {function}。序列化{noun}时只能省略值为 None 的字段；"
        "0、False 和空字符串都是有效业务值，必须保留。"
    )
    buggy = f"""
    def {function}(record):
        return {{key: value for key, value in record.items() if value}}
    """
    fixed = f"""
    def {function}(record):
        return {{key: value for key, value in record.items() if value is not None}}
    """
    public = f"""
    from {module} import {function}

    def test_zero_is_preserved_and_none_removed():
        assert {function}({{"count": 0, "note": None}}) == {{"count": 0}}
    """
    hidden = f"""
    from {module} import {function}

    def test_false_and_empty_string_are_preserved():
        value = {{"enabled": False, "label": "", "missing": None}}
        assert {function}(value) == {{"enabled": False, "label": ""}}
    """
    return _task(task_id, "none-aware-serialization", stage, prompt, module, buggy, fixed, public, hidden)


def _state_transition(
    task_id: str,
    stage: str,
    module: str,
    class_name: str,
    states: tuple[str, str, str],
    noun: str,
) -> EvolutionTask:
    first, second, terminal = states
    transitions = repr({first: {second}, second: {terminal}, terminal: set()})
    all_states = repr(set(states))
    prompt = (
        f"修复 {module}.py 中 {class_name} 的状态流转校验。{noun}只能按照 "
        f"{first} → {second} → {terminal} 变化，跳步或从终态继续流转必须抛出 ValueError。"
    )
    buggy = f"""
    class {class_name}:
        TRANSITIONS = {transitions}

        def __init__(self):
            self.state = {first!r}

        def move_to(self, target):
            if target not in {all_states}:
                raise ValueError("unknown state")
            self.state = target
    """
    fixed = f"""
    class {class_name}:
        TRANSITIONS = {transitions}

        def __init__(self):
            self.state = {first!r}

        def move_to(self, target):
            if target not in self.TRANSITIONS[self.state]:
                raise ValueError("invalid transition")
            self.state = target
    """
    public = f"""
    import pytest
    from {module} import {class_name}

    def test_cannot_skip_intermediate_state():
        item = {class_name}()
        with pytest.raises(ValueError):
            item.move_to({terminal!r})
    """
    hidden = f"""
    import pytest
    from {module} import {class_name}

    def test_valid_flow_and_terminal_guard():
        item = {class_name}()
        item.move_to({second!r})
        item.move_to({terminal!r})
        assert item.state == {terminal!r}
        with pytest.raises(ValueError):
            item.move_to({first!r})
    """
    return _task(task_id, "state-transition-guard", stage, prompt, module, buggy, fixed, public, hidden)


TASKS: tuple[EvolutionTask, ...] = (
    _inclusive_window("boundary-learn", "learn", "audit_window", "select_audit_window", "审计记录"),
    _inclusive_window("boundary-transfer-1", "transfer", "score_window", "select_score_window", "评分"),
    _inclusive_window("boundary-transfer-2", "transfer", "event_window", "select_event_window", "事件"),
    _optional_text("optional-learn", "learn", "display_name", "normalize_display_name", "anonymous", "显示名称"),
    _optional_text("optional-transfer-1", "transfer", "region_name", "normalize_region", "global", "区域名称"),
    _optional_text("optional-transfer-2", "transfer", "queue_label", "normalize_queue_label", "default", "队列标签"),
    _bounded_integer("range-learn", "learn", "port_config", "validate_port", 1, 65535, "端口"),
    _bounded_integer("range-transfer-1", "transfer", "percentage", "validate_percentage", 0, 100, "百分比"),
    _bounded_integer("range-transfer-2", "transfer", "batch_size", "validate_batch_size", 1, 500, "批大小"),
    _stable_unique("dedupe-learn", "learn", "route_order", "unique_routes", "路由"),
    _stable_unique("dedupe-transfer-1", "transfer", "recipient_order", "unique_recipients", "收件人"),
    _stable_unique("dedupe-transfer-2", "transfer", "plugin_order", "unique_plugins", "插件"),
    _deep_merge("merge-learn", "learn", "service_config", "merge_service_config", "服务配置"),
    _deep_merge("merge-transfer-1", "transfer", "theme_config", "merge_theme_config", "主题配置"),
    _deep_merge("merge-transfer-2", "transfer", "policy_config", "merge_policy_config", "策略配置"),
    _cache_invalidation("cache-learn", "learn", "price_book", "PriceBook", "total", "set_price", 2, "价格"),
    _cache_invalidation("cache-transfer-1", "transfer", "quota_book", "QuotaBook", "weighted", "set_quota", 3, "配额"),
    _cache_invalidation("cache-transfer-2", "transfer", "score_book", "ScoreBook", "rank_score", "set_score", 5, "分数"),
    _path_containment("path-learn", "learn", "upload_guard", "is_safe_upload", "上传文件"),
    _path_containment("path-transfer-1", "transfer", "workspace_guard", "is_in_workspace", "工作区文件"),
    _path_containment("path-transfer-2", "transfer", "archive_guard", "is_in_archive", "归档文件"),
    _retry_budget("retry-learn", "learn", "request_retry", "run_request", "请求"),
    _retry_budget("retry-transfer-1", "transfer", "job_retry", "run_job", "任务"),
    _retry_budget("retry-transfer-2", "transfer", "upload_retry", "run_upload", "上传"),
    _omit_none("serialization-learn", "learn", "user_payload", "serialize_user", "用户数据"),
    _omit_none("serialization-transfer-1", "transfer", "metric_payload", "serialize_metric", "指标数据"),
    _omit_none("serialization-transfer-2", "transfer", "feature_payload", "serialize_feature", "特性配置"),
    _state_transition("state-learn", "learn", "order_state", "Order", ("created", "paid", "completed"), "订单"),
    _state_transition("state-transfer-1", "transfer", "deploy_state", "Deployment", ("queued", "running", "finished"), "部署"),
    _state_transition("state-transfer-2", "transfer", "review_state", "Review", ("draft", "submitted", "approved"), "审核单"),
)


if len(TASKS) != 30 or len({task.task_id for task in TASKS}) != 30:
    raise RuntimeError("evolution benchmark must contain 30 uniquely named tasks")
