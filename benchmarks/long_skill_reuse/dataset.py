"""Multi-file transfer tasks for measuring procedural Skill reuse.

Each repository uses the same architectural pattern but a different domain.
The learning task can produce a reusable idempotent-command/transactional-
outbox Skill; transfer tasks run in fresh sessions and isolated workspaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True)
class LongSkillTask:
    task_id: str
    family: str
    stage: str
    prompt: str
    buggy_files: dict[str, str]
    fixed_files: dict[str, str]
    public_tests: str
    hidden_tests: str


def _clean(value: str) -> str:
    return dedent(value).strip() + "\n"


def _workflow_task(
    *,
    task_id: str,
    stage: str,
    domain: str,
    resource: str,
    initial: int,
    amount: int,
    direction: int,
) -> LongSkillTask:
    after = initial + direction * amount
    operation = "+ amount" if direction > 0 else "- amount"
    prompt = f"""
    修复 workflow 包中 {domain} 的多文件处理流程。这个任务属于
    idempotent command + transactional outbox 场景，必须满足以下合同：

    1. request_id 首次执行时只修改一次 {resource}，并只发布一个事件；
    2. 相同 request_id 重试时返回首次结果，不得再次修改状态或发布事件；
    3. amount 必须大于 0，非法输入不能产生任何状态变化；
    4. outbox 发布失败时，repository 的值与 processed 记录必须一起回滚，之后可用同一 request_id 重试；
    5. repository.restore 与 repository.forget 是回滚边界的一部分，必须真正恢复旧值并移除处理记录。

    请先检查 workflow 下的模型、仓储、outbox 和 service，再完成修复并运行全部测试。
    不要修改测试文件或公开接口。
    """
    models = """
    from dataclasses import dataclass


    @dataclass(frozen=True)
    class CommandResult:
        request_id: str
        resource_id: str
        before: int
        after: int
    """
    outbox = """
    class InMemoryOutbox:
        def __init__(self):
            self.events = []
            self.fail_next = False

        def publish(self, event):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("outbox unavailable")
            self.events.append(dict(event))
    """
    repository_buggy = f"""
    class InMemoryRepository:
        def __init__(self, initial):
            self._values = dict(initial)
            self._processed = {{}}

        def value_for(self, resource_id):
            return self._values.get(resource_id, 0)

        def processed(self, request_id):
            return self._processed.get(request_id)

        def apply(self, resource_id, amount):
            next_value = self.value_for(resource_id) {operation}
            if next_value < 0:
                raise ValueError("insufficient value")
            self._values[resource_id] = next_value
            return next_value

        def remember(self, request_id, result):
            self._processed[request_id] = result

        def restore(self, resource_id, previous):
            self._values[resource_id] = self._values.get(resource_id, 0)

        def forget(self, request_id):
            return None
    """
    repository_fixed = f"""
    class InMemoryRepository:
        def __init__(self, initial):
            self._values = dict(initial)
            self._processed = {{}}

        def value_for(self, resource_id):
            return self._values.get(resource_id, 0)

        def processed(self, request_id):
            return self._processed.get(request_id)

        def apply(self, resource_id, amount):
            next_value = self.value_for(resource_id) {operation}
            if next_value < 0:
                raise ValueError("insufficient value")
            self._values[resource_id] = next_value
            return next_value

        def remember(self, request_id, result):
            self._processed[request_id] = result

        def restore(self, resource_id, previous):
            self._values[resource_id] = previous

        def forget(self, request_id):
            self._processed.pop(request_id, None)
    """
    service_buggy = """
    from .models import CommandResult


    class WorkflowService:
        def __init__(self, repository, outbox):
            self.repository = repository
            self.outbox = outbox

        def execute(self, request_id, resource_id, amount):
            before = self.repository.value_for(resource_id)
            after = self.repository.apply(resource_id, amount)
            if amount <= 0:
                raise ValueError("amount must be positive")
            result = CommandResult(request_id, resource_id, before, after)
            self.repository.remember(request_id, result)
            self.outbox.publish({
                "request_id": request_id,
                "resource_id": resource_id,
                "before": before,
                "after": after,
            })
            return result
    """
    service_fixed = """
    from .models import CommandResult


    class WorkflowService:
        def __init__(self, repository, outbox):
            self.repository = repository
            self.outbox = outbox

        def execute(self, request_id, resource_id, amount):
            existing = self.repository.processed(request_id)
            if existing is not None:
                return existing
            if amount <= 0:
                raise ValueError("amount must be positive")

            before = self.repository.value_for(resource_id)
            after = self.repository.apply(resource_id, amount)
            result = CommandResult(request_id, resource_id, before, after)
            try:
                self.repository.remember(request_id, result)
                self.outbox.publish({
                    "request_id": request_id,
                    "resource_id": resource_id,
                    "before": before,
                    "after": after,
                })
            except Exception:
                self.repository.restore(resource_id, before)
                self.repository.forget(request_id)
                raise
            return result
    """
    public_tests = f"""
    from workflow.outbox import InMemoryOutbox
    from workflow.repository import InMemoryRepository
    from workflow.service import WorkflowService


    def build():
        repository = InMemoryRepository({{{resource!r}: {initial}}})
        outbox = InMemoryOutbox()
        return WorkflowService(repository, outbox), repository, outbox


    def test_success_updates_state_and_publishes_once():
        service, repository, outbox = build()
        result = service.execute("req-1", {resource!r}, {amount})
        assert result.before == {initial}
        assert result.after == {after}
        assert repository.value_for({resource!r}) == {after}
        assert len(outbox.events) == 1


    def test_duplicate_request_returns_original_result_without_second_effect():
        service, repository, outbox = build()
        first = service.execute("req-1", {resource!r}, {amount})
        second = service.execute("req-1", {resource!r}, {amount})
        assert second == first
        assert repository.value_for({resource!r}) == {after}
        assert len(outbox.events) == 1
    """
    hidden_tests = f"""
    import pytest

    from workflow.models import CommandResult
    from workflow.outbox import InMemoryOutbox
    from workflow.repository import InMemoryRepository
    from workflow.service import WorkflowService


    def build():
        repository = InMemoryRepository({{{resource!r}: {initial}}})
        outbox = InMemoryOutbox()
        return WorkflowService(repository, outbox), repository, outbox


    def test_invalid_amount_has_no_side_effects():
        service, repository, outbox = build()
        with pytest.raises(ValueError):
            service.execute("bad", {resource!r}, 0)
        assert repository.value_for({resource!r}) == {initial}
        assert repository.processed("bad") is None
        assert outbox.events == []


    def test_publish_failure_rolls_back_and_same_request_can_retry():
        service, repository, outbox = build()
        outbox.fail_next = True
        with pytest.raises(RuntimeError):
            service.execute("retry", {resource!r}, {amount})
        assert repository.value_for({resource!r}) == {initial}
        assert repository.processed("retry") is None
        assert outbox.events == []

        result = service.execute("retry", {resource!r}, {amount})
        assert result.after == {after}
        assert repository.value_for({resource!r}) == {after}
        assert len(outbox.events) == 1


    def test_repository_rollback_helpers_are_real_boundaries():
        repository = InMemoryRepository({{{resource!r}: {initial}}})
        repository.apply({resource!r}, {amount})
        marker = CommandResult("marker", {resource!r}, {initial}, {after})
        repository.remember("marker", marker)
        repository.restore({resource!r}, {initial})
        repository.forget("marker")
        assert repository.value_for({resource!r}) == {initial}
        assert repository.processed("marker") is None
    """
    common = {
        "workflow/__init__.py": "",
        "workflow/models.py": _clean(models),
        "workflow/outbox.py": _clean(outbox),
    }
    return LongSkillTask(
        task_id=task_id,
        family="idempotent-transactional-workflow",
        stage=stage,
        prompt=_clean(prompt).strip(),
        buggy_files={
            **common,
            "workflow/repository.py": _clean(repository_buggy),
            "workflow/service.py": _clean(service_buggy),
        },
        fixed_files={
            **common,
            "workflow/repository.py": _clean(repository_fixed),
            "workflow/service.py": _clean(service_fixed),
        },
        public_tests=_clean(public_tests),
        hidden_tests=_clean(hidden_tests),
    )


TASKS: tuple[LongSkillTask, ...] = (
    _workflow_task(
        task_id="wallet-credit-learn",
        stage="learn",
        domain="钱包入账",
        resource="wallet-A",
        initial=100,
        amount=25,
        direction=1,
    ),
    _workflow_task(
        task_id="inventory-reservation-transfer",
        stage="transfer",
        domain="库存预留",
        resource="sku-42",
        initial=30,
        amount=7,
        direction=-1,
    ),
    _workflow_task(
        task_id="quota-consumption-transfer",
        stage="transfer",
        domain="配额扣减",
        resource="tenant-blue",
        initial=80,
        amount=15,
        direction=-1,
    ),
)
