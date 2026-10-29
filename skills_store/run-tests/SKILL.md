---
name: run-tests
description: 运行 pytest，并从首个有因果价值的失败开始解释和定位问题。
scene: code-repair
triggers:
  - 跑测试
  - 运行测试
  - pytest
  - test failure
permission_requirements:
  - tool: test_run
  - tool: file_read
    paths:
      - .
---

# 运行并解读测试

1. 用 `test_run` 运行固定的 `pytest -q` 测试套件。
2. 从输出中找出最早的根因失败，不把后续连锁失败当成多个独立问题。
3. 用 `file_read` 阅读失败测试和直接相关的实现文件。
4. 报告失败位置、预期值、实际值及最可能的原因；本技能不包含写文件权限。

