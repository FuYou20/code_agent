---
name: fix-import-cycle
description: 定位并修复 Python 循环导入，同时用测试验证行为未回归。
scene: code-repair
triggers:
  - 循环导入
  - circular import
  - partially initialized module
permission_requirements:
  - tool: file_read
    paths:
      - .
  - tool: file_write
    paths:
      - .
  - tool: test_run
---

# 修复 Python 循环导入

1. 用 `file_read` 阅读报错栈涉及的模块，画出最短导入环。
2. 区分仅用于类型标注的导入、模块初始化时必需的导入和可延迟导入。
3. 优先把共享类型或常量下沉到无反向依赖的模块；只在确有必要时使用局部导入。
4. 用 `file_write` 做最小修改，不顺带重构无关代码。
5. 用 `test_run` 执行完整测试，确认导入错误消失且行为没有回归。

