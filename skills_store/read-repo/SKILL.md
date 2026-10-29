---
name: read-repo
description: 通过 Git 元数据和关键入口文件快速建立仓库结构认知。
scene: code-development
triggers:
  - 了解仓库
  - 仓库结构
  - read repo
  - repository overview
permission_requirements:
  - tool: git_read
  - tool: file_read
    paths:
      - .
---

# 阅读仓库

1. 用 `git_read(status)` 确认当前工作树状态，避免把已有改动误判为本次工作。
2. 根据会话中的仓库快照识别语言、包结构、测试目录和可能的入口文件。
3. 用 `file_read` 阅读 README、项目配置和最关键的入口模块。
4. 输出模块职责、主要执行路径、测试入口及仍需确认的问题；本技能不修改仓库。

