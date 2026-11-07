---
name: paper-deep-reading
description: 阅读已推荐论文的公开 PDF，并基于页码证据说明方法、实验、消融或局限；用户要求精读、查看详情或追问某篇论文时使用。
metadata:
  scene: paper-research
  triggers:
    - 精读
    - 论文详情
    - 详情
    - 仔细看看
    - 实验部分
    - 消融实验
    - 方法部分
    - 论文局限
  permission_requirements:
    - tool: paper_read
---

# 论文证据精读

根据用户所指的 `P1`、`P2` 或当前论文，调用一次 `paper_read`，并选择“实验”“方法”或“局限”作为 focus。

- 实验：整理数据集、对比方法、评价指标、关键结果与消融。
- 方法：说明核心问题、主要模块及模块之间的关系。
- 局限：区分作者明确承认的限制和基于证据的分析。
- 关键数字和论文结论必须带工具摘录中的 `[PDF p.N]`；证据未覆盖时明确说明。
- 工具返回失败时说明具体原因，不把 PaperNotes 摘要冒充 PDF 全文。
