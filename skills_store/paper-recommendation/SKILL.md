---
name: paper-recommendation
description: 为用户检索并呈现当天的病理图像、全切片图像或持续学习论文推荐；用户请求找论文、今日推荐或每日推送时使用。
metadata:
  scene: paper-research
  triggers:
    - 今日推荐
    - 推荐论文
    - 论文推荐
    - 找一篇论文
    - 今天有什么论文
    - 每日论文
    - 每日推送
  permission_requirements:
    - tool: paper_recommend_today
---

# 每日论文推荐

用户需要新的论文推荐时，调用一次 `paper_recommend_today`，以工具返回的论文为唯一事实来源。

- 优先突出病理图像与持续学习的交叉论文，但不隐藏工具实际返回的方向。
- 对每篇论文说明编号、发表时间、解决的主要问题和来源链接。
- 不凭模型记忆补充工具结果中没有的论文、时间或实验结论。
- 告诉用户可以继续要求查看某篇论文的实验、方法或局限。
