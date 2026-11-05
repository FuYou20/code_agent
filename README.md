# MyAgent

面向 KV Cache 与最小权限优化的轻量级可进化 Coding Agent。

MyAgent 不是功能堆叠型 Agent 框架。项目聚焦一个长期运行 Coding Agent
最容易出现的三个问题：能力越多导致上下文越臃肿、任务拆分导致权限扩散、
历史任务完成后经验无法复用。

## 架构

```text
                         MyAgent
                            │
              Lightweight Agent Runtime
          CLI / Model Loop / Tool Calling / Agent Manager
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
 Context Engine      Permission Tree      Evolution Engine
 KV Cache + Skill    Atomic Authority     Memory + Skill
```

Runtime 只承担对话、模型调用、工具执行和子 Agent 生命周期，不作为独立的
复杂编排框架。项目的三个核心能力如下。

### 1. KV Cache 友好的 Context Engine

上下文被拆为三个生命周期不同的区域：

- 稳定 System Prefix：固定约束、工具协议和稳定期 Skill L0；
- Conversation History：近期原始消息与压缩后的早期历史；
- Ephemeral Sidecar：本轮记忆检索、临时 Skill、Skill L1 和工具结果。

默认策略为将全量 Skill L0 固定放入 System 前缀的 `full_system`。项目同时提供
实验性的 `scene_epoch`：

```text
场景识别 → 全局RAG与主模型重排 → 累计预热工作集 → Skill固化
                                                ↓
                     工作集外Skill按轮注入Sidecar，不自动解冻
```

预热首轮根据 `scene` 标签和全局检索选择当前场景可能需要的 Skill，随后通过本地
影子检索维护累计工作集；达到预热轮数后，将工作集固化到会话 System Prompt。
只有出现高置信工作集外证据时才调用模型路由，已确认 fallback 可在同一 Epoch 内
复用，但仍只按轮注入 Sidecar，不改变 System。新 Epoch 只能由六小时超时、
`/skill-reset` 或用户显式 `/scene` 命令创建；Skill L1 始终按轮注入且不进入历史。

在 100-Skill、3 类连续会话共 19 轮的 DeepSeek 强基线 A/B 中，相较把全部
Skill 固定在 System 前缀，旧Warm-Stable在包含查询扩展和重排开销后，将总输入
Token 从 76,913 降至 49,846（降低 35.19%）；但必需 Skill 可见率仅为
15/19，且主 Agent缓存率由71.89%降至47.83%。该结果促使路由策略重构为
Scene-Epoch；新策略通过本地门控与 fallback 确认复用解决了漏召回和逐轮路由问题。

新的确定性结构实验在同一 100-Skill、19 轮任务序列上保持必需 Skill
可见率 19/19，并将未缓存 Skill Token 代理值降低 75.87%；三个会话均只发生
一次 System 变更。随后在线 A/B 在保持 19/19 必需 Skill 可见率的同时，将总输入
Token 降低 48.83%、未缓存输入降低 9.47%；当前仍需通过更多数据和重复运行验证。

### 2. 原子权限树

权限树是运行时唯一授权事实源。一个 `PermissionAtom` 将工具调用权与完整文件
访问范围绑定，二者不能拆分或跨原子重组：

```python
PermissionAtom.read("file_read", "src")
PermissionAtom.write("file_write", "generated")
PermissionAtom.read_write("test_run", read=(".",), write=(".",))
```

可信工具适配器先把具体调用转换成请求原子，再由当前 Agent 挂载节点完整覆盖
校验。模型只能看到节点拥有的工具；执行前再次检查参数对应的文件范围。高风险
人工审批和终端命令白名单属于额外执行约束，不会替代或扩大权限树授权。

子节点只能继承父节点的权限子集，并支持节点隔离、相同策略复用、级联撤权和
Agent 卸载后的空节点回收。Skill 中的 `permission_requirements` 只是运行需求
说明，不能给 Agent 授权，也不会生成第二套 Effective Scope。

### 3. 自进化 Evolution Engine

工具轨迹先经过统一成功门禁，再分流成两种产物：

```text
Execution Trace → Success Gate ─┬→ Experience Memory
                                └→ Procedural Skill
```

当前成功门禁要求：存在持久化变更、写入工具无错误、最后一次 `test_run` 返回码
为 0（允许先红后绿），并且模型给出最终结果。通过后，只把问题结果、关键动作和相关文件写入
SQLite FTS5；相似任务检索到的经验只进入当前 Sidecar，不污染稳定前缀和历史。

用户执行 `/learn` 时，同一门禁会阻止失败或纯闲聊轨迹进入 Skill 蒸馏。蒸馏器
只接收真实工具轨迹，剔除模型虚构的工具需求，并将历史版本归档。经验记忆解决
“这次做对了什么”，Skill 解决“以后按什么流程重复”。

两类产物统一注册为进化资产：新资产为 `candidate`，只能在沙盒显式加载；经过成对
A/B 后，正确性不下降且累计净等价 Token（包含蒸馏成本）为正才晋升为 `active`。
正确但不省 Token 的资产进入 `retired`，导致正确性回退的资产立即进入
`quarantined`；保留期结束后物理清理内容并保留哈希与事件墓碑。生产运行只自动
加载 `active`，生命周期状态不授予任何工具或文件权限。

## 安装与运行

要求 Python 3.11 或更高版本：

```powershell
python -m pip install -e ".[test]"
$env:OPENAI_API_KEY = "..."
python -m myagent
```

也可以执行单轮任务：

```powershell
python -m myagent "读取 README.md 并概括项目架构"
```

全屏 TUI 使用现有 `prompt-toolkit` 依赖，无需额外安装：

```powershell
myagent-tui
# 或
python -m myagent.tui
```

TUI 左侧显示对话，右侧实时展示每次供应商调用返回后的本轮/会话 Token、缓存与
未缓存输入、Skill Epoch 预热进度、固化工作集、当前 Memory/Skill、Sidecar组成、
路由 Token 与 System 变更次数。`/telemetry` 查看完整用量，`/sidecar` 查看最近一次
实际发送给主 Agent 的 Sidecar。`Enter` 发送，`Esc+Enter` 换行，`Tab` 切换焦点，
`Ctrl+L` 清空显示，`Ctrl+Q` 退出；高风险工具等待审批时按 `Y` 或 `N`。

### 展示页与在线 TUI

项目提供“核心设计展示页 + 浏览器 Web Terminal”。公开实例会为每个连接创建一次性
Git 工作区和独立 `MYAGENT_HOME`，只向模型开放 `file_read` 与 `git_read`；会话结束后
自动清理，不共享 Memory，也不允许执行命令、修改代码或沉淀 Skill。

本地启动展示页：

```powershell
python -m pip install -e ".[web,test]"
$env:DEEPSEEK_API_KEY = "..."
uvicorn myagent.web_demo:app --host 127.0.0.1 --port 8080
```

访问 `http://127.0.0.1:8080` 查看设计页，点击“进入在线 TUI”打开终端。Web PTY 仅在
Linux 环境运行；Windows 本地可以预览展示页和健康检查。

Railway 可直接读取仓库根目录的 `railway.toml` 和 `Dockerfile.demo`。在 Railway
项目变量中配置 `DEEPSEEK_API_KEY`，并建议开启 Serverless、设置月度 Hard Limit。
可选限制变量包括：

- `DEMO_MAX_SESSIONS=2`：全站最大并发会话；
- `DEMO_MAX_SESSIONS_PER_HOUR=3`：单 IP 每小时会话数；
- `DEMO_SESSION_SECONDS=900`：单次体验时长；
- `DEMO_TOKEN_LIMIT=30000`：单会话 Provider Token 上限；
- `DEMO_LLM_MODEL=deepseek-flash`：公开体验使用的模型。

部署细节见 [web_demo/README.md](web_demo/README.md)。密钥只在 Railway Variables 中
设置，不能写入 Git 仓库或前端 `config.js`。

DeepSeek 配置写入 `%MYAGENT_HOME%\config.yaml`，密钥只放环境变量：

```yaml
llm:
  provider: deepseek
  model: deepseek-flash
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY

skills:
  strategy: full_system
  candidate_k: 20
  activation_min_score: 80
  max_frozen: 8
  warm_turns: 3
  idle_seconds: 21600

evolution:
  min_canary_trials: 5
  uncached_input_weight: 1.0
  cached_input_weight: 1.0
  output_weight: 1.0
  candidate_retention_days: 30
  quarantine_retention_days: 30
  retired_retention_days: 30
  gc_on_startup: true
  gc_interval_seconds: 86400
```

交互命令：

- `/skill-status`：查看 warming/frozen 阶段、epoch、固化 Skill 与路由 Token；
- `/skill-reset`：显式结束当前 Skill Epoch 并重新预热；
- `/scene code-repair`：指定新场景并重新预热；
- `/learn`：仅在最近一轮通过进化门禁后蒸馏 Skill；
- `/evolution-status`：查看 Memory/Skill 的版本、状态与累计 Token 经济账；
- `/evolution-gc`：立即执行降级、淘汰和到期内容清理；
- `/exit`：退出会话。

## 验证

```powershell
python -m pytest -q
python scripts/evaluate_skill_rag.py
python scripts/evaluate_skill_epoch_cache.py
python scripts/measure_cache.py
python scripts/validate_evolution_benchmark.py
```

- `evaluate_skill_rag.py`：比较全量 Skill 与检索加主模型重排的 Token 和必需命中；
- `evaluate_skill_epoch_cache.py`：确定性验证 Skill 固化的结构性缓存上界；
- `measure_cache.py`：使用实际供应商返回的 cached tokens 观察连续会话；
- `evaluate_skill_epoch_live.py`：同一任务序列下的 Full-System/Scene-Epoch 在线 A/B。
- `validate_evolution_benchmark.py`：离线验证10类30条进化任务的错误版与参考修复；
- `evaluate_evolution.py`：运行 Fixed-Skill/Evolved 双臂在线 A/B，统计迁移成功率、
  Provider Token、工具调用与耗时。

详细设计见：

- [总体架构](docs/architecture.md)
- [记忆信用与生命周期](docs/memory-lifecycle.md)
- [原子权限树](docs/permission-tree.md)
- [Agent Manager](docs/agent-manager.md)
- [Skill 加载评测](docs/skill-loading-evaluation.md)
- [Skill 缓存 Epoch 实验](docs/skill-epoch-system-experiment.md)
- [Scene-Epoch 在线 A/B 报告](docs/scene-epoch-ab-report.md)
- [自进化 A/B 数据集](benchmarks/evolution/README.md)

## 当前边界

- 交互式根 Agent 已接入原子权限树；子 Agent Manager 已实现独立生命周期与权限
  挂载，但尚未接入 CLI 的真实模型循环。
- Python 线程只能协作式取消。需要强制终止时应迁移到独立进程或容器。
- `terminal_exec` 虽受权限树、命令白名单和人工审批共同约束，但进程级文件隔离仍
  需要操作系统沙箱。
- Scene-Epoch 优化后在线 A/B 达到 19/19 必需 Skill 可见率、总输入降低 48.83%、
  未缓存输入降低 9.47%；单次合成评测不足以直接替换默认策略，当前仍以
  `full_system` 作为安全默认值。
