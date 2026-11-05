# MyAgent Public Demo

公开演示由同一个 Railway 服务提供两部分：

1. `/`：静态核心设计页；
2. `/terminal`：xterm.js 浏览器终端；
3. `/ws/terminal`：为每个浏览器启动独立 Linux PTY 与 MyAgent TUI。

## Railway 部署

1. 在 Railway 选择 **New Project → Deploy from GitHub repo**，连接
   `FuYou20/code_agent`；
2. Railway 会根据 `railway.toml` 使用 `Dockerfile.demo`；
3. 在 **Variables** 添加 `DEEPSEEK_API_KEY`，不要把值写入仓库；
4. 在 **Settings → Networking** 生成公开域名；
5. 建议开启 Serverless，并在 Usage 设置可接受的 Hard Limit。

健康检查地址是 `/health`。默认限制为 2 个并发会话、单 IP 每小时 3 次、每次 15
分钟、每会话最多 30,000 Provider Token。

## 安全边界

- 每个连接复制 `demo_workspace` 到随机临时目录，并初始化独立 Git 仓库；
- 权限树根节点只挂载 `file_read` 和 `git_read` 原子，模型看不到其他工具 schema；
- 即使模型输出伪造工具调用，执行前仍按 Agent ID 和真实参数二次鉴权；
- 临时 Agent Home、SQLite 和工作区在断开后删除；
- API Key 只存在服务端进程环境，不发送到网页；
- 并发、时长、输入字节和 Provider Token 均有限额。

这仍然是作品集演示，不是多租户生产沙箱。若将来开放写代码或执行测试，必须把每个
会话迁移到独立容器或 microVM，并增加网络出口控制，不能只依靠应用层权限。
