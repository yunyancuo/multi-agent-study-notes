# 专题五：2025 至 2026 四大主流生产框架运行时、MCP/A2A 协议栈与评测科学

![图 9：现代智能体双层开放协议栈](../assets/fig09-framework-and-protocol-stack.svg)

## 1. MCP 与 Google A2A 协议报文级解剖

### 1.1 Anthropic MCP（垂直层：Agent 调用确定性工具）

MCP 采用标准 JSON-RPC 2.0 报文。当智能体需要执行确定性数值计算或查询结构化资源时，客户端向 MCP Server 发送 `tools/call` 请求：

```json
{
  "jsonrpc": "2.0",
  "id": 101,
  "method": "tools/call",
  "params": {
    "name": "solve_acoustic_vent_geometry",
    "arguments": {
      "canal_mesh_uri": "blob://cases/104/canal_scan.stl",
      "low_freq_threshold_db": 45.0,
      "device_style": "RIC"
    }
  }
}
```

MCP Server 在沙箱内运行确定性算法后，返回结构化结果与资源 URI，全程无大模型幻觉介入。

### 1.2 Google A2A（水平层：跨智能体能力发现与异步任务委派）

当总控智能体需要把一项复杂子任务委派给远端自治专家智能体时，首先读取远端的 `/.well-known/agent.json`（Agent Card），随后发起 `tasks/sendSubscribe` 请求并通过 Server-Sent Events（SSE）接收状态迁移流：

```json
{
  "jsonrpc": "2.0",
  "id": "task_8842",
  "result": {
    "id": "task_8842",
    "status": {
      "state": "input-required",
      "message": {
        "role": "agent",
        "parts": [{ "type": "text", "text": "缺少右耳 4000Hz 骨导阈值，请补充或转人工复核。" }]
      }
    }
  }
}
```

## 2. 四大主流编排框架底层运行时对比

![图 11：2025-2026 四大主流多智能体框架底层运行时对比](../assets/fig11-framework-runtime-internals.svg)

| 评估维度 | LangGraph (LangChain) | AutoGen v0.4 / AG2 (Microsoft) | OpenAI Agents SDK | Anthropic 主从架构模式 |
| :--- | :--- | :--- | :--- | :--- |
| **底层计算模型** | Google Pregel / BSP 超步有向循环图 | 异步事件驱动 Actor 模型 (`autogen-core`) | 原生函数循环 + Handoff 工具转移 | Orchestrator-Workers 动态扇出 |
| **状态一致性保障** | 强类型 `State` + 通道级 `Reducer` | Actor 私有状态 + 事件溯源日志 | `RunContextWrapper` 共享上下文对象 | 主控维护全局计划 + 外部存储快照 |
| **长周期人工介入** | 原生 `Checkpointer` + `interrupt()` | 订阅外部人工审批事件唤醒 Actor | 需在外层业务代码序列化会话状态 | 依赖外部任务队列挂起与恢复 |
| **并发安全护栏** | 条件边守卫节点 | 独立 Observer / Guard Actor | **并行抢占式 Guardrail**（违规即中断主模型） | 独立 Verifier 子智能体闭环 |

## 3. 智能体评测科学：$\tau$-bench 的 $\text{pass}^k$ 悬崖与 Agent-as-a-Judge

![图 10：智能体评测科学](../assets/fig10-evaluation-science-and-judge.svg)

### 3.1 为什么必须同时监控 $\text{pass}@k$ 与 $\text{pass}^k$？

设单次运行成功率为 $p$，连续独立运行 $k$ 次：
* **探索上限指标 $\text{pass}@k = 1 - (1-p)^k$**：衡量模型“有没有潜力解出这道题”（用于强化学习采样或代码生成候选池筛选）。
* **生产交付指标 $\text{pass}^k = p^k$**：衡量系统“能不能稳定可靠地交付生产”（用于临床辅助、金融交易、企业工单）。

对任意 $p < 1$，$\frac{\partial (\text{pass}^k)}{\partial k} = p^k \ln p < 0$。要把 $\text{pass}^8$ 维持在 $80\%$ 以上，单次端到端成功率 $p$ 必须达到 $0.80^{1/8} \approx \mathbf{97.25\%}$。这就要求我们在架构设计上把一切可公理化的计算剥离为确定性 MCP 工具，仅让大模型负责语义解析、模糊匹配与跨模态综合解释。
