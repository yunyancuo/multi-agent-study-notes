# 专题四：多智能体上下文工程、MAST 失效分类学与分布式容错模式

![图 6：多智能体上下文工程](../assets/fig06-memory-and-context-engineering.svg)

## 1. UC Berkeley MAST 14 类失效模式与工程防御映射

![图 7：UC Berkeley MAST 多智能体失效分类学](../assets/fig07-mast-failure-taxonomy.svg)

加州大学伯克利分校（Cemri et al., 2025, `arXiv:2503.13657`）对 1,642 条多智能体运行轨迹的实证统计证明：多智能体系统的崩溃不是单一原因造成的，而是分布在**规约设计（37%）**、**跨节点协同（34%）**与**验证终止（29%）**三个阶段。下表给出了全部 14 项失效模式对应的架构级根治手段：

| MAST 编号 | 失效模式名称 | 典型故障表现 | 架构级根治方案（非 Prompt 修补） |
| :--- | :--- | :--- | :--- |
| **FM-1.1** | 违背任务或角色规约 | 分析节点越权直接生成最终处方，或输出未按要求格式化 | 绑定强类型 `Pydantic Output Schema` + 最小权限工具白名单 |
| **FM-1.2** | 步骤重复与死循环 | 两个节点互相礼貌推诿，或反复以相同参数调用报错接口 | 在消息信封强制注入 `hop_budget` 跳数预算 + 相同调用哈希去重熔断 |
| **FM-1.3** | 历史上下文关键约束丢失 | 历经 4 棒交接后，用户最初声明的预算上限或禁忌条件被截断 | 将核心约束提取至全局不可变 `ImmutableConstraints` 顶层状态字段 |
| **FM-1.4 / 1.5** | 停止准则模糊 / 丧失终止意识 | 已达成目标仍继续无意义润色，或遇到轻微阻碍即放弃 | 由外部确定性状态守卫函数（Guard Predicate）决定何时路由至 `END` |
| **FM-2.1** | 对话意外重置 | 下游节点忽略上游已提取的结构化字段，重新向用户索要 | 统一从全局 `TypedDict State` 投影读取已验证字段，禁止裸读聊天流 |
| **FM-2.2** | 漏问澄清与盲目假设 | 关键必填项缺失时不中断追问，而是自行编造默认值往下跑 | 前置 Schema 完备性网关：必填项为 `None` 时强制触发 `interrupt()` |
| **FM-2.3** | 偏离主线任务 | 多轮分支讨论后遗忘最初的任务目标 | 每次子智能体调用时，在系统提示词头部固定锚定 `root_objective` |
| **FM-2.4 / 2.5** | 隐瞒负向观测 / 忽略审校批注 | 上游丢弃异常指标，或下游重算时原样输出旧方案无视批注 | 引入轨迹差异比对：重算输出与上一版距离过近时直接拦截打回 |
| **FM-2.6** | 推理链与实际工具调用脱节 | 思考链写“将增益下调 5dB”，实际传给工具的 JSON 仍是原值 | 启用原生 `Strict Tool Use` 并由后置钩子比对状态增量（State Diff） |
| **FM-3.1** | 过早宣告完成 | 尚未通过仿真验证便直接向前端返回最终完成状态 | 收回生成节点的终止权，仅有独立 Verifier 节点拥有流向 `END` 的边 |
| **FM-3.2 / 3.3** | 验证不完整 / 裁判自身幻觉 | 裁判大模型只看文字通顺度，漏检数值越界或放行错误方案 | 采用 **Agent-as-a-Judge + 确定性规则引擎** 双重校验 |

## 2. 分布式容错三件套：Context Envelope、Checkpointing 与 Saga 补偿事务

![图 8：分布式多智能体容错工程](../assets/fig08-distributed-reliability-patterns.svg)

### 2.1 Saga 逆向补偿事务的参考实现

当长链路中的第 $k$ 个节点验证失败、需要回退到上游节点重新选型时，必须按逆序清理沿途已产生的外部状态副作用：

```python
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

@dataclass
class SagaStep:
    name: str
    forward_action: Callable[[Dict[str, Any]], Dict[str, Any]]
    compensate_action: Callable[[Dict[str, Any]], None]

class SagaOrchestrator:
    def __init__(self, steps: List[SagaStep]):
        self.steps = steps

    def execute_with_rollback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        completed_steps: List[SagaStep] = []
        try:
            for step in self.steps:
                state = step.forward_action(state)
                completed_steps.append(step)
            return state
        except Exception as exc:
            # 逆序执行补偿事务，清理已创建的外部副作用并记录失败约束
            for step in reversed(completed_steps):
                step.compensate_action(state)
            state.setdefault("negative_constraints", []).append(str(exc))
            raise
```
