# 专题一：多智能体第一性原理、信息论边界与 CoALA 认知架构

![图 1：多智能体第一性原理](../assets/fig01-first-principles-scaling.svg)

## 1. 为什么超长上下文窗口不能替代多智能体？

### 1.1 自注意力熵增与信噪比退化推导

给定长度为 $L$ 的上下文序列，单头自注意力在第 $i$ 个位置的输出表示为：

$$o_i = \sum_{j=1}^{L} \alpha_{ij} v_j, \quad \alpha_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{m=1}^{L} \exp(q_i^\top k_m / \sqrt{d_k})}$$

设序列中仅有集合 $S^* \subset \{1, \dots, L\}$（规模 $|S^*| = s \ll L$）包含当前子任务所需的有效证据，其余 $L - s$ 个位置为其他子任务的历史搜索草稿与报错噪声。定义有效注意力质量（Effective Attention Mass）为：

$$M_{\text{eff}}(L) = \sum_{j \in S^*} \alpha_{ij} = \frac{\sum_{j \in S^*} \exp(z_{ij})}{\sum_{j \in S^*} \exp(z_{ij}) + \sum_{m \notin S^*} \exp(z_{im})}$$

假设噪声位置的 logit $z_{im}$ 服从均值为 $\mu_n$、方差为 $\sigma_n^2$ 的次高斯分布，根据对数和指数（LogSumExp）的期望性质，分母中的噪声项随噪声长度 $(L-s)$ 呈线性增长：

$$\mathbb{E}\left[\sum_{m \notin S^*} \exp(z_{im})\right] = (L - s) \exp\left(\mu_n + \frac{\sigma_n^2}{2}\right)$$

这意味着，即使模型对有效证据 $j \in S^*$ 打出了很高的相关性内积 $z_{ij}$，随着无关上下文长度 $L$ 增长一个数量级（例如从 $4\text{K}$ 增长到 $128\text{K}$），有效注意力质量 $M_{\text{eff}}(L)$ 也会按 $O(1/L)$ 衰减。这就是多智能体进行**子空间上下文隔离（Subagent Context Isolation）**的第一性原理：通过将总长度为 $L$ 的混杂轨迹切分为 $K$ 个互不干扰的局部窗口 $L_k \approx L/K$，每个子智能体内部的噪声分母缩小了 $K$ 倍，局部有效注意力信噪比随之显著回升。

### 1.2 孔多塞陪审团定理在同源大模型中的失效分析

经典的孔多塞陪审团定理（Condorcet, 1785）假设 $N$ 个独立分类器 $Y_1, \dots, Y_N \in \{-1, +1\}$ 对真值 $y^* = +1$ 的正确率均为 $p = \Pr(Y_i = +1) > 0.5$。当 $Y_i$ 相互独立时，由霍夫丁不等式（Hoeffding's Inequality），多数票决出错的概率随 $N$ 指数衰减：

$$\Pr\left(\frac{1}{N}\sum_{i=1}^N Y_i < 0\right) \le \exp\left(-2N(p - 0.5)^2\right)$$

然而，当 $N$ 个智能体共享同一个预训练权重 $\theta$（仅改变角色提示词）时，任意两个智能体出错指示变量 $E_i = \mathbb{I}(Y_i \neq y^*)$ 之间存在正相关系数 $\rho = \text{Corr}(E_i, E_j) > 0$。此时群体平均误差率 $\bar{E} = \frac{1}{N}\sum_{i=1}^N E_i$ 的方差为：

$$\text{Var}(\bar{E}) = \frac{p(1-p)}{N} + \frac{N-1}{N} \rho \cdot p(1-p) \xrightarrow{N \to \infty} \rho \cdot p(1-p) > 0$$

只要同源模型的预训练误差相关性 $\rho > 0$，即便把智能体数量 $N$ 堆到无穷大，群体误差方差也存在一个无法突破的下界 $\rho p(1-p)$。工程结论非常明确：**想要靠多智能体投票提升下限，必须引入异构信息源（如一个智能体查关系型数据库、一个跑物理仿真器、一个做数值优化），从而在物理上把误差相关系数 $\rho$ 压向 $0$ 甚至负值。**

## 2. CoALA 认知架构与四代推理循环实现对比

![图 2：单智能体认知架构与三级记忆存储层次](../assets/fig02-coala-memory-hierarchy.svg)

普林斯顿 CoALA 框架将单智能体的内部动作空间（Action Space）严格划分为两类：
* **外部动作（External Grounding Actions）**：与外部环境或 MCP 工具交互，改变环境状态并获取 Observation。
* **内部动作（Internal Cognitive Actions）**：
  * **检索（Retrieval）**：从长期记忆（Episodic / Semantic / Procedural）读取信息写入 Working Memory；
  * **推理（Reasoning）**：在 Working Memory 内部通过大模型对已有信息进行归纳、规划或反思；
  * **学习（Learning）**：将 Working Memory 中经过验证的新经验固化写回长期记忆库。

### 2.1 ReAct vs. ReWOO 的前缀 Token 复杂度对比

设完成一项任务需要调用 $M$ 次外部工具，初始系统提示词与任务描述长度为 $P$，每次工具调用的思考与参数长度为 $A$，每次工具返回的观察值长度为 $O$。
* **在 ReAct 循环中**，第 $m$ 步请求大模型时的输入上下文包含了前 $m-1$ 步的全部累积历史，总输入 Token 消耗为：
  $$\text{Cost}_{\text{ReAct}}(M) = \sum_{m=1}^{M+1} \left[ P + (m-1)(A + O) \right] = (M+1)P + \frac{M(M+1)}{2}(A + O)$$
* **在 ReWOO（规划与观察解耦）架构中**，Planner 仅需 1 次调用生成含变量引用的 DAG 计划，Worker 在模型外执行 $M$ 次工具调用，最后 Solver 汇总初始问题与 $M$ 个工具输出作答，总输入 Token 消耗为：
  $$\text{Cost}_{\text{ReWOO}}(M) = \underbrace{P}_{\text{Planner}} + \underbrace{(P + M \cdot A + M \cdot O)}_{\text{Solver}} = 2P + M(A + O)$$

当工具调用步数 $M = 8, P = 1500, A+O = 800$ 时，ReAct 需消耗 $42,300$ 输入 Token，而 ReWOO 仅消耗 $9,400$ 输入 Token，节省比例高达 **77.8%**。这解释了为什么在子任务流程相对固定的节点内部，应当优先采用“一次规划 + DAG 批量执行”而非无限制的自由 ReAct 循环。
