# 专题二：六大通信拓扑图论分析与动态图剪枝算法（DyLAN / AgentPrune / MacNet）

![图 3：多智能体六大通信拓扑图谱与复杂度对比](../assets/fig03-six-communication-topologies.svg)

## 1. 多智能体通信图的矩阵表示与谱性质

设包含 $N$ 个智能体的系统在第 $t$ 轮的通信拓扑由有向邻接矩阵 $\mathbf{A}^{(t)} \in \{0, 1\}^{N \times N}$ 描述，其中 $A^{(t)}_{ij} = 1$ 表示智能体 $i$ 在第 $t$ 轮的输出会作为智能体 $j$ 的上下文输入。

在多轮同步交互中，若采用固定邻接矩阵 $\mathbf{A}$，则经过 $R$ 轮迭代后，节点 $i$ 的初始误差或幻觉传播到节点 $j$ 的路径总数由矩阵幂 $(\mathbf{A}^R)_{ij}$ 给出：

* **全连接辩论网（Complete Mesh）**：$A_{ij} = 1 \ (\forall i \neq j)$，其最大特征值 $\lambda_{\max} = N - 1$。经过 $R$ 轮后，单点噪声的扩散路径数按 $(N-1)^R$ 指数级增长。
* **有向无环图（DAG）**：邻接矩阵 $\mathbf{A}$ 可通过拓扑排序置换为严格上三角矩阵，所有特征值均为 $0$，且存在幂零指数 $d \le N$ 使得 $\mathbf{A}^d = \mathbf{0}$。这保证了任何局部噪声最多传播有限步（图直径 $d$），绝不会在环路中无限自激放大。

## 2. 动态图裁剪与拓扑优化三大代表性算法

![图 4：动态智能体网络裁剪原理](../assets/fig04-dylan-agentprune-dag.svg)

### 2.1 DyLAN：基于逆向重要度传播的在线组队（COLM 2024）

DyLAN 将 $T$ 轮协作展开为 $T$ 层网络。在第 $t$ 层，每个存活智能体 $v_{t,i}$ 接收上一层存活节点的输出并生成自身方案，同时对上一层各节点方案的帮助程度打分，经归一化后得到边权重 $w_{t-1, j, i}$（满足 $\sum_j w_{t-1, j, i} = 1$）。

推理结束后的**逆向贡献度分解（Backward Contribution Decomposition）**分为两步：
1. 顶层初始化：若最终被采纳的答案来自第 $T$ 层的节点 $k^*$，则令 $I^{(T)}_{k^*} = 1$，其余顶层节点 $I^{(T)}_i = 0$。
2. 逐层回传：对 $t = T, T-1, \dots, 2$，依次累加每个前驱节点的贡献度：
   $$I^{(t-1)}_j = \sum_{i} w_{t-1, j, i} \cdot I^{(t)}_i$$
3. 汇总每个角色跨时间步的总得分 $S_j = \sum_{t=1}^T I^{(t)}_j$，在验证集上保留得分最高的 Top-$K$ 专家组成精简工作流。

### 2.2 AgentPrune：时空通信边的一阶策略梯度剪枝（ICLR 2025）

在 AgentPrune 中，通信图被拆解为空间邻接矩阵 $\mathbf{A}^{\text{sp}} \in \{0,1\}^{N \times N}$（同轮内依赖）与时间邻接矩阵 $\mathbf{A}^{\text{tm}} \in \{0,1\}^{N \times N}$（跨轮次上下文继承）。系统为每条潜在边赋予参数化概率 $\mathbf{P} = \sigma(\boldsymbol{\Theta})$，每次前向推理按伯努利采样生成稀疏掩码图 $G \sim \text{Bernoulli}(\mathbf{P})$。

记任务效用函数（如验证器通过率减去 Token 惩罚）为 $R(G, x)$，则掩码参数 $\boldsymbol{\Theta}$ 的 REINFORCE 梯度估计为：

$$\nabla_{\boldsymbol{\Theta}} \mathcal{J}(\boldsymbol{\Theta}) = \mathbb{E}_{G \sim \text{Bernoulli}(\sigma(\boldsymbol{\Theta}))} \left[ (R(G, x) - b) \cdot \nabla_{\boldsymbol{\Theta}} \ln \Pr(G \mid \boldsymbol{\Theta}) \right]$$

其中 $b$ 为移动平均基线。仅需在少量样本上进行数步单次更新，然后对 $\sigma(\Theta_{ij}) < \tau$ 的边执行硬截断，即可裁除 50% 至 70% 的冗余历史传递，并显著提升抗对抗注入攻击能力。

### 2.3 MacNet：千级智能体 DAG 的小世界效应（ICLR 2025）

MacNet 探讨了当节点数 $N$ 从 $10^1$ 扩展至 $10^3$ 时不同拓扑家族的表现差异：
* **规则链与环形图**：平均路径长度 $L_{\text{path}} \propto N$，早期节点发现的关键约束在经过几十次转述后发生严重信息损耗（Information Decay）。
* **随机图（Erdős–Rényi）**：虽然平均路径短，但缺乏局部高密度交叉核验子图（Clustering Coefficient 低），局部错误容易未经校验直接汇入终点。
* **小世界有向无环图（Small-World DAG）**：同时具备高局部聚类系数（3 至 4 个紧密耦合的领域子智能体先在局部子图内完成生成与验证闭环）和短全局路径长度（通过跨簇长程边在 $O(\log N)$ 步内汇入总控），在大规模协作基准上取得了最优的精度-成本帕累托前沿。
