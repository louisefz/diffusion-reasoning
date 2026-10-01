# ELF-B 内部推理机制：第一阶段报告

日期：2026-09-19

## 核心结论

在当前 randomized function-composition 模型中，推理深度主要反映在
**单次 denoiser forward 的 Transformer 深度**，而不是 flow/ODE 时间。

更具体地说：

1. 最终答案在 flow 很早期即可由完整 denoiser 预测，且这一时刻不随
   composition depth 系统变化；因此不能把更多 flow steps 直接解释为更多推理。
2. 在同一次 denoiser forward 内，答案的线性可读性随任务深度逐层后移：
   lookup 和 compose-d1 在第 9 层出现，compose-d2 在第 10 层出现，
   compose-d4 在第 11 层出现。
3. 因果 patching 得到几乎相同的阶梯：把 donor 的答案槽表示写入模型后，
   compose-d1/d2 与 lookup 从第 10 层起切换为 donor 答案；compose-d4 要到
   第 11 层才切换。
4. 对四步计算的真实中间值做 probe 时，s1 很弱，s2/s3 和最终答案集中在
   第 10--11 层出现。这不支持“每一层依次显式保存一个计算步骤”的简单图景；
   当前证据更符合分布式计算在后部层快速整合并定型答案。

因此，当前最可辩护的表述是：

> Reasoning depth is reflected in where answer-relevant computation becomes
> readable and causally controlling across denoiser layers, rather than in when
> the continuous flow reaches the answer.

这还不能证明每一层对应一个 symbolic step，也不能证明模型采用了人类式算法。

## 1. Layer-wise linear probe

实验只分析第一次 conditional denoiser forward：`t=0`，self-conditioning 为零。
在 answer slot 记录 input projection 和 12 个 Transformer blocks 的表示，并用
独立 held-out 数据训练共享的低容量 Ridge probes。

达到 90% held-out accuracy 的首层：

| 任务 | 首个 >=90% 的层 |
|---|---:|
| compose d1 | 9 |
| compose d2 | 10 |
| compose d4 | 11 |
| lookup d4 | 9 |

关键准确率：

| 任务 | layer 9 | layer 10 | layer 11 | layer 12 |
|---|---:|---:|---:|---:|
| compose d1 | 1.000 | 1.000 | 1.000 | 1.000 |
| compose d2 | 0.265 | 1.000 | 1.000 | 1.000 |
| compose d4 | 0.285 | 0.240 | 1.000 | 1.000 |
| lookup d4 | 1.000 | 1.000 | 1.000 | 1.000 |

lookup-d4 与 compose-d4 具有相同的表格上下文规模，但前者不需要四步依赖链；
两层的差距因而不能简单归因于输入长度或词汇格式。

产物：`runs/official-layer-probe-full-62073605/`

## 2. Causal answer-slot patching

对于同一任务组的 source/donor 样本，在每一层把 source 的 answer-slot hidden
state 替换为 donor state，再继续执行余下网络并使用原生 decoder 解码。

- 不同答案 donor：用于测试该层表示是否已能因果决定答案 identity。
- 相同答案 donor：用于控制 patch 本身造成的 distribution shift。

结果：

| 任务 | donor 答案开始控制输出的层 | 代表性 donor rate |
|---|---:|---:|
| compose d1 | 10 | 1.000 |
| compose d2 | 10 | 0.935 |
| compose d4 | 11 | 1.000 |
| lookup d4 | 10 | 1.000 |

所有同答案 controls 在所有层均保持 100% 正确。compose-d4 在 layer 10 patch 后仍
100% 输出 source 答案，而 layer 11 patch 后 100% 输出 donor 答案，形成非常清晰的
因果边界。

一个重要反例是 compose-d1：layer 9 已可被线性 probe 100% 读出，但 donor patch
尚不能改变输出；下游层会恢复 source 答案。因此本实验直接展示了：

`decodable != causally used`。

产物：`runs/official-layer-patch-full-62073661/`

## 3. Ground-truth intermediate-state probes

针对 compose-d4，分别 probe 真值计算链
`s1 = f1(x), s2 = f2(s1), s3 = f3(s2), s4 = f4(s3)`。

各状态的最佳 held-out accuracy：

| 状态 | 最佳准确率 | 最佳层 |
|---|---:|---:|
| s1 | 0.365 | 10 |
| s2 | 0.885 | 11 |
| s3 | 0.915 | 11 |
| s4（答案） | 1.000 | 11--12 |

八分类随机水平为 0.125。s2 在 layer 10 已达 0.650，s2/s3 在 layer 11 同时变得
高度可读，而 s1 始终较弱。这个结果排除了最简单的“第 8 层保存 s1、第 9 层保存
s2……”叙述，但不能排除以下可能：

- 中间值编码在其他 token positions；
- 中间值以非线性或分布式方式编码；
- 模型采用并行/压缩算法，而非逐步模拟标准计算链；
- probe 读到的是与中间值相关、但未被后续计算使用的特征。

产物：`runs/official-state-probe-full-62073734/`

## 4. 与 flow-level 实验合并后的解释

先前的 flow trajectory 与 perturb-and-rollout 实验显示：

- 完整 denoiser 很早即可预测正确 endpoint；
- 增加 flow time 主要改善 latent-to-token lexical compatibility；
- 即使对早期 `z`/self-conditioning 施加强扰动，模型仍能恢复正确答案；
- 没有发现稳定的 reasoning-depth-dependent flow commitment time。

结合本报告的 layer results，最简解释是：每次 denoiser 调用都重新读取问题条件，
并在其后部 Transformer blocks 内完成/恢复答案计算；continuous flow 主要负责把
状态搬运到可被原生 token interface 解码的位置。

这意味着当前项目不应优先增加 ODE NFE 来获得推理能力，而应研究和扩展
denoiser 内部的 semantic compute。

## 5. 下一步机制实验

### 已完成：attention-vs-MLP component patching

job 62073773 对 blocks 9--11 的 attention output 与 MLP output 分别进行了
answer-slot donor patch（每组 200 条，共 9,600 次干预）。不同答案 donor 的
输出切换率如下：

| 任务 | block | attention donor rate | MLP donor rate |
|---|---:|---:|---:|
| compose d1 | 10 | 0.710 | 0.015 |
| compose d2 | 10 | 0.795 | 0.015 |
| compose d4 | 10 | 0.000 | 0.000 |
| lookup d4 | 10 | 0.765 | 0.000 |
| compose d1 | 11 | 0.910 | 0.990 |
| compose d2 | 11 | 1.000 | 0.985 |
| compose d4 | 11 | 1.000 | 0.985 |
| lookup d4 | 11 | 0.945 | 0.980 |

block 9 的所有不同答案 component patches 均不能改变答案；所有 same-answer
controls 在所有组件和层均保持 100% 正确。

该结果说明 block 10 attention output 已携带浅层任务的因果有效答案信号，而
compose-d4 的相同转变延迟到 block 11；到 block 11 时 attention 和 MLP output
均携带足以控制输出的答案信息。它与整层 patching 的深度阶梯一致。

需要注意：donor component patch 测量的是一个组件输出是否携带**足够**的答案
信息，不等价于该组件在未干预运行中单独完成了计算。要区分“产生答案”与“转发
答案”，仍需 component ablation、path patching 或 activation replacement chain。

产物：`runs/official-component-patch-full-62073773/`

### 尚待完成

接下来优先定位答案信息从哪些 token positions 汇入 attention：

1. **Position-wise patching/probing**：扫描 table entries、program tokens、answer
   slot，判断 s1/s2 是否暂存在其他位置，再被汇入答案槽。
2. **Attention path patching**：在 block 10/11 定位哪些 source positions 和 heads
   向 answer slot 输送答案相关信息。
3. **Intermediate-state causal interventions**：不只 probe s1--s3，而是编辑其候选
   subspace，检查后续答案是否按函数链预测发生变化。

### 全位置 latent-state map（job 62074219）

进一步在 10 类语义位置上追踪了 `s1,s2,s3,s4`：start、四个 program tokens、
四个 ground-truth execution path 实际访问的 table cells，以及 answer slot。实验使用
4,096 条训练样本拟合固定 `Ridge alpha=1` 的 probes，并在 200 条独立 held-out
compose-d4 上评估，共 520 个 probes。

四个 `oracle_cell_k` 是输入锚点：它们的选择使用了 ground-truth path，cell token
本身又直接写着函数输出，因此高准确率只用于验证位置解析，**不能**当作 latent
reasoning evidence。剔除这些锚点后的主要结果为：

| 被预测状态 | 最强非 oracle 位置 | 层 | held-out accuracy |
|---|---|---:|---:|
| s1 | answer | 10 | 0.365 |
| s2 | answer | 11 | 0.880 |
| s3 | answer | 11 | 0.910 |
| s4 | answer | 11 | 1.000 |

在 program positions 上没有观察到 `s1 -> s2 -> s3` 按程序位置依次移动：s1--s3
的最佳 program-position accuracy 分别只有 0.250、0.280、0.195。相反，s2、s3
与最终答案在 block 11 的 answer slot 同时高度可读。到 block 12，最终答案才广泛
出现在 start/program positions（0.855--0.990）。

当前结果因此不支持显式 token-to-token latent CoT；它更符合以下图景：

`distributed input processing -> compressed answer-slot computation at block 11 -> final-answer broadcast at block 12`。

这仍然只是 decodability map。下一项必须使用最小 counterfactual table edits 和
path patching，验证 block-11 answer-slot 中的 s2/s3 是否参与了真实计算，还是仅为
与最终答案共同出现的可读 correlates。

产物：`runs/official-position-state-full-62074219/`

### 最小反事实 causal patching（job 62074267）

为了区分 readable correlate 与实际使用的表示，构造了 600 个 in-distribution
最小反事实对，每个干预步骤 150 对。每个反事实只在第 k 步实际访问的函数表中
交换两个输出，并满足：

- 函数仍是合法 permutation；
- 第 k 步之前的所有 states 与原题完全相同；
- `s_k` 和最终答案均改变；
- 原题与反事实题的 first-denoiser endpoint accuracy 均为 100%。

把反事实题的 answer-slot activation 移植到原题后：

| 被改变的状态 | layer 10 跟随反事实 | layer 11 跟随反事实 | block 11 attention | block 11 MLP |
|---|---:|---:|---:|---:|
| s1 | 0.0% | 100% | 100% | 99.3% |
| s2 | 0.0% | 100% | 100% | 98.7% |
| s3 | 0.0% | 100% | 100% | 99.3% |
| s4 | 0.0% | 100% | 100% | 98.7% |

blocks 0--9 同样全部为 0%；layer 10 只有 s1 组出现 0.7% other answers，仍无
反事实答案。该结果证明：由单个函数表变化所导致的新答案 identity，在 block 11
的 answer-slot representation 及其 attention output 中成为因果充分的信息。

与此同时，不论改变计算链的第 1、2、3 或第 4 步，因果边界都完全相同。因此
结果并不支持可分离的逐步 latent CoT；它更强地支持四步依赖在 block 11 被集中
整合为一个 causally controlling answer representation。

严格说，这证明的是 counterfactual answer identity 的集中形成，而非模型内部完全
没有较早的分布式计算。进一步定位早期计算需要 source-position/head-level path
patching，而不能只观察 answer slot。

产物：`runs/official-counterfactual-full-62074267/`

### Block-11 attention-head localization（job 62075181）

将 block 11 的 12 个 attention heads 在 projection 前分别进行 paired-counterfactual
patching，并同时测量：

- **single-head sufficiency**：仅将一个 source head 换成 counterfactual head；
- **leave-one-source necessity**：其余 11 个 heads 全部换成 counterfactual，唯独保留
  指定 source head。

结果高度稀疏：只有 head 3 单独具有显著反事实切换能力。

| 改变的状态 | 仅 patch head 3 | 保留 source head 3、替换其余 11 heads |
|---|---:|---:|
| s1 | 74.7% | 0.0% |
| s2 | 79.3% | 0.7% |
| s3 | 80.0% | 0.0% |
| s4 | 75.3% | 0.0% |

其余 11 个 heads 单独 patch 的反事实切换率全部为 0%。保留 source head 1 时仍有
76.0%--82.7% 切换，说明 head 1 对 donor computation 有一定对抗/调制作用，但它
自身并不具备单头反事实充分性。完整 12-head donor patch 为 100%。

因此 block-11 answer integration 并非均匀分布在全部 heads，而是主要通过 head 3
形成一个低维、因果主导的 bottleneck。下一步需要对 head 3 的 value path 做
source-position patching，确定它从哪些 table/program positions 汇聚信息。

产物：`runs/official-head-patch-full-62075181/`

### Head-3 semantic source-position patching（job 62075186）

在 600 个相同最小反事实对上，只替换 block-11 head-3 的 value vectors，并按
语义位置对齐 source/counterfactual tokens。测试位置包括：实际改变的 table cell、
swap partner、两者、全部 32 个 table cells、program、start、answer 和全部语义
位置。

结果出现明显的步骤差异：

| 反事实编辑 | changed-cell V patch | all-table V patch | program/start/answer V patch |
|---|---:|---:|---:|
| s1 | 0.0% | 0.7% | 0.0% |
| s2 | 0.0% | 0.0% | 0.0% |
| s3 | 0.0% | 0.0% | 0.0% |
| s4 | 76.0% | 75.3% | 0.0% |

对 s4，单独 patch 最后一步实际访问 cell 已与 patch 全部 table cells 等效，说明
head 3 存在一条从 final lookup cell 到 answer slot 的直接 value path。swap partner
本身为 0%，进一步排除了“只要 patch 被交换的任一 token 就行”的解释。

对 s1--s3，即使替换全部 table-cell values 也不能转移反事实答案。这说明较早步骤
的影响不以 block-11 原始 table-cell V vectors 的形式存在；可能性收敛为：

1. counterfactual 改变了 head-3 query/key routing，而非只改变 values；或
2. 较早步骤已在 blocks <=10 中被压缩到其他分布式位置，head 3 最终读取的是
   该压缩结果。

下一项应对 block-11 head 3 做 Q/K/V factorial patching，并随后向 block 10 回溯。

产物：`runs/official-head-position-full-62075186/`

### Head-3 Q/K/V factorial decomposition（job 62075643）

在 600 个最小反事实对上，分别 patch block-11 head-3 的 answer query、全序列
Keys、全序列 Values 及其组合。为排除答案 `0` 的双-token位置偏移，主要结论使用
source/counterfactual answer position 相同的子集（每组 112--119 对）。

| 编辑状态 | K only | V only | K+V | Q only |
|---|---:|---:|---:|---:|
| s1 | 91.1% | 0.9% | 99.1% | 0.0% |
| s2 | 91.6% | 0.0% | 100% | 0.0% |
| s3 | 94.1% | 0.0% | 99.2% | 0.0% |
| s4 | 0.0% | 98.3% | 98.3% | 0.0% |

加入 donor Q 不改变对应 K/V 结果。因此 head 3 内部存在明确功能分解：

- 前三步 counterfactual 主要通过 **Keys 改变 attention routing**；
- 最后一步 counterfactual 主要通过 **Values 改变被读取的答案内容**；
- answer query 在 paired counterfactual 间不是答案差异的载体。

产物：`runs/official-head-qkv-full-62075643/`

### Key routing 位于最后一张函数表（job 62075653）

进一步把 32 个 table positions 的 head-3 Keys 按函数表分组。对于编辑 s1--s3，
仅 patch 第四步即将读取的 final-function table 的 8 个 Keys，与 patch 全部 32 个
table Keys 的结果逐样本聚合完全相同：69.3%/76.0%/76.0%。

为排除某个早期 program function 与 final function 重复的混淆，进一步只保留
`edited function != final function` 的样本：

| 编辑状态 | 样本数 | edited-function table K | final-function table K | all-table K |
|---|---:|---:|---:|---:|
| s1 | 120 | 0.0% | 75.0% | 75.0% |
| s2 | 108 | 0.0% | 81.5% | 81.5% |
| s3 | 112 | 0.0% | 80.4% | 80.4% |

因此可以得到目前最具体的机制图景：

```text
first three function applications
          ↓  (blocks <= 10)
counterfactual-dependent key pattern
written across the 8 cells of the final function table
          ↓  (block 11, attention head 3)
select one final-table cell
          ↓  (value path)
read the final answer into the answer slot
```

这不是逐 token 的显式 CoT，而是一条可因果干预的 **latent routing computation**：
前三步被压缩成最终查表地址，最后一步由一个稀疏 attention head 执行寻址与读取。
尚未解决的是 blocks <=10 如何从 s1 到 s3 形成该地址；下一层回溯应研究 block-10
写入 final-table Keys 的 upstream components。

产物：`runs/official-head-key-position-full-62075653/`

### Routing information migrates across layers（job 62075713）

为回溯 final-table Key pattern 的形成过程，在 layer 0--10 分别将 paired
counterfactual residual states patch 到：edited-function table、final-function
table 或全部 tables。以下结果排除了 `edited function == final function` 的样本。

Final-function table patch 的反事实切换率：

| 编辑状态 | L8 | L9 | L10 |
|---|---:|---:|---:|
| s1 | 0.0% | 25.0% | 75.8% |
| s2 | 0.0% | 0.0% | 81.5% |
| s3 | 0.0% | 80.4% | 80.4% |

Edited-function table patch 呈互补变化：

| 编辑状态 | L8 | L9 | L10 |
|---|---:|---:|---:|
| s1 | 75.8% | 18.3% | 0.0% |
| s2 | 81.5% | 74.1% | 0.0% |
| s3 | 80.4% | 0.0% | 0.0% |

All-table patch 在所有层保持约 75%--82%，说明 patch 方法持续保留同一反事实
信息；变化来自信息的因果位置迁移，而非整体 patch 强度消失。

该结果给出了目前最接近 latent chain of computation 的证据：

- edit s3（只剩一次函数应用）的影响在 block 9 后迁移到 final table；
- edit s2（剩两次应用）在 block 10 后完成迁移；
- edit s1（剩三次应用）在 layer 9 呈 source/final 分布式过渡，到 layer 10 完成。

因此 blocks 9--10 并非只在静态地编码答案，而是在执行可因果追踪的信息转移，
把上游函数结果逐步转换成 final-table lookup address。下一步应在 blocks 9/10 内
做 attention-vs-MLP table-position patching，定位执行迁移的具体组件。

产物：`runs/official-routing-layer-full-62075713/`

### Flow time × layer causal map（job 62082063）

为区分“答案在早期 flow 中一次形成并被持续携带”与“每个 denoiser step
重新计算/刷新答案”，在 200 对最小 table-edit counterfactual 上使用 8-step
uniform ODE，并在 block 8--12 的 answer slot 做两类干预：

- `pulse`：仅在指定 flow step patch 一次 donor residual，随后恢复原模型；
- `sustained`：从指定 flow step 起，在余下每一步持续 patch 同一 block。

原题和 counterfactual 的完整 ODE baseline 均为 100%。聚合所有 blocks 后，
single-pulse 的最终 donor-answer rate 随时间为：

| flow time | 0.000 | 0.125 | 0.250 | 0.375 | 0.500 | 0.625 | 0.750 | 0.875 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| donor rate | 17.8% | 46.7% | 68.4% | 74.3% | 75.6% | 76.4% | 78.4% | 100% |

在 `t=0`，blocks 8/9/10 的 pulse switch 均为 0%，block 11/12 分别为
46.5%/42.5%；这与此前 block-11 causal boundary 一致。早期 pulse 经后续
denoiser steps 后多数被修复，而越接近终点越难被修复。相反，sustained patch
跨全部 cell 的平均 donor rate 为 98.9%。

因此当前结果不支持“早期答案一旦写入 flow state 就永久固定”的简单图景；更支持：

```text
each denoiser call recomputes / refreshes answer-related state
                  ↓
an isolated early perturbation is corrected by later calls
                  ↓
repeated or late perturbation controls the final flow endpoint
```

这说明 continuous flow 与内部 reasoning 不是完全独立：late blocks 在每个 flow
step 提供语义更新，而 flow state 对这些重复更新进行积分。但实验 patch 的是
answer-slot residual，因此 sustained-patch 的高控制率本身不能证明每一步都执行
完全相同的算法；下一步应做 **semantic-cache/reuse ablation**，直接测试复用一次
late-block routing state 能否替代后续重复计算。

产物：`runs/official-flow-layer-causal-full-62082063/`

### Head-3 temporal causal switching（job 62082273）

二维 residual patch 表明早期单次扰动会被修复，但它没有回答具体 K/V computation
在什么时候具有因果控制力。为此，在 200 对 counterfactual 上固定 8-step uniform
flow，针对 block-11 head 3 做三个双向 sustained-switch protocols：

- `source_to_donor`：switch 前使用 native source computation，之后每步注入 donor；
- `donor_to_native`：switch 前每步注入 donor，之后恢复 native computation；
- `donor_to_source`：switch 前注入 donor，之后每步显式恢复 clean source QKV。

对 edits s1--s3 聚合 head-3 Keys，最终 donor-answer rate 为：

| switch t | 0.000 | 0.125 | 0.250 | 0.375 | 0.500+ |
|---|---:|---:|---:|---:|---:|
| source→donor K | 72.7% | 34.0% | 4.0% | 0.0% | 0.0% |
| donor→native K | 0.0% | 19.3% | 68.7% | 72.7% | 72.7% |
| donor→source K | 0.0% | 10.7% | 64.0% | 72.0% | 72.7% |

对 edit s4 的 Values：

| switch t | 0.000 | 0.125 | 0.250 | 0.375 | 0.500 | 0.625+ |
|---|---:|---:|---:|---:|---:|---:|
| source→donor V | 76.0% | 68.0% | 32.0% | 6.0% | 2.0% | 0.0% |
| donor→native V | 0.0% | 20.0% | 72.0% | 76.0% | 76.0% | 76.0% |
| donor→source V | 0.0% | 18.0% | 72.0% | 76.0% | 76.0% | 76.0% |

该实验揭示明确的 **early semantic commitment window**：

- routing Keys 的主要因果窗口约为 t=0--0.25；
- final-value readout 的可接管窗口稍宽，约延伸到 t=0.5；
- donor K/V 控制最初两个 flow steps 后，后续 native 或显式 source restoration
  几乎不能恢复原答案；
- 若错过早期窗口，后期持续注入同一 donor component 也难以改变答案。

因此更精确的机制不是“所有 flow steps 等权重复 reasoning”，而是：late blocks
每步仍产生语义更新，但其 **causal leverage strongly front-loaded**。早期 K routing
与 V readout 把 trajectory 推入一个 solution basin，后期 computation 主要在已经
选定的 basin 内 refinement。需要注意，`donor_to_source` 使用 source-baseline QKV
注入到已经偏离的 trajectory，存在 state mismatch；不可单独解释为严格不可逆性。

这一结果把加速假设进一步具体化：完整 semantic computation 应保留在早期 commitment
window；t>0.5 后优先尝试 cache、skip 或 shallow refinement，而不是均匀削减所有
flow steps。

产物：`runs/official-temporal-switch-full-62082273/`

### Late-flow semantic cache equivalence（job 62082294）

基于 temporal commitment window，在 800 条 held-out examples（lookup、compose
d1/d2/d4 各 200）上测试 inference-legal cache：在 cutoff 时刻保存当前样本自身的
block-11 head-3 text-position K/V，之后复用；model/time/self-conditioning prefix
保持 native。比较 static K、V、K+V cache，2/4-step periodic refresh，以及晚期
将 head 3 输出置零。cutoff 为 t=.25/.5/.625，8-step uniform flow。

Native exact accuracy 为 d1 98.0%、d2 99.5%、d4 100%、lookup 99.5%。所有 cache
和 periodic conditions 在三个 cutoff、四个任务上均与 native answer token **100%**
一致，exact accuracy 与 native 完全相同。更强的 zero-head-3 ablation 也保持 100%
answer-token agreement；d4 始终为 100% exact，其他组仅在 multi-token exact
细节上出现最多 +0.5% 的非负变化。

最终 answer-state L2 显示 cache 与 native trajectory 极接近：

| cutoff | cache K（各组范围） | cache K+V（各组范围） | zero head 3（各组范围） |
|---|---:|---:|---:|
| .25 | .010--.019 | .039--.069 | .885--1.282 |
| .50 | .007--.011 | .024--.040 | .815--1.142 |
| .625 | .005--.008 | .016--.025 | .787--1.070 |

因此不只是 stale K/V 可以复用；即便 late head-3 state 明显改变，答案也不再依赖它。
这与 early causal commitment 完全一致：head 3 在前 t<=.25 负责选择 solution basin，
之后对最终 answer identity 已是冗余计算。

本实验通过 hooks 替换已算出的 activations，因此只证明 functional equivalence，
**尚未节省真实 FLOPs/latency**。单独跳过 1 个 head 的理论收益也很小。下一步必须
扩大 late-flow structured ablation：先跳过整个 block 11，再测试 blocks 9--11；
找到最大无损 block set 后，才实现真正 conditional execution 并 benchmark latency。

产物：`runs/official-semantic-cache-full-62082294/`

### 从 whole-block bypass 到 component gating（jobs 62082311/62082323）

Whole-block identity bypass 给出明确负结果：即使只从 t=.625 起 bypass block 11，
各任务准确率也显著下降。这说明“答案已经 committed”不等于 late block 整体无用；
后续 flow 仍需要部分 state refinement。

将 block 拆为 attention 与 MLP 后，800 条 held-out 样本显示强烈功能分化：从
t=.625 起，将 block-11 attention 在 conditional/unconditional CFG 两支都置零，
四组任务均保持 100% exact accuracy 和 100% native answer agreement；最终
answer-state L2 平均仍改变 5.45。相反，置零同一 block 的 MLP 后平均准确率仅
65.9%。从 t=.5 起 zero block-11 attention 也达到 99.1% exact / 99.75% agreement。

因此 late refinement 并非均匀分布在整个 Transformer：block-11 attention 在答案
commitment 后功能上冗余，但其 MLP 仍不可缺少。产物：
`runs/official-late-component-bypass-full-62082323/`。

### 组合冗余不可加与真实 conditional execution（jobs 62082329--62082442）

逐层扫描发现许多 attention/MLP 单独 bypass 时不改变 80-sample smoke 的答案；但
组合搜索揭示这些冗余不可直接相加。第一次 greedy search 找到 8-component policy，
独立 800-sample validation 只有 97.5% agreement；两次搜索稳定交集形成的
5-component policy 在第三数据集上更降至 91.1%。这表明各层之间存在 compensatory
interactions，post-hoc independent pruning 会破坏联合 dynamics。

最稳健的单组件策略 block-11 attention 在第二个独立 800-sample validation 上仍为
四组 100% native agreement。随后用真正不调用 attention forward 的 conditional
execution（而非 post-compute hook）在第三个 800-sample 集上复现 100% agreement。
H100、batch 20、8 flow steps 的 30 次 paired benchmark 为：

| policy | mean flow latency | native agreement |
|---|---:|---:|
| native | 308.52 ms | -- |
| skip B11 attention for t>=.625 | 304.59 ms | 100% |

即实际 speedup 1.0129x、latency 降低 1.27%，与 1.56% theoretical component-call
reduction 一致。这是第一个端到端可执行的 **commitment-aware component gating**
prototype，但收益仍小。下一方法版本不应继续盲目叠加 post-hoc skip，而应在训练时
加入 time-conditioned component dropout/gating，使模型学习一组能够在 late flow
阶段共同关闭的子网络，再优化 accuracy--FLOPs Pareto。

产物：`runs/official-true-skip-benchmark-62082442/`。

### Reasoning-to-velocity causal transport（jobs 62082452/62082457）

为直接检验“早期 reasoning 如何通过 flow 变成答案”，在 200 对最小反事实上，于
每个 flow time 单次 patch block-11 head-3 donor K/V，并显式提取 answer-position
velocity。测量 intervention-induced velocity shift 与 paired donor endpoint 的方向
一致性、一次 Euler update 到 donor trajectory 的归一化距离缩短，以及继续 native
flow 后的最终 donor-answer rate。

whole-head 实验显示，在 t=0 注入 K（edits s1--s3）时，velocity endpoint alignment
为 .942，一步 donor progress 为 78.1%，最终 donor-answer rate 为 19.3%；V
（edit s4）分别为 .985、84.7%、18.0%。K 到 t=.25 时 progress 已降至 1.8%，
V 在 t=.125 仍有 33.3%，复现了 K routing 早于 V readout 的时间窗口。velocity
endpoint projection 与最终 donor switch 在 sample level 呈正相关（K r=.455，
V r=.287）。产物：`runs/official-reasoning-velocity-transport-full-62082452/`。

更严格的 semantic-position intervention 排除了“整头 donor activation 太强”的
解释。t=0 的 full 结果为：

| intervention | velocity–endpoint alignment | one-step donor progress | final donor answer |
|---|---:|---:|---:|
| final-table K | .951 | 79.0% | 14.7% |
| edited-table K control | .244 | 16.2% | 2.0% |
| accessed-cell V | .983 | 83.7% | 20.0% |
| swap-partner V control | .082 | 0.0% | 0.0% |
| all-table V | .983 | 83.5% | 20.0% |

accessed-cell V 与 all 32 table Values 几乎完全等价，说明 velocity rotation 由实际
读取 cell 充分解释，而不是广泛 donor contamination。结合 routing-layer 与 Q/K/V
实验，当前支持的完整机制链为：blocks 8--10 将早期 composition 信息迁移并写入
final-function table Keys；block-11 head 3 用该 routing pattern 选择正确 cell，沿
Value path 读出答案；该 readout 直接把 answer-position velocity 旋转到对应 donor
endpoint；Euler/ODE integration 将 z 推入该 solution basin。K transport 在 t<.25
最强，V/readout 窗口延伸至约 t=.25--.5，之后主要为 basin 内 lexical refinement。

产物：`runs/official-semantic-velocity-transport-full-62082457/`。

## 6. 下一步方法方向

如果 component/position experiments 继续确认计算集中在 late blocks，方法应是：

- 在 blocks 9--11 加 weight-tied recurrent loops；
- 训练时随机化 loop count，使同一模型支持更多 test-time semantic compute；
- 比较固定 solver、不同 loop count 下的 d4/d8 accuracy；
- 最终再学习 `one more loop` 的 value-of-computation controller。

第一版方法只需要回答：在保持数值 solver 不变时，增加内部 recurrence 是否提升
更深 composition 的准确率。它比“增加 flow steps”更直接地受到当前机制证据支持。

二维因果图又产生了一个互补的方法方向：**Reason Once, Flow Cheaply**。先在某个
flow step 完整运行 blocks 9--11，缓存 block-11 head-3 routing/readout state；后续
steps 复用缓存、跳过或缩短 late-block computation。关键实验是在相同 ODE schedule
下比较 full recomputation、cached late-state、shallow refinement 三者的准确率与
FLOPs。若缓存版本无明显退化，可将机制发现直接转化成 continuous-DLM inference
加速方法；若明显退化，则说明 late-block computation 是 state-dependent refresh，
更适合做随 flow state 更新的 recurrent/low-rank semantic module。

Temporal switching 进一步建议采用分阶段实现：t<0.25 保留完整 K-routing compute，
t=0.25--0.5 保留 V/readout refresh，t>0.5 使用 cheap flow refinement。第一版
method ablation 应分别缓存 K、缓存 V、缓存 K+V，并与“晚期跳过 block-11 head 3”
比较，从而验证 component-specific schedule 是否优于统一 cache time。

Cache-equivalence full result 表明 t>=.25 后甚至直接 zero head 3 也不改变任何
held-out answer token，因此方法优先级应从“优化单个 K/V cache”升级为“寻找可安全
跳过的最大 late-block 子网络”。建议依次测试 block 11、blocks 10--11、blocks
9--11 的 identity bypass，并对 cutoff 做 accuracy/trajectory/latency Pareto 曲线。
