# 第一轮进展（2026-09-07）

## Function composition 数据（2026-09-09）

已生成 official ELF 可直接读取的 `data/official-composition-v1`：50,000 个随机
episode，每个产生一条 composition 与一条 matched lookup control，共 100,000
条训练样本；validation/test 各 1,200 episode、2,400 条。训练深度为 1/2/4/8，
验证和测试另含 12/16 外推深度。三个 split 共 52,400 组函数表完全不重复；
每条数据保留真实中间 states，供后续按深度评估和 flow 机制分析。condition 长度
为 85--117 T5 tokens，因此后续配置需使用至少 128 的 max_input_length。
另建立了 256 条均衡 tiny-overfit 集：task × depth 的八个组各 32 条。数据格式与
分组检查已通过。首轮 composition tiny-overfit 61947573/61947578 虽正常退出，
但配置错误沿用了 lookup 的 `max_length=68`，而 composition condition 有
85--101 tokens，导致所有答案 token 被截断；`target_tokens=0` 和全程 loss=0
证实这轮没有形成有效训练结果，不能解释为模型能力失败。Fine-tune 评估
61947579 因同一长度错误触发越界，尚未启动的 scratch 评估 61947583 已取消。
配置已修正为 `max_length=132`，并验证每条样本至少保留一个 target token。
修正后的 10k-step fine-tune 61951023 与 scratch 61951037 均已成功完成。
Fine-tune 的 standard 16-step CFG2 exact match 为 92.97%；自动评估 61951065
显示 clean oracle 100%，单步去噪 t=.25/.5/.75/.95 为
95.31/98.05/100/100%，18 组完整 flow 最佳为 92.97%（logit-normal、16-step、
CFG1），略低于 95% tiny-overfit 门槛但曲线到 10k 仍在上升。Scratch standard
生成在 10k 为 17.19%，明显更慢；其自动评估 61951102 当前等待 Priority。
已为正式泛化阶段加入 `task/depth` 分组 flow accuracy，并支持只运行指定 sampler
设置。当前 256-sample fine-tune checkpoint 的 2,400 条 held-out 分组评估
job 61958081 已提交。100,000 条训练集上的 official ELF-B fine-tune
job 61958090 与 scratch job 61958137 已并行提交：batch 128、32 epochs，精确
24,992 optimizer steps；每 4 epochs（约 3,124 steps）保存并评估一次。最终
held-out 分组评估分别为 61958144/61958286，依赖对应训练成功后运行。训练与
当前评估均等待 Priority；不应在 held-out 结果出来前作 reasoning claim。

上述任务现均已完成。256-sample fine-tune 在 2,400 条 held-out 上为 11.50%，
接近 8 类随机基线 12.5%，确认此前 92.97% 主要是记忆。100k-data、24,992-step
正式结果：fine-tune overall 70.96%，scratch overall 65.88%。但 overall 被所有
lookup 组的 100% 显著抬高。Fine-tune composition 分深度 exact accuracy 为：
d1 100%、d2 91%、d4 16.5%、d8 11%、d12 16%、d16 17%；scratch 为：d1
100%、d2 36%、d4 14%、d8 11.5%、d12 13%、d16 16%。因此两者均学会 lookup
和 d1，fine-tune 明确学会 d2，scratch 只部分学到 d2；d4 及以上尚无可靠能力。
预训练对浅层 composition 有明显帮助，但当前结果不支持深链 reasoning claim。

下一步先隔离训练步数效应。Fine-tune 已从 checkpoint_24992 原 optimizer/EMA
状态续训到 64 epochs、49,984 steps：job 61961807，当前运行正常（已超过 30k
steps，约 4.5 step/s）。最终 held-out 分组评估 job 61961844 自动依赖运行。
另提交 job 61961804，对 3,124--24,992 的八个已保存 checkpoint 使用相同
logit-normal/16-step/CFG1 设置逐一评估 task × depth learning curve。此轮不改变
数据分布；若 d4 随 steps 上升则继续训练，若停滞再单独测试 curriculum/更多
composition 数据，避免将两种效应混合。

## Official ELF 诊断（2026-09-08 晚）

已按 decoder → teacher-forced denoising → sampling sweep → tiny overfit 的顺序检查。
job 61936991 显示：2-epoch fine-tune 在 clean answer latent 上目标-token accuracy
为 100%；单步去噪在 t=.5 为 88.3%、t=.95 为 100%。Scratch clean oracle
仅 59.5%，尚未充分学会 decoder。完整 flow sweep job 61937154 比较 16/32/64
steps、CFG 1/2/3、uniform/logit-normal；fine-tune 最佳 first-target-token accuracy
仅 3.9%，scratch 最佳约 12.9%（chance 12.5%），增加 NFE 不能挽救完整 flow。

旧配置 tiny overfit job 61937155 在固定 256 条训练样本上跑 5,000 steps；完整
文本 first-integer 从早期约 2% 升至 41.4%。token-level eval job 61937196 的
最佳 first-target-token accuracy 为 45.3%（16-step loglando-normal, CFG=1），
说明 pipeline 能学习但尚未充分 overfit。

进一步发现旧 lookup 配置偏离官方 XSum 设置：使用 `pad_token: pad` 与
`decoder_noise_scale: 5.0`，而官方为 `pad_token: eos` 与 1.0。旧设置使答案后
未训练的位置在生成时产生重复数字/杂词，exact-match 因而恒近 0。已修正 pilot
和 overfit 配置，并把 W&B staging 指向 `/tmp`。修正版 256-sample overfit
job 61937211 已完成 5,000 steps；16-step CFG2 exact match 为 59.77%。最终
token-level evaluator 61937368 显示 clean oracle 100%，单步去噪 exact match
在 t=.25/.5/.75/.95 分别为 77.73%/96.88%/100%/100%。完整 flow sweep 的
最佳 first-target-token/all-valid-target accuracy 为 61.72%（16-step、
logit-normal、CFG1）；32/64 steps 没有改善，未达到预设 95% overfit 通过线。
因此尚未启动新的 100k-data 长训练。

逐步轨迹诊断 job 61941166（5k 修正版 checkpoint、256 个训练样本、CFG1）
进一步定位到 rollout degradation：uniform 16-step 的首个更新产生的终点预测
`x_pred` accuracy 为 64.1%，中间最高 66.0%，但随积分下降并在 t=1 变为
57.8%；实际 `z_t` 则从 3.5% 逐渐上升到 57.8%。logit-normal 观察到同样
结构（早期 x_pred 约 64%–66%，最终 56.6%）。这是训练集上的诊断，不是泛化
或 reasoning claim。从同一 checkpoint/optimizer 续训 5k→10k 的 job
61941174 已完成；自动 evaluator 61941179 也已完成。Fine-tune 在 7k steps
达到 96.09% exact match，8.5k 起达到 100%；10k checkpoint 的 clean oracle、
所有单步去噪时间点以及全部 18 组完整 flow 设置均为 100%。因此修正版
official ELF-B 管线已经通过预设的 95% tiny-overfit 门槛。

为形成预训练作用的严格对照，另已提交修正版 scratch ELF-B：job 61941256。
它使用完全相同的 official ELF-B 架构、固定 256 条训练样本、EOS padding、
decoder noise scale 1.0 和 10,000 steps；唯一主要区别是 ELF 主干随机初始化，
冻结的 T5-small encoder 仍沿用预训练权重。Scratch job 61941256 及 evaluator
61941303 均已完成。10k 时 standard generation exact match 为 65.63%；flow
sweep 最佳为 70.70%（logit-normal、32/64 steps、CFG1）。Clean oracle 为
100%，但单步去噪 exact match 随 t=.25/.5/.75/.95 仅为
73.83/76.95/81.25/89.45%。这说明 scratch 仍在学习，但显著慢于 pretrained
fine-tune；当前结果支持正式泛化实验优先使用官方 pretrained checkpoint。
本轮仍未启动 100k-data 长训练。

## Official ELF-B 并行 pilot（2026-09-08）

已完成 official `pytorch_elf` 环境与 checkpoint 兼容性 smoke；job 61932976
在 A100 上 COMPLETED、exit 0。已生成显式单表 lookup 数据：train 100,000、
validation/test 各 2,000，三者 table 集合不重叠，输出仅为最终数字答案。

为公平区分预训练作用与 architecture inductive bias，准备同一 ELF-B（约 105M）
的两个条件生成 pilot：fine-tune 从官方 XSum checkpoint 39800 仅载入 EMA
模型权重，optimizer/scheduler/step 均重新初始化；scratch 随机初始化 ELF 主干。
两者均沿用 official ELF 的冻结 T5-small encoder，使用同一数据、seed、batch、
learning rate 和 2 epochs（约 1,562 optimizer steps）。因此这里的 scratch 不是
“连 T5 一起从零预训练”。已加入 exact-match/first-integer 指标；weights-only
初始化与指标单元测试在 job 61934084 上均通过。

首次提交的 fine-tune job 61934207 在首次 `torch.compile` 前因 home quota 已满、
无法创建默认 `~/.triton` 缓存而退出；它已正确载入数据、104,579,940 参数模型
和 XSum EMA checkpoint，但尚未完成任何训练 step。未启动的 scratch 61934208
已取消。Slurm 脚本现把 Triton 与 TorchInductor 缓存分别放到作业独立的 `/tmp`
目录。修复后 fine-tune 61934533、scratch 61934549 均完成 1,562 steps；旧
padding 配置下的生成结果不能作为最终能力判断，详情见上方诊断更新。

## 4.8M 模型 100k 最终结果与 ELF-scale 计划

jobs 61929648/61929649 均 COMPLETED、exit 0。每个见过 640 万条在线 lookup
查询。flow 用时 1:07:55，最终 84/600=14.0%，100 次验证最高 15.33%；
direct 用时 0:39:21，最终 77/600=12.83%，100 次验证最高 13.67%，最终
CE 2.0774，接近 ln(8)=2.0794。两者各占约 3.0GB 磁盘；Slurm MaxRSS
约 6.2GB。结果不支持 4.8M 模型已学会 unseen-table lookup。

用户要求直接测试 ELF-B 同级容量。新配置 width 864 / 12 layers /
12 heads，按代码精确公式为 108,477,008 个可训练参数，接近 ELF-B 105M，
但仍是本项目的符号 conditional Transformer，不是官方 ELF-B architecture
或 checkpoint。计划 flow/direct 各一张 A100、100k 步并行；每 1k 验证、
每 25k 保存，48GB host RAM、12h 上限。该实验只检验容量假设；鉴于 direct
在 4.8M 下完全处于 chance，扩大容量不保证成功。

已提交并启动：flow job 61932844（W&B gd5x5xxc，GPU 2）与 direct
job 61932845（W&B qo0l48aq，GPU 3）。两者同时运行于 wICE k28g30 的
不同 A100。首步完成，峰值 PyTorch allocated GPU memory 分别约 6.01GB
和 3.27GB。首次验证在 step 1000，当前尚无容量实验结论。

## 已提交：100k 容量/训练预算实验

用户确认两个任务可同时各占一张 A100，不设相互依赖。
flow job 61929648；direct job 61929649；wICE gpu_a100，账户
lp_zhou_diffusionlang，每任务 1 A100 SXM4 80GB / 8 CPU / 32GB RAM / 12h 上限。
两者均从头训练 width 256、6 层、4 heads（4,833,872 总参数），lookup-only，
seed 0，batch 64，lr 3e-4，100,000 步（各 640 万训练查询）。
每 1,000 步验证，每 10,000 步及结束保存完整 checkpoint；保存先写临时文件
再替换。16 项测试和独立验证/保存频率的三步 CPU smoke 均通过。
哈希历史粗估最终约 1GB Python RAM、约 409MiB 序列化；保存会有额外临时
内存。10 个 checkpoint 每任务预计数 GB。df 为共享盘空闲量，不是个人配额；
quota 查询未取得数据目录配额，不能声称个人剩余额度已核实。
12h 是保守作业上限，不是实测运行时间。W&B group: lookup-scale-100k。
提交不等于已运行/完成；以队列和后续日志为准。

## 更新：直接分类器对照

job 61929622 已完成 2,000 步，final checkpoint 已保存，W&B 同步成功。
同 seed、同在线 lookup episode 序列、width 128 / 4 层，直接 CE 训练，
常量零 answer query；无 flow 采样。128,000 条训练查询，验证
82/600 = 13.67%，CE 约 2.081，接近均匀 8 分类的 ln(8)=2.079。
这不支持把此前失败单独归因于 flow；也不证明任务不可学或训练预算充足。
下一步应检查更简单的查表子任务、输入表示和优化/收敛，而非机制解释。
16 项测试通过；未自动扩大训练预算。
W&B: https://wandb.ai/nlp_louise/latent-diffusion-reasoning/runs/cmczxet9

## 更新：lookup-only 诊断

job 61929610 已完成 2,000 步并保存 final.pt，W&B 同步成功。
新训练禁用最终 decode history，与 decode 训练一致；旧 checkpoint 保留旧行为。
lookup-only 每步 64 条查询，共 128,000 个 episode；验证 84/600 = 14%，
尚无可靠泛化证据。固定噪声抽查 128 条 lookup：原题及打乱题目均 12.5%，
仅 1/128 个答案因题目打乱而变化。此检查支持条件依赖仍弱，但不是根因证明。
15 项测试通过。建议下一项独立对照：同任务上的直接监督分类器，区分任务
表示/优化难度与 flow 训练问题；尚未实现或提交，不扩大训练预算。
W&B: https://wandb.ai/nlp_louise/latent-diffusion-reasoning/runs/viy1b21d

## 更新：2026-09-08 GPU pilot

wICE job 61929568 已 COMPLETED，exit 0，耗时 1 分 10 秒。
单张 A100 debug，width 128 / 4 层 / self-conditioning，在线训练 2,000 步、
batch 64，共 128,000 条查询。最终验证 142/1200 = 11.83%，仍接近随机
基线 12.5%；loss 下降不能作为学会任务的证据。暂未启动长训练或容量对照。
下一步诊断条件信息是否被使用、lookup 学习和训练/采样一致性。
W&B 指标上传成功：https://wandb.ai/nlp_louise/latent-diffusion-reasoning/runs/6cfag6ay
checkpoint 和日志：`runs/pilot-61929568/`。13 项单元测试通过。
以下为早期记录，其中“尚未提交”等状态已由本次更新取代。

## 当前问题

在随机函数复合任务上，模型是否跨 flow 更新使用中间计算结果？
本轮只构建实验基础，尚未进行机制实验，也没有证明模型具备任务能力。

## 已完成

- 随机置换表数据生成器；每个 episode 使用新的表。
- 10,000 个训练 episode（20,000 条 compose/lookup 查询）。
- 验证、测试各 600 个 episode（各 1,200 条查询）。
- 训练长度 1、2、4、8；验证/测试另含 12、16。
- 各 split 映射表集合不重叠；数据 manifest 保存 SHA256 与分组计数。
- 小型 ELF-style 条件 Transformer、flow loss、decode loss、可选 self-conditioning。
- 训练、恢复、分长度评估、完整状态记录和从中途重放。
- 六项单元测试全部通过；Python 编译、Slurm shell 语法检查通过。
- 两步 CPU smoke training、checkpoint 恢复至第三步、独立评估均通过。
- 保存的中途状态恢复后，最终输出通过一致性检查。

## 这些检查不意味着学会了任务

`runs/smoke` 是 width=32、单层、17,520 参数的程序检查模型，仅训练两步。
验证准确率为 152/1200 = 12.67%，接近 8 分类的 12.5% 随机基线。
它不能用于机制结论。默认训练配置是更大的 width=128、四层模型。

## 下一步

训练已改为默认在线生成；原 20,000 条静态训练样本仅供 offline 调试。
每个 batch 生成新的置换表 episode，各有 compose/lookup 两种查询。
固定验证/测试表和已生成训练表均会排除，保证集合隔离与训练表不重复。
生成器 RNG、已见表哈希、episode 计数随 checkpoint 保存，支持精确续训。
正式训练以步数和 batch size 控制预算；例如 20,000×64 是 128 万条查询、
64 万个表 episode，而不是已经完成了这些训练。

在线版本验证：九项单元测试通过；两步 CPU 在线训练通过。
从第一步 checkpoint 恢复到第二步，模型参数、生成器历史和 PyTorch RNG
与连续训练两步逐项完全一致。此检查仅验证续训正确性，不代表任务已学会。

`train.slurm` 已准备一个 2,000-step、30 分钟上限的 GPU pilot，尚未提交。
当前节点无可用 CUDA GPU。等待用户指定本次使用的 cluster/partition/account；
旧项目的集群配置未自动沿用。

小样本过拟合检查已完成：`runs/overfit_sc/report.json`。
width=64、两层、开启 self-conditioning，在固定 8 个 episode 的 16 条查询上
训练 700 步；用 16-step sampler 和三个噪声种子评估，训练准确率均为 100%。
这说明训练与采样流程可以拟合小样本，不证明新映射表上的泛化或跨 flow 推理。
新增 `overfit.py` 可复现此检查；九项单元测试再次全部通过。

首次 GPU pilot 先看固定验证集上的单步 lookup、短链 composition 是否学得动。
若表现接近随机，继续训练诊断，不解释 flow 几何。
若表现可靠，再固定 checkpoint，开展中间变量 probe 和定向 patching。
更长链是外推评估，不能以其失败否定跨 flow 计算的可能性。

注意：该模型用固定符号向量代替 T5，因此是 ELF-style 受控变体，
不是预训练 ELF-B，也不是原论文的性能复现。

## 50k-step official ELF-B continuation（2026-09-10）

三个作业均正常完成（exit code 0）：checkpoint curve `61961804`、
25k→50k 续训 `61961807`、最终 held-out 评估 `61961844`。

最终 checkpoint 为 step 49,984。统一使用 logit-normal schedule、16 个采样步、
CFG=1，在 2,400 条 held-out 查询（每组 200 条）上得到 71.625% overall exact：

- lookup d=1/2/4/8/12/16：全部 100%；
- compose d=1：100%；
- compose d=2：100%；
- compose d=4：12.5%；
- compose d=8：18%；
- compose d=12：13%；
- compose d=16：16%。

相较 step 24,992 的 70.958%，整体仅提升 0.67 个百分点。主要变化是 compose
d=2 从 91% 提升到 100%；d=4 及以上没有形成可靠能力。checkpoint curve 显示
lookup 和 d=1 约在 9k--12k steps 学会，d=2 在 22k--25k 附近突增，而 d=4
截至 50k 没有持续上升趋势。因此暂不把继续相同设置训练到 100k 作为首选；
下一步应改变 composition 训练分布/课程或引入显式迭代计算，并可先在
lookup、compose d=1、d=2 上开展有限的 flow 机制分析。

## 第一轮 flow trajectory 诊断（2026-09-11）

job 61972229 使用固定 step-49,984 checkpoint，在全部 2,400 条 held-out
validation 样本上完成 16-step uniform 与 logit-normal trajectory 诊断，正常
退出（exit 0，2 分 42 秒）。输出为
`runs/official-composition-trajectory-61972229/trajectory.json`。脚本现明确区分
模型被调用的 `t_model` 和 Euler 更新后的 `t_state`，并按 task/depth 记录
`x_pred`、实际 `z_t` 的首答案 token accuracy、answer-slot velocity norm 与
相邻 velocity cosine。

uniform sampler 的初步现象：在首次模型调用 `t_model=0` 时，`x_pred` 对
lookup、compose d1、compose d2 已分别达到 100%，与最终准确率相同；但第一次
更新后的 `z` accuracy 仅约 16%--22.5%，到 `t_state=.5` 才达到约
97.5%--99%，到 `.625` 达到 100%。compose d4 的 `x_pred` 和 `z` 全程仍约
14%--15%，与其最终失败一致。成功任务 answer slot 的相邻 velocity cosine
几乎始终大于 .9997，轨迹接近同向 transport。

这支持一个当前模型特定、但尚非因果的解释：答案由每次 denoiser forward
直接预测，而 ODE flow 主要把 noisy latent transport 到该预测；目前没有证据
显示 16 个 solver step 在逐步执行 d1/d2 reasoning。`x_pred` 可解码不等于
causal commitment，下一项应做 early-state intervention/branching rollout，检验
首次 endpoint prediction 对后续答案是否稳定，并与不同噪声初值比较。

Gaussian branching pilot job 61972276 在 lookup/d1、compose/d1、compose/d2
各 20 条 baseline-correct 样本上完成。对 answer slot 加每维 sigma=0.5/1/2
的独立高斯噪声后，从 t=.0625/.5/1 继续或直接解码；保留 self-conditioning
cache 与清空 cache 两种设置下均保持 100% 正确。它说明这些方向上的局部
robustness 很强，但高维随机方向通常不指向竞争答案，不能单独定义 commitment。

Counterfactual answer-slot pilot job 61972480 同样使用每组 20 条。完整 donor
replacement 在 t<=.25 后继续 flow 时全部恢复 source answer。compose/d2 的
joint z+cache patch 在 t=.5 有 10% 转为 donor、t=.75 为 40%；lookup/d1 与
compose/d1 在这两个时点均为 0%。但终点 answer-slot replacement 的 donor
transfer 也只有 lookup/d1 5%、compose/d1 40%、compose/d2 70%，说明单个
contextual slot 并非跨题可移植的纯 answer representation。因此已扩展为同时
patch answer slot 和完整 target suffix，并提交每组 200 条的 job 61972639；
该扩大版已正常完成，三组 baseline 均为 200/200。完整 target-suffix、alpha=1
的 patch 在 t=1 对三组均产生 100% donor answer，校准了 intervention validity。
同一 joint z+cache patch 在 t=.5 的 donor rate 为 lookup/d1 0%、compose/d1
0.5%、compose/d2 18%；在 t=.75 分别为 1%、12%、94%。t<=.25 三组均为 0%。
因此 flow 早期能完全利用 source condition 纠正竞争 state，但这种 corrective
contraction 对 d2 明显更早减弱。该结果不能表述为 d2 更早完成 reasoning；更
保守的解释是较深任务的 answer basin 更脆弱、对当前 target state 更敏感。下一步
需要增加 t=.5--.75 的时间分辨率、独立 seed/donor pairing，并测 conditional
contraction/Jacobian proxy，确认这是不是 reasoning-depth-dependent stability。

High-resolution 3-seed job 61973097 已完成（每个 group/seed 200 条，三组共
1,800 条 baseline trajectories）。完整 suffix 的 joint z+cache donor-transfer
在三个 seed 上高度一致。compose/d2 的三-seed 均值随
t=.375/.5/.625/.75/.875 为 5.8%/21.2%/61.3%/90.8%/100%；compose/d1 为
0%/1.5%/3.3%/7.8%/19.3%；lookup/d1 为 0%/0%/0%/0.8%/4%。cache-only
全程为 0%；z-only 到 .875 仍仅 d2 19.3%、d1 .5%、lookup 0%。joint patch
的强效应说明 z 与 self-conditioning cache 形成一致性/纠错耦合，不能把任一
分量单独当作完整 sampler state。

该差异稳定但尚缺严格长度对照。compose/d2 的 prompt/program 比 lookup/d1
更长，因此补做了真正 matched 的 lookup/d2 三-seed control：job 61973191，
正常完成（exit 0，2 分 16 秒）。lookup/d2 的 joint suffix donor-transfer 在
t=.375/.5/.625/.75/.8125/.875 的三-seed 均值约为
0%/0%/0.17%/0.17%/1.67%/3.5%，与 lookup/d1 的
0%/0%/0%/0.83%/1.83%/4% 基本重合，却远低于 compose/d2 的
5.83%/21.17%/61.33%/90.83%/97.33%/100%。由于 lookup/d2 与 compose/d2
共享相同表、program 长度与答案空间，主要区别是 FIRST vs FULL dependency；
这排除了简单输入长度解释，并支持 composition dependency 与较早的 conditional
stability loss 相关。当前仍只比较 depth 1 与 2，不能外推为一般深度规律。

### Local perturbation 与 counterfactual patching pilot

Gaussian branching job 61972276 在 lookup/d1、compose/d1、compose/d2 各 20 条
held-out 样本上完成。对 state index 1/.0625、8/.5、16/1.0，在答案 slot 加
每维 sigma=0/.5/1/2 的 Gaussian noise；无论保留还是清空 self-conditioning
cache，最终答案均为 100% 正确。该结果只说明随机方向下局部稳定域较宽；高维
随机扰动很少指向竞争答案，因此不能单独作为 causal commitment 证据。

Counterfactual patching job 61972480 使用同样三组各 20 条，将 answer-slot state
朝同组、不同正确答案样本的真实 state 替换。alpha=1 时，state t<=.25 的所有
patch 均被后续 flow 修正回 source answer。compose/d2 的 joint z+cache patch
在 t=.5 有 10% 转为 donor answer，在 t=.75 为 40%；lookup/d1 与 compose/d1
在这些时点均为 0%。但在 t=1 直接 patch 时 donor 转移率也只有 lookup/d1 5%、
compose/d1 40%、compose/d2 70%，说明单一 contextual answer slot 并不是跨题
可直接移植的完整答案表示。此轮只能视为 pipeline/pilot 信号。下一步需扩大
样本并 patch 完整 target suffix，或用 targeted minimal intervention 校准每组的
有效 counterfactual direction，然后再比较归一化的 causal onset。

### d4-heavy 百万级 DDP 训练

2026-09-12 生成 `data/official-composition-curriculum-d4-1280k-v1`：共
1,280,000 条训练样本和 1,280,000 个唯一随机函数表，与原始 train/validation/test
的 52,400 个表无重叠。组成是 compose/d4 720k、compose/d2 320k、
compose/d1 80k，以及 lookup/d1,d2,d4,d8 各 40k；答案 0--7 基本均衡。

正式训练 job 61993941 已提交到 Wice `gpu_a100`：2 x A100 80GB，DDP，
从 `runs/official-composition-finetune-61958090/checkpoint_49984` 的 EMA 权重初始化，
global batch 128（每卡 64），10 epochs x 10,000 steps/epoch = 100,000 optimizer
steps；每 2,000 steps 保存 checkpoint，W&B run 名为
`official-composition-d4-1m-61993941`。提交时状态为 PENDING (Priority)。

Job 61993941 于 2026-09-12 启动后在第一个 optimizer step 前失败（exit 1）；
根因是新 DDP Slurm 脚本遗漏 Triton/Inductor cache 环境变量，`torch.compile`
回退写入已满额的 home `.triton`，报 `Disk quota exceeded`。没有生成 checkpoint，
也没有发生部分训练。已将 `XDG_CACHE_HOME`、`TRITON_CACHE_DIR` 和
`TORCHINDUCTOR_CACHE_DIR` 全部定向到 node-local `/tmp`，并重新提交 job
61996825；训练规格不变，提交时状态 PENDING (Resources)。

61996825 因 8 小时资源窗口预计等待过久而在尚未启动时取消，缩短 walltime 后重提
为正式双卡 job 61996829。该任务于 2026-09-12 22:13:28 在 k28g25 启动，
GPU 0,2，world_size=2，每卡 batch 64；修复后的 Triton/Inductor node-local cache
生效，首个 compile+training step 成功。step 500 时 loss=0.0035、l2=0.0044、
无 NaN/OOM/NCCL 错误，稳定吞吐约 7.3 steps/s。

由于普通用户不能把运行中 job 的 3 小时 walltime 延长到 5 小时，而该吞吐下
100k steps 预计超过 3 小时，已提交 afterany 依赖的单卡续训 job 61996970。
它会在 61996829 结束后从同一 run 目录的最新 2k-step checkpoint 恢复 optimizer、
EMA 和 counters，并继续同一个 W&B run，直至配置中的 100k steps 完成。

双卡训练已通过 20k steps，吞吐稳定在约 7.2--7.5 steps/s，未发现
NaN/OOM/NCCL/traceback。held-out 2400 条在线验证（ODE 16、CFG 2）结果：
step 10k 总准确率 71.0%，compose d1/d2/d4 分别为 100%/100%/16%；
step 20k 总准确率 72.04%，compose d1/d2/d4/d8 分别为
100%/100%/18.5%/13.5%，lookup 的所有验证深度仍为 100%。d4 相比 10k
仅小幅增长，目前尚未出现能力跃迁，需继续观察 30k--100k checkpoint。

step 30k 总准确率 71.04%；compose d1/d2/d4/d8 为
100%/100%/16%/11%，lookup 所有深度仍为 100%。因此 d4 在
10k/20k/30k 的 16%/18.5%/16% 只是随机基线附近波动，尚无持续上升证据；
训练过程本身正常，已进入 epoch 4，故继续保留后续 checkpoint 以检验晚期跃迁。

step 40k 出现明确能力跃迁：总准确率 73.88%，compose d1/d2/d4/d8 为
100%/100%/50%/12.5%，lookup 所有深度仍为 100%。d4 从 30k 的 16% 跃升到
50%，与约 35k 开始出现的训练 loss 从约 .0028 降至 .001 左右相吻合；d8 仍在
8 类随机基线。训练已正常进入 epoch 5，下一步检验 d4 是否继续升至 80--90%，
以及 d8 是否出现更晚的学习跃迁。

### LLaDA 离散 diffusion 对照

为在完全相同的 held-out split 上比较 continuous ELF 与 discrete LLaDA，已下载
官方 `GSAI-ML/LLaDA-8B-Instruct` 完整 snapshot（约 15GB），并新增
`llada_eval_composition.py` 和 `llada_composition_zero_shot.slurm`。第一轮固定
temperature=0、8 个 denoising steps、8-token answer block，对 compose
d1/d2/d4/d8 各 200 条做 zero-shot 测试。正式 800 条前先用各深度 2 条的 debug
A100 smoke job 61997334 验证模型加载、显存和批量 sampler；当前预计 23:25 启动。

Smoke job 61997334 已在 debug A100 上正常完成（exit 0，3 分 15 秒）；8 条输出
都是可解析的单整数，pipeline 正常，但这 8 条恰好全错。已提交完整 800 条
zero-shot job 61997359（compose d1/d2/d4/d8 各 200 条，batch 8，2 小时），
当前在普通 A100 partition 等待 Priority。

为避免普通队列长时间等待，先完成 debug A100 quick50 job 61997407（每个深度
50 条，共 200 条，exit 0）：LLaDA-8B-Instruct zero-shot 的 compose
d1/d2/d4/d8 为 14%/14%/20%/6%，总体 13.5%，接近 8 类随机基线 12.5%。
由于实际吞吐远快于保守估计，已取消预计一周后才启动的 61997359，并提交完整
200/group debug job 61997461（共 800 条，15 分钟）以缩小抽样误差。

完整 LLaDA zero-shot job 61997461 已正常完成：compose d1/d2/d4/d8 分别为
16%/10.5%/12%/11.5%，总体 100/800 = 12.5%，恰好等于 8 类答案空间的随机
基线。所有输出均为可解析的单整数，因此不是格式或 parser 失败。该结果表明原始
LLaDA-8B-Instruct 没有 zero-shot 学会 episodic randomized function-table 规则；
它不能直接用于 continuous-vs-discrete 能力比较，公平对照需要在同一训练分布上
做参数高效微调后再评估。

### d4-heavy 50k held-out 结果

双卡主任务 61996829 在 step 50k 正常产出 2400 条完整 held-out generation。
compose d1/d2/d4 分别达到 100%/100%/100%，其中 d4 从 step 40k 的 50% 继续
跃迁至 100%，说明模型已学会当前 curriculum 的目标深度。未纳入 compose 训练
分布的 d8/d12/d16 分别为 17%/15%/12.5%，仍接近 8 类随机基线；lookup 所有
深度均为 100%。继续监控后续训练是否保持 d4 稳定，并在 100k 后报告最终结果。

step 60k 的完整 2400 条 held-out 评估再次得到 compose d1/d2/d4 =
100%/100%/100%，确认 50k 的 d4 满分不是单次波动。compose d8/d12/d16 为
15.5%/10%/14.5%，仍处在随机基线附近；lookup 所有深度继续为 100%。
`checkpoint_60018` 已完整保存，主任务 61996829 继续正常运行。

step 70k 的 held-out compose d1/d2/d4 仍为 100%/100%/100%；d8/d12/d16 为
15%/12%/11.5%。主任务 61996829 最终在 step 77300 左右达到三小时时限，最近的
完整文件是 `checkpoint_76024`。原单卡续跑任务 61996970 恢复双卡 Muon optimizer
state 后，因 positional optimizer state 映射到不同形状参数（768 vs 512）失败。
已新增 `resume_reset_optimizer` 恢复模式：保留模型、EMA、step、epoch、scheduler
和 optimizer group 超参数，只重建 tensor-valued moments；相关本地恢复测试通过。
修复后的单卡 A100 续跑任务 62008276 已提交，当前等待资源，将从 step 76024
继续到 100k。

续跑任务 62008276 已于 2026-09-14 14:49 启动，并成功跨过原 optimizer 报错点；
单卡吞吐约 4.57 step/s。step 80k 的完整 2400 条 held-out 结果为 compose
d1/d2/d4 = 100%/100%/100%，d8/d12/d16 = 14.5%/12.5%/11%；lookup
d1/d2/d4/d8/d12/d16 = 98.5%/99%/100%/100%/100%/100%。
`checkpoint_80026` 已完整保存，训练已进入 epoch 9。

step 90k 的完整 held-out 结果为 compose d1/d2/d4 = 99.5%/100%/100%
（d1 仅错 1/200），d8/d12/d16 = 15%/11.5%/10.5%；lookup
d1/d2/d4/d8/d12/d16 = 98.5%/98.5%/100%/100%/100%/100%。
`checkpoint_90028` 已完整保存，任务已进入最后一个 epoch。

### d4-heavy 100k 正式训练完成

修复后的续跑任务 62008276 于 2026-09-14 16:21 正常完成，Slurm 状态
`COMPLETED`、exit code `0:0`。最终 `checkpoint_100000` 完整保存（约 1.26 GB），
并生成全部 2400 条 held-out 样本。最终 compose d1/d2/d4 为
99%/100%/100%（198/200、200/200、200/200）；compose d8/d12/d16 为
14.5%/11%/12%，说明训练深度内能力稳定，但未出现更长组合链的零样本泛化。
lookup d1/d2/d4/d8/d12/d16 为 99.5%/99.5%/99.5%/100%/100%/100%，
全体 2400 条准确率为 77.9167%。100k 训练与最终评估目标已完成。

### 100k checkpoint flow trajectory smoke experiment

已实现 `official_flow_trajectory.py`：记录完整中间 `z_t`、Euler velocity、
native-decoder answer exact match/margin、predicted-endpoint 指标及两张曲线图。
特别处理了 T5 将答案 `0` 编成两个 token 的情况，答案准确率使用完整 target span
exact match，margin/velocity 使用实际数字 token。四组 held-out 数据为 compose
d1/d2/d4 与 matched lookup/d4，各 30 条、共 120 条唯一数据。

首轮任务 62059404 已完成 120/120 条 GPU 计算，但 CSV 导出因初始状态的空
xpred/velocity 累计字段未清理而失败。修复 finalizer 并加入 regression test 后，
任务 62060730 在 A100 debug partition 正常完成（31 秒，exit `0:0`）。自动验收
通过：`z` shape `[120,17,132,512]`、velocity shape `[120,16,132,512]`，
2,040 条 per-sample/time records，所有 JSON/CSV/tensor/PNG 产物完整。

主要现象：在 t=0.0910 时，predicted endpoint 的 compose d1/d2/d4 和 lookup d4
准确率已为 96.7%/96.7%/100%/96.7%，但当前 `z_t` 的 native-decoder 准确率仅
23.3%/23.3%/33.3%/33.3%。最终 current-z 为 96.7%/100%/100%/96.7%。
这支持“endpoint prediction 早于 latent lexical compatibility”，但没有显示
reasoning depth 越大、flow-time commitment 越晚：d4 并不晚于 d1/d2，matched
lookup 也相似；velocity norm 和方向余弦高度重叠。因此当前证据更符合 reasoning
发生在单次 denoiser Transformer evaluation 内、flow 主要负责 transport/realization。
完整结论见 `FLOW_TRAJECTORY_REPORT.md`，下一步应基于已保存 tensor 做 perturbation/
branch continuation，以区分 decodability 与 causal stability。

### Latent branching stability 完成

已实现 `official_branching_stability.py`，并分别扰动完整采样状态中的 `z_t`、上一轮
self-conditioning endpoint、以及两者。正式任务 62073506 在 A100 上正常完成
（exit `0:0`，8 分 20 秒）：compose d1/d2/d4 与 matched lookup d4 各 30 条，
四个 flow 时刻、六档相对扰动、每档八个随机分支，共 69,120 个 continuation。
两个 smoke job 62073452/62073468 均先行通过，所有 sigma=0 分支精确复现基线。

结果显示条件 flow 从最早测试时刻 t=.091 起就极强鲁棒：即使扰动 SD 等于对应
状态自身 RMS，d1/d2/d4/lookup 的最低 answer agreement 仍为
97.5%--100%，baseline-correct retention 为 97.1%--100%；d4 在所有条件下均为
100%。稳定性不随 flow time 系统上升，也没有 d1<d2<d4 的深度顺序。这不支持
“reasoning commitment 随 flow 逐步形成”。但因为每个后续 denoiser 仍可访问完整
clean condition，这个指标测到的是 conditional recoverability，不能单独证明早期
latent 已因果承诺答案。完整报告见 `BRANCHING_STABILITY_REPORT.md`。研究主线应转向
12-layer ELF-B denoiser 内部的 layer-wise emergence 与 causal patching，method 则
应优先考虑 denoiser 内 recurrent/looped compute，而不是增加 ODE steps。
## 内部推理机制实验（2026-09-19）

已完成三组 full-scale、独立 held-out 实验，综合报告见
`INTERNAL_REASONING_REPORT.md`。

- Layer-wise probe，job 62073605：答案达到 90% 可读性的首层依次为
  compose-d1=9、compose-d2=10、compose-d4=11、lookup-d4=9。
- Causal layer patching，job 62073661：不同答案 donor 在 compose-d1/d2/lookup
  的第 10 层开始控制输出，在 compose-d4 的第 11 层才控制输出；所有同答案
  patch controls 均为 100% 正确。
- Ground-truth intermediate-state probe，job 62073734：compose-d4 的 s1/s2/s3/s4
  最佳 held-out accuracy 分别为 36.5%/88.5%/91.5%/100%；s2--s4 主要在
  layer 10--11 集中出现，而非逐层显式展开。

三组证据与此前 flow-level negative result 一致：推理深度主要反映在 denoiser
Transformer 的内部层深，而不是 flow time。下一步优先对 blocks 9--11 做
attention-vs-MLP 与 token-position causal patching；方法方向是给 late blocks
增加 recurrent/looped semantic compute，而非简单增加 ODE NFE。

组件级 causal patching（job 62073773）也已完成，每组 200 条、共 9,600 次
干预。block 10 attention 的不同答案 donor switch rate 在 compose-d1/d2、
lookup-d4 分别为 71.0%/79.5%/76.5%，对应 MLP 仅 1.5%/1.5%/0%；
compose-d4 在 block 10 两者均为 0%，到 block 11 时 attention/MLP 分别升至
100%/98.5%。所有 same-answer controls 仍为 100%。这进一步把深度阶梯定位到
late-block attention 输出，但不能单凭 donor sufficiency 断言 attention 独立产生
答案。下一步应做 token/head-level path patching。

全位置 latent-state map（job 62074219）已完成：4,096 train、200 held-out、
10 个语义位置 × 13 层 × 4 个中间状态，共 520 个固定 alpha Ridge probes。
剔除使用 ground-truth path 选出的 oracle table-cell anchors 后，s1/s2/s3/s4 的
最佳位置均为 answer slot，分别为 L10 36.5%、L11 88.0%、L11 91.0%、L11
100%。program positions 未出现逐步 s1->s2->s3 传播；block 12 则把最终答案
广播到 start/program positions（85.5%--99.0%）。当前证据更支持 block-11
answer-slot 的压缩计算，而非显式 token-to-token latent CoT。下一步采用最小
counterfactual table edit + path patching 做因果验证。

最小反事实 causal patching（job 62074267）已完成：每个被改变的计算步骤 150
对，共 600 对、11,400 次 interventions。反事实只交换一个实际访问 table cell
与同表另一输出，保持 permutation 合法、此前 states 不变，并保证当前 state 与
最终答案改变。原题和反事实 first-endpoint baseline 均为 100%。四组在 layers
0--10 的 answer-slot patch 均有 0% 跟随反事实答案，在 layer 11 均突然变为
100%；block-11 attention patch 四组均 100%，MLP 为 98.7%--99.3%。这建立了
非常清晰的 causal boundary，但四个步骤没有不同的形成时刻，更支持 block-11
集中整合，而非可分离的逐步 latent CoT。后续若继续机制定位，应转向早期
source-position/head-level path patching。

Block-11 head-level counterfactual patching 已实现并通过编译/单元检查。实验分别测试
单个 donor head 的充分性，以及“全部 donor heads、仅保留一个 source head”的
必要性对照。smoke job 62075155 已提交到单张 A100 debug；当前为 PENDING
(Resources)，调度器估计开始时间 2026-09-19 23:46:11。任务仍有效，未失败；
由于 debug QOS 每用户提交上限，不能在它等待时重复提交同类任务。

上述 head-level smoke 与 full job 现均已完成。full job 62075181 使用 600 对最小
反事实：12 heads 中只有 block-11 head 3 单独具有反事实答案充分性，四个干预步骤
的 switch rate 为 74.7%/79.3%/80.0%/75.3%；其余所有单 heads 均为 0%。当其余
11 heads 全换成 donor、唯独保留 source head 3 时，switch rate 为
0%/0.7%/0%/0%，说明 head 3 也近乎必要。下一步沿 head-3 value path 做
source-position patching。

Head-3 semantic value-path patching full job 62075186 已完成（600 对）。仅编辑 s4
时，patch 最终实际访问 table cell 的 block-11 head-3 V vector 可使 76.0% 输出
跟随反事实；patch 全部 32 个 table values 为 75.3%，swap partner 单独为 0%。
但编辑 s1/s2/s3 时，changed-cell 与 all-table V patch 的 switch rate 均约为 0%。
这定位出最后一步的直接 lookup value path，同时说明较早步骤依赖 Q/K routing
或 block-10 之前形成的压缩表示。下一实验为 head-3 Q/K/V factorial patching。

Head-3 Q/K/V factorial 首次 smoke job 62075601 正常完成，并显示单独 K patch
控制 s1--s3、单独 V patch 控制 s4 的初步分工。但检查组合条件时发现原实现用
substring 解析 mode，导致 `q_answer_kv_all` 漏 patch K；单独 Q/K/V 列不受影响，
组合列不可使用。解析已改为显式 component sets 并通过编译。修正版 smoke job
62075606 已提交，当前 PENDING (Resources)，调度预计 2026-09-20 01:31:59。

修正版 Q/K/V full job 62075643 已完成。在 answer positions 对齐子集上，编辑
s1/s2/s3 时 K-only switch 为 91.1%/91.6%/94.1%，V-only 为 0.9%/0%/0%；
编辑 s4 时 K-only 为 0%，V-only 为 98.3%。Q-only 全部 0%，K+V 全部约
98.3%--100%。这支持“前三步通过 Keys 控制路由、最后一步通过 Values 提供内容”。

Key semantic-position full/control job 62075653 进一步定位：排除 edited function
与 final function 相同的样本后，patch edited-function table 的 8 Keys 对 s1/s2/s3
均为 0%，而 patch final-function table 为 75.0%/81.5%/80.4%，并与 patch 全部
32 table Keys 完全相同。当前最佳机制解释是：blocks <=10 将前三步 composition
压缩成写在 final table 8 个 cells 上的 key routing pattern；block-11 head 3 选择
最终 cell，并沿 value path 把答案读入 answer slot。

Routing-layer full job 62075713 已完成。排除 edited function 与 final function
相同的样本后，final-table patch 对 s1/s2/s3 在 L8 均为 0%；到 L9 分别变为
25.0%/0%/80.4%，L10 为 75.8%/81.5%/80.4%。edited-table patch 则从 L8 的
75.8%/81.5%/80.4% 降至 L9 的 18.3%/74.1%/0%，并在 L10 全部为 0%。
all-table patch 全程稳定，证明这是同一反事实信息从 edited table 向 final table
迁移，而非干预失效。迁移所需层数随剩余 composition depth 系统变化，是目前
latent chain of computation 最强的因果证据。下一步拆 blocks 9/10 的 attention
与 MLP table-position outputs。

Flow-time × layer causal map full job 62082063 已完成：200 对最小反事实，8 个
uniform flow steps，blocks 8--12，pulse/sustained 两种干预；原题与 donor 完整
trajectory baseline 均为 100%。单次 pulse 的 donor control 从 t=0 的 17.8%
随 flow time 增至 t=.75 的 78.4%，最终一步为 100%；t=0 时 blocks 8--10
为 0%，block 11/12 为 46.5%/42.5%。sustained patch 平均为 98.9%。结果说明
孤立的早期语义扰动会被后续 denoiser calls 修复，late blocks 更像在每个 flow
step 重算/刷新答案相关状态，而不是只在早期计算一次后被动携带。下一方法实验应
测试 late-block/head-3 semantic state 的 cache/reuse，以区分可复用的重复计算与
必须随 z_t 更新的 state-dependent refresh。产物：
`runs/official-flow-layer-causal-full-62082063/`。

Head-3 temporal switching full job 62082273 已完成（200 对，8-step uniform flow，
baseline 双方均 100%）。对 s1--s3 的 Keys，source→donor 从 t=0 开始可达到
72.7%，但从 t=.25 才开始只剩 4.0%，t=.375 后为 0%；反向 donor→native 在
donor 控制前两步后已达 68.7%，之后饱和至 72.7%。对 s4 Values，接管窗口稍宽：
source→donor 在 t=0/.125/.25/.375/.5 分别为 76%/68%/32%/6%/2%，t>=.625
为 0%；donor 控制前两步后已达 72%。显式 source-baseline restoration 与 native
repair 基本重合，说明 early donor drive 一旦改变 trajectory，后期同一 K/V component
难以救回。当前最佳解释是 component-specific early commitment：K routing 主要在
t<.25，V/readout 约在 t<.5，之后主要是 basin 内 refinement。下一方法实验应测试
early-full/late-cheap 的 K/V cache、head skipping 与周期刷新。产物：
`runs/official-temporal-switch-full-62082273/`。

Late-flow semantic-cache full job 62082294 已完成：lookup、compose d1/d2/d4 各
200 条，共 800 held-out samples；cutoff t=.25/.5/.625。static K、V、K+V cache
及 periodic-2/4 refresh 在所有组和 cutoff 上均为 100% native answer-token agreement，
exact accuracy 与 native 一致（native d1/d2/d4/lookup 为 98%/99.5%/100%/99.5%）。
从 t=.25 起直接 zero block-11 head 3 同样保持 100% answer agreement，d4 exact
仍为100%。cache K/K+V 在 t=.25 的 answer-state L2 仅约 .01--.02/.04--.07；
zero head 虽造成约 .89--1.28 的状态变化，仍不改变答案，说明 early commitment 后
该 head 对 answer identity 已冗余。当前 hooks 尚未节省真实计算；下一步应扩大到
whole-block late bypass，寻找可无损跳过的最大 blocks 后再实现并 benchmark 真正
conditional execution。产物：`runs/official-semantic-cache-full-62082294/`。

Whole-block bypass smoke job 62082311 为明确负结果：答案 commitment 后仍不能跳过
整个 block 11。细分 attention/MLP 后，full job 62082323 在 800 条 held-out 上发现
t>=.625 zero block-11 attention 时四组均为 100% exact 和 native agreement，而
zero block-11 MLP 平均仅 65.9%，说明 late block 内存在组件级功能分化。逐层单组件
扫描显示大量独立冗余，但 greedy 组合发生明显过拟合：8-component policy 独立验证
97.5%，双搜索交集的 5-component policy 第三验证仅 91.1%，证明组件冗余不可加。

保守的 block-11-attention policy 在另一独立 800 条样本上再次达到 100% native
agreement。true-skip benchmark job 62082442 真正绕过 attention forward，在第三个
800-sample 集上仍 100% agreement；H100 paired latency 从 308.52ms 降至 304.59ms，
speedup 1.0129x（latency -1.27%，理论 component calls -1.56%）。当前已经得到首个
可运行的 commitment-aware component gating prototype。下一步应做 time-conditioned
gating/dropout fine-tuning，训练出可联合关闭的 late-flow 子网络，而不是继续叠加
不稳定的 post-hoc pruning。产物：`runs/official-true-skip-benchmark-62082442/`。

Reasoning-to-velocity transport full job 62082452 已完成（200 对最小反事实）。在
t=0 单次 donor K patch（edits s1--s3）使 answer velocity 与 donor endpoint 的
alignment 达 .942，一步缩短 78.1% donor-trajectory 距离，最终 19.3% 跟随 donor；
V patch（edit s4）为 .985/84.7%/18.0%。K progress 到 t=.25 已仅 1.8%，V 在
t=.125 仍为 33.3%，直接复现 early K-routing / later V-readout 窗口。

Semantic-position full job 62082457 进一步定位：t=0 只 patch final-table 8 个 Keys
仍产生 .951 alignment、79.0% one-step progress、14.7% final donor answer；matched
edited-table K control 仅 .244/16.2%/2%。只 patch 实际 accessed-cell Value 产生
.983/83.7%/20%，与 all-table V 的 .983/83.5%/20% 几乎相同；未访问 swap-partner
V control 为 .082/0%/0%。因此目前已有直接因果证据支持：composition 被写成
final-table routing，head-3 readout 旋转 answer-position velocity，flow integration
把 state 推进对应 solution basin。产物：
`runs/official-semantic-velocity-transport-full-62082457/`。
## 2026-09-26: causal response kernel and dose response

- Added `official_causal_response_kernel.py` plus H100 smoke/full/dose-response
  Slurm jobs.
- Completed full job 62165781 (200 d4 counterfactual pairs) and dose-response
  job 62165792.
- A one-step block-11/head-3 semantic pulse at early flow time causally changes
  the entire later trajectory.  At s=0, final donor-axis response is 0.321 for
  final-table K and 0.417 for accessed-cell V; donor-answer rates are 14.7% and
  28.0%.  The position-matched V control is approximately zero.
- Routing K closes earlier than readout V: K is essentially inactive by t=.25;
  V retains a weak effect at t=.25 and vanishes after t=.375.
- Strength sweeps reveal nonlinear thresholds rather than linear transport.
  Early K rises sharply around alpha=.5-.75; final V basin steering appears
  mainly above alpha=.75.  Delayed interventions require greater strength and
  attain smaller effects.
- Full interpretation and caveats are in `CAUSAL_RESPONSE_KERNEL_REPORT.md`.

## 2026-09-26: circuit organization and cross-depth dynamics

- Generalized minimal counterfactual construction and semantic token mapping to
  arbitrary composition depth; d1/d2 smoke and full jobs pass.
- Block-10 head scan job 62165937 shows shallow computation is distributed:
  head 3 is necessary but no single head is sufficient.
- Head-group job 62165949 identifies a sparse cooperative mode.  For d1,
  h1+h2+h3 reaches 68% donor control versus 0% for all singleton heads; for d2
  it reaches 72--73%.  This equals all-head patching, while size-matched control
  heads remain at 0%.
- Unified flow-response job 62165962 patches each depth's causally sufficient
  circuit.  All depths show similarly strong immediate donor-axis motion, but
  endpoint semantic retention rises from 0.9% (d1) to 6.1% (d2) to 28.1% (d4).
  Only d4 produces a substantial final donor-answer rate (22.5%) and response
  amplification (2.0x).
- Current interpretation: shallow computation creates a fast transient mode
  that source-conditioned dynamics erase; d4 writes into a persistent slow
  semantic mode.  See `CIRCUIT_DYNAMICS_REPORT.md`.

## 2026-09-26: sustained forcing and fixed-release control

- Added `official_sustained_circuit_forcing.py` and completed sustained job
  62166039 plus fixed-release job 62166063.
- Consecutive forcing of the causal circuit can drive all depths into the donor
  basin.  The fixed-release t=.75 control rules out the trivial explanation
  that longer forcing merely leaves less recovery time.
- With 1/2/3/4/5/6 forced steps, donor rates are d1:
  0/4/12/38/60/68%, d2: 0/25/61/70/71/71%, and d4:
  0/4.5/36/60.5/76.5/77.5%.  Size-matched controls remain 0%.
- Above threshold, d2/d4 native dynamics sometimes increase donor-axis response
  after release, consistent with basin entry rather than continued external
  maintenance.  The next method step is an autonomous tied-weight recurrent
  circuit trained to create a stable slow semantic state without donor oracle.

## 2026-09-26: tied-weight recurrent reasoning baseline

- Added checkpoint-compatible recurrence over blocks 10--11. K=1 is exactly
  the original ELF; K>1 reuses the same weights inside each fixed flow state,
  without increasing parameter count or outer solver NFE.
- Full held-out job 62166102 evaluated 2,400 examples at 16 fixed flow steps.
  Without recurrent training, K=1/2/4/6 overall exact accuracy is
  78.33/78.46/78.54/78.58%.  d1/d2/d4 remain 100% at every K; d8 changes
  16.5->17.0%, d12 13.0->14.0%, and d16 10.5->12.0%.
- K6 versus K1 changes 3.42% of first-answer predictions, with 12 corrected
  versus 6 broken examples. This is directionally positive but too small for a
  method claim; the pretrained blocks were never optimized for repeated use.
- Recurrent fine-tuning pilot job 62166105 was submitted from the 100k EMA
  checkpoint. It samples K uniformly from 1--4 during training for 2,000 steps,
  while holding the flow-matching objective and outer dynamics fixed.

## 2026-09-26: recurrent fine-tuning outcomes

- Random-K evaluation 62166120 improves the checkpoint at K=1 but shows no
  consistent marginal value from extra loops; random K mainly trains output
  invariance across compute budgets.
- Depth-conditioned evaluation 62166136 produces a promising but localized d16
  effect (8.0% at K1 to 14.5% at K4; paired +18/-5), while d8 degrades and d12
  stays nearly flat. This is not yet general test-time scaling.
- Two badly scaled intermediate-loss pilots (62166143, 62166149) were stopped
  early. The corrected cosine-readout pilot 62166153 trains stably and lowers
  8-way intermediate-state CE below chance, but evaluation 62166159 still lacks
  endpoint K-scaling.
- Conclusion: simply tying and repeating feed-forward blocks is insufficient.
  The next method needs an explicit persistent reasoning memory and inner-time
  signal, supervised as a state-transition system while outer flow NFE remains
  fixed. Plot: `analysis_artifacts/recurrent_reasoning_pilots.png`.

## 2026-09-26: causal recurrent-memory bottleneck

- Non-bottleneck memory job 62166249/eval 62166403 learns highly decodable
  intermediate states (CE 1.45 vs log(8)=2.08) but no K-scaling, confirming the
  original token path bypasses memory.
- Weak direct coupling job 62166443/eval 62166461 also remains flat; its learned
  coupling gate stays near the 0.018 initialization.
- Causal-bottleneck job 62166469/eval 62166476 discards direct recurrent token
  updates and forces velocity to use memory. It produces the first clear
  scaling result: d4 exact accuracy 13.5/14.5/19.5/22.5% for K=1/2/4/6.
  K6 vs K1 is paired +18/-0 on d4 (p=7.6e-6), and +31/-9 overall
  (p=6.8e-4). Lookup remains 99.9%.
- Low-LR continuation job 62166495 completed 5k additional steps.  The 5k
  checkpoint strengthens the compute effect substantially: K=1/2/4/6 overall
  accuracy is 70.25/70.96/73.83/73.96%; d2 is 96.0/99.0/99.0/98.5%; and d4
  is 12.5/16.0/51.5/51.5%.  On d4, K1->K4 gives 78 paired corrections and
  zero regressions (one-sided exact p=3.3e-24), while lookup stays 99.9%.
- This is task-selective test-time compute scaling at fixed outer NFE and fixed
  parameter count. Extended job 62167151 shows d4 accuracy
  12.5/16.0/34.0/51.5/51.5/47.5/46.5/45.0% at
  K=1/2/3/4/6/8/12/16: accumulation, saturation near K=4--6, then mild
  over-iteration. It still does not transfer to unseen d8/d12/d16.
- Two-dimensional memory tracing (job 62167163) shows k1 reads s1 (77% at the
  final model call), k2 reads s2 (98.5%), k3 already reads the final s4 (88%),
  and k4 stabilizes s4 (99.5%); explicit s3 is weak. This structure is already
  present at t=0 and nearly stationary over flow time. The model implements a
  compressed iterative computation, not a literal four-state symbolic trace.
- Temporal compute windows (corrected job 62167172) show equal compute is more
  valuable early: one earliest K4 call gives 21.0% d4 accuracy versus 17.0%
  for one latest call (paired +9/-1, p=.0107); eight early versus eight late
  calls give 30.0% versus 25.5% (+13/-4, p=.0245). Full K4 remains best at
  47.0%, so computation is distributed rather than purely one-shot.
- Single-call impulse job 62167191 estimates a causal compute-response kernel.
  A K1->K4 impulse at t=0 adds 4.5 accuracy points (9 corrections, 0
  regressions), decaying to 0.5 points late in the flow. Across 16 times there
  are no regressions; kernel magnitude correlates with remaining integration
  horizon (Pearson r=.876, p=8.8e-6; Spearman rho=.930, p=1.8e-7).
- Replications 62167203/62167204 and uniform-grid job 62167202 preserve the
  early-compute advantage. Mean impulse gains over the earliest versus latest
  four calls are 3.38/.63, 2.75/.75, 1.38/.38, and 4.88/.50 percentage points
  across the four runs. The uniform-grid kernel has Spearman rho=.979 with
  remaining horizon. Across all 64 impulses there are zero correct-to-wrong
  transitions. Plot: `analysis_artifacts/recurrent_compute_response_kernel_replication.png`.
- Plot: `analysis_artifacts/recurrent_reasoning_bottleneck_comparison.png`.
