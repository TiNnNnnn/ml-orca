# ML-ORCA

pgORCA 的 trace 采集、处理、规则图编码、学习与实验组件。生产代码不再位于
`test/dsl`，也不导入测试代码；不改变 ORCA 的优化器行为。

独立仓库：https://github.com/TiNnNnnn/ml-orca 。pgORCA 在 `tools/ml-orca`
通过 Git submodule 固定版本。外层是仓库根目录，`ml_orca/` 是可安装的 Python 包。
初始代码提取自 pgORCA `f5a0911b` 的 learnable 组件及其 GPU 续训增量；保留 Apache-2.0 许可证。
不携带 pgORCA 历史、私有规则、SQL 数据、trace、模型或 checkpoint。

独立部署（包括 GPU 主机）：

```sh
git clone https://github.com/TiNnNnnn/ml-orca.git
cd ml-orca
python3 -m pip install -e .
python3 -m ml_orca --help
python3 -B -m unittest discover -s ml_orca/tests -t .
```

训练按主机安装 CPU/CUDA PyTorch。`PGORCA_ROOT=/absolute/path/to/pgorca` 指定采集/集成测试
需要的 SQL/schema/规则资产；包位于 pgORCA 子模块内时可自动发现。使用冻结 manifest 训练
不需要 PostgreSQL 源码、服务器或重新采集。模型、目标和编码层之间的依赖边界保持不变。
更新代码用 `git pull --ff-only`；**不要覆盖运行中训练的源码**，它会在结束时校验代码快照。
应在新 checkout 中测试，再从完整 checkpoint/manifest/vocabulary 包续训。

## 分层与数据流

```text
ORCA trace / catalog / rule DSL
    → collect → trace / data → encoding → models → training
                      graph ──────┘                  ↑
                                              objectives
```

| 目录 | 职责 |
| --- | --- |
| `collect` | 执行查询、采集 catalog/statistics、候选 trace、workload 对照 |
| `trace` | 解析候选、Memo 来源、规则触发及替代证据 |
| `data` | workload 导入、历史恢复、数据准入和不可变样本导出 |
| `graph` | 合并静态/动态有向依赖图、邻域与可视化 |
| `encoding` | DSL 树、绑定约束、query、真实 GroupExpression 与历史上下文特征；不依赖 PyTorch |
| `models` | 网络及共享层；不读取数据文件，不选择训练目标或调用采集器 |
| `objectives` | 标签、变换、策略选择评价、DRO 数学；不依赖网络 |
| `training` | 组合输入、网络与目标，管理 split、优化器、checkpoint 和评价 |
| `experiments` | 基数扰动、响应曲线、DRO 校准及其他离线实验 |
| `common` / `tests` | 共享工件校验、路径、替代契约 / 独立组件测试 |

## 网络与目标不能混为一谈

| 网络/共享层 | 实现 | 用途 |
| --- | --- | --- |
| 平面序列基线 | `models/rule_policy_model.py` | `--rule-encoding flat`；序列规则编码，集合或静态图聚合 |
| 有序树网络 | `models/rule_tree_model.py` | `--rule-encoding tree`；DSL 树、约束符号绑定、精确依赖根、多轮有向消息 |
| 通用序列编码 | `models/sequence.py` | 局部字段、query/catalog/attempt 编码 |
| 图消息层 | `models/message_passing.py` | 前驱/后继、根绑定与重复边的聚合 |

| 训练入口 | 目标模块 | 预测量与可用信息 |
| --- | --- | --- |
| `train-observed` | `objectives/observed_search.py` | 当前 trace 的 match+constraint、instantiate 累计耗时；**回顾性**，可使用当前真实 GE/规则图，不能解释为事前策略收益 |
| `train-policy --objective planning` | `objectives/policy.py` | 候选策略下的实际规划时间；仅使用预测时可获得的 query/catalog/规则与准入历史 |
| `train-policy --objective plan_execution` | `objectives/policy.py` | 实际规划、执行时间；完整 workload 策略比较，不逐查询偷看标签选策略 |

耗时均使用 `log1p(ms)` 目标；训练采用 Smooth L1。目标缺失或失败不会补成零。
`ObservedSearchPredictor` 复用树网络，仅显式改变当前观测/历史准入边界，不复制网络。
`graph/context/none` 是信息通道消融，不是新的训练目标。图边、DSL 子树和约束不因迁移裁剪。

### 新主线：净效用排序（合同与观测审计阶段）

`objectives/rule_utility.py` 独立定义 `rule-net-utility-v1`：预测 D/F/C，使用冻结的
plan-cost/search-work 尺度及偏好权重得到净效用。C 是预测量，不是仅作硬预算；
不要求精确预测毫秒。已有时间模型、训练目标和 checkpoint 不改名。
`rank_policy_predictions` 对相同查询及权重上的完整候选策略评分，不相加独立规则分数；
策略身份必须包含集合和优先级。缺失、失败、重复或额外查询/策略直接拒绝，不丢弃困难样本。
这是配置排序接口，尚不是训练好的效用模型、图束搜索、promise 发布或 DRO 证书。

使用现有 `comparison.json` 审计尝试、来源和 best-cost 生命周期，不执行 SQL：

```sh
python3 -m ml_orca audit-utility --comparison /path/to/comparison.json \
  --run /stats_experiments/0/modes/replacement --output /path/to/new-audit.json
```

`--run` 是实际 JSON pointer，可重复；省略时审计其中所有运行。复用现有完整性检查，
区分首次可行计划、同 context 的成本改善、不同统计水位/属性的不可比事件。
源触发实例编号缺失只报告缺失，不根据规则 id 或时间相邻猜测 F/C 标签。
带 `cost_origin_trace_version=1` 的新 trace 记录 costing 当时的 DSL 生成实例，
审计检查实例/规则、来源链位置和时间水位；旧 trace 缺该字段时不推测补齐。
`instance_work_ledger` 按实例集合去重菱形后继与成本事件，区分生成来源与已有输入暴露。
这些是非独占的观测工作范围，不是完整 C 或因果贡献；跨规则不可直接相加。
`physical_plan_source_audit` 沿 costing 当时保存的 child candidate 引用展开 best/终态计划，
区分根 GE 来源、完整物理子计划的生成来源、已记录插入来源和已有输入暴露。
不会使用 Group 后来的 best 反推旧计划，也不把重复生成者当成多个必要前提。
`terminal_plan_quality` 单独保留经最终根候选核对的完整策略 cost：首次可行计划无改善事件，
也不能视为无收益；跨策略比较仍需相同查询/数据/统计/cost 模型与独立结果校验。
输出拒绝覆盖已有文件；该阶段不生成训练就绪标签或自动改动优化器策略。

可选 `--dsl-credit-share 0.5` 启用 `new-inserter-equal-credit-v1` 的观测信用记账。
份额必须显式设置，是敏感性参数而非实测概率；没有默认的背景/DSL 贡献比例。
仅对可比的根成本下降，选择新计划已记录插入者中不属于旧计划生成来源的实例，
与去重严格前驱等份分享指定份额，余量保留未归属。首次可行、失败及不可比事件保持掩码。
`root_gain_credit.complete` 只表示记账合法，不能用 D/F 占比证明因果价值或替代终态策略监督。

`objectives/search_work.py` 冻结 `observed-search-events-v1` 的可观测工作向量，
`audit-utility` 的 `search_work_audit` 同时输出分派/实际评估/预算跳过、各状态的成本入口、
剪枝与属性检查次数。任一事件流不完整则向量缺失，不补零；同一次成本入口不一定实际算了 cost。
向量不是标量 C 或毫秒，组合前必须冻结权重和尺度；未覆盖的绑定构造、调度及 exclusive 工作单列。

新 `cost_progress_version=1` trace 在成本保留/最优更新时记录实际工作水位。
`audit-utility` 的 `root_search_progress` 以该时点的规则尝试、成本入口、搜索检查分别表示进度，
不把候选创建编号当成更新时间；首次可行之前标记无计划、终止后不外推。
旧 trace、计数倒退/越界、统计版本变化或终态成本不一致均不输出可用曲线。
它是完整策略的观测轨迹，不是逐规则因果标签，也不等价于经过多少毫秒。

当前优先研究排序：固定规则集合、CBO 阶段和预算，不扫描/优化预算。
`python3 -m ml_orca.experiments.generate_priority_control --audit-bin ... --rules ...
--policy ... --seed 20260914 --output ...` 从原生快照生成随机顺序控制，保持禁用项及每条
规则的完整配置，并通过原生回读检查仅启用规则的 priority 改变。输出拒绝覆盖。
随机顺序不是模型预测；当前 CBO priority 只作用于同一绑定的候选，不是全局 Memo 调度。

`compare-workload --feature-graph` 可以原样冻结原生 v1 静态图或合并后的 v2 图，
无需为使用静态树/约束编码而引入历史响应。版本、节点身份和边端点仍严格检查。
`export-policy --include-search` 在原时间响应之外追加独立 `response.search`：终态质量、
工作向量、首计划/后续更新轨迹和 trace 审计（包括 costing 时的生成实例与物理子计划来源）。
来源审计只导出完整性和排除原因，不复制庞大的来源闭包。缺失保持掩码，结果不进入 inputs，
不自动解除训练准入限制，也不把完整策略响应当作单条规则的 D/F/C 因果标签。
单策略标签不要求参考策略成功：仅当本策略诊断完整、无 trace 计划匹配且独立 PG 校验通过时，
精确分离已核实的参考策略失败。原 timing_samples 的成对比较排除原因保留，
参考差值仍无效；未知错误、目标不一致和本策略失败仍拒绝。

新的输入树 trace 在有 table descriptor 的位置记录 `relation_oid`。它只用于关联同一数据库
采集前的 catalog，不进入神经特征。标准 `ml_orca.encoding.rule_history_encoding` 导出入口
复用已验证的 comparison/context 快照，在对应树节点附加独立的 `ge:catalog:*` 特征；
不替换 `ge:rows`，不触发统计推导，也不从未来 costing 回填统计。表重命名/OID 重编号不改变
编码；未知关联、缺失行数与零行数保持区别。旧 trace 没有关系身份时仍标记未知。
旧 corpus/recovery 调用默认不附加 catalog，已有历史工件/模型不重写；这不等于任意派生
GroupExpression 已有基数，也尚未提供任意基数干预配置的历史编码。

`--capture-pre-context` 同时保存 `stats_experiment_requests`：调用原生
`pgorca_rule_audit --stats-requests`，复用运行时基数配置解析器，不另写 YAML 解析器。
`requested_rows` 仅表示前置请求，不是已解析到查询的目标或实际注入结果；解析错误、
旧二进制和超时保持 error，不改成空请求。原始文件与首尾完整性校验仍保留。
新 trace 的 source_tree 使用 `request_binding=resolved_operator_only` 和节点 `request_index`
关联冻结配置；`encode_observations(..., stats_requests=...)` 可输出独立的
`ge:requested_rows`，不把序号/指纹当作特征，不把发现目标当作请求。全局策略干预编码
尚未完成，现有干预训练准入限制不放宽；局部尝试树可能完全不含请求目标，不能硬填行数。

新 trace 还在预处理后、Memo 初始化前捕获一次 `query_input_context`，比较器保存在
`query_input_contexts`。读取器检查分片完整性、实验身份和先于 CBO 搜索的顺序。
`encoding.group_expression_encoding.query_input_tree` 复用实际树编码，在 CBO-only 零前序
水位下关联全查询根表达式中的请求；后续超时不抹除已经完整捕获的输入。
完整性限于根表达式树，外部 CTE producer/完整标量语义/物理需求仍标记未观测。
`encoding.rule_policy_encoding.pre_memo_input_sequences` 与
`TreePolicyPredictor(..., query_input=True)` 显式接入该树：复用有序 Tree-LSTM，
融合查询/catalog 表示后再进入 DSL 树、约束、根绑定有向消息与已准入历史 GE。
默认模型/旧 checkpoint 不变，默认模式拒绝新输入；决策点是
`post_preprocessing_pre_cbo`，不能静默改变旧训练输入合同。当前训练 CLI 尚未启用它。

`objectives.priority.priority_pair(left, right)` 接收 `export-policy --include-search`
导出的同一冻结 comparison 内的两个策略单元。它检查相同 SQL、catalog、统计干预、
runtime、规则集合及全部预算，只允许 priority 不同；无限制参考组不能混入同预算排序标签。
返回最终 cost 的 `log1p(left)-log1p(right)`、观测胜负/平局，以及分阶段搜索事件计数差。
这些是完整优先级策略的响应，不是单条规则 D/F，也不将计数相加伪装成时间或完整 C。
诊断/独立 PG 结果校验与时间采样分开；失败/缺失保留排除原因，cost/work 独立掩码，
完整 trace 单独标记。平局仅表示记录精度下 cost 相同。接口不自动批准训练，仍需
独立查询划分、明确目标头及留出选择评估；旧规划/执行时间训练目标不变。

`TreePolicyPredictor(..., output_size=1)` 可显式使用单个相对策略分数；对应的
`training.priority_loss.priority_cost_loss` 对同一输入/场景的完整已审计策略对拟合
log1p(cost) 差，保留平局，拒绝缺失 cost/work/来源，分数不解释为绝对 cost 或毫秒。
Tensor 损失位于 training，objectives 的标签合同继续不依赖 PyTorch。
默认 output_size=2 的旧模型布局/初始化不变，训练 checkpoint 必须声明目标和模型配置。

仅供结构读出诊断的 `query_pooling='root_mean'` 在 query_input=True 时增加所有
Tree-LSTM 子树状态的均值通道，保留原有完整有序树和根状态；默认仍为 root。
新配置有不同的融合矩阵，不能无声明加载旧 checkpoint。单查询两端点的固定 40 轮
诊断仅有小幅 loss 改善，仍明显落后于训练集的每策略常量，未证明泛化或推荐价值；
不要将该选项视作已验证的默认优化，也不启用新的大规模训练。

新 observed 训练 manifest 明确记录网络、目标、输入范围、宽度和消息轮数；续训拒绝
不兼容的配置。兼容旧 checkpoint 的 state_dict/Adam/RNG/样本位置，不改历史工件和校验值。
旧 manifest 的源代码路径是历史来源记录，不会被重写成新路径；重新审计历史采集代码时
需恢复其对应版本。新运行记录 `tools/ml-orca` 源码快照。

## 使用

从 pgORCA 根目录执行：

```sh
git submodule update --init tools/ml-orca
python3 -m pip install -e tools/ml-orca
python3 -m ml_orca --help
python3 -m ml_orca collect --help
python3 -m ml_orca export-history --help
python3 -m ml_orca train-policy --help
```

训练另需 PyTorch：按运行主机的 CPU/CUDA wheel 安装
`tools/ml-orca/requirements-training.txt`。采集、处理和入口帮助不要求 PyTorch；
绘图命令另需 matplotlib/Graphviz 及中文字体。没有自动安装或训练的副作用。

完整命令仍支持模块入口，例如：

```sh
python3 -m ml_orca.experiments.generate_stats_sweep --help
python3 -m ml_orca.collect.run_trace_corpus --help
python3 /absolute/path/to/pgorca/tools/ml-orca/ml_orca --help
```

只用现有数据续训，不触发任何 SQL 采集：

```sh
python3 -u -B -m ml_orca train-observed \
  --corpus output/gnn-history-2000-v1 \
  --admission output/gnn-history-2000-v1/admission-empty-pilot-v1.json \
  --resume output/gnn-observed-search-v2/latest.pt \
  --output output/gnn-observed-search-v3 \
  --epochs 100 --threads 16 --mode graph --seed 929
```

输出目录必须不存在。checkpoint 旁的 manifest/vocabulary 必须保留；仅复制 `.pt`
不是可复现的续训包。输入快照变化会拒绝续训。线程数可调，不改变样本、轮数或目标。
当前空表 pilot 的搜索耗时标签不能作为有数据的 SQL 执行收益证据。

GPU 续训使用 `--device cuda:0`，通过 `CUDA_VISIBLE_DEVICES` 限定允许使用的卡。
模型保持 FP32，不开启 AMP/TF32；仍然逐查询更新 Adam，不改变轮数或样本顺序。
CPU/CUDA 内核存在浮点舍入差异，不能宣称跨设备逐位一致；设备、PyTorch/CUDA
版本写入 manifest，CUDA RNG 写入 checkpoint。CUDA 不可用时显式报错，不静默回退。

跨机器复制后，可加 `--artifact-root /original/pgorca /copied/pgorca`。该选项只在
本次调用中映射工件位置，保留原 manifest/词表/样本内容及原始路径身份；大小和 CRC
校验仍执行。`profile-observed` 同样支持这两个选项，CUDA 计时同步实际完成的计算。
续训始终使用原 manifest 的标签；只在对已校验原始 summary 重算 `log1p` 时允许
相邻一个 binary64 值的跨平台 libm 舍入差，零值必须完全一致，更大的差异仍拒绝。
部署只需已有训练 manifest 引用的数据、catalog、静态规则图，以及 checkpoint 旁的
manifest/词表和当前代码；不需要重新执行 SQL，也不需要复制全部原始 trace。

已有公开 SQL/schema/rules 回归资产仍放在 `test/dsl`，生产模块只是按默认路径读取它们；
私有完整规则、采集数据、模型、图片继续放 `output/`，不进 Git/CI。
原 `python3 test/dsl/<工具>.py` 调用需改成上面的模块入口，不在测试目录保留生产转发器。

## 验证与 CI

```sh
python3 -B -m unittest discover -s tools/ml-orca/ml_orca/tests -t tools/ml-orca
python3 -B test/dsl/test_trace_framework.py
```

组件测试不要求 PostgreSQL 服务或私有数据。无 PyTorch 时网络测试跳过；独立
`ml-orca` CI 安装 CPU PyTorch 并运行网络/梯度/Adam 续训测试。原生 DSL E2E
仍由原测试入口负责，生产采集入口的 CI 已更新。

可选 CTest 集成：配置 `-DPGORCA_BUILD_ML_ORCA_TESTS=ON` 后执行
`ctest --test-dir build -L ml-orca --output-on-failure`。默认 C++ 构建不增加 Python/ML 依赖。

原生 E2E 保留独立的 `replacement_contract.py` 协议校验，不依赖 ML 包。
ML-ORCA 的工件读取使用同一协议副本，pgORCA 集成测试校验两者一致，防止跨仓漂移。

## 保持模型的计算优化

树编码采用森林内的拓扑层批处理：同层、同 arity 的节点一起调用原有 sibling GRU 和
Tree-LSTM，孩子顺序不变。按依赖深度归纳，每个节点仍计算相同的函数；这不是将树
改为池化模型。规则拼接时只调整张量地址，输入 token 中的规则局部符号编号保持原样。
约束 incidence 为互不相连的各规则分块；重复引用、根绑定、全部 attempt/edge 仍参与梯度。
共享子层、网络参数名称、初始化、三轮有向消息和逐查询 Adam 更新均未改变。

浮点批处理可能改变求和/内核舍入，不能宣称逐位相同；测试对照保留在
`tests/scalar_tree_reference.py`，验证有序/变 arity/多树/空约束、全部参数梯度和连续 Adam 更新。
不跨训练步骤缓存 learned embedding，不改 batch size、精度、目标或样本顺序。

固定输入的相同序列共享只读 Python 对象；每次前向仍按当前参数重新编码，再按原顺序
取回全部重复项。对象身份索引只活在一次前向内，不跨更新复用。约束端口批量写入
原有的 padded token context，保留所有符号引用及梯度；消息层将不变的节点数量移出
逐边校验循环，仍校验全部端点。输入/模型测试覆盖非法类型、非有限数、padding 梯度、
共享对象修改后的重新校验，以及与原逐约束实现的参数/Adam 对照。

### 固定输入缓存与有序预取

`train-observed --input-cache-mb 32768 --input-workers 0` 可在大内存主机保留
至多 32 GiB（按 Python 对象大小保守计数）的固定输入编码，避免每轮重复解压、JSON、
SQL 绑定及 token 编码。缓存不含 Tensor 或 learned embedding；**每次命中仍读取原图并
校验 size/CRC**。超预算按 LRU 淘汰，单个超大输入不缓存，绝不删边/树/样本。
第一遍包含对象大小统计和缓存填充开销；命中率不足时不能预期同样的收益。

`--input-workers N --input-prefetch 1` 使用 PyTorch DataLoader 的 spawn 多进程与
persistent workers。CPU 准备可与训练重叠；采用独立 RNG、有序 sampler 和不转换输入的
collate，逐查询 Adam 更新/恢复位置/验证顺序不变。缓存总预算平分到各 worker，
不包含运行时、在途消息和模型内存。完整 Python 大图跨进程复制可能比主进程缓存慢；
**因此默认不启用多进程，先测量再选择**，不能把更多 worker 当作必然加速。

进度记录将主进程等待时间与 worker 的校验、编码时间、缓存命中和保留字节分别记录。
默认 `--input-cache-mb 0 --input-workers 0` 不增加缓存占用。CPU 测试逐位对照原输入、
预测、全部梯度和连续 Adam 更新，并覆盖 byte budget、文件变化拒绝、跨机器映射与 RNG。

仅测量已有图的输入准备/缓存/IPC（不运行 SQL、不更新模型；不代表 GPU 重叠收益）：

```sh
python3 -m ml_orca profile-observed --inputs-only \
  --manifest /data/observed/manifest.json \
  --case lobsters:131 --case discourse:3990 \
  --input-cache-mb 32768 --input-workers 0 --repeats 2 \
  --output output/input-cache.json
```

报告区分首次填充与后续重复读取；完整训练时间仍需结合前向、反向、验证和真实命中率。

固定 checkpoint 的完整真实图测速（不执行 SQL、不改训练 checkpoint）：

```sh
python3 -m ml_orca profile-observed \
  --manifest output/gnn-observed-search-v3/manifest.json \
  --checkpoint output/gnn-forest-batching-v1/checkpoint.pt \
  --case lobsters:131 --case discourse:3990 \
  --threads 16 --repeats 3 --profile --output output/model-profile.json
```

`--reference-source` 可载入可信的迁移前模型源码作对照。每次测量前恢复相同参数和 Adam
状态；预热与额外的 cProfile 步骤不计入中位数。读入/编码单独计时，串行 A/B 时暂停其他
训练/基准进程，不能把后台争用、不同样本或 profiler 开销当成加速效果。
