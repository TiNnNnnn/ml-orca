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
