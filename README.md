# LM-LitAgent：液态金属材料科学文献调研智能体

LM-LitAgent 面向 GOAI AI for Research 材料科学方向，围绕“文献驱动的科学发现智能体”构建从文献检索、知识抽取、证据追溯到构效关系发现的完整工作流。

当前版本为 **5.3.1**，重点实现路线 A：**构效关系发现（结构—性能关联挖掘）**。系统以液态金属组成—性能分析为示范场景，在缺少新增实验的条件下，从多来源文献信息中生成可解释、可追溯的计算候选，为后续实验验证提供依据。

## 核心能力

- **文献检索与结构化抽取**：获取候选文献，提取材料组成、性能数值、单位、测试条件和证据位置。
- **知识融合与证据审计**：规范化属性和单位，识别跨文献一致性、冲突项与研究空白。
- **ERCPD 构效发现**：结合来源留一评估、风险调整 Pareto 搜索和守恒组成反事实分析，降低候选对单一来源的依赖。
- **多目标候选筛选**：同时考虑熔点、电导率、风险和组成约束，保留多目标权衡关系。
- **可复现运行**：统一保存配置、日志、中间结果、最终报告及输入文件哈希。
- **在线与离线双模式**：支持 MiniMax 与 Sciverse 在线工作流；合成数据用于程序演示，公开数值快照用于科学计算复现。

## 5.4.0 更新：文献观测接入路线 A

统一属性名、单位和证据校验后，配比明确的在线观测可直接进入 GA、BO 和 ERCPD 的共享代理模型。每个属性只使用实际存在的观测，不用默认值填补缺失性能；来源留一会同时移除该来源的全部新增观测。

对已保存的 50 篇结果回放，并对其中 5 篇进行 MiniMax 定向重抽取后，保留 **20 条有效物性记录**，其中 **12 条为代理关注的五类物性**；**8 条带明确质量配比的物性观测**进入代理，覆盖 3 个配比和 3 个文献来源组。路线 A 输出 **3 组跨来源配比—物性对照**和 **4 条模型反事实趋势**。跨来源对照未控制测试条件，模型趋势不是实测规律。

## 无 API 的科学计算复现

```bash
# 重算提交时的固定锚点数值基线：候选、反事实、等预算对照
python3 scripts/reproduce_science.py --output outputs/reproduce-baseline

# 重算 5.4.0 新增观测接入后的预测、对照、来源留一和候选
python3 scripts/reproduce_online.py --output outputs/reproduce-online
```

两条命令均自动比对冻结预期值并输出 PASS/FAIL；重复运行请使用新的输出目录。公开输入位于 `data/`，包含归一化数值、配比、来源标识和溯源哈希，不含文献正文或证据引文。数据范围、版本区别和具体结果见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 方法亮点

传统文献分析容易将“样本数量”误认为“证据独立性”。ERCPD 将文献来源作为显式扰动因素：每次留出一个来源组，重新评估候选的代理性能，再以风险调整目标进行 Pareto 筛选。最终输出不仅包含候选组成，还包含来源波动、性能代价和反事实变化方向，使候选结论更便于解释与验证。

```text
文献检索
   ↓
结构化知识卡片
   ↓
知识融合与证据核验
   ↓
来源留一代理评估
   ↓
风险调整 Pareto 搜索
   ↓
守恒组成反事实与研究报告
```

## 快速开始

运行环境为 Python 3.10+，核心代码仅使用 Python 标准库，无需 GPU。

```bash
git clone https://github.com/Xiang-zy/ai4r-liquid-metal-research-agent.git
cd ai4r-liquid-metal-research-agent

# 运行全部自动化测试
python3 -m unittest discover -s tests -v

# 无需 API 的离线演示
python3 run.py --offline --max-papers 5
```

每次运行默认创建独立的时间戳目录，并输出知识卡片、研究空白、路线 A 分析、消融结果和 HTML 调研报告。

## 在线运行

将环境变量模板复制为本地 `.env`，填写自己有权使用的 MiniMax Coding Plan 与 Sciverse 凭据：

```bash
cp .env.example .env
python3 scripts/check_apis.py
python3 run.py --strict --max-papers 10
```

`.env` 已被 Git 忽略，不应提交真实密钥。模型名称和服务端点均可在本地配置。

## 等预算算法对照

仓库提供 Random、GA 和 BO 的统一预算比较脚本。各方法使用相同搜索域、目标函数、固定种子和目标调用次数：

```bash
python3 scripts/equal_budget_benchmark.py --output outputs/equal-budget.json
```

该脚本保存逐次搜索轨迹与汇总结果，便于独立检查评估预算和候选生成过程。

## 测试与质量保证

5.4.0 发布版通过 **80 项自动化测试**，覆盖：

- API 请求与响应兼容性；
- 数值、单位和测试条件抽取；
- 证据定位与跨来源融合；
- BO、GA 及 ERCPD 优化逻辑；
- 组成守恒和反事实方向；
- 缓存隔离、错误处理与输出安全；
- 完整离线命令行工作流。

测试与复现摘要见 [TESTING.md](TESTING.md)。

## 项目结构

```text
.
├── agents.py                    # 抽取、融合、Gap 与报告 Agent
├── literature_data.py           # 文献数据结构与属性处理
├── optimizer.py                 # GA、BO 与 ERCPD
├── sciverse_client.py           # Sciverse 检索客户端
├── run.py                       # 工作流入口
├── papers.py                    # 可公开的合成演示数据
├── scripts/
│   ├── check_apis.py            # 在线接口检查
│   ├── equal_budget_benchmark.py # 等预算算法对照
│   └── secret_scan.py            # 敏感信息扫描
└── tests/                        # 自动化测试
```

## 科学定位

LM-LitAgent 输出的是文献数据与代理模型支持的**计算候选**，用于缩小后续研究和实验验证范围。系统强调来源稳健性、证据可追溯性和多目标权衡，不将计算候选表述为已经完成的材料实验结论。

## 许可证

本项目采用 [LM-LitAgent 非商业源码共享许可证 1.0](LICENSE)：

- 允许非商业使用、研究和内部修改；
- 非商业内部修改无需公开源码；
- 对外发布修改版本或使用修改版本向外部提供服务时，需要提供对应源码并沿用本许可证；
- 商业使用需要另行取得书面授权。

具体权利与义务以 [LICENSE](LICENSE) 正文为准。
