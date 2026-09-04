# LM-LitAgent — 液态金属文献调研智能体

当前版本为 5.3.1，面向 GOAI AI4R 材料方向。包含检索、结构化
抽取、融合、主张审计、GA/BO baseline 和 ERCPD 来源稳健性搜索。

## 快速运行

Python 3.10+，仅依赖标准库。无需 API 即可运行测试和离线演示：

```bash
python3 -m unittest discover -s tests -v
python3 run.py --offline --max-papers 5
python3 scripts/secret_scan.py .
```

结果写入本地 `outputs/`，已被 Git 忽略，不应上传包含文献内容的在线输出。

在线使用时，将 `.env.example` 复制为本地 `.env`，配置自己有权使用的
MiniMax Coding Plan 与 Sciverse 凭据。不要提交真实 `.env`。

```bash
cp .env.example .env
# 只在本地编辑 .env；自行确认账号权限、额度及文献使用条款
python3 scripts/check_apis.py
python3 run.py --strict --max-papers 10
```

默认模型与端点见 `.env.example`；它们是可配置的别名，不保证供应商永久
提供相同版本。离线运行不调用服务，不能证明在线 API 当前可用。
所有原创 Prompt 及动态构造逻辑保留在 `agents.py`。

## 此仓库与参赛复现包的区别

- 核心模块基于初版继续修复为 5.3.1：BO 后验均值改为使用观测 y；抽取、证据、快照比较、缓存和 HTML 输出均有回归修复，不与旧冻结代码混称同一版本。
- `papers.py` 已换成明确标注、无真实 DOI/作者的合成测试记录，避免传播
  来源未确认的旧叙述。它们不是文献、实验数据或已验证的材料属性。
- 因输入不同，此仓库的离线抽取、报告及下游结果不保证与原参赛冻结运行
  一致；原参赛包保留原输入、运行与校验值，不用本快照冒充原始版本。
- 不包含原生会话、历史 Git 对象、初赛 PDF、完整提交 ZIP、在线检索正文、
  知识卡片、API 缓存或账号信息。
- 内置 25 个整理锚点仍待逐项原始来源核验；源码中的书目仅是记录，不是
  已审核的原文证据。代码许可不重新授权第三方文献。

## 科学与实现限制

ERCPD 是来源留一代理模型上的风险调整和守恒组分反事实搜索，至多产生
C1 计算候选假说，不是实验验证，也不以新增 DSC 实验为前提。

5.3.1 已修复 Galinstan 误归类，排除复合材料/纯合金直接比较，并修复 BO 均值
未使用观测适应度的错误。旧 64.71%“异常”和旧 BO 基线均不作为有效证据。
来源标准差降低不等于预测误差降低；新增等预算对照也仅适用于当前代理和
固定种子，不证明普遍优势。详细验证结果及剩余边界见 [TESTING.md](TESTING.md)。

```bash
python3 scripts/equal_budget_benchmark.py --output outputs/equal-budget.json
python3 scripts/check_apis.py --generate  # 实际生成，会消耗账号额度
```

在线运行生成的 `source_text` 和证据片段仅供本地追溯，不应未经授权发布。

## 许可

采用 [LM-LitAgent 非商业源码共享许可证 1.0](LICENSE)，自定义标识
`LicenseRef-LMLitAgent-NC-Reciprocal-1.0`：

- 禁止商用；商业使用须取得权利人另行书面授权。
- 非商业内部使用及内部修改，不要求公开源码。
- 对外发布修改版或用修改版向外部提供服务时，须免费公开该版本完整对应
  源码、保留声明，并让修改部分继续适用同一许可证。
- 商业内部使用仍受禁止商用约束；内部修改豁免只豁免源码公开义务。

这是“非商业源码可用”，不是 OSI 意义的开源许可。正式条款以 `LICENSE`
为准；自定义文本未经过独立法律审查。第三方内容和服务适用其自身条款。

## 安全与发布

仓库当前按团队要求保持 private；不会自动改为 public。即使只公开几天，
下载与公开 fork 也无法靠改回 private 收回。公开前应再次检查全部 Git
历史、Actions 日志和附件。完整参赛包不应整包上传此仓库。
