"""
管线编排器 + HTML报告生成 + 入口脚本 (v5.3 ERCPD版)
运行: python3 run.py
环境变量:
  MINIMAX_API_KEY  - MiniMax LLM API密钥
  SCIVERSE_API_KEY - Sciverse 文献检索API密钥
  LLM_MODEL        - MiniMax模型名（默认 MiniMax-M3）
  MINIMAX_BASE_URL - MiniMax OpenAI兼容端点（可选）
  SCIVERSE_BASE_URL - Sciverse API根地址（可选）
v5.3 改进:
  - 数值证据保存原文短句、文献标识与定位信息
  - 参考锚点明确标记为待原始来源复核
  - 冻结参考快照一致性检查与单位归一化
  - 路线A构效关系发现Agent, GA+BO迭代优化, 消融实验
"""

import json
import argparse
import html as html_lib
import os
import platform
import sys
import time
import math
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ===== 加载 .env 文件 =====
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"\''))
    print(f"[Config] .env 已加载: {_env_path}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    LLMClient, TaskPlannerAgent, LiteratureRetrievalAgent,
    LiteratureFilterAgent, KnowledgeExtractionAgent,
    KnowledgeFusionAgent, GapIdentificationAgent,
    EvidenceVerificationAgent, ReportGenerationAgent,
    StructurePropertyAgent,
)
from sciverse_client import SciverseClient
from optimizer import CompositionPropertySurrogate, run_ablation_study


class Pipeline:
    def __init__(self, offline=False, strict=False, target_paper_count=50):
        if offline and strict:
            raise ValueError("--offline and --strict are mutually exclusive")
        if target_paper_count < 1:
            raise ValueError("target_paper_count must be positive")
        self.offline = offline
        self.strict = strict
        self.target_paper_count = target_paper_count
        self.run_mode = "offline_demo" if offline else "online"
        self.llm = LLMClient()
        if offline:
            self.llm.mode = "template"
        self.sciverse = None

        sciverse_key = None if offline else os.environ.get("SCIVERSE_API_KEY")
        if sciverse_key:
            self.sciverse = SciverseClient(sciverse_key)
        if strict and (self.llm.mode != "api" or self.sciverse is None):
            raise RuntimeError("严格在线模式要求同时配置MiniMax及Sciverse凭据")

        self.agents = {
            "planner": TaskPlannerAgent(self.llm),
            "retriever": LiteratureRetrievalAgent(self.llm, self.sciverse),
            "filter": LiteratureFilterAgent(self.llm),
            "extractor": KnowledgeExtractionAgent(self.llm, self.sciverse),
            "fusion": KnowledgeFusionAgent(self.llm),
            "gap": GapIdentificationAgent(self.llm),
            "verifier": EvidenceVerificationAgent(self.llm),
            "route_a": StructurePropertyAgent(self.llm),
            "reporter": ReportGenerationAgent(self.llm),
        }
        self.steps = []
        self.total_time = 0

    def run(self, query="液态金属领域文献调研: 聚焦材料物性、柔性电子、软体机器人与可穿戴传感"):
        llm_mode = self.llm.mode.upper()
        sciverse_status = "已连接" if self.sciverse else "未连接"

        print("=" * 70)
        print("  多Agent文献调研系统 v5.3 (ERCPD证据稳健发现版)")
        print("  领域: 液态金属 (Liquid Metal)")
        print(f"  LLM: {self.llm.model} [{llm_mode}模式]")
        print(f"  Sciverse: [{sciverse_status}]")
        print(f"  运行模式: {self.run_mode}")
        print(f"  路线A: 构效关系发现 [已启用]")
        print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()

        pipeline_start = time.time()

        # Step 1: 任务规划
        print("[Step 1/8] 任务规划")
        t0 = time.time()
        scope = self.agents["planner"].run(query)
        scope["target_paper_count"] = self.target_paper_count
        self._record_step("任务规划", t0)
        print()

        # Step 2: 文献检索
        print("[Step 2/8] 文献检索 (Sciverse API, 多查询)")
        t0 = time.time()
        retrieved = self.agents["retriever"].run(scope)
        self._record_step("文献检索", t0)
        print()

        if not retrieved:
            if self.strict and not self.offline:
                raise RuntimeError("严格模式下 Sciverse 未返回文献，停止运行")
            self.run_mode = "offline_demo" if self.offline else "mixed_fallback"
            print("[INFO] 使用内置演示文献数据；该模式不代表在线检索结果")
            from papers import PAPERS
            retrieved = []
            for p in PAPERS:
                retrieved.append({
                    "id": p["id"],
                    "title": p["title"],
                    "authors": p["authors"][:3],
                    "journal": p["journal"],
                    "year": p["year"],
                    "doi": p["doi"],
                    "doc_id": "",
                    "citation_count": 0,
                    "score": 0.5,
                    "abstract": p["abstract"],
                    "chunk": p["abstract"],
                    "primary_topic": "",
                    "data_origin": p.get("data_type", "historical_demo_fixture"),
                })

        # Step 3: 文献筛选
        print("[Step 3/8] 文献筛选与评分")
        t0 = time.time()
        filter_result = self.agents["filter"].run(retrieved, scope)
        self._record_step("文献筛选", t0)
        print()

        # Step 4: 知识抽取
        print(f"[Step 4/8] LLM 深度知识抽取 ({self.llm.model} + 多chunk)")
        t0 = time.time()
        knowledge_cards = self.agents["extractor"].run(filter_result["selected_papers"])
        self._record_step("知识抽取", t0)
        print()

        # Step 5: 知识融合
        print("[Step 5/8] 跨文献知识融合")
        t0 = time.time()
        fusion_results = self.agents["fusion"].run(knowledge_cards)
        self._record_step("知识融合", t0)
        print()

        # Step 6: Gap识别
        print("[Step 6/8] Research Gap 识别 (LLM增强)")
        t0 = time.time()
        gaps = self.agents["gap"].run(fusion_results, knowledge_cards)

        self._record_step("Gap识别", t0)
        print()

        # Step 7: 证据核验
        print("[Step 7/8] 证据核验")
        t0 = time.time()
        verifications = self.agents["verifier"].run(gaps, knowledge_cards)
        self._record_step("证据核验", t0)
        print()

        # Step 8: 路线A - 构效关系发现
        print("[Step 8/8] 路线A: 构效关系发现 (LLM分析)")
        t0 = time.time()
        route_a_result = self.agents["route_a"].run(knowledge_cards, fusion_results)
        self._record_step("构效关系发现", t0)
        print()

        # 报告生成
        print("[Final] 报告生成")
        t0 = time.time()
        report = self.agents["reporter"].run(
            scope, filter_result, knowledge_cards, fusion_results, gaps, verifications, route_a_result
        )
        self._record_step("报告生成", t0)
        print()

        self.total_time = time.time() - pipeline_start
        total_props = sum(len(c['properties']) for c in knowledge_cards)
        fallback_cards = sum(c.get("extraction_mode") != "llm" for c in knowledge_cards)
        service_failures = self.llm.failed_call_count + (self.sciverse.failed_call_count if self.sciverse else 0)
        if (fallback_cards or service_failures) and not self.offline:
            self.run_mode = "mixed_fallback"
        if self.strict and (fallback_cards or service_failures):
            raise RuntimeError(f"严格模式失败: {fallback_cards}张回退知识卡片, {service_failures}次服务失败")
        route_a_rels = len(route_a_result.get("relationships", [])) if route_a_result else 0

        print("=" * 70)
        print(f"  管线执行完成! 总耗时: {self.total_time:.2f}s")
        print(f"  LLM调用次数: {self.llm.call_count} | Token用量: {self.llm.total_tokens}")
        print(f"  Sciverse调用次数: {self.sciverse.call_count if self.sciverse else 0}")
        print(f"  论文数: {len(filter_result['selected_papers'])} | "
              f"知识卡片: {len(knowledge_cards)} | "
              f"属性条目: {total_props} | "
              f"Research Gaps: {len(gaps)} | "
              f"构效关系: {route_a_rels}")
        print("=" * 70)

        return {
            "scope": scope,
            "filter_result": filter_result,
            "knowledge_cards": knowledge_cards,
            "fusion_results": fusion_results,
            "gaps": gaps,
            "verifications": verifications,
            "route_a": route_a_result,
            "report": report,
            "pipeline_stats": {
                "steps": self.steps,
                "total_time": round(self.total_time, 2),
                "llm_mode": self.llm.mode,
                "llm_model": self.llm.model,
                "llm_calls": self.llm.call_count,
                "llm_failed_calls": self.llm.failed_call_count,
                "llm_request_attempts": self.llm.request_attempt_count,
                "llm_last_finish_reason": self.llm.last_finish_reason,
                "llm_last_error": self.llm.last_error,
                "llm_tokens": self.llm.total_tokens,
                "sciverse_connected": self.sciverse is not None,
                "sciverse_calls": self.sciverse.call_count if self.sciverse else 0,
                "sciverse_failed_calls": self.sciverse.failed_call_count if self.sciverse else 0,
                "sciverse_request_attempts": self.sciverse.request_attempt_count if self.sciverse else 0,
                "sciverse_cache_hits": self.sciverse.cache_hits if self.sciverse else 0,
                "run_mode": self.run_mode,
                "fallback_cards": fallback_cards,
            },
        }

    def _record_step(self, name, t0):
        elapsed = time.time() - t0
        self.steps.append({"step": name, "time": round(elapsed, 3)})
        print(f"  -> 耗时: {elapsed:.3f}s")


# ============================================================
# HTML 报告生成器 (增强版: 知识图谱 + 属性分布 + 路线A)
# ============================================================

def generate_html_report(results, output_path):
    """生成专业的 HTML 调研报告 (v5.3 ERCPD版)"""
    def escaped(value):
        if isinstance(value, str):
            return html_lib.escape(value, quote=True)
        if isinstance(value, dict):
            return {k: escaped(v) for k, v in value.items()}
        if isinstance(value, list):
            return [escaped(v) for v in value]
        return value

    # Report interpolation handles text, never trusted HTML from a paper or LLM.
    results = escaped(results)

    scope = results["scope"]
    filter_result = results["filter_result"]
    cards = results["knowledge_cards"]
    fusion = results["fusion_results"]
    gaps = results["gaps"]
    verifications = results["verifications"]
    route_a = results.get("route_a")
    report = results["report"]
    stats = results["pipeline_stats"]

    total_props = sum(len(c["properties"]) for c in cards)

    # ===== 生成知识图谱SVG =====
    kg_svg = _generate_knowledge_graph_svg(cards, fusion)

    # ===== 生成属性分布图SVG =====
    prop_chart_svg = _generate_property_chart_svg(fusion)

    # ===== 生成路线A关系图SVG =====
    route_a_svg = _generate_route_a_svg(route_a) if route_a else ""

    html_parts = []

    # ===== HTML Head =====
    html_parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>液态金属文献调研报告 - Demo v5.0</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #f5f7fa; color: #333; line-height: 1.7; }}
  .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}

  .report-header {{ background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%); color: white;
                   padding: 40px 30px; border-radius: 12px; margin-bottom: 30px; }}
  .report-header h1 {{ font-size: 26px; margin-bottom: 8px; }}
  .report-header .meta {{ font-size: 13px; opacity: 0.85; }}
  .report-header .badge {{ display: inline-block; background: rgba(255,255,255,0.2); padding: 3px 12px;
                           border-radius: 20px; font-size: 12px; margin-right: 8px; margin-bottom: 4px; }}

  .section {{ background: white; border-radius: 10px; padding: 25px 30px; margin-bottom: 20px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .section h2 {{ font-size: 20px; color: #1a237e; border-bottom: 2px solid #e8eaf6; padding-bottom: 10px;
                margin-bottom: 15px; }}
  .section h3 {{ font-size: 16px; color: #283593; margin: 15px 0 8px; }}

  .pipeline-flow {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 15px 0; }}
  .pipeline-step {{ background: #e8eaf6; border: 1px solid #c5cae9; border-radius: 8px;
                    padding: 6px 12px; font-size: 12px; color: #1a237e; }}
  .pipeline-step.active {{ background: #1a237e; color: white; }}
  .pipeline-step.route-a {{ background: #e8f5e9; border-color: #a5d6a7; color: #2e7d32; }}
  .pipeline-arrow {{ color: #9fa8da; font-size: 14px; }}

  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin: 15px 0; }}
  .stat-card {{ background: #f5f7fa; border-radius: 8px; padding: 15px; text-align: center; }}
  .stat-card .num {{ font-size: 28px; font-weight: 700; color: #1a237e; }}
  .stat-card .label {{ font-size: 12px; color: #757575; margin-top: 4px; }}
  .stat-card.route-a .num {{ color: #2e7d32; }}

  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }}
  th {{ background: #e8eaf6; color: #1a237e; padding: 8px 10px; text-align: left; font-weight: 600;
        border-bottom: 2px solid #c5cae9; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f9f9f9; }}

  .kcard {{ border: 1px solid #e0e0e0; border-radius: 10px; padding: 18px; margin: 12px 0; }}
  .kcard-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }}
  .kcard-title {{ font-size: 14px; font-weight: 600; color: #1a237e; flex: 1; }}
  .kcard-id {{ background: #e8eaf6; color: #1a237e; padding: 2px 8px; border-radius: 4px; font-size: 11px;
               font-weight: 600; white-space: nowrap; margin-left: 10px; }}
  .kcard-meta {{ font-size: 12px; color: #757575; margin-bottom: 8px; }}
  .kcard-tags {{ margin: 5px 0; }}
  .kcard-tag {{ display: inline-block; background: #e3f2fd; color: #1565c0; padding: 2px 8px;
                border-radius: 4px; font-size: 11px; margin-right: 4px; }}
  .prop-table td:first-child {{ font-weight: 500; color: #283593; }}

  .gap-card {{ border-left: 4px solid; border-radius: 0 8px 8px 0; padding: 15px 20px; margin: 12px 0;
               background: #fafafa; }}
  .gap-card.high {{ border-color: #e53935; }}
  .gap-card.medium {{ border-color: #fb8c00; }}
  .gap-card.low {{ border-color: #43a047; }}
  .gap-id {{ font-size: 11px; font-weight: 700; color: #757575; }}
  .gap-title {{ font-size: 15px; font-weight: 600; margin: 4px 0 8px; color: #333; }}
  .gap-desc {{ font-size: 13px; color: #555; margin-bottom: 8px; }}
  .gap-evidence {{ font-size: 12px; color: #757575; background: white; border-radius: 6px;
                   padding: 8px 12px; margin-top: 6px; }}
  .gap-evidence li {{ margin: 3px 0; }}
  .gap-suggestion {{ font-size: 12px; color: #1565c0; margin-top: 8px; font-style: italic; }}
  .severity-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px;
                     font-weight: 600; color: white; }}
  .severity-high {{ background: #e53935; }}
  .severity-medium {{ background: #fb8c00; }}
  .severity-low {{ background: #43a047; }}

  .verify-status {{ display: inline-block; padding: 2px 10px; border-radius: 4px; font-size: 12px;
                    font-weight: 600; }}
  .verify-verified {{ background: #e8f5e9; color: #2e7d32; }}
  .verify-with_notes {{ background: #fff3e0; color: #e65100; }}
  .verify-weak {{ background: #ffebee; color: #c62828; }}

  .conclusion {{ background: #e8eaf6; border-radius: 10px; padding: 20px 25px; margin-top: 15px;
                 font-size: 14px; color: #283593; line-height: 1.8; }}

  .route-a-section {{ border: 2px solid #a5d6a7; border-radius: 10px; padding: 20px 25px; margin: 15px 0;
                       background: #f1f8e9; }}
  .route-a-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }}
  .route-a-header h2 {{ color: #2e7d32; border: none; margin: 0; }}
  .route-a-badge {{ background: #2e7d32; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; }}

  .relationship-card {{ background: white; border-left: 3px solid #66bb6a; border-radius: 0 8px 8px 0;
                         padding: 12px 18px; margin: 10px 0; }}
  .rel-title {{ font-size: 14px; font-weight: 600; color: #2e7d32; margin-bottom: 5px; }}
  .rel-mechanism {{ font-size: 12px; color: #555; margin-top: 5px; }}
  .rel-confidence {{ display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
  .conf-high {{ background: #c8e6c9; color: #2e7d32; }}
  .conf-medium {{ background: #fff9c4; color: #f57f17; }}
  .conf-low {{ background: #ffebee; color: #c62828; }}

  .trend-card {{ background: white; border-radius: 8px; padding: 12px 18px; margin: 8px 0; border: 1px solid #c8e6c9; }}
  .trend-name {{ font-size: 13px; font-weight: 600; color: #1b5e20; }}
  .trend-desc {{ font-size: 12px; color: #555; margin-top: 3px; }}

  .svg-container {{ text-align: center; margin: 15px 0; overflow-x: auto; }}
  .svg-container svg {{ max-width: 100%; height: auto; }}

  .footer {{ text-align: center; padding: 20px; color: #bdbdbd; font-size: 12px; }}

  .api-info {{ background: #e8f5e9; border-radius: 8px; padding: 12px 18px; margin: 10px 0; font-size: 12px; color: #2e7d32; }}

  .data-sufficiency {{ background: #fff3e0; border-radius: 8px; padding: 12px 18px; margin: 10px 0; font-size: 13px; }}
  .data-sufficiency.adequate {{ background: #e8f5e9; }}
  .data-sufficiency.insufficient {{ background: #ffebee; }}
</style>
</head>
<body>
<div class="container">
""")

    # ===== Header =====
    html_parts.append(f"""
<div class="report-header">
  <h1>液态金属领域文献调研报告</h1>
  <div class="meta">
    <span class="badge">{stats.get('llm_model', 'MiniMax')}</span>
    <span class="badge">Sciverse API</span>
    <span class="badge">8-Agent Pipeline</span>
    <span class="badge">路线A: 构效关系</span>
    <span class="badge">GA + BO 迭代优化</span>
    <span class="badge">消融实验</span>
    <span class="badge">LLM: {stats['llm_mode']}</span>
    <br>生成时间: {report['meta']['generated_at']} &nbsp;|&nbsp; {report['meta']['generated_by']}
  </div>
</div>
""")

    # ===== API Info =====
    if stats.get("sciverse_connected"):
        html_parts.append(f"""
<div class="api-info">
  <strong>API 调用统计:</strong>
  LLM调用 {stats.get('llm_calls', 0)} 次 | Token用量 {stats.get('llm_tokens', 0)} |
  Sciverse调用 {stats.get('sciverse_calls', 0)} 次 | 总耗时 {stats['total_time']}s
</div>
""")

    # ===== Pipeline Overview =====
    agent_names = ["任务规划", "文献检索", "文献筛选", "知识抽取", "知识融合", "Gap识别", "证据核验"]
    html_parts.append("""
<div class="section">
  <h2>管线执行概览 (8-Agent + Route A)</h2>
  <div class="pipeline-flow">
""")
    for i, name in enumerate(agent_names):
        cls = "pipeline-step active"
        html_parts.append(f'    <span class="{cls}">{i+1}. {name}</span>\n')
        html_parts.append('    <span class="pipeline-arrow">&rarr;</span>\n')
    html_parts.append('    <span class="pipeline-step route-a">8. 构效关系发现</span>\n')
    html_parts.append('    <span class="pipeline-arrow">&rarr;</span>\n')
    html_parts.append('    <span class="pipeline-step">报告生成</span>\n')
    html_parts.append("  </div>\n")

    # Stats
    verified_count = sum(1 for v in verifications if v["verification_status"] == "verified")
    route_a_rels = len(route_a.get("relationships", [])) if route_a else 0
    html_parts.append("""  <div class="stats-grid">
""")
    html_parts.append(f'    <div class="stat-card"><div class="num">{len(cards)}</div><div class="label">文献知识卡片</div></div>\n')
    html_parts.append(f'    <div class="stat-card"><div class="num">{total_props}</div><div class="label">抽取属性条目</div></div>\n')
    html_parts.append(f'    <div class="stat-card"><div class="num">{len(fusion)}</div><div class="label">融合属性类别</div></div>\n')
    html_parts.append(f'    <div class="stat-card"><div class="num">{len(gaps)}</div><div class="label">Research Gaps</div></div>\n')
    html_parts.append(f'    <div class="stat-card"><div class="num">{verified_count}</div><div class="label">证据可追溯Gap（非科学验证）</div></div>\n')
    html_parts.append(f'    <div class="stat-card route-a"><div class="num">{route_a_rels}</div><div class="label">构效关系</div></div>\n')
    html_parts.append(f'    <div class="stat-card"><div class="num">{stats["total_time"]}s</div><div class="label">总执行耗时</div></div>\n')
    html_parts.append("""  </div>

  <h3>各步骤耗时</h3>
  <table>
    <tr><th>步骤</th><th>耗时(s)</th><th>进度</th></tr>
""")
    for step in stats["steps"]:
        max_time = max(s["time"] for s in stats["steps"]) or 1
        bar_len = max(1, int(step["time"] / max_time * 30))
        html_parts.append(
            f'    <tr><td>{step["step"]}</td><td>{step["time"]}</td>'
            f'<td><span style="color:#1a237e;">{"&#9608;" * bar_len}</span></td></tr>\n'
        )
    html_parts.append(f'    <tr><td><strong>总计</strong></td><td><strong>{stats["total_time"]}</strong></td><td></td></tr>\n')
    html_parts.append("""  </table>
</div>
""")

    # ===== Scope =====
    html_parts.append("""
<div class="section">
  <h2>1. 调研范围与任务规划</h2>
""")
    html_parts.append(f'  <p><strong>用户意图:</strong> {scope["user_query"]}</p>\n')
    html_parts.append(f'  <p><strong>目标领域:</strong> {scope["domain"]}</p>\n')
    html_parts.append(f'  <p><strong>目标文献数:</strong> {scope["target_paper_count"]} 篇</p>\n')
    html_parts.append('  <h3>子主题</h3>\n  <ul>\n')
    for topic in scope["subtopics"]:
        html_parts.append(f'    <li>{topic}</li>\n')
    html_parts.append('  </ul>\n')
    html_parts.append('  <h3>分析维度</h3>\n  <ul>\n')
    for dim in scope["analysis_dimensions"]:
        html_parts.append(f'    <li>{dim}</li>\n')
    html_parts.append('  </ul>\n')
    html_parts.append(f'  <h3>检索查询 ({len(scope.get("search_queries", []))} 条, Sciverse)</h3>\n  <ul>\n')
    for q in scope.get("search_queries", []):
        html_parts.append(f'    <li><code>{q}</code></li>\n')
    html_parts.append('  </ul>\n')
    html_parts.append("</div>\n")

    # ===== Literature Selection =====
    html_parts.append("""
<div class="section">
  <h2>2. 文献检索与筛选 (Sciverse API)</h2>
  <table>
    <tr><th>ID</th><th>标题</th><th>期刊</th><th>年份</th><th>引用数</th><th>评分</th><th>语义相关性</th><th>状态</th></tr>
""")
    for p in filter_result["scored_papers"][:50]:
        status = '<span style="color:#2e7d32;font-weight:600;">PASS</span>' if p.get("selected") else '<span style="color:#c62828;">SKIP</span>'
        html_parts.append(
            f'    <tr><td><strong>{p["id"]}</strong></td><td style="max-width:280px;">{p["title"][:60]}...</td>'
            f'<td>{p.get("journal", "Unknown")}</td><td>{p.get("year", "?")}</td>'
            f'<td>{p.get("citation_count", 0)}</td>'
            f'<td><strong>{p["relevance_score"]}</strong></td>'
            f'<td>{p.get("score_breakdown", {}).get("semantic_relevance", "-")}</td><td>{status}</td></tr>\n'
        )
    html_parts.append("""  </table>
</div>
""")

    # ===== Knowledge Cards =====
    llm_mode_label = f"{stats.get('llm_model', 'MiniMax')} LLM 深度抽取" if stats.get("llm_mode") == "api" else "Regex 回退模式 (LLM未连接)"
    html_parts.append(f"""
<div class="section">
  <h2>3. 知识抽取卡片 ({llm_mode_label})</h2>
""")
    for card in cards:
        tags_html = " ".join(f'<span class="kcard-tag">{t}</span>' for t in card.get("domain_tags", []))
        authors = card.get("authors", [])
        if isinstance(authors, list):
            authors_str = ", ".join(str(a) for a in authors[:4])
        else:
            authors_str = str(authors)[:100]

        html_parts.append(f"""
  <div class="kcard">
    <div class="kcard-header">
      <div class="kcard-title">{card["title"]}</div>
      <span class="kcard-id">{card["paper_id"]}</span>
    </div>
    <div class="kcard-meta">
      {authors_str} | {card.get("journal", "Unknown")} ({card.get("year", "?")}) | DOI: {card.get("doi", "N/A")}
      <br><em>抽取方式: {card.get("extraction_method", "unknown")}</em>
    </div>
    <div class="kcard-tags">{tags_html}</div>
""")
        if card.get("properties"):
            html_parts.append(f'    <p style="font-size:12px;color:#757575;margin:5px 0;">抽取到 {len(card["properties"])} 条属性:</p>\n')
            html_parts.append("""    <table class="prop-table">
      <tr><th>材料</th><th>属性</th><th>数值</th><th>单位</th><th>条件</th></tr>
""")
            for prop in card["properties"]:
                html_parts.append(
                    f'      <tr><td>{prop.get("material", "")}</td><td>{prop.get("property_original", prop.get("property", ""))}</td>'
                    f'<td><strong>{prop.get("value", "")}</strong></td><td>{prop.get("unit", "")}</td>'
                    f'<td>{prop.get("conditions", "")}</td></tr>\n'
                )
            html_parts.append("    </table>\n")
        else:
            html_parts.append('    <p style="font-size:12px;color:#999;">未抽取到结构化属性</p>\n')

        if card.get("methods_summary"):
            html_parts.append(f'    <h3 style="color:#283593;font-size:13px;">方法摘要</h3>\n    <p style="font-size:12px;color:#555;">{card["methods_summary"][:300]}</p>\n')
        if card.get("key_findings"):
            html_parts.append(f'    <h3 style="color:#1565c0;font-size:13px;">主要发现</h3>\n    <p style="font-size:12px;color:#555;">{card["key_findings"][:300]}</p>\n')
        if card.get("limitations"):
            html_parts.append('    <h3 style="color:#e65100;font-size:13px;">识别的局限性</h3>\n    <ul style="font-size:12px;color:#757575;">\n')
            for lim in card["limitations"]:
                html_parts.append(f'      <li>{lim}</li>\n')
            html_parts.append("    </ul>\n")
        html_parts.append("  </div>\n")
    html_parts.append("</div>\n")

    # ===== Knowledge Graph =====
    html_parts.append("""
<div class="section">
  <h2>4. 材料-属性知识图谱</h2>
  <p style="font-size:13px;color:#757575;margin-bottom:10px;">下图展示了文献中材料与属性之间的关联网络。节点大小反映连接数, 颜色区分材料(蓝色)与属性(橙色)。</p>
  <div class="svg-container">
""")
    html_parts.append(kg_svg)
    html_parts.append("  </div>\n</div>\n")

    # ===== Property Distribution =====
    html_parts.append("""
<div class="section">
  <h2>5. 属性分布与跨文献一致性</h2>
  <p style="font-size:13px;color:#757575;margin-bottom:10px;">下图展示了各属性的跨文献覆盖情况和一致性水平。</p>
  <div class="svg-container">
""")
    html_parts.append(prop_chart_svg)
    html_parts.append("  </div>\n")

    # Fusion Table
    html_parts.append("""  <table>
    <tr><th>属性</th><th>涉及材料</th><th>来源文献数</th><th>数值范围</th><th>一致性</th><th>冲突/空白</th></tr>
""")
    for f in fusion:
        conflicts = "; ".join(f["conflicts"]) if f["conflicts"] else "-"
        gaps_text = "; ".join(f["data_gap"]) if f["data_gap"] else "-"
        consist_color = {"high": "#2e7d32", "medium": "#e65100", "low": "#c62828", "single_source": "#757575", "n/a": "#bdbdbd"}.get(f["consistency"], "#757575")
        html_parts.append(
            f'    <tr><td><strong>{f["property"]}</strong></td>'
            f'<td>{", ".join(f["materials"])}</td><td>{f["paper_count"]}</td>'
            f'<td>{f["value_range"]}</td>'
            f'<td style="color:{consist_color};">{f["consistency"]}</td>'
            f'<td style="font-size:12px;">{conflicts}{" | " if f["conflicts"] else ""}{gaps_text}</td></tr>\n'
        )
    html_parts.append("""  </table>
</div>
""")

    # ===== Research Gaps =====
    html_parts.append("""
<div class="section">
  <h2>6. Research Gap 识别 (LLM增强)</h2>
""")
    for gap in gaps:
        sev_class = gap["severity"]
        sev_badge = f'<span class="severity-badge severity-{sev_class}">{gap["severity"].upper()}</span>'
        gap_type_badge = f'<span style="background:#e3f2fd;color:#1565c0;padding:2px 8px;border-radius:4px;font-size:11px;">{gap.get("gap_type", "")}</span>'

        html_parts.append(f"""
  <div class="gap-card {sev_class}">
    <div class="gap-id">{gap["id"]} &nbsp; {gap_type_badge} &nbsp; {sev_badge}</div>
    <div class="gap-title">{gap["title"]}</div>
    <div class="gap-desc">{gap.get("description", "待核验的候选问题")}</div>
    <div class="gap-evidence">
      <strong>证据链 ({len(gap.get("evidence", []))} 条):</strong>
      <ul>
""")
        for ev in gap.get("evidence", []):
            quote = ev.get("quote", ev.get("conflicts", ""))
            if isinstance(quote, list):
                quote = "; ".join(str(q) for q in quote)
            pid = ev.get("paper_id", "")
            html_parts.append(f'        <li>[{pid}] {quote}</li>\n')
        html_parts.append(f"""      </ul>
    </div>
    <div class="gap-suggestion">&#128161; 建议: {gap.get("suggestion", "")}</div>
  </div>
""")
    html_parts.append("</div>\n")

    # ===== Verification =====
    html_parts.append("""
<div class="section">
  <h2>7. 证据核验结果</h2>
  <table>
    <tr><th>Gap ID</th><th>标题</th><th>证据数</th><th>来源数</th><th>核验状态</th><th>备注</th></tr>
""")
    for v in verifications:
        status_class = {
            "verified": "verify-verified",
            "verified_with_notes": "verify-with_notes",
            "weak": "verify-weak",
        }.get(v["verification_status"], "verify-with_notes")
        status_text = {
            "verified": "证据可追溯（非科学验证）",
            "verified_with_notes": "部分可追溯（待核验）",
            "weak": "证据不足",
        }.get(v["verification_status"], v["verification_status"])
        issues = "; ".join(v.get("issues", [])) if v.get("issues") else "-"
        html_parts.append(
            f'    <tr><td><strong>{v["gap_id"]}</strong></td>'
            f'<td style="max-width:300px;">{v["gap_title"][:50]}...</td>'
            f'<td>{v["evidence_count"]}</td><td>{len(v["source_papers"])}</td>'
            f'<td><span class="verify-status {status_class}">{status_text}</span></td>'
            f'<td style="font-size:12px;">{issues}</td></tr>\n'
        )
    html_parts.append("""  </table>
</div>
""")

    # ===== Route A: Structure-Property Discovery =====
    if route_a:
        ds = route_a.get("data_sufficiency", {})
        ds_class = "adequate" if ds.get("adequate_for_analysis") else "insufficient"
        ds_text = "充足" if ds.get("adequate_for_analysis") else "不足"

        html_parts.append(f"""
<div class="route-a-section">
  <div class="route-a-header">
    <span class="route-a-badge">路线A</span>
    <h2>构效关系发现</h2>
  </div>
  <p style="font-size:13px;color:#555;margin-bottom:12px;">
    根据可追溯数值分析材料配比与物性的关联，分别呈现观测关联、跨来源对照和模型反事实。
    分析方法: {route_a.get("method", "unknown")} | 发现 {len(route_a.get("relationships", []))} 条构效关系,
    {len(route_a.get("trends", []))} 个趋势。
  </p>
  <div style="background:#e8f5e9;border-radius:6px;padding:8px 12px;margin:8px 0 12px;font-size:12px;color:#1b5e20;">
    <strong>数据来源说明</strong>: 代理模型使用整理参考锚点与本批次通过证据和质量配比校验的新增物性观测，按属性进行距离反比加权插值；缺失性能不补造。
    书目信息已记录，但各数值及测试条件仍需回到原始来源逐项复核；本报告不宣称完成实时材料数据库交叉验证。
  </div>

  <div class="data-sufficiency {ds_class}">
    <strong>数据充分性评估: {ds_text}</strong>
    {f'<br>缺失数据: {", ".join(ds.get("missing_data", []))}' if ds.get("missing_data") else ''}
    {f'<br>建议: {ds.get("recommendation", "")}' if ds.get("recommendation") else ''}
  </div>
""")

        # Knowledge Graph for Route A
        if route_a_svg:
            html_parts.append(f"""
  <h3 style="color:#2e7d32;">构效关系网络图</h3>
  <div class="svg-container">
    {route_a_svg}
  </div>
""")

        # Relationships
        if route_a.get("relationships"):
            html_parts.append('  <h3 style="color:#2e7d32;">发现的构效关系</h3>\n')
            for rel in route_a["relationships"]:
                conf = rel.get("confidence", "medium")
                conf_class = f"conf-{conf}"
                trend_arrow = {"positive": "&#8593; 正相关", "negative": "&#8595; 负相关",
                               "nonlinear": "&#8656; 非线性", "unclear": "&#8653; 不明确"}.get(rel.get("trend", ""), "")

                html_parts.append(f"""
  <div class="relationship-card">
    <div class="rel-title">{rel.get("relationship", "")}
      <span class="rel-confidence {conf_class}">{conf.upper()}</span>
      <span style="font-size:11px;color:#757575;margin-left:8px;">{trend_arrow}</span>
    </div>
    <div style="font-size:12px;color:#757575;">
      组成变量: <strong>{rel.get("component", "")}</strong> &rarr; 性能: <strong>{rel.get("property", "")}</strong>
    </div>
    <div class="rel-mechanism">&#9881; 机制: {rel.get("mechanism", "")}</div>
""")
                if rel.get("evidence"):
                    html_parts.append('    <div style="font-size:11px;color:#9e9e9e;margin-top:5px;">证据:\n    <ul>\n')
                    for ev in rel["evidence"][:3]:
                        html_parts.append(f'      <li>[{ev.get("paper_id", "")}] {ev.get("data_point", "")}</li>\n')
                    html_parts.append('    </ul>\n    </div>\n')
                html_parts.append("  </div>\n")

        # Trends
        if route_a.get("trends"):
            html_parts.append('  <h3 style="color:#2e7d32;">研究趋势</h3>\n')
            for trend in route_a["trends"]:
                html_parts.append(f"""
  <div class="trend-card">
    <div class="trend-name">{trend.get("trend_name", "")}</div>
    <div class="trend-desc">{trend.get("description", "")}</div>
    <div style="font-size:11px;color:#1565c0;margin-top:3px;">&#128161; {trend.get("implication", "")}</div>
  </div>
""")

        ingestion = route_a.get('data_ingestion', {}).get('summary', {})
        html_parts.append(f"<p>有效物性记录：{ingestion.get('accepted_properties', 0)}；新增代理物性观测：{ingestion.get('extracted_property_anchors', 0)}。</p>")
        for key, heading in [('exploratory_comparisons', '跨来源配比—物性对照（非因果规律）'),
                             ('exploratory_trends', '跨来源探索性拟合（条件未统一）'),
                             ('model_trends', '模型反事实趋势（非实测验证）')]:
            if route_a.get(key):
                html_parts.append(f'<h3>{heading}</h3>')
                for item in route_a[key]:
                    html_parts.append('<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' +
                                      json.dumps(item, ensure_ascii=False, indent=2) + '</pre>')

        # Composition Optimization
        if route_a.get("composition_optimization"):
            html_parts.append(f"""
  <h3 style="color:#2e7d32;">组成优化建议</h3>
  <div class="conclusion" style="background:#e8f5e9;color:#1b5e20;">
    {route_a["composition_optimization"]}
  </div>
""")

        robust = route_a.get("evidence_robust_discovery", {})
        if robust and not robust.get("error"):
            best = robust.get("best_risk_adjusted_candidate", {})
            comp = best.get("composition", {})
            tradeoff = robust.get("robustness_tradeoff_vs_naive", {})
            guide = robust.get("llm_search_guidance", {})
            audit = robust.get("llm_scientific_audit", {})
            html_parts.append(f"""
  <h3 style="color:#6a1b9a;">证据约束的稳健反事实发现（ERCPD）</h3>
  <p style="font-size:13px;color:#555;">
    对每个文献来源执行留一法，在来源扰动下联合评估组成候选，并进行Pareto筛选。
    输出属于<strong>计算候选假说</strong>，不等同于实验验证。
  </p>
  <div class="stat-card" style="background:#f3e5f5;border-left:4px solid #6a1b9a;text-align:left;">
    <strong>稳健候选组成</strong>: Ga {comp.get('ga', '?')}% / In {comp.get('in', '?')}% / Sn {comp.get('sn', '?')}%<br>
    来源留一适应度: mean={best.get('fitness_mean', '?')}, std={best.get('fitness_std', '?')}, worst={best.get('fitness_worst_case', '?')}<br>
    相对朴素最优: 标准差降低 {tradeoff.get('fitness_std_reduction', '?')}, 平均适应度代价 {tradeoff.get('mean_fitness_delta', '?')}<br>
    来源组={robust.get('parameters', {}).get('source_groups', '?')}, 网格候选={robust.get('parameters', {}).get('grid_candidates', '?')}, Pareto前沿={robust.get('pareto_front_size', '?')}<br>
    搜索引导={html_lib.escape(str(guide.get('method', '?')))}, 主张审计={html_lib.escape(str(audit.get('method', '?')))}, 等级={html_lib.escape(str(robust.get('claim_level', '?')))}
  </div>
  <h4 style="color:#6a1b9a;margin:12px 0 6px;">守恒组分反事实检验</h4>
  <table style="font-size:12px;">
    <tr><th>扰动</th><th>目标组成</th><th>预测Δ熔点</th><th>预测Δ电导率</th><th>跨来源符号一致率</th><th>状态</th></tr>
""")
            for test in robust.get("counterfactual_tests", []):
                target = test.get("to_composition", {})
                delta = test.get("predicted_delta", {})
                consistency = test.get("sign_consistency", {})
                html_parts.append(
                    "<tr>"
                    f"<td>{html_lib.escape(str(test.get('change', '')))}</td>"
                    f"<td>{target.get('ga', '?')}/{target.get('in', '?')}/{target.get('sn', '?')}</td>"
                    f"<td>{delta.get('melting_point_mean_C', '?')} ± {delta.get('melting_point_std_C', '?')} °C</td>"
                    f"<td>{delta.get('conductivity_mean_S_per_m', '?')} ± {delta.get('conductivity_std_S_per_m', '?')} S/m</td>"
                    f"<td>熔点 {consistency.get('melting_point', '?')} / 电导率 {consistency.get('conductivity', '?')}</td>"
                    f"<td>{html_lib.escape(str(test.get('hypothesis_status', '')))}</td>"
                    "</tr>"
                )
            html_parts.append("</table>")
            ablation_rows = robust.get("parameter_ablation", {}).get("rows", [])
            if ablation_rows:
                html_parts.append("""
  <h4 style="color:#6a1b9a;margin:12px 0 6px;">ERCPD 参数消融</h4>
  <table style="font-size:12px;">
    <tr><th>配置</th><th>λ</th><th>网格</th><th>候选 Ga/In/Sn</th><th>mean</th><th>source std</th><th>worst</th></tr>
""")
                for row in ablation_rows:
                    row_comp = row.get("composition", {})
                    html_parts.append(
                        "<tr>"
                        f"<td>{html_lib.escape(str(row.get('configuration', '')))}</td>"
                        f"<td>{row.get('risk_penalty', '?')}</td>"
                        f"<td>{row.get('resolution_wt_pct', '?')} wt%</td>"
                        f"<td>{row_comp.get('ga', '?')}/{row_comp.get('in', '?')}/{row_comp.get('sn', '?')}</td>"
                        f"<td>{row.get('fitness_mean', '?')}</td>"
                        f"<td>{row.get('fitness_std', '?')}</td>"
                        f"<td>{row.get('fitness_worst_case', '?')}</td>"
                        "</tr>"
                    )
                html_parts.append("</table>")

        html_parts.append("</div>\n")

    # ===== Optimization & Ablation =====
    opt_data = route_a.get("optimization") if route_a else None
    ablation = results.get("ablation")

    if opt_data or ablation:
        html_parts.append("""
<div class="section">
  <h2>9. 迭代优化与消融实验</h2>
""")
        # 收敛曲线
        if opt_data:
            conv_svg = _generate_convergence_svg(route_a)
            ga = opt_data.get("ga", {})
            bo = opt_data.get("bo", {})

            html_parts.append(f"""
  <h3>9.1 优化收敛曲线 (GA + BO)</h3>
  <p style="font-size:13px;color:#757575;margin-bottom:10px;">
    遗传算法(GA)在组成空间进行全局搜索, 贝叶斯优化(BO)使用高斯过程代理模型+EI采集函数进行高效采样。
  </p>
  <div class="svg-container">
    {conv_svg}
  </div>

  <table>
    <tr><th>方法</th><th>最优适应度</th><th>最优组成 (Ga/In/Sn)</th><th>预测电导率</th><th>预测熔点</th><th>评估次数</th><th>耗时</th></tr>
    <tr>
      <td><strong>GA</strong></td>
      <td>{ga.get('best_fitness', '-')}</td>
      <td>{ga.get('best_composition', {}).get('ga', '?')}% / {ga.get('best_composition', {}).get('in', '?')}% / {ga.get('best_composition', {}).get('sn', '?')}%</td>
      <td>{ga.get('best_properties', {}).get('conductivity', '-')} S/m</td>
      <td>{ga.get('best_properties', {}).get('melting_point', '-')} C</td>
      <td>{ga.get('total_evaluations', '-')}</td>
      <td>{ga.get('elapsed_time', '-')}s</td>
    </tr>
    <tr>
      <td><strong>BO</strong></td>
      <td>{bo.get('best_fitness', '-')}</td>
      <td>{bo.get('best_composition', {}).get('ga', '?')}% / {bo.get('best_composition', {}).get('in', '?')}% / {bo.get('best_composition', {}).get('sn', '?')}%</td>
      <td>{bo.get('best_properties', {}).get('conductivity', '-')} S/m</td>
      <td>{bo.get('best_properties', {}).get('melting_point', '-')} C</td>
      <td>{bo.get('total_evaluations', '-')}</td>
      <td>{bo.get('elapsed_time', '-')}s</td>
    </tr>
  </table>
""")

        # 消融实验 v2.0
        if ablation:
            abl_svg = _generate_ablation_svg(ablation)
            global_opt = ablation.get("_surrogate_grid_optimum", {})
            robustness = ablation.get("_robustness", {})
            top5 = global_opt.get("top5_local_optima", [])

            html_parts.append(f"""
  <h3>9.2 消融实验 v3.1: 整理参考锚点驱动, 优化策略多维度对比</h3>
  <p style="font-size:13px;color:#757575;margin-bottom:10px;">
    25个整理参考锚点(含二元/三元体系) | 距离反比插值代理模型 | 纯随机初始化 | 5个评价指标 | 5种子鲁棒性测试
  </p>
""")

            # 全局最优信息
            if global_opt:
                gopt_comp = global_opt.get("composition", {})
                html_parts.append(f"""
  <div class="stat-card" style="background:#e8f5e9;border-left:4px solid #2e7d32;margin-bottom:12px;padding:10px 15px;">
    <strong style="color:#2e7d32;">代理模型网格参考最优</strong>: fitness={global_opt.get('fitness', '?')},
    Ga={gopt_comp.get('ga', '?')}% In={gopt_comp.get('in', '?')}% Sn={gopt_comp.get('sn', '?')}%
    <br><span style="font-size:12px;color:#757575;">Top-5局部最优组成已识别, 适应度景观存在多峰结构</span>
  </div>
""")

            html_parts.append(f"""
  <div class="svg-container">
    {abl_svg}
  </div>

  <h4 style="color:#1a237e;margin:15px 0 8px;">9.2.1 核心指标对比</h4>
  <table style="font-size:12px;">
    <tr><th>配置</th><th>方法</th><th>适应度</th><th>评估次数</th><th>耗时(s)</th><th>样本效率</th><th>收敛AUC</th><th>解多样性</th><th>覆盖率</th><th>最优比</th></tr>
""")
            base_fit = ablation.get("baseline", {}).get("best_fitness", 0)
            for key in ["baseline", "random_search", "ga", "bo", "ga_bo_hybrid"]:
                if key not in ablation:
                    continue
                r = ablation[key]
                m = r.get("metrics", {})
                improvement = ((r["best_fitness"] - base_fit) / base_fit * 100) if base_fit > 0 else 0
                imp_color = "#2e7d32" if improvement > 0 else "#757575"
                config_label = {"baseline": "A", "random_search": "B", "ga": "C", "bo": "D", "ga_bo_hybrid": "E"}.get(key, key)
                html_parts.append(
                    f'    <tr><td><strong>{config_label}</strong></td>'
                    f'<td>{r["method"]}</td>'
                    f'<td><strong>{r["best_fitness"]}</strong></td>'
                    f'<td>{r["total_evaluations"]}</td>'
                    f'<td>{r["elapsed_time"]}</td>'
                    f'<td>{m.get("sample_efficiency", "-")}</td>'
                    f'<td>{m.get("convergence_auc", "-")}</td>'
                    f'<td>{m.get("solution_diversity", "-")}</td>'
                    f'<td>{m.get("exploration_coverage", "-")}</td>'
                    f'<td>{m.get("optimality_gap", "-")}</td></tr>\n'
                )
            html_parts.append("  </table>\n")

            # 多种子鲁棒性
            if robustness:
                html_parts.append("""
  <h4 style="color:#1a237e;margin:15px 0 8px;">9.2.2 多种子鲁棒性 (5个随机种子)</h4>
  <table style="font-size:12px;">
    <tr><th>方法</th><th>平均适应度</th><th>标准差</th><th>变异系数(CV%)</th><th>最差</th><th>最好</th><th>各种子结果</th></tr>
""")
                for key in ["random_search", "ga", "bo"]:
                    if key not in robustness:
                        continue
                    r = robustness[key]
                    cv_color = "#c62828" if r["cv"] > 10 else ("#f9a825" if r["cv"] > 5 else "#2e7d32")
                    all_fits = " / ".join(str(f) for f in r.get("all_fitnesses", []))
                    html_parts.append(
                        f'    <tr><td><strong>{key.upper()}</strong></td>'
                        f'<td>{r["mean_fitness"]}</td>'
                        f'<td>{r["std_fitness"]}</td>'
                        f'<td style="color:{cv_color};font-weight:600;">{r["cv"]}%</td>'
                        f'<td>{r["min_fitness"]}</td>'
                        f'<td>{r["max_fitness"]}</td>'
                        f'<td style="font-size:11px;color:#757575;">{all_fits}</td></tr>\n'
                    )
                html_parts.append("  </table>\n")

            # 评价指标说明
            html_parts.append("""
  <div style="background:#f5f5f5;padding:10px 15px;border-radius:6px;margin:10px 0;font-size:12px;color:#616161;">
    <strong>评价指标说明</strong>: 
    <strong>样本效率</strong>=到达95%代理模型网格参考最优所需的评估次数(越少越好);
    <strong>收敛AUC</strong>=归一化收敛曲线下面积(越接近1越快收敛); 
    <strong>解多样性</strong>=探索组成的标准差(越大探索越广); 
    <strong>覆盖率</strong>=访问的网格单元比例; 
    <strong>最优比</strong>=最优解/代理模型网格参考最优(越接近1越准确)
  </div>
""")

            # 推荐实验方案
            recs = ablation.get("recommendations", [])
            if recs:
                html_parts.append("""
  <h3>9.3 优化推荐实验方案</h3>
  <p style="font-size:13px;color:#757575;margin-bottom:10px;">基于优化结果, 推荐以下实验方案供后续验证:</p>
""")
                for rec in recs:
                    comp = rec.get("composition", {})
                    props = rec.get("predicted_properties", {})
                    html_parts.append(f"""
  <div class="relationship-card" style="border-left-color:#7b1fa2;">
    <div class="rel-title" style="color:#7b1fa2;">
      方案 {rec.get("rank", "?")}: Ga {comp.get("ga", "?")}% / In {comp.get("in", "?")}% / Sn {comp.get("sn", "?")}%
      <span class="rel-confidence conf-high">适应度 {rec.get("fitness", "?")}</span>
    </div>
    <div style="font-size:12px;color:#757575;">
      来源: {rec.get("method", "")} |
      预测电导率: <strong>{props.get("conductivity", "-")} S/m</strong> |
      预测熔点: <strong>{props.get("melting_point", "-")} C</strong> |
      置信度: {props.get("confidence", "-")}
    </div>
    <div class="rel-mechanism">&#9881; {rec.get("rationale", "")}</div>
  </div>
""")
        html_parts.append("</div>\n")

    # ===== Conclusion =====
    html_parts.append(f"""
<div class="section">
  <h2>10. 冻结参考快照一致性检查</h2>
""")

    ablation = results.get("ablation")
    cv_data = ablation.get("_cross_validation", {}) if ablation else {}
    if cv_data:
        cv_details = cv_data.get("validation_details", [])
        html_parts.append(f"""
  <p style="font-size:13px;color:#555;margin-bottom:12px;">
    将抽取值与代码内冻结参考快照进行单位归一化后比较；本步骤没有实时查询外部数据库，也不等同于原始来源核验。
    代理模型锚点: <strong>{cv_data.get('surrogate_anchors', 0)}个</strong>,
    按 <strong>{len(cv_data.get('anchor_sources', []))}组</strong>待核验书目分组（未证明独立性）。
  </p>
  <div style="background:#e3f2fd;border-radius:6px;padding:10px 15px;margin:10px 0;">
    <strong>验证汇总:</strong>
    验证属性数: {cv_data.get('total_properties_validated', 0)} |
    匹配（见属性容差）: <span style="color:#2e7d32;font-weight:bold;">{cv_data.get('matches', 0)}</span> |
    接近（见属性容差）: <span style="color:#f57f17;font-weight:bold;">{cv_data.get('close_matches', 0)}</span> |
    不匹配（见属性容差）: <span style="color:#c62828;font-weight:bold;">{cv_data.get('mismatches', 0)}</span> |
    匹配率: <strong>{cv_data.get('match_rate_pct', 0)}%</strong> |
    平均偏差: <strong>{cv_data.get('average_deviation_pct', 0)}%</strong>
  </div>
""")

        if cv_details:
            html_parts.append('<p>熔点按绝对差&lt;1°C / &lt;5°C分级，百分比仅用开尔文参考值；其他属性按5% / 15%分级。均为示意容差，不是实验误差或准确率。</p>')
            html_parts.append("""
  <table style="width:100%;border-collapse:collapse;font-size:12px;margin:10px 0;">
    <tr style="background:#1a237e;color:white;">
      <th style="padding:6px;text-align:left;">材料</th>
      <th style="padding:6px;text-align:left;">属性</th>
      <th style="padding:6px;text-align:right;">抽取值</th>
      <th style="padding:6px;text-align:right;">快照参考值</th>
      <th style="padding:6px;text-align:right;">偏差%</th>
      <th style="padding:6px;text-align:center;">状态</th>
      <th style="padding:6px;text-align:left;">参考来源</th>
    </tr>
""")
            for cv in cv_details:
                status_color = "#2e7d32" if cv["status"] == "match" else ("#f57f17" if cv["status"] == "close" else "#c62828")
                status_text = "匹配" if cv["status"] == "match" else ("接近" if cv["status"] == "close" else "不匹配")
                html_parts.append(f"""
    <tr style="border-bottom:1px solid #e0e0e0;">
      <td style="padding:6px;">{cv.get('material', '')}</td>
      <td style="padding:6px;">{cv.get('property', '')}</td>
      <td style="padding:6px;text-align:right;">{cv.get('normalized_value', '')} {cv.get('normalized_unit', '')}</td>
      <td style="padding:6px;text-align:right;">{cv.get('reference_value', '')}</td>
      <td style="padding:6px;text-align:right;">{cv.get('deviation_pct', 0)}%</td>
      <td style="padding:6px;text-align:center;color:{status_color};font-weight:bold;">{status_text}</td>
      <td style="padding:6px;font-size:11px;">{cv.get('reference_source', '')}</td>
    </tr>
""")
            html_parts.append("""
  </table>
""")

        # Literature references for anchors
        lit_refs = ablation.get("_literature_references", []) if ablation else []
        if lit_refs:
            html_parts.append("""
  <h3>10.1 代理模型锚点文献来源</h3>
  <table style="width:100%;border-collapse:collapse;font-size:12px;margin:10px 0;">
    <tr style="background:#1a237e;color:white;">
      <th style="padding:6px;text-align:left;">文献代码</th>
      <th style="padding:6px;text-align:left;">完整引用</th>
      <th style="padding:6px;text-align:center;">数据点数</th>
    </tr>
""")
            for ref in lit_refs:
                html_parts.append(f"""
    <tr style="border-bottom:1px solid #e0e0e0;">
      <td style="padding:6px;font-weight:bold;">[{ref.get('code', '')}]</td>
      <td style="padding:6px;">{ref.get('full_citation', '')}</td>
      <td style="padding:6px;text-align:center;">{ref.get('data_points_count', 0)}</td>
    </tr>
""")
            html_parts.append("""
  </table>
""")
    else:
        html_parts.append("""
  <p style="font-size:13px;color:#999;">交叉验证数据不可用。</p>
""")

    html_parts.append("</div>")

    # ===== Conclusion =====
    html_parts.append(f"""
<div class="section">
  <h2>11. 调研结论</h2>
  <div class="conclusion">
    {report["conclusion"]}
  </div>
</div>
""")

    # ===== Footer =====
    html_parts.append(f"""
<div class="footer">
  本报告由多Agent文献调研系统自动生成 | Demo Version 5.3 (ERCPD + 可追溯证据)<br>
  Pipeline: 8 Agents + Route A + GA/BO Optimization + Ablation | LLM: {stats.get('llm_model', 'MiniMax')} [{stats['llm_mode']}] | Sciverse: {'Connected' if stats.get('sciverse_connected') else 'N/A'}<br>
  LLM Calls: {stats.get('llm_calls', 0)} | Tokens: {stats.get('llm_tokens', 0)} | Sciverse Calls: {stats.get('sciverse_calls', 0)} | Total Time: {stats['total_time']}s<br>
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
""")

    html_parts.append("""
</div>
</body>
</html>
""")

    html_content = "".join(html_parts)
    tmp_path = f"{output_path}.tmp-{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    return html_content


def _generate_knowledge_graph_svg(cards, fusion):
    """生成材料-属性知识图谱 SVG"""
    # 收集节点和边
    material_nodes = set()
    property_nodes = set()
    edges = []

    for card in cards:
        for prop in card["properties"]:
            mat = prop.get("material", "unknown")
            prop_name = prop.get("property", "unknown")
            material_nodes.add(mat)
            property_nodes.add(prop_name)
            edges.append((mat, prop_name, card["paper_id"]))

    all_nodes = list(material_nodes) + list(property_nodes)
    n_nodes = len(all_nodes)
    if n_nodes == 0:
        return "<p style='color:#999;'>无数据</p>"

    # 计算连接数
    connection_count = defaultdict(int)
    for mat, prop, _ in edges:
        connection_count[mat] += 1
        connection_count[prop] += 1

    # 布局: 圆形布局
    width, height = 700, 450
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 2 - 60

    positions = {}
    for i, node in enumerate(all_nodes):
        angle = 2 * math.pi * i / n_nodes - math.pi / 2
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        positions[node] = (x, y)

    svg_parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border-radius:8px;">']

    # 边
    for mat, prop, pid in edges:
        x1, y1 = positions[mat]
        x2, y2 = positions[prop]
        svg_parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#c5cae9" stroke-width="1" opacity="0.4"/>')

    # 材料节点 (蓝色)
    for node in material_nodes:
        x, y = positions[node]
        r = 8 + min(connection_count[node] * 2, 12)
        label = node[:20] if len(node) > 20 else node
        svg_parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="#1565c0" opacity="0.8"/>')
        svg_parts.append(f'<text x="{x:.0f}" y="{y - r - 5:.0f}" text-anchor="middle" font-size="9" fill="#1565c0" font-weight="600">{label}</text>')

    # 属性节点 (橙色)
    for node in property_nodes:
        x, y = positions[node]
        r = 6 + min(connection_count[node] * 2, 10)
        label = node[:22] if len(node) > 22 else node
        svg_parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="#ff7043" opacity="0.8"/>')
        svg_parts.append(f'<text x="{x:.0f}" y="{y + r + 12:.0f}" text-anchor="middle" font-size="8" fill="#bf360c">{label}</text>')

    # 图例
    svg_parts.append(f'<rect x="10" y="{height - 30}" width="12" height="12" fill="#1565c0" opacity="0.8" rx="2"/>')
    svg_parts.append(f'<text x="28" y="{height - 20}" font-size="10" fill="#333">材料</text>')
    svg_parts.append(f'<rect x="70" y="{height - 30}" width="12" height="12" fill="#ff7043" opacity="0.8" rx="2"/>')
    svg_parts.append(f'<text x="88" y="{height - 20}" font-size="10" fill="#333">属性</text>')
    svg_parts.append(f'<text x="{width - 120}" y="{height - 20}" font-size="10" fill="#999">节点大小 = 连接数</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def _generate_property_chart_svg(fusion):
    """生成属性分布柱状图 SVG"""
    if not fusion:
        return "<p style='color:#999;'>无数据</p>"

    width, height = 700, 350
    bar_height = 22
    bar_gap = 8
    chart_top = 30
    max_bar_width = 450
    label_width = 180

    # 按paper_count排序
    sorted_fusion = sorted(fusion, key=lambda x: x["paper_count"], reverse=True)[:12]

    svg_parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border-radius:8px;">']

    svg_parts.append(f'<text x="10" y="20" font-size="12" fill="#1a237e" font-weight="600">属性跨文献覆盖情况 (按来源文献数排序)</text>')

    for i, f in enumerate(sorted_fusion):
        y = chart_top + i * (bar_height + bar_gap)
        prop_label = f["property"][:25] if len(f["property"]) > 25 else f["property"]
        count = f["paper_count"]
        bar_w = max(20, min(count * 60, max_bar_width))

        # 颜色根据一致性
        color = {"high": "#43a047", "medium": "#fb8c00", "low": "#e53935",
                 "single_source": "#42a5f5", "n/a": "#bdbdbd"}.get(f["consistency"], "#757575")

        svg_parts.append(f'<text x="10" y="{y + 15}" font-size="10" fill="#333">{prop_label}</text>')
        svg_parts.append(f'<rect x="{label_width}" y="{y}" width="{bar_w}" height="{bar_height}" fill="{color}" opacity="0.8" rx="3"/>')
        svg_parts.append(f'<text x="{label_width + bar_w + 5}" y="{y + 15}" font-size="10" fill="#555">{count}篇 | {f["consistency"]}</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def _generate_route_a_svg(route_a):
    """生成路线A构效关系网络图 SVG"""
    relationships = route_a.get("relationships", [])
    if not relationships:
        return ""

    width, height = 700, 400
    svg_parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#f1f8e9;border-radius:8px;">']

    svg_parts.append(f'<text x="10" y="20" font-size="12" fill="#2e7d32" font-weight="600">组成-性能关系网络</text>')

    # 收集唯一组成和性能节点
    components = list(set(r.get("component", "") for r in relationships))
    properties = list(set(r.get("property", "") for r in relationships))

    # 布局: 左侧组成, 右侧性能
    comp_x = 150
    prop_x = 550
    n_comp = len(components)
    n_prop = len(properties)

    comp_positions = {}
    for i, comp in enumerate(components):
        y = 60 + i * (height - 80) / max(n_comp, 1)
        comp_positions[comp] = (comp_x, y)
        label = comp[:20] if len(comp) > 20 else comp
        svg_parts.append(f'<rect x="{comp_x - 70}" y="{y - 12}" width="140" height="24" fill="#66bb6a" opacity="0.3" rx="12"/>')
        svg_parts.append(f'<text x="{comp_x}" y="{y + 4}" text-anchor="middle" font-size="10" fill="#1b5e20" font-weight="600">{label}</text>')

    prop_positions = {}
    for i, prop in enumerate(properties):
        y = 60 + i * (height - 80) / max(n_prop, 1)
        prop_positions[prop] = (prop_x, y)
        label = prop[:22] if len(prop) > 22 else prop
        svg_parts.append(f'<rect x="{prop_x - 70}" y="{y - 12}" width="140" height="24" fill="#42a5f5" opacity="0.3" rx="12"/>')
        svg_parts.append(f'<text x="{prop_x}" y="{y + 4}" text-anchor="middle" font-size="10" fill="#0d47a1" font-weight="600">{label}</text>')

    # 关系边
    for rel in relationships:
        comp = rel.get("component", "")
        prop = rel.get("property", "")
        if comp in comp_positions and prop in prop_positions:
            x1, y1 = comp_positions[comp]
            x2, y2 = prop_positions[prop]

            trend = rel.get("trend", "")
            color = {"positive": "#43a047", "negative": "#e53935", "nonlinear": "#fb8c00", "unclear": "#9e9e9e"}.get(trend, "#757575")
            arrow = {"positive": "&#8593;", "negative": "&#8595;", "nonlinear": "&#8656;", "unclear": "?"}.get(trend, "")

            # 曲线连接
            mid_x = (x1 + x2) / 2
            svg_parts.append(f'<path d="M {x1 + 70} {y1} Q {mid_x} {(y1 + y2) / 2} {x2 - 70} {y2}" '
                           f'stroke="{color}" stroke-width="2" fill="none" opacity="0.6"/>')
            svg_parts.append(f'<text x="{mid_x}" y="{(y1 + y2) / 2 - 5}" text-anchor="middle" font-size="9" fill="{color}">{arrow}</text>')

    # 图例
    svg_parts.append(f'<text x="10" y="{height - 15}" font-size="10" fill="#1b5e20">&#9650; 组成变量</text>')
    svg_parts.append(f'<text x="{width - 100}" y="{height - 15}" font-size="10" fill="#0d47a1">&#9650; 性能变量</text>')
    svg_parts.append(f'<text x="{width // 2 - 80}" y="{height - 15}" font-size="9" fill="#757575">&#8593;正相关 &#8595;负相关 &#8656;非线性</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def _generate_convergence_svg(route_a):
    """生成GA和BO收敛曲线 SVG"""
    opt = route_a.get("optimization") if route_a else None
    if not opt:
        return ""

    ga_history = opt.get("ga", {}).get("convergence_history", [])
    bo_history = opt.get("bo", {}).get("convergence_history", [])

    if not ga_history and not bo_history:
        return ""

    width, height = 700, 320
    margin_l, margin_r, margin_t, margin_b = 50, 180, 30, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    svg_parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border-radius:8px;">']

    # 标题
    svg_parts.append(f'<text x="{width//2}" y="18" text-anchor="middle" font-size="13" fill="#1a237e" font-weight="600">迭代优化收敛曲线</text>')

    # 确定Y轴范围
    all_fitness = []
    for h in ga_history:
        all_fitness.append(h.get("best_fitness", 0))
    for h in bo_history:
        all_fitness.append(h.get("fitness", 0))
    if not all_fitness:
        return ""
    y_min = min(all_fitness) - 0.02
    y_max = max(all_fitness) + 0.02
    y_range = y_max - y_min if y_max > y_min else 0.1

    # 确定X轴范围
    max_gen = max(len(ga_history), len(bo_history))
    if max_gen == 0:
        return ""

    # 网格线
    for i in range(5):
        y = margin_t + plot_h * (1 - i / 4)
        val = y_min + y_range * i / 4
        svg_parts.append(f'<line x1="{margin_l}" y1="{y:.0f}" x2="{margin_l + plot_w}" y2="{y:.0f}" stroke="#eee" stroke-width="1"/>')
        svg_parts.append(f'<text x="{margin_l - 5}" y="{y + 3:.0f}" text-anchor="end" font-size="9" fill="#999">{val:.3f}</text>')

    # X轴标签
    svg_parts.append(f'<text x="{margin_l + plot_w // 2}" y="{height - 8}" text-anchor="middle" font-size="10" fill="#757575">迭代次数</text>')
    svg_parts.append(f'<text x="15" y="{margin_t + plot_h // 2}" text-anchor="middle" font-size="10" fill="#757575" transform="rotate(-90, 15, {margin_t + plot_h // 2})">适应度</text>')

    # GA曲线 (蓝色, best_fitness)
    if ga_history:
        points = []
        for i, h in enumerate(ga_history):
            x = margin_l + (i / max(max_gen - 1, 1)) * plot_w
            y = margin_t + plot_h * (1 - (h["best_fitness"] - y_min) / y_range)
            points.append(f"{x:.0f},{y:.0f}")
        svg_parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1565c0" stroke-width="2"/>')

        # GA avg_fitness (虚线)
        avg_points = []
        for i, h in enumerate(ga_history):
            x = margin_l + (i / max(max_gen - 1, 1)) * plot_w
            y = margin_t + plot_h * (1 - (h.get("avg_fitness", h["best_fitness"]) - y_min) / y_range)
            avg_points.append(f"{x:.0f},{y:.0f}")
        svg_parts.append(f'<polyline points="{" ".join(avg_points)}" fill="none" stroke="#1565c0" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>')

        # 标记最优点
        best_idx = max(range(len(ga_history)), key=lambda i: ga_history[i]["best_fitness"])
        bx = margin_l + (best_idx / max(max_gen - 1, 1)) * plot_w
        by = margin_t + plot_h * (1 - (ga_history[best_idx]["best_fitness"] - y_min) / y_range)
        svg_parts.append(f'<circle cx="{bx:.0f}" cy="{by:.0f}" r="4" fill="#1565c0"/>')

    # BO曲线 (绿色)
    if bo_history:
        points = []
        # 累积最优
        cum_best = -1
        for i, h in enumerate(bo_history):
            cum_best = max(cum_best, h.get("fitness", 0))
            x = margin_l + (i / max(len(bo_history) - 1, 1)) * plot_w
            y = margin_t + plot_h * (1 - (cum_best - y_min) / y_range)
            points.append(f"{x:.0f},{y:.0f}")
        svg_parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#2e7d32" stroke-width="2"/>')

        # BO每次评估点
        for i, h in enumerate(bo_history):
            x = margin_l + (i / max(len(bo_history) - 1, 1)) * plot_w
            y = margin_t + plot_h * (1 - (h.get("fitness", 0) - y_min) / y_range)
            phase = h.get("phase", "")
            color = "#81c784" if phase == "initial" else "#2e7d32"
            svg_parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3" fill="{color}" opacity="0.7"/>')

    # 图例
    legend_x = margin_l + plot_w + 15
    legend_y = margin_t + 20
    svg_parts.append(f'<rect x="{legend_x - 5}" y="{legend_y - 15}" width="160" height="80" fill="white" stroke="#e0e0e0" rx="5"/>')
    svg_parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="#1565c0" stroke-width="2"/>')
    svg_parts.append(f'<text x="{legend_x + 25}" y="{legend_y + 4}" font-size="10" fill="#333">GA Best</text>')
    svg_parts.append(f'<line x1="{legend_x}" y1="{legend_y + 20}" x2="{legend_x + 20}" y2="{legend_y + 20}" stroke="#1565c0" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>')
    svg_parts.append(f'<text x="{legend_x + 25}" y="{legend_y + 24}" font-size="10" fill="#333">GA Average</text>')
    svg_parts.append(f'<line x1="{legend_x}" y1="{legend_y + 40}" x2="{legend_x + 20}" y2="{legend_y + 40}" stroke="#2e7d32" stroke-width="2"/>')
    svg_parts.append(f'<text x="{legend_x + 25}" y="{legend_y + 44}" font-size="10" fill="#333">BO Cumulative Best</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def _generate_ablation_svg(ablation):
    """生成消融实验多维度对比 SVG (v2.0: 柱状图 + 雷达图)"""
    if not ablation:
        return ""

    methods = []
    for key in ["baseline", "random_search", "ga", "bo", "ga_bo_hybrid"]:
        if key in ablation:
            r = ablation[key]
            m_data = r.get("metrics", {})
            methods.append({
                "key": key,
                "name": r["method"].replace(" (Baseline)", "").replace(" (GA)", "").replace(" (BO)", "").replace(" (Baseline)", ""),
                "fitness": r["best_fitness"],
                "evals": r["total_evaluations"],
                "time": r["elapsed_time"],
                "metrics": m_data,
            })

    if not methods:
        return ""

    global_opt = ablation.get("_surrogate_grid_optimum", {})
    global_opt_fit = global_opt.get("fitness", max(m["fitness"] for m in methods))

    width, height = 720, 360
    svg_parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border-radius:8px;">']
    svg_parts.append(f'<text x="{width//2}" y="18" text-anchor="middle" font-size="13" fill="#1a237e" font-weight="600">消融实验 v2.0: 多维度优化策略对比</text>')

    # === 左侧: 适应度柱状图 ===
    chart_left = 40
    chart_width = 300
    margin_b = 50
    chart_top = 35
    max_height = 220

    all_fitness = [m["fitness"] for m in methods]
    f_min = min(all_fitness + [global_opt_fit]) - 0.02
    f_max = max(all_fitness + [global_opt_fit]) + 0.02
    f_range = f_max - f_min if f_max > f_min else 0.1

    bar_width = 45
    bar_gap = 14
    colors = ["#9e9e9e", "#42a5f5", "#1565c0", "#2e7d32", "#7b1fa2"]

    svg_parts.append(f'<text x="{chart_left + chart_width//2}" y="{chart_top - 8}" text-anchor="middle" font-size="11" fill="#555" font-weight="600">最优适应度</text>')

    for i, m in enumerate(methods):
        x = chart_left + i * (bar_width + bar_gap)
        bar_h = ((m["fitness"] - f_min) / f_range) * max_height
        y = height - margin_b - bar_h
        color = colors[i % len(colors)]

        svg_parts.append(f'<rect x="{x}" y="{y:.0f}" width="{bar_width}" height="{bar_h:.0f}" fill="{color}" opacity="0.85" rx="3"/>')
        svg_parts.append(f'<text x="{x + bar_width//2}" y="{y - 5:.0f}" text-anchor="middle" font-size="10" fill="{color}" font-weight="600">{m["fitness"]:.4f}</text>')

        label_map = {"baseline": "A:Baseline", "random_search": "B:Random", "ga": "C:GA", "bo": "D:BO", "ga_bo_hybrid": "E:GA+BO"}
        label = label_map.get(m["key"], m["name"][:10])
        svg_parts.append(f'<text x="{x + bar_width//2}" y="{height - margin_b + 14}" text-anchor="middle" font-size="9" fill="#555">{label}</text>')

    # 全局最优线
    gopt_y = height - margin_b - ((global_opt_fit - f_min) / f_range) * max_height
    svg_parts.append(f'<line x1="{chart_left}" y1="{gopt_y:.0f}" x2="{chart_left + len(methods) * (bar_width + bar_gap)}" y2="{gopt_y:.0f}" stroke="#c62828" stroke-width="1.5" stroke-dasharray="5,3"/>')
    svg_parts.append(f'<text x="{chart_left + len(methods) * (bar_width + bar_gap) - 5}" y="{gopt_y - 4:.0f}" text-anchor="end" font-size="9" fill="#c62828">全局最优={global_opt_fit:.4f}</text>')

    # Y轴
    for j in range(4):
        y = chart_top + max_height * (1 - j / 3)
        val = f_min + f_range * j / 3
        svg_parts.append(f'<line x1="{chart_left}" y1="{y:.0f}" x2="{chart_left + len(methods) * (bar_width + bar_gap)}" y2="{y:.0f}" stroke="#eee" stroke-width="1"/>')
        svg_parts.append(f'<text x="{chart_left - 5}" y="{y + 3:.0f}" text-anchor="end" font-size="8" fill="#999">{val:.3f}</text>')

    # === 右侧: 雷达图 (5个指标) ===
    radar_cx = 530
    radar_cy = 160
    radar_r = 90

    svg_parts.append(f'<text x="{radar_cx}" y="{chart_top - 8}" text-anchor="middle" font-size="11" fill="#555" font-weight="600">五维度归一化对比</text>')

    # 5个维度
    dimensions = ["适应度", "样本效率", "收敛AUC", "解多样性", "覆盖率"]
    n_dims = len(dimensions)

    # 计算每个方法每个维度的归一化值 (0-1)
    all_vals = {dim: [] for dim in dimensions}
    for m in methods:
        mt = m["metrics"]
        all_vals["适应度"].append(m["fitness"])
        all_vals["样本效率"].append(1.0 / max(mt.get("sample_efficiency", 1), 1))  # 反转: 评估次数越少越好
        all_vals["收敛AUC"].append(mt.get("convergence_auc", 0))
        all_vals["解多样性"].append(mt.get("solution_diversity", 0))
        all_vals["覆盖率"].append(mt.get("exploration_coverage", 0))

    # 归一化
    norm_vals = {}
    for dim in dimensions:
        vals = all_vals[dim]
        v_min, v_max = min(vals), max(vals)
        v_range = v_max - v_min if v_max > v_min else 1
        norm_vals[dim] = [(v - v_min) / v_range if v_range > 0 else 0.5 for v in vals]

    # 绘制雷达网格
    for level in [0.25, 0.5, 0.75, 1.0]:
        points = []
        for j in range(n_dims):
            angle = 2 * 3.14159 * j / n_dims - 3.14159 / 2
            x = radar_cx + radar_r * level * math.cos(angle)
            y = radar_cy + radar_r * level * math.sin(angle)
            points.append(f"{x:.0f},{y:.0f}")
        svg_parts.append(f'<polygon points="{" ".join(points)}" fill="none" stroke="#e0e0e0" stroke-width="1"/>')

    # 绘制轴线
    for j in range(n_dims):
        angle = 2 * 3.14159 * j / n_dims - 3.14159 / 2
        x = radar_cx + radar_r * math.cos(angle)
        y = radar_cy + radar_r * math.sin(angle)
        svg_parts.append(f'<line x1="{radar_cx}" y1="{radar_cy}" x2="{x:.0f}" y2="{y:.0f}" stroke="#e0e0e0" stroke-width="1"/>')

        # 维度标签
        lx = radar_cx + (radar_r + 15) * math.cos(angle)
        ly = radar_cy + (radar_r + 15) * math.sin(angle)
        svg_parts.append(f'<text x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle" font-size="9" fill="#666">{dimensions[j]}</text>')

    # 绘制每个方法的雷达多边形
    for i, m in enumerate(methods):
        if m["key"] == "baseline":
            continue  # 基线不画雷达
        points = []
        for j, dim in enumerate(dimensions):
            angle = 2 * 3.14159 * j / n_dims - 3.14159 / 2
            val = norm_vals[dim][i]
            x = radar_cx + radar_r * val * math.cos(angle)
            y = radar_cy + radar_r * val * math.sin(angle)
            points.append(f"{x:.0f},{y:.0f}")
        color = colors[i % len(colors)]
        svg_parts.append(f'<polygon points="{" ".join(points)}" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="1.5"/>')

    # 图例
    legend_x = 400
    legend_y = 270
    for i, m in enumerate(methods):
        if m["key"] == "baseline":
            continue
        color = colors[i % len(colors)]
        label_map = {"random_search": "B:Random", "ga": "C:GA", "bo": "D:BO", "ga_bo_hybrid": "E:GA+BO"}
        label = label_map.get(m["key"], m["name"][:10])
        svg_parts.append(f'<rect x="{legend_x}" y="{legend_y + i * 16 - 10}" width="12" height="12" fill="{color}" opacity="0.5" rx="2"/>')
        svg_parts.append(f'<text x="{legend_x + 16}" y="{legend_y + i * 16}" font-size="9" fill="#555">{label}</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


# ============================================================
# 主入口
# ============================================================

def _write_json_atomic(path, value, serializer):
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=serializer)
    os.replace(tmp_path, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="液态金属多Agent文献调研管线")
    parser.add_argument("--query", default="液态金属领域文献调研: 聚焦材料物性、柔性电子、软体机器人与可穿戴传感")
    parser.add_argument("--max-papers", type=int, default=50)
    parser.add_argument("--output-dir", help="输出目录；默认创建带时间戳的新目录")
    parser.add_argument("--offline", action="store_true", help="不调用任何外部API，使用内置演示数据")
    parser.add_argument("--strict", action="store_true", help="API或LLM发生回退时立即失败")
    parser.add_argument("--skip-ablation", action="store_true", help="跳过耗时的消融实验")
    args = parser.parse_args(argv)

    if args.max_papers < 1:
        parser.error("--max-papers 必须大于0")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if args.offline and args.strict:
        parser.error("--offline 与 --strict 不能同时使用")
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = os.path.abspath(args.output_dir or os.path.join(project_dir, "outputs", f"run-{run_stamp}"))
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        # Logged runners may pre-create only their own metadata, not previous results.
        allowed = {"input_config.json", "execution.jsonl"}
        if set(os.listdir(output_dir)) - allowed:
            parser.error("输出目录非空；请指定新的目录，避免覆盖已有结果")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = Pipeline(offline=args.offline, strict=args.strict, target_paper_count=args.max_papers)
    try:
        results = pipeline.run(args.query)
    except Exception as exc:
        _write_json_atomic(os.path.join(output_dir, "failure_manifest.json"), {
            "status": "failed", "error_type": type(exc).__name__,
            "run_mode": pipeline.run_mode, "strict": args.strict,
            "llm_failed_calls": pipeline.llm.failed_call_count,
            "sciverse_failed_calls": pipeline.sciverse.failed_call_count if pipeline.sciverse else 0,
        }, str)
        raise

    def default_serializer(obj):
        if isinstance(obj, (set,)):
            return list(obj)
        return str(obj)

    # 消融实验
    if not args.skip_ablation:
        print("\n[Ablation] 运行消融实验 (迭代优化策略对比)...")
    ablation_results = None
    if results.get("knowledge_cards") and not args.skip_ablation:
        surrogate = CompositionPropertySurrogate(results["knowledge_cards"])
        ablation_results = run_ablation_study(surrogate, results["knowledge_cards"])
        ablation_path = os.path.join(output_dir, "ablation_study.json")
        _write_json_atomic(ablation_path, ablation_results, default_serializer)
        print(f"[Output] 消融实验结果已保存: {ablation_path}")
    results["ablation"] = ablation_results

    manifest = {
        "schema_version": "1.0",
        "application_version": "5.4.0",
        "source_files_sha256": {name: hashlib.sha256((Path(project_dir) / name).read_bytes()).hexdigest()
                                 for name in ("agents.py", "literature_data.py", "optimizer.py", "papers.py", "run.py", "sciverse_client.py", "route_a_data.py")},
        "created_at": datetime.now().astimezone().isoformat(),
        "run_mode": results["pipeline_stats"]["run_mode"],
        "strict": args.strict,
        "query": args.query,
        "max_papers": args.max_papers,
        "python_version": platform.python_version(),
        "llm_model": pipeline.llm.model,
        "llm_endpoint": pipeline.llm.base_url,
        "sciverse_endpoint": pipeline.sciverse.base_url if pipeline.sciverse else None,
        "api_keys_in_output": False,
        "innovation_module": {
            "name": "Evidence-Robust Counterfactual Pareto Discovery (ERCPD)",
            "enabled": bool(results.get("route_a", {}).get("evidence_robust_discovery")),
            "claim_level": "computational_hypothesis_not_experimental_validation",
            "search_guidance": results.get("route_a", {}).get("evidence_robust_discovery", {}).get(
                "llm_search_guidance"
            ),
            "scientific_audit_method": results.get("route_a", {}).get(
                "evidence_robust_discovery", {}
            ).get("llm_scientific_audit", {}).get("method"),
        },
        "output_directory": output_dir,
        "stats": results["pipeline_stats"],
    }
    results["run_manifest"] = manifest

    json_path = os.path.join(output_dir, "pipeline_results.json")
    cards_path = os.path.join(output_dir, "knowledge_cards.json")
    gaps_path = os.path.join(output_dir, "research_gaps.json")
    manifest_path = os.path.join(output_dir, "run_manifest.json")
    _write_json_atomic(json_path, results, default_serializer)
    _write_json_atomic(cards_path, results["knowledge_cards"], default_serializer)
    _write_json_atomic(gaps_path, results["gaps"], default_serializer)
    _write_json_atomic(manifest_path, manifest, default_serializer)
    print(f"\n[Output] JSON 结果已保存: {json_path}")
    print(f"[Output] 知识卡片已保存: {cards_path}")
    print(f"[Output] Research Gap 分析已保存: {gaps_path}")
    print(f"[Output] 运行清单已保存: {manifest_path}")

    if results.get("route_a"):
        route_a_path = os.path.join(output_dir, "route_a_analysis.json")
        _write_json_atomic(route_a_path, results["route_a"], default_serializer)
        print(f"[Output] 路线A构效关系分析已保存: {route_a_path}")

    # HTML
    html_path = os.path.join(output_dir, "survey_report.html")
    generate_html_report(results, html_path)
    print(f"[Output] HTML 调研报告已保存: {html_path}")

    print(f"\n{'=' * 70}")
    print(f"  Demo v5.3 完成! 运行模式: {manifest['run_mode']}")
    print("  参考锚点均标记为待原始来源逐项复核")
    print(f"  所有输出文件位于: {output_dir}")
    print(f"{'=' * 70}")

    return output_dir


if __name__ == "__main__":
    main()
