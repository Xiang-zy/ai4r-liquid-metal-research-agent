"""
多Agent文献调研管线 - 核心实现 (v5.3 ERCPD版)
LLM: MiniMax OpenAI-compatible API (中国区端点)
文献检索: Sciverse API (多chunk深度获取)
新增: 路线A 构效关系发现Agent, GA+BO迭代优化循环, 消融实验
"""

import os
import re
import json
import time
import math
import hashlib
import statistics
import urllib.request
import urllib.error
from datetime import datetime
from collections import defaultdict

from sciverse_client import SciverseClient
from literature_data import _normalize_value, reference_material_key
from optimizer import (
    CompositionPropertySurrogate,
    GeneticAlgorithm,
    BayesianOptimizer,
    RandomSearch,
    run_evidence_robust_discovery,
    run_ercpd_parameter_ablation,
    run_ablation_study,
)


# ============================================================
# 工具函数
# ============================================================

def deduplicate_authors(authors):
    """作者去重"""
    if not isinstance(authors, list):
        return [str(authors)] if authors else []
    seen = set()
    result = []
    for a in authors:
        name = str(a).strip().lower()
        # 标准化: "wang, jiangxin" 和 "jiangxin wang" 视为同一人
        parts = re.split(r"[,\s]+", name)
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            key = frozenset(parts[:3])
        else:
            key = name
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result[:6]


def normalize_property_name(name):
    """标准化属性名，便于跨文献匹配"""
    name = name.lower().strip()
    aliases = {
        "conductivity": "electrical conductivity",
        "electrical conductivity": "electrical conductivity",
        "conductivity (electrical)": "electrical conductivity",
        "surface tension": "surface tension",
        "surface stress": "surface tension",
        "critical surface stress for shape stabilization by oxide skin": "surface tension",
        "viscosity": "viscosity",
        "melting point": "melting point",
        "melting temperature": "melting point",
        "density": "density",
        "max strain": "max strain",
        "maximum stretchability (strain)": "max strain",
        "stretchability": "max strain",
        "max bending angle": "bending angle",
        "gauge factor": "gauge factor",
        "response time": "response time",
        "actuation response time": "response time",
        "self-healing recovery": "self-healing recovery",
        "cyclic degradation": "cyclic degradation",
        "cyclic drift": "cyclic degradation",
        "temperature sensitivity": "temperature sensitivity",
        "min detectable strain": "min detectable strain",
        "liquid metal loading": "filler loading",
        "oxide skin thickness": "oxide skin thickness",
        "thickness": "oxide skin thickness",
    }
    return aliases.get(name, name)


# ============================================================
# LLM 客户端 (MiniMax OpenAI-compatible API, 中国区端点)
# ============================================================

class LLMClient:
    """MiniMax LLM 客户端, 无API时回退到模板模式"""

    def __init__(self):
        self.api_key = os.environ.get("MINIMAX_API_KEY")
        self.base_url = os.environ.get(
            "MINIMAX_BASE_URL",
            "https://api.minimaxi.com/v1/chat/completions",
        )
        self.model = os.environ.get("LLM_MODEL", "MiniMax-M3")
        self.mode = "api" if self.api_key else "template"
        self.call_count = 0
        self.failed_call_count = 0
        self.request_attempt_count = 0
        self.total_tokens = 0
        self.last_error = None
        self.last_finish_reason = None
        self.max_retries = int(os.environ.get("MINIMAX_MAX_RETRIES", "2"))

    def chat(self, prompt, system_prompt="", temperature=0.3, max_tokens=2000):
        """调用 MiniMax API, 返回纯文本 (去除 think 标签)"""
        if self.mode != "api":
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "你是材料科学领域的专业研究助手。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            # `max_tokens` is deprecated by the current MiniMax OpenAI-
            # compatible API. Keep the Python argument name for callers, but
            # send the supported wire-format field.
            "max_completion_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                self.request_attempt_count += 1
                with urllib.request.urlopen(req, timeout=90) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    choice = result["choices"][0]
                    content = choice["message"].get("content", "")
                    usage = result.get("usage", {})
                    self.total_tokens += usage.get("total_tokens", 0)
                    self.call_count += 1
                    self.last_finish_reason = choice.get("finish_reason")
                    if self.last_finish_reason == "length":
                        raise ValueError("truncated completion")
                    if not isinstance(content, str):
                        raise ValueError("non-text completion")
                    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
                    if not content:
                        raise ValueError("模型响应在移除思考内容后为空")
                    return content
            except urllib.error.HTTPError as exc:
                self.last_error = f"HTTP {exc.code}"
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                break
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                self.last_error = type(exc).__name__  # never print supplier payloads/credentials
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                break
        self.failed_call_count += 1
        print(f"  [LLM] API调用失败: {self.last_error}")
        return None


# ============================================================
# Agent 基类
# ============================================================

class BaseAgent:
    def __init__(self, name, llm_client):
        self.name = name
        self.llm = llm_client
        self.logs = []

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {self.name}: {msg}"
        self.logs.append(entry)
        print(f"  {entry}")

    def run(self, *args, **kwargs):
        raise NotImplementedError


# ============================================================
# Agent 1: 任务规划Agent (增强: 8条查询)
# ============================================================

class TaskPlannerAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("TaskPlanner", llm_client)

    def run(self, user_query):
        self.log("解析用户意图, 生成调研任务规划...")

        scope = {
            "user_query": user_query,
            "domain": "液态金属 (Liquid Metal)",
            "subtopics": [
                "材料基础物性 (电导率、表面张力、粘度、密度、熔点)",
                "氧化行为与界面科学 (氧化物皮膜、pH响应、界面电学)",
                "柔性电子与可拉伸导体 (应变传感、拉伸互连、可重构天线)",
                "软体机器人与驱动器 (弯曲驱动、自修复、气动执行)",
                "微流控与液滴操控 (电场驱动、电润湿、微通道)",
                "生物医学与可穿戴 (健康监测、生物相容性、电子皮肤)",
            ],
            "target_paper_count": 50,
            "analysis_dimensions": [
                "材料组成与配比 (Ga/In/Sn比例)",
                "关键性能参数及测试条件",
                "应用场景与验证方法",
                "已识别的局限性",
                "跨文献数据一致性",
                "组成-性能定量关系 (路线A)",
            ],
            "retrieval_keywords": [
                "liquid metal", "EGaIn", "galinstan", "gallium alloy",
                "stretchable electronics", "soft robotics", "wearable sensor",
                "self-healing", "liquid metal droplet", "oxide skin",
            ],
            "search_queries": [
                "liquid metal EGaIn electrical conductivity surface tension viscosity density properties",
                "liquid metal Galinstan stretchable electronics wearable sensor strain gauge",
                "liquid metal self-healing soft robotics actuator bending pneumatic",
                "liquid metal droplet electrowetting locomotion manipulation microfluidic",
                "liquid metal oxidation oxide skin mechanical stability interface",
                "gallium indium alloy composition melting point thermal properties",
                "liquid metal antenna reconfigurable flexible conductor RF",
                "liquid metal biocompatibility wearable healthcare sensor human motion",
                "liquid metal gallium alloy thermal conductivity heat transfer coolant",
                "liquid metal pH responsive deformation shape change electrochemical",
                "liquid metal composite polymer conductive elastomer filler",
                "liquid metal printing direct write fabrication patterning 3D",
                "gallium liquid metal alloy rheology flow behavior non-Newtonian",
                "liquid metal electrode battery electrochemistry energy storage",
                "liquid metal gallium tin eutectic phase diagram solidification",
            ],
            "quality_criteria": {
                "min_journal_tier": "SCI Q2",
                "year_range": "2008-2026",
                "min_citations": 10,
                "require_experimental_data": True,
            },
        }

        self.log(f"规划完成: 领域={scope['domain']}, 子主题={len(scope['subtopics'])}个, "
                 f"检索查询={len(scope['search_queries'])}条")
        return scope


# ============================================================
# Agent 2: 文献检索Agent (增强: 多chunk深度获取)
# ============================================================

class LiteratureRetrievalAgent(BaseAgent):
    def __init__(self, llm_client, sciverse_client=None):
        super().__init__("LiteratureRetriever", llm_client)
        self.sciverse = sciverse_client

    def run(self, scope):
        self.log(f"通过 Sciverse API 检索文献 (多查询语义检索)...")

        if not self.sciverse:
            self.log("Sciverse 客户端未配置, 跳过检索")
            return []

        all_hits = []
        seen_doc_ids = set()

        for query in scope.get("search_queries", []):
            self.log(f"  查询: {query[:60]}...")
            hits = self.sciverse.agentic_search(query, top_k=10)
            for hit in hits:
                doc_id = hit.get("doc_id", "")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    all_hits.append(hit)
                elif not doc_id:
                    all_hits.append(hit)
            time.sleep(0.5)

        all_hits.sort(
            key=lambda h: (h.get("citation_count", 0), h.get("score", 0)),
            reverse=True,
        )

        retrieved = []
        for i, hit in enumerate(all_hits):
            paper = {
                "id": f"P{i+1:03d}",
                "title": hit.get("title", "Unknown"),
                "authors": deduplicate_authors(hit.get("author", [])),
                "journal": hit.get("publication_venue_name_unified", "Unknown"),
                "year": int(hit.get("publication_published_year", 0)) if hit.get("publication_published_year") else 0,
                "doi": hit.get("doi", ""),
                "doc_id": hit.get("doc_id", ""),
                "citation_count": hit.get("citation_count", 0),
                "score": round(hit.get("score", 0), 4),
                "abstract": hit.get("abstract", ""),
                "chunk": hit.get("chunk", ""),
                "chunk_id": hit.get("chunk_id", ""),
                "page_no": hit.get("page_no", 0),
                "primary_topic": hit.get("primary_topic", ""),
            }
            retrieved.append(paper)

        self.log(f"检索到 {len(retrieved)} 篇唯一文献 (去重后, {len(scope['search_queries'])}条查询)")
        return retrieved


# ============================================================
# Agent 3: 文献筛选Agent
# ============================================================

class LiteratureFilterAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("LiteratureFilter", llm_client)

    def run(self, retrieved_papers, scope):
        self.log(f"对 {len(retrieved_papers)} 篇文献进行相关性评分...")

        scored = []
        for p in retrieved_papers:
            citation_score = min(p.get("citation_count", 0) / 100.0, 1.0) * 30
            relevance_score = min(p.get("score", 0), 1.0) * 35
            year = p.get("year", 0)
            year_score = min(max((year - 2005) / 20.0, 0), 1.0) * 20
            abstract_len = len(p.get("abstract", "")) + len(p.get("chunk", ""))
            data_score = min(abstract_len / 800.0, 1.0) * 15

            total = round(citation_score + relevance_score + year_score + data_score, 1)

            scored.append({
                **p,
                "relevance_score": total,
                "score_breakdown": {
                    "citation_impact": round(citation_score, 1),
                    "semantic_relevance": round(relevance_score, 1),
                    "recency": round(year_score, 1),
                    "data_richness": round(data_score, 1),
                },
                "selected": total >= 40.0,
            })

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)

        target = scope.get("target_paper_count", 5)
        # For large target counts, use a minimum score threshold to ensure quality
        min_score = 25.0 if target > 20 else 40.0
        qualified = [s for s in scored if s["relevance_score"] >= min_score]
        selected = qualified[:target] if len(qualified) >= target else scored[:target]
        for s in selected:
            s["selected"] = True

        self.log(f"筛选完成: {len(selected)}/{len(scored)} 篇入选 (目标={target})")
        for p in scored[:20]:
            status = "PASS" if p["selected"] else "SKIP"
            self.log(f"  {p['id']} {p['title'][:50]}... 评分={p['relevance_score']} 引用={p.get('citation_count',0)} [{status}]")

        return {"scored_papers": scored, "selected_papers": selected}


# ============================================================
# Agent 4: 知识抽取Agent (增强: 多chunk + 二次抽取 + 属性标准化)
# ============================================================

class KnowledgeExtractionAgent(BaseAgent):
    def __init__(self, llm_client, sciverse_client=None):
        super().__init__("KnowledgeExtractor", llm_client)
        self.sciverse = sciverse_client

    def run(self, selected_papers):
        self.log(f"对 {len(selected_papers)} 篇文献进行深度 LLM 知识抽取...")

        knowledge_cards = []
        for idx, sp in enumerate(selected_papers):
            if (idx + 1) % 10 == 0:
                self.log(f"  === 进度: {idx+1}/{len(selected_papers)} 篇已处理 ===")
            self.log(f"  抽取 {sp['id']}: {sp['title'][:60]}...")

            paper_text = self._gather_text(sp)
            card = self._extract_with_llm(sp, paper_text)

            if card is None:
                self.log(f"    LLM抽取失败, 使用回退模式")
                card = self._extract_fallback(sp, paper_text)
            elif len(card["properties"]) < 3 and self.llm.mode == "api":
                self.log(f"    首轮抽取仅 {len(card['properties'])} 条属性, 进行二次抽取...")
                card2 = self._extract_with_llm(sp, paper_text, second_pass=True)
                if card2 and len(card2["properties"]) > len(card["properties"]):
                    card = card2
                if not card["properties"]:
                    self.log("    未找到可逐字回溯的LLM数值证据, 使用正则回退")
                    card = self._extract_fallback(sp, paper_text)

            knowledge_cards.append(card)
            time.sleep(0.5)

        total_props = sum(len(c["properties"]) for c in knowledge_cards)
        self.log(f"知识抽取完成, 生成 {len(knowledge_cards)} 张知识卡片, 共 {total_props} 条属性")
        return knowledge_cards

    def _gather_text(self, paper):
        """收集论文的所有可用文本内容 (增强: 多chunk深度获取)"""
        parts = []
        if paper.get("abstract"):
            parts.append(f"[Abstract] {paper['abstract']}")
        if paper.get("chunk"):
            parts.append(f"[Content Chunk] {paper['chunk']}")

        # 深度获取: 多个content chunk
        if self.sciverse and paper.get("doc_id"):
            full_text = self.sciverse.get_paper_full_text(paper["doc_id"], max_chars=5000)
            if full_text:
                parts.append(f"[Full Text] {full_text}")

        text = "\n\n".join(parts)
        paper["_evidence_text"] = text
        return text

    def _extract_with_llm(self, paper, paper_text, second_pass=False):
        """使用 LLM 从论文文本中抽取结构化知识"""
        paper["_evidence_text"] = paper_text
        if self.llm.mode != "api":
            return None

        if second_pass:
            prompt = f"""请仔细阅读以下论文内容，找出之前可能遗漏的定量数据。

论文标题: {paper['title']}
论文内容:
{paper_text[:6000]}

请额外抽取至少3条之前未提到的定量属性。重点关注:
- 具体数值（电导率、张力、应变、温度、电压、速度、厚度等）
- 组成比例（Ga/In/Sn重量百分比）
- 性能指标（响应时间、循环次数、灵敏度、恢复率等）
- 几何参数（通道宽度、液滴体积、膜厚度等）

返回JSON格式:
{{
  "properties": [
    {{"material": "材料名", "property": "属性名(英文)", "value": 数值, "unit": "单位", "conditions": "测试条件", "section": "results/methods", "evidence_quote": "包含该数值的原文短句"}}
  ]
}}
只返回JSON, 不要其他文字。"""
        else:
            prompt = f"""请从以下液态金属领域论文中抽取结构化知识。严格按照JSON格式返回, 不要包含其他文字。

论文标题: {paper['title']}
期刊: {paper.get('journal', 'Unknown')}
年份: {paper.get('year', 'Unknown')}
DOI: {paper.get('doi', 'Unknown')}

论文内容:
{paper_text[:5000]}

请返回以下JSON格式 (确保是合法JSON):
{{
  "materials_identified": ["材料1", "材料2"],
  "properties": [
    {{"material": "材料名", "property": "属性名(英文)", "value": 数值, "unit": "单位", "conditions": "测试条件", "section": "results/methods", "evidence_quote": "包含该数值的原文短句"}}
  ],
  "methods_summary": "方法简述(200字以内)",
  "key_findings": "主要发现(200字以内)",
  "limitations": ["局限性1", "局限性2", "局限性3"]
}}

注意:
- properties 中抽取所有有明确数值的属性, 目标至少5条
- 包括: 电导率、表面张力、粘度、密度、熔点、应变、灵敏度、响应时间、组成比例、循环次数等
- value 必须是数值类型(不要字符串), 如果原文给出范围取中间值
- evidence_quote 必须逐字复制论文内容中包含该数值的短句；找不到原文就不要输出该属性
- limitations 要基于论文实际内容, 不要编造
- 如果信息不足, properties 可以为空列表, limitations 至少给2条"""

        system = "你是材料科学文献知识抽取专家。只返回合法JSON, 不要其他文字。"
        result = self.llm.chat(prompt, system_prompt=system, temperature=0.1, max_tokens=2500)

        if not result:
            return None

        try:
            result = re.sub(r"```json\s*", "", result)
            result = re.sub(r"```\s*", "", result)
            result = result.strip()

            data = json.loads(result)

            # 标准化属性名
            normalized_props = []
            if not isinstance(data, dict) or not isinstance(data.get("properties", []), list):
                return None
            for prop in data.get("properties", []):
                if not isinstance(prop, dict):
                    continue
                # 跳过value为null或None的属性
                val = prop.get("value")
                if val is None or isinstance(val, bool):
                    continue
                if isinstance(val, str):
                    try:
                        val = float(val)
                    except ValueError:
                        continue
                if not isinstance(val, (int, float)) or not math.isfinite(val):
                    continue

                if not isinstance(prop.get("property", ""), str):
                    continue
                normalized_name = normalize_property_name(prop.get("property", ""))
                quote = str(prop.get("evidence_quote", "")).strip()
                quote_verified = self._quote_in_text(quote, paper_text)
                if not quote_verified or not self._value_in_quote(val, quote):
                    continue
                normalized_props.append({
                    "material": prop.get("material", "liquid metal"),
                    "property": normalized_name,
                    "property_original": prop.get("property", ""),
                    "value": val,
                    "unit": prop.get("unit", ""),
                    "conditions": prop.get("conditions", ""),
                    "section": prop.get("section", "results"),
                    "source_section": prop.get("section", "results"),
                    "evidence": self._evidence_record(
                        paper, quote, quote_verified, prop.get("section", "results")
                    ),
                })

            return {
                "paper_id": paper["id"],
                "title": paper["title"],
                "authors": deduplicate_authors(paper.get("authors", [])),
                "journal": paper.get("journal", "Unknown"),
                "year": paper.get("year", 0),
                "doi": paper.get("doi", ""),
                "source": self._source_record(paper),
                "source_text": paper_text,
                "domain_tags": self._infer_domain_tags(paper),
                "materials_identified": data.get("materials_identified", []),
                "properties": normalized_props,
                "methods_summary": data.get("methods_summary", ""),
                "key_findings": data.get("key_findings", ""),
                "limitations": data.get("limitations", []),
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_method": f"{self.llm.model} + Sciverse (LLM mode, enhanced)" + (" + 2nd pass" if second_pass else ""),
                "extraction_mode": "llm",
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"    JSON解析失败: {e}")
            return None

    def _extract_fallback(self, paper, paper_text):
        """回退模式: 从文本中简单抽取 (带去重)"""
        properties = []
        seen_keys = set()  # (property, value) 去重
        chunk = paper_text
        paper["_evidence_text"] = paper_text

        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[xX×*]\s*10\s*\^?\s*[-+]?\d+|[eE][-+]?\d+)?"
        patterns = [
            (rf"({number})\s*S/m\b", "electrical conductivity", "S/m"),
            (rf"({number})\s*mN/m\b", "surface tension", "mN/m"),
            (rf"({number})\s*Pa(?:\*|·)?s\b", "viscosity", "Pa*s"),
            (rf"({number})(?:\s*°\s*C|\s+deg(?:ree)?\s*C|\s+C\b)", "temperature", "C"),
            (rf"({number})\s*g/cm(?:3|³)", "density", "g/cm3"),
            (rf"({number})\s*%", "percentage", "%"),
        ]

        materials = []
        for mat_name in ["EGaIn", "Galinstan", "gallium", "liquid metal"]:
            if mat_name.lower() in chunk.lower():
                materials.append(mat_name)
        if not materials:
            materials.append("liquid metal")

        for pattern, prop_name, unit in patterns:
            matches = re.finditer(pattern, chunk, flags=re.IGNORECASE)
            for match_obj in matches:
                try:
                    val = self._parse_number(match_obj.group(1))
                except (ValueError, OverflowError):
                    continue
                if not math.isfinite(val):
                    continue
                before = re.split(r"[.;!?\n]", chunk[max(0, match_obj.start()-100):match_obj.start()])[-1]
                after = chunk[match_obj.end():match_obj.end()+30]
                actual_name = prop_name
                if prop_name == "temperature":
                    if not re.search(r"melting\s*(?:point|temperature)|melts?\s*(?:at)?|熔点", before, re.I):
                        continue
                    actual_name = "melting point"
                if prop_name == "percentage":
                    if not (re.search(r"(?:strain|stretchability)[^.;!?]{0,40}$", before, re.I)
                            or re.match(r"\s*(?:tensile\s+)?strain\b", after, re.I)):
                        continue
                    actual_name = "max strain"
                specific = [m for m in materials if m != "liquid metal"]
                material = specific[0] if len(specific) == 1 else "liquid metal (identity ambiguous)"
                if re.search(r"composite|elastomer|polymer|TPU|PDMS|sensor|oxide\s+skin", chunk, re.I):
                    material += " composite/device (not bulk alloy)"

                norm_prop = normalize_property_name(actual_name)
                dedup_key = (norm_prop, val)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                properties.append({
                    "material": material,
                    "property": norm_prop,
                    "property_original": prop_name,
                    "value": val,
                    "unit": unit,
                    "conditions": "from text",
                    "section": "extracted",
                    "source_section": "extracted",
                    "evidence": self._evidence_record(
                        paper, match_obj.group(0), True, f"text:char-{match_obj.start()}"
                    ),
                })

        return {
            "paper_id": paper["id"],
            "title": paper["title"],
            "authors": deduplicate_authors(paper.get("authors", [])),
            "journal": paper.get("journal", "Unknown"),
            "year": paper.get("year", 0),
            "doi": paper.get("doi", ""),
            "source": self._source_record(paper),
            "source_text": paper_text,
            "domain_tags": self._infer_domain_tags(paper),
            "materials_identified": materials,
            "properties": properties[:8],
            "methods_summary": paper.get("abstract", "")[:300],
            "key_findings": paper.get("chunk", "")[:300],
            "limitations": [
                "Information extracted from abstract and content chunk only",
                "Full text not available for all papers",
            ],
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_method": "regex fallback (LLM unavailable)",
            "extraction_mode": "regex_fallback",
        }

    @staticmethod
    def _parse_number(value):
        compact = re.sub(r"\s+", "", value)
        compact = re.sub(r"[xX×*]10\^?", "e", compact)
        return float(compact)

    @classmethod
    def _value_in_quote(cls, value, quote):
        pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[xX×*]\s*10\s*\^?\s*[-+]?\d+|[eE][-+]?\d+)?"
        for match in re.finditer(pattern, quote):
            try:
                if math.isclose(value, cls._parse_number(match.group()), rel_tol=1e-9, abs_tol=1e-12):
                    return True
            except (ValueError, OverflowError):
                pass
        return False

    @staticmethod
    def _quote_in_text(quote, text):
        if not quote:
            return False
        normalize = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
        return normalize(quote) in normalize(text)

    @staticmethod
    def _source_record(paper):
        return {
            "paper_id": paper.get("id", ""),
            "doi": paper.get("doi", ""),
            "doc_id": paper.get("doc_id", ""),
            "chunk_id": paper.get("chunk_id", ""),
            "page_no": paper.get("page_no"),
            "title": paper.get("title", ""),
            "data_origin": paper.get("data_origin", "retrieved_unverified"),
            "source_text_sha256": hashlib.sha256(paper.get("_evidence_text", "").encode()).hexdigest(),
        }

    def _evidence_record(self, paper, quote, quote_verified, locator):
        text = paper.get("_evidence_text", "")
        normalized = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
        offset = normalized(text).find(normalized(quote)) if quote else -1
        return {
            **self._source_record(paper),
            "quote": quote,
            "quote_verified": bool(quote_verified and offset >= 0),
            "page_no": None,  # a retrieval-hit page cannot locate quotes from appended full text
            "locator": f"gathered_text:normalized_char-{offset}" if offset >= 0 else "",
            "reported_section": str(locator or ""),
            "verification_scope": "substring_in_saved_source_text_only",
        }

    def _infer_domain_tags(self, paper):
        tags = []
        text = (paper.get("title", "") + " " + paper.get("abstract", "") + " " + paper.get("primary_topic", "")).lower()
        if any(k in text for k in ["conductivity", "surface tension", "viscosity", "property", "properties", "density", "melting"]):
            tags.append("基础物性")
        if any(k in text for k in ["stretchable", "electronics", "flexible", "wearable", "conductor"]):
            tags.append("柔性电子")
        if any(k in text for k in ["sensor", "wearable", "strain", "healthcare", "monitoring"]):
            tags.append("可穿戴传感")
        if any(k in text for k in ["robotic", "actuator", "self-healing", "soft", "bending"]):
            tags.append("软体机器人")
        if any(k in text for k in ["droplet", "locomotion", "electrowetting", "microfluidic"]):
            tags.append("液滴操控")
        if any(k in text for k in ["oxide", "oxidation", "interface", "skin"]):
            tags.append("氧化行为")
        if any(k in text for k in ["antenna", "rf", "wireless", "reconfigurable"]):
            tags.append("可重构天线")
        if any(k in text for k in ["biocompat", "health", "medical", "skin contact"]):
            tags.append("生物医学")
        return tags or ["其他"]


# ============================================================
# Agent 5: 跨文献知识融合Agent (增强: 属性标准化匹配)
# ============================================================

class KnowledgeFusionAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("KnowledgeFusion", llm_client)

    def run(self, knowledge_cards):
        self.log(f"融合 {len(knowledge_cards)} 张知识卡片...")

        property_groups = defaultdict(list)
        for card in knowledge_cards:
            for prop in card["properties"]:
                key = prop["property"]
                property_groups[key].append({
                    **prop,
                    "paper_id": card["paper_id"],
                    "paper_title": card["title"],
                    "year": card["year"],
                })

        fusion_results = []
        for prop_name, entries in property_groups.items():
            materials = sorted(set(e["material"] for e in entries))
            conditions = sorted(set(e.get("conditions", "unknown") for e in entries))
            papers = sorted(set(e["paper_id"] for e in entries))
            units = set()
            values = []
            for entry in entries:
                raw = entry.get("value")
                if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                    continue
                value, unit = _normalize_value(prop_name, raw, entry.get("unit"))
                if value is None:
                    value, unit = raw, entry.get("unit", "")
                entry["normalized_value"], entry["normalized_unit"] = value, unit
                values.append(value)
                units.add(unit)
            comparable = (len(materials) == 1 and len(conditions) == 1 and len(units) == 1
                          and bool(next(iter(units), "")) and bool(conditions[0])
                          and conditions[0] not in {"unknown", "from text"})

            if not comparable:
                val_range, cv, consistency = "Not pooled: incompatible or missing metadata", None, "not_comparable"
            elif len(values) > 1:
                val_range = f"[{min(values)}, {max(values)}]"
                val_mean = sum(values) / len(values)
                denominator = abs(val_mean + 273.15) if prop_name == "melting point" else abs(val_mean)
                cv = statistics.pstdev(values) / denominator * 100 if denominator else None
                consistency = "n/a" if cv is None else ("high" if cv < 20 else ("medium" if cv < 50 else "low"))
            elif values:
                val_range = str(values[0])
                cv = 0
                consistency = "single_source"
            else:
                val_range = "N/A"
                cv = 0
                consistency = "n/a"

            conflicts = []
            if len(units) > 1:
                conflicts.append("单位无法统一，不合并数值")
            if not comparable:
                conflicts.append("材料、单位或测试条件不足以支持直接数值比较")
            if len(materials) > 1:
                conflicts.append(f"不同材料体系: {', '.join(materials)}")
            if len(conditions) > 1 and prop_name in ["electrical conductivity", "surface tension", "viscosity"]:
                conflicts.append(f"测试条件不一致: {', '.join(conditions)}")

            fusion_results.append({
                "property": prop_name,
                "entries": entries,
                "materials": materials,
                "papers": papers,
                "paper_count": len(papers),
                "value_range": val_range,
                "conditions": conditions,
                "consistency": consistency,
                "variation_coefficient": round(cv, 1) if cv is not None else None,
                "variation_definition": "population_std / absolute_mean * 100; temperature uses Kelvin",
                "comparison_scope": "recorded_metadata_only; conditions_not_independently_verified",
                "conflicts": conflicts,
                "data_gap": self._identify_data_gap(prop_name, entries),
            })

        fusion_results.sort(key=lambda x: (-x["paper_count"], x["property"]))

        cross_ref = sum(1 for f in fusion_results if f["paper_count"] > 1)
        self.log(f"融合完成: {len(fusion_results)} 个属性类别, "
                 f"跨文献交叉引用 {cross_ref} 个, "
                 f"冲突 {sum(1 for f in fusion_results if f['conflicts'])} 个, "
                 f"数据空白 {sum(1 for f in fusion_results if f['data_gap'])} 个")
        return fusion_results

    def _identify_data_gap(self, prop_name, entries):
        gaps = []
        conditions = [e.get("conditions", "").lower() for e in entries]
        has_temp_data = any(re.search(r"\d\s*(?:°\s*C|degC|K\b)|temperature", c, re.I) for c in conditions)
        if not has_temp_data and prop_name in ["electrical conductivity", "surface tension", "viscosity"]:
            gaps.append("未记录明确温度条件；不能据此断言整篇文献缺少温度依赖性研究")
        if len(entries) == 1:
            gaps.append("仅单一文献报道, 需要交叉验证")
        return gaps


# ============================================================
# Agent 6: Research Gap 识别Agent (LLM 增强)
# ============================================================

class GapIdentificationAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("GapIdentifier", llm_client)

    def run(self, fusion_results, knowledge_cards):
        self.log("基于融合结果和知识卡片识别 Research Gap...")

        if not fusion_results:
            self.log("没有可追溯属性可供融合，跳过Research Gap归纳")
            return []

        if self.llm.mode == "api":
            llm_gaps = self._identify_with_llm(fusion_results, knowledge_cards)
            if llm_gaps:
                self.log(f"LLM 识别出 {len(llm_gaps)} 个 Research Gap")
                return self._finalize_gaps(llm_gaps)
            self.log("LLM 识别失败, 回退到规则模式")

        return self._finalize_gaps(self._identify_with_rules(fusion_results, knowledge_cards))

    @staticmethod
    def _finalize_gaps(gaps):
        for gap in gaps:
            gap.setdefault("description", "当前抽取材料提示的待核验问题；不证明整个研究领域存在该空白。")
            gap["claim_level"] = "unverified_research_question"
        return gaps

    def _identify_with_llm(self, fusion_results, knowledge_cards):
        fusion_summary = []
        for f in fusion_results[:20]:
            fusion_summary.append({
                "property": f["property"],
                "materials": f["materials"],
                "paper_count": f["paper_count"],
                "consistency": f["consistency"],
                "conflicts": f["conflicts"],
                "data_gap": f["data_gap"],
                "value_range": f["value_range"],
            })

        limitations_summary = []
        for c in knowledge_cards:
            for lim in c.get("limitations", []):
                limitations_summary.append({"paper_id": c["paper_id"], "limitation": lim})

        prompt = f"""你是材料科学研究专家。基于以下跨文献知识融合结果和论文局限性, 识别5-7个重要的Research Gap。

知识融合结果 (JSON):
{json.dumps(fusion_summary, ensure_ascii=False, indent=2)}

论文局限性 (JSON):
{json.dumps(limitations_summary, ensure_ascii=False, indent=2)}

请返回JSON数组, 每个Gap包含:
[
  {{
    "title": "Gap标题(中文, 30字以内)",
    "description": "详细描述(中文, 150字以内, 说明问题的重要性和现状)",
    "gap_type": "数据空白|方法论缺陷|知识空白|验证缺失|机制不明",
    "severity": "high|medium|low",
    "affected_properties": ["属性1", "属性2"],
    "suggestion": "建议研究方向(中文, 80字以内)"
  }}
]

要求:
1. 基于实际数据, 不要编造
2. 每个Gap必须有来自论文的支撑
3. 优先识别高影响力的问题
4. 识别5-7个Gap
5. 只返回JSON数组, 不要其他文字"""

        result = self.llm.chat(prompt, system_prompt="你是材料科学研究专家。只返回合法JSON。", temperature=0.3, max_tokens=3500)
        if not result:
            return None

        try:
            result = re.sub(r"```json\s*", "", result)
            result = re.sub(r"```\s*", "", result).strip()
            gaps_data = json.loads(result)

            gaps = []
            for i, g in enumerate(gaps_data[:7]):
                evidence = []
                for prop in g.get("affected_properties", []):
                    norm_prop = normalize_property_name(prop)
                    for f in fusion_results:
                        if f["property"] == norm_prop or f["property"] == prop:
                            for e in f["entries"][:3]:
                                record = e.get("evidence")
                                if isinstance(record, dict):
                                    evidence.append(dict(record))

                for lim in limitations_summary:
                    for keyword in g.get("affected_properties", []) + [g.get("gap_type", "")]:
                        if keyword.lower() in lim["limitation"].lower():
                            evidence.append({
                                "paper_id": lim["paper_id"], "quote": lim["limitation"],
                                "quote_verified": False, "locator": "llm_inferred_limitation",
                            })

                seen = set()
                unique_evidence = []
                for e in evidence:
                    key = (e["paper_id"], e["quote"][:50])
                    if key not in seen:
                        seen.add(key)
                        unique_evidence.append(e)

                gaps.append({
                    "id": f"GAP-{i+1:03d}",
                    "title": g["title"],
                    "description": g["description"],
                    "gap_type": g.get("gap_type", "知识空白"),
                    "severity": g.get("severity", "medium"),
                    "affected_properties": g.get("affected_properties", []),
                    "evidence": unique_evidence[:6],
                    "suggestion": g.get("suggestion", ""),
                })

            return gaps if gaps else None
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"  LLM Gap JSON解析失败: {e}")
            return None

    def _identify_with_rules(self, fusion_results, knowledge_cards):
        gaps = []

        temp_props = [f for f in fusion_results
                      if f["property"] in ["electrical conductivity", "surface tension", "viscosity"]
                      and any("温度" in g or "temperature" in g.lower() for g in f.get("data_gap", []))]
        temp_evidence = []
        for card in knowledge_cards:
            for lim in card.get("limitations", []):
                if "temperature" in lim.lower() or "温度" in lim:
                    temp_evidence.append(self._limitation_evidence(card, lim))

        if temp_evidence or temp_props:
            gaps.append({
                "id": "GAP-001",
                "title": "当前条目的温度条件及覆盖范围待核验",
                "description": "当前抽取条目未记录足够温度条件；需要回溯原文，不能据此断言文献或领域缺少温度扫描研究。",
                "gap_type": "数据空白",
                "severity": "high",
                "affected_properties": ["electrical conductivity", "surface tension", "viscosity"],
                "evidence": temp_evidence[:5],
                "suggestion": "建议开展系统性物性表征, 建立温度-物性数据库。",
            })

        cond_props = [f for f in fusion_results if f["conflicts"]
                      and any("条件" in c or "condition" in c.lower() for c in f["conflicts"])]
        if cond_props:
            evidence = []
            for f in cond_props:
                for pid in f["papers"]:
                    for entry in f["entries"]:
                        if entry.get("paper_id") == pid and isinstance(entry.get("evidence"), dict):
                            evidence.append(dict(entry["evidence"]))
                            break
            gaps.append({
                "id": "GAP-002",
                "title": "跨条目材料与测试条件的可比性待核验",
                "description": "当前记录含不同材料、单位或不完整测试条件，暂不合并；这不是已证实的领域方法论缺陷。",
                "gap_type": "方法论缺陷",
                "severity": "high",
                "affected_properties": [f["property"] for f in cond_props],
                "evidence": evidence[:5],
                "suggestion": "建议制定液态金属物性表征标准协议。",
            })

        rel_evidence = []
        for card in knowledge_cards:
            for lim in card.get("limitations", []):
                if any(k in lim.lower() for k in ["cyclic", "long-term", "reliability", "循环", "长期"]):
                    rel_evidence.append(self._limitation_evidence(card, lim))
        if rel_evidence:
            gaps.append({
                "id": "GAP-003",
                "title": "液态金属器件长期可靠性评估不足",
                "gap_type": "验证缺失",
                "severity": "medium",
                "affected_properties": ["cyclic degradation", "self-healing recovery"],
                "evidence": rel_evidence[:5],
                "suggestion": "建议建立加速老化测试标准。",
            })

        oxide_evidence = []
        for card in knowledge_cards:
            for lim in card.get("limitations", []):
                if any(k in lim.lower() for k in ["oxide", "oxidation", "氧化"]):
                    oxide_evidence.append(self._limitation_evidence(card, lim))
        if oxide_evidence:
            gaps.append({
                "id": "GAP-004",
                "title": "液态金属氧化行为机制尚不明确",
                "gap_type": "机制不明",
                "severity": "medium",
                "affected_properties": ["surface tension", "electrical conductivity"],
                "evidence": oxide_evidence[:5],
                "suggestion": "建议利用原位表征技术研究界面氧化行为。",
            })

        for i, g in enumerate(gaps):
            g["id"] = f"GAP-{i+1:03d}"

        return gaps

    @staticmethod
    def _limitation_evidence(card, limitation):
        source = card.get("source", {}) if isinstance(card.get("source"), dict) else {}
        return {
            "paper_id": card.get("paper_id", ""),
            "doi": source.get("doi", card.get("doi", "")),
            "doc_id": source.get("doc_id", ""),
            "chunk_id": source.get("chunk_id", ""),
            "page_no": source.get("page_no"),
            "quote": str(limitation),
            "quote_verified": False,
            "locator": "llm_inferred_limitation",
        }


# ============================================================
# Agent 7: 证据核验Agent
# ============================================================

class EvidenceVerificationAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("EvidenceVerifier", llm_client)

    def run(self, gaps, knowledge_cards):
        self.log(f"核验 {len(gaps)} 个 Research Gap 的证据链...")

        verified = []
        cards_by_id = {c["paper_id"]: c for c in knowledge_cards}
        for gap in gaps:
            evidence = gap.get("evidence", [])
            source_papers = []
            traceable = []
            for item in evidence:
                card = cards_by_id.get(item.get("paper_id"), {})
                source = card.get("source", {})
                doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(source.get("doi") or card.get("doi") or "").strip().lower())
                identity = doi or source.get("doc_id", "")
                quote = str(item.get("quote", "")).strip()
                approved = [p.get("evidence", {}) for p in card.get("properties", [])]
                approved = [e for e in approved if isinstance(e, dict)]
                valid = bool(identity and quote and source.get("data_origin") not in {"synthetic_test_fixture", "historical_demo_fixture"}
                             and KnowledgeExtractionAgent._quote_in_text(quote, card.get("source_text", ""))
                             and any(e.get("quote_verified") is True and e.get("quote") == quote and e.get("locator") for e in approved))
                traceable.append(valid)
                if valid and identity not in source_papers:
                    source_papers.append(identity)

            verification = {
                "gap_id": gap["id"],
                "gap_title": gap["title"],
                "evidence_count": len(evidence),
                "source_papers": source_papers,
                "traceable_evidence_count": sum(traceable),
                "verification_status": "weak",
                "issues": [],
                "verification_scope": "local_evidence_traceability_only; not_gap_novelty_or_scientific_validation",
            }

            if len(evidence) < 2:
                verification["issues"].append("证据数量不足 (<2)")
                verification["verification_status"] = "weak"

            if len(source_papers) < 2:
                verification["issues"].append("证据来源单一, 建议补充多文献交叉验证")

            if not all(traceable):
                verification["issues"].append("存在无法回溯到原文短句及定位信息的证据")

            if len(evidence) >= 2 and len(source_papers) >= 2 and all(traceable):
                verification["verification_status"] = "verified"
            elif any(traceable):
                verification["verification_status"] = "verified_with_notes"

            verified.append(verification)
            self.log(f"  {gap['id']}: {verification['verification_status']} "
                     f"(证据={verification['evidence_count']}, 来源={len(source_papers)})")

        verified_count = sum(1 for v in verified if v["verification_status"] == "verified")
        notes_count = sum(1 for v in verified if v["verification_status"] == "verified_with_notes")
        weak_count = sum(1 for v in verified if v["verification_status"] == "weak")
        self.log(f"核验完成: {verified_count} verified, {notes_count} with_notes, {weak_count} weak")
        return verified


# ============================================================
# Agent 8 (路线A): 构效关系发现Agent
# ============================================================

class StructurePropertyAgent(BaseAgent):
    """路线A: 基于LLM的构效关系发现Agent
    分析材料组成( Ga/In/Sn比例)与关键性能的定量关系"""

    def __init__(self, llm_client):
        super().__init__("StructurePropertyDiscovery", llm_client)

    def run(self, knowledge_cards, fusion_results):
        self.log("路线A: 构效关系发现分析...")

        # 收集所有组成和性能数据
        composition_data = self._extract_composition_data(knowledge_cards)
        property_data = self._collect_property_data(knowledge_cards)
        property_sources = {p["paper_id"] for p in property_data}

        if len(property_data) < 5 or len(property_sources) < 2:
            self.log("可追溯定量数据不足，跳过LLM构效关系归纳")
            llm_result = {
                "agent": "StructurePropertyDiscovery (Route A)",
                "compositions_found": len(composition_data),
                "properties_analyzed": len(property_data),
                "relationships": [],
                "trends": [],
                "composition_optimization": "",
                "data_sufficiency": {
                    "adequate_for_analysis": False,
                    "missing_data": ["至少5条来自2篇以上文献的可追溯定量属性"],
                    "recommendation": "补充带原文短句和定位信息的多组成实验数据后再做定量构效关系归纳。",
                },
                "analysis_timestamp": datetime.now().isoformat(),
                "method": "insufficient-data guard",
            }
        elif self.llm.mode == "api":
            llm_result = self._analyze_with_llm(composition_data, property_data, fusion_results)
            if llm_result:
                self.log(f"构效关系分析完成: {len(llm_result.get('relationships', []))} 条关系, "
                         f"{len(llm_result.get('trends', []))} 个趋势")
            else:
                self.log("LLM分析失败, 使用规则模式")
                llm_result = self._analyze_with_rules(composition_data, property_data, fusion_results)
        else:
            llm_result = self._analyze_with_rules(composition_data, property_data, fusion_results)

        # 迭代优化循环 (GA + BO)
        self.log("启动迭代优化循环 (GA + BO)...")
        optimization_result = self._run_optimization(knowledge_cards)
        if optimization_result:
            llm_result["optimization"] = optimization_result
            self.log(f"优化完成: GA适应度={optimization_result.get('ga', {}).get('best_fitness', 'N/A')}, "
                     f"BO适应度={optimization_result.get('bo', {}).get('best_fitness', 'N/A')}")

        guidance = self._search_guidance(property_data, fusion_results)
        self.log("运行证据感知来源留一法与稳健Pareto反事实搜索...")
        try:
            robust_discovery = run_evidence_robust_discovery(
                CompositionPropertySurrogate(),
                resolution=2.5,
                risk_penalty=guidance["risk_penalty"],
                sn_step=guidance["sn_counterfactual_step_wt_pct"],
            )
            robust_discovery["llm_search_guidance"] = guidance
            robust_discovery["parameter_ablation"] = run_ercpd_parameter_ablation(
                CompositionPropertySurrogate()
            )
            robust_discovery["llm_scientific_audit"] = self._audit_robust_discovery(
                robust_discovery, property_data
            )
            llm_result["evidence_robust_discovery"] = robust_discovery
            self.log(
                f"稳健搜索完成: {robust_discovery['parameters']['grid_candidates']} 个候选, "
                f"Pareto前沿 {robust_discovery['pareto_front_size']} 个"
            )
        except (ValueError, KeyError) as exc:
            self.log(f"稳健搜索跳过: {exc}")
            llm_result["evidence_robust_discovery"] = {"error": str(exc)}

        return llm_result

    def _search_guidance(self, properties, fusion_results):
        default = {
            "risk_penalty": 5.0,
            "sn_counterfactual_step_wt_pct": 5.0,
            "focus": "在平均性能与来源敏感性之间取保守平衡",
            "method": "deterministic_default",
        }
        if self.llm.mode != "api" or len(properties) < 5:
            return default
        prompt = f"""基于以下已抽取属性与跨文献一致性摘要，为稳健组成搜索选择参数。

属性摘要:
{json.dumps(properties[:30], ensure_ascii=False)}

融合摘要:
{json.dumps([{"property": f["property"], "consistency": f["consistency"], "paper_count": f["paper_count"]} for f in fusion_results[:15]], ensure_ascii=False)}

只返回JSON:
{{"risk_penalty": 1到10的数值, "sn_counterfactual_step_wt_pct": 2.5或5或10, "focus": "选择理由"}}
risk_penalty越高越惩罚来源留一法中的不稳定。不得依据未提供的数据作判断。"""
        result = self.llm.chat(
            prompt,
            system_prompt="你负责为可复现搜索选择参数，只返回合法JSON。",
            temperature=0.1,
            max_tokens=400,
        )
        if not result:
            return default
        try:
            data = json.loads(re.sub(r"```(?:json)?\s*|```", "", result).strip())
            risk = min(10.0, max(1.0, float(data.get("risk_penalty", 5.0))))
            step = float(data.get("sn_counterfactual_step_wt_pct", 5.0))
            if step not in {2.5, 5.0, 10.0}:
                step = 5.0
            return {
                "risk_penalty": risk,
                "sn_counterfactual_step_wt_pct": step,
                "focus": str(data.get("focus", ""))[:300],
                "method": f"{self.llm.model}_guided",
            }
        except (ValueError, TypeError, json.JSONDecodeError):
            return default

    def _audit_robust_discovery(self, robust_discovery, properties):
        if self.llm.mode != "api" or len(properties) < 5:
            return {
                "method": "deterministic_guard",
                "decision": "candidate_hypotheses_only",
                "reason": "可追溯抽取属性不足，未调用LLM提升主张等级。",
            }
        compact = {
            "candidate": robust_discovery["best_risk_adjusted_candidate"],
            "counterfactual_tests": robust_discovery["counterfactual_tests"],
            "tradeoff": robust_discovery["robustness_tradeoff_vs_naive"],
            "limits": robust_discovery["interpretation_limits"],
        }
        prompt = f"""审计以下计算结果，输出可证伪的候选假说和下一轮定向检索词。
{json.dumps(compact, ensure_ascii=False)}

只返回JSON:
{{"candidate_hypotheses": [{{"claim": "候选假说", "support": "计算支持", "falsifier": "否证条件"}}],
  "source_sensitive_findings": ["不稳健或负结果"],
  "next_retrieval_queries": ["英文检索式"],
  "claim_level": "computational_hypothesis"}}
禁止使用“已验证”“已发现新规律”等超出计算证据的表述。"""
        result = self.llm.chat(
            prompt,
            system_prompt="你是科学主张审计员，只能降低或保持主张等级，不能越级。",
            temperature=0.1,
            max_tokens=1000,
        )
        if not result:
            return {"method": "llm_unavailable", "decision": "candidate_hypotheses_only"}
        try:
            data = json.loads(re.sub(r"```(?:json)?\s*|```", "", result).strip())
            data["claim_level"] = "computational_hypothesis"
            data["method"] = f"{self.llm.model}_post_search_audit"
            return data
        except (json.JSONDecodeError, TypeError):
            return {"method": "llm_parse_failed", "decision": "candidate_hypotheses_only"}

    def _run_optimization(self, knowledge_cards):
        """运行 GA + BO 迭代优化循环"""
        try:
            surrogate = CompositionPropertySurrogate(knowledge_cards)

            # 遗传算法
            self.log("  [GA] 遗传算法搜索 (15代, 种群20)...")
            ga = GeneticAlgorithm(surrogate, pop_size=20, generations=15, mutation_rate=0.15, seed=42)
            ga_result = ga.run()
            self.log(f"  [GA] 完成: 最优适应度={ga_result['best_fitness']}, "
                     f"组成=Ga:{ga_result['best_composition']['ga']}%/"
                     f"In:{ga_result['best_composition']['in']}%/"
                     f"Sn:{ga_result['best_composition']['sn']}%, "
                     f"评估次数={ga_result['total_evaluations']}, 耗时={ga_result['elapsed_time']}s")

            # 贝叶斯优化
            self.log("  [BO] 贝叶斯优化 (15次迭代)...")
            bo = BayesianOptimizer(surrogate, n_iterations=15, n_initial=5, seed=42)
            bo_result = bo.run()
            self.log(f"  [BO] 完成: 最优适应度={bo_result['best_fitness']}, "
                     f"组成=Ga:{bo_result['best_composition']['ga']}%/"
                     f"In:{bo_result['best_composition']['in']}%/"
                     f"Sn:{bo_result['best_composition']['sn']}%, "
                     f"评估次数={bo_result['total_evaluations']}, 耗时={bo_result['elapsed_time']}s")

            return {
                "ga": ga_result,
                "bo": bo_result,
                "surrogate_data_points": len(surrogate.data_points),
                "surrogate_anchors": len(surrogate.prior_knowledge),
            }
        except Exception as e:
            self.log(f"  优化循环出错: {e}")
            return None

    def _extract_composition_data(self, knowledge_cards):
        """从知识卡片中提取材料组成信息"""
        compositions = []
        for card in knowledge_cards:
            for prop in card["properties"]:
                prop_orig = prop.get("property_original", prop.get("property", "")).lower()
                if any(k in prop_orig for k in ["composition", "wt%", "weight", "gallium composition",
                                                  "indium composition", "tin composition", "loading"]):
                    compositions.append({
                        "paper_id": card["paper_id"],
                        "material": prop.get("material", ""),
                        "component": prop_orig,
                        "value": prop.get("value", 0),
                        "unit": prop.get("unit", ""),
                    })
            # 从材料名中推断组成
            for mat in card.get("materials_identified", []):
                mat_lower = mat.lower()
                if "egain" in mat_lower or "ga-in" in mat_lower:
                    compositions.append({
                        "paper_id": card["paper_id"],
                        "material": mat,
                        "component": "Ga-In eutectic",
                        "value": "75.5% Ga / 24.5% In",
                        "unit": "wt%",
                        "inferred": True,
                    })
                elif "galinstan" in mat_lower:
                    compositions.append({
                        "paper_id": card["paper_id"],
                        "material": mat,
                        "component": "Ga-In-Sn eutectic",
                        "value": "68.5% Ga / 21.5% In / 10% Sn",
                        "unit": "wt%",
                        "inferred": True,
                    })
        return compositions

    def _collect_property_data(self, knowledge_cards):
        """收集关键性能数据"""
        key_properties = [
            "electrical conductivity", "surface tension", "viscosity",
            "melting point", "density", "max strain", "gauge factor",
            "response time", "self-healing recovery", "oxide skin thickness",
            "filler loading", "bending angle",
        ]
        prop_data = []
        for card in knowledge_cards:
            for prop in card["properties"]:
                if prop["property"] in key_properties:
                    prop_data.append({
                        "paper_id": card["paper_id"],
                        "material": prop.get("material", ""),
                        "property": prop["property"],
                        "value": prop.get("value", 0),
                        "unit": prop.get("unit", ""),
                        "conditions": prop.get("conditions", ""),
                    })
        return prop_data

    def _analyze_with_llm(self, compositions, properties, fusion_results):
        """使用LLM分析构效关系"""
        prompt = f"""你是材料科学构效关系分析专家。基于以下从文献中抽取的数据, 分析材料组成与性能之间的定量关系。

材料组成数据:
{json.dumps(compositions, ensure_ascii=False, indent=2)}

关键性能数据:
{json.dumps(properties, ensure_ascii=False, indent=2)}

跨文献融合结果:
{json.dumps([{"property": f["property"], "materials": f["materials"], "value_range": f["value_range"],
              "paper_count": f["paper_count"], "consistency": f["consistency"]}
             for f in fusion_results[:15]], ensure_ascii=False, indent=2)}

请分析并返回以下JSON格式:
{{
  "relationships": [
    {{
      "relationship": "关系描述(中文, 如'EGaIn中Ga含量增加, 电导率提高')",
      "component": "组成变量(如'Ga含量', 'In含量', 'Sn含量', '液态金属体积分数')",
      "property": "性能变量(如'electrical conductivity')",
      "trend": "positive|negative|nonlinear|unclear",
      "evidence": [{{"paper_id": "P00x", "data_point": "描述"}}],
      "mechanism": "物理机制解释(中文, 100字以内)",
      "confidence": "high|medium|low"
    }}
  ],
  "trends": [
    {{
      "trend_name": "趋势名称(中文)",
      "description": "趋势描述(中文, 100字以内)",
      "supporting_papers": ["P00x"],
      "implication": "对材料设计的启示(中文, 80字以内)"
    }}
  ],
  "composition_optimization": "基于已有数据的组成优化建议(中文, 200字以内)",
  "data_sufficiency": {{
    "adequate_for_analysis": true|false,
    "missing_data": ["缺失的数据类型"],
    "recommendation": "数据补充建议(中文)"
  }}
}}

要求:
1. 只基于提供的数据进行分析, 不要编造
2. 如果数据不足以做定量分析, 如实说明
3. 只输出数据直接支持的关系；不足3条时可以少于3条或为空
4. 只返回JSON, 不要其他文字"""

        result = self.llm.chat(prompt, system_prompt="你是材料科学构效关系分析专家。只返回合法JSON。",
                               temperature=0.3, max_tokens=3000)
        if not result:
            return None

        try:
            result = re.sub(r"```json\s*", "", result)
            result = re.sub(r"```\s*", "", result).strip()
            data = json.loads(result)

            return {
                "agent": "StructurePropertyDiscovery (Route A)",
                "compositions_found": len(compositions),
                "properties_analyzed": len(properties),
                "relationships": data.get("relationships", []),
                "trends": data.get("trends", []),
                "composition_optimization": data.get("composition_optimization", ""),
                "data_sufficiency": data.get("data_sufficiency", {}),
                "analysis_timestamp": datetime.now().isoformat(),
                "method": f"{self.llm.model} LLM analysis",
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"  构效关系LLM JSON解析失败: {e}")
            return None

    def _analyze_with_rules(self, compositions, properties, fusion_results):
        """规则模式构效关系分析"""
        relationships = []

        # EGaIn 电导率
        egain_cond = [p for p in properties if "egain" in p["material"].lower() and p["property"] == "electrical conductivity"]
        if egain_cond:
            relationships.append({
                "relationship": "EGaIn的电导率约为3.4×10⁶ S/m, 约为铜的17%",
                "component": "Ga-In eutectic (75.5% Ga / 24.5% In)",
                "property": "electrical conductivity",
                "trend": "positive",
                "evidence": [{"paper_id": p["paper_id"], "data_point": f"{p['value']} {p['unit']}"} for p in egain_cond],
                "mechanism": "Ga是主要导电组分, In的加入降低熔点但略微降低电导率",
                "confidence": "high",
            })

        # Galinstan 熔点
        galinstan_mp = [p for p in properties if "galinstan" in p["material"].lower() and p["property"] == "melting point"]
        if galinstan_mp:
            relationships.append({
                "relationship": "Galinstan中添加Sn使熔点降至-19°C, 显著低于EGaIn的15.7°C",
                "component": "Sn含量 (10 wt%)",
                "property": "melting point",
                "trend": "negative",
                "evidence": [{"paper_id": p["paper_id"], "data_point": f"{p['value']} {p['unit']}"} for p in galinstan_mp],
                "mechanism": "Sn的加入破坏了Ga晶格的周期性, 降低了熔化所需能量",
                "confidence": "medium",
            })

        # 氧化皮与表面张力
        oxide_props = [f for f in fusion_results if f["property"] == "surface tension"]
        if oxide_props:
            relationships.append({
                "relationship": "氧化皮使EGaIn表面张力从~500 mN/m(无氧)增至~624 mN/m(有氧)",
                "component": "oxide skin (Ga2O3)",
                "property": "surface tension",
                "trend": "positive",
                "evidence": [{"paper_id": e["paper_id"], "data_point": f"{e['value']} {e.get('unit', '')}"} for f in oxide_props for e in f["entries"][:2]],
                "mechanism": "氧化皮在液态金属表面形成固态壳层, 增加有效表面应力",
                "confidence": "high",
            })

        return {
            "agent": "StructurePropertyDiscovery (Route A, rule-based)",
            "compositions_found": len(compositions),
            "properties_analyzed": len(properties),
            "relationships": relationships,
            "trends": [],
            "composition_optimization": "建议系统研究Ga/In/Sn三元相图, 优化组成配比以平衡电导率、熔点和表面张力。",
            "data_sufficiency": {
                "adequate_for_analysis": len(properties) >= 10,
                "missing_data": ["不同组成比例的系统电导率数据", "温度依赖性数据"],
                "recommendation": "需要更多不同组成的实验数据点来建立定量构效关系模型",
            },
            "analysis_timestamp": datetime.now().isoformat(),
            "method": "rule-based analysis (LLM unavailable)",
        }


# ============================================================
# Agent 9: 报告生成Agent
# ============================================================

class ReportGenerationAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("ReportGenerator", llm_client)

    def run(self, scope, filter_result, knowledge_cards, fusion_results, gaps, verifications,
            route_a_result=None):
        self.log("生成结构化文献调研报告...")

        report = {
            "meta": {
                "title": "液态金属领域文献调研报告",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generated_by": f"多Agent文献调研系统 v5.3 ({self.llm.model} + Sciverse + ERCPD)",
                "pipeline_version": "8-Agent Pipeline (traceable evidence + Route A + Optimization v3.1)",
            },
            "scope": scope,
            "literature_summary": {
                "total_retrieved": len(filter_result["scored_papers"]),
                "total_selected": len(filter_result["selected_papers"]),
                "papers": [
                    {
                        "id": p["id"],
                        "title": p["title"],
                        "journal": p.get("journal", "Unknown"),
                        "year": p.get("year", 0),
                        "relevance_score": p["relevance_score"],
                        "citation_count": p.get("citation_count", 0),
                    }
                    for p in filter_result["selected_papers"]
                ],
            },
            "knowledge_statistics": {
                "total_cards": len(knowledge_cards),
                "total_properties": sum(len(c["properties"]) for c in knowledge_cards),
                "unique_materials": sorted(set(m for c in knowledge_cards for m in c.get("materials_identified", []))),
                "domain_coverage": sorted(set(tag for c in knowledge_cards for tag in c.get("domain_tags", []))),
            },
            "fusion_summary": {
                "total_property_categories": len(fusion_results),
                "categories_with_conflicts": sum(1 for f in fusion_results if f["conflicts"]),
                "categories_with_data_gaps": sum(1 for f in fusion_results if f["data_gap"]),
                "cross_referenced": sum(1 for f in fusion_results if f["paper_count"] > 1),
            },
            "research_gaps": gaps,
            "verification_summary": {
                "total_gaps": len(verifications),
                "verified": sum(1 for v in verifications if v["verification_status"] == "verified"),
                "with_notes": sum(1 for v in verifications if v["verification_status"] == "verified_with_notes"),
                "weak": sum(1 for v in verifications if v["verification_status"] == "weak"),
            },
            "route_a": route_a_result,
            "conclusion": self._generate_conclusion(
                scope, knowledge_cards, gaps, verifications, filter_result, route_a_result
            ),
        }

        self.log("报告生成完成")
        return report

    def _generate_conclusion(self, scope, cards, gaps, verifications, filter_result, route_a_result=None):
        total_props = sum(len(c["properties"]) for c in cards)
        high_severity = [g for g in gaps if g["severity"] == "high"]
        paper_count = len(filter_result["selected_papers"])

        verified_gap_count = sum(v.get("verification_status") == "verified" for v in verifications)

        if self.llm.mode == "api" and total_props > 0:
            route_a_summary = ""
            if route_a_result:
                route_a_summary = f"\n路线A构效关系分析发现 {len(route_a_result.get('relationships', []))} 条组成-性能关系。"

            prompt = f"""请用中文写一段文献调研结论 (300字以内)。

调研概况:
- 检索文献数: {len(filter_result['scored_papers'])}
- 入选文献数: {paper_count}
- 抽取属性数: {total_props}
- 识别Gap数: {len(gaps)} (其中 {len(high_severity)} 个高优先级)
- 满足本地原文可追溯条件的Gap数: {verified_gap_count}（不代表科学验证或新颖性）
{route_a_summary}

Gap列表:
{json.dumps([{"title": g["title"], "severity": g["severity"]} for g in gaps], ensure_ascii=False, indent=2)}

要求: 所有Gap仅为候选研究问题，原文可追溯不代表领域空白或新颖性已经验证。不得把抽取字段缺失称为整个研究领域缺失。不要使用标题或列表, 只写一段话。"""

            result = self.llm.chat(prompt, system_prompt="你是材料科学研究报告撰写专家。", temperature=0.4, max_tokens=500)
            if result:
                return result

        mode_text = "在线检索" if self.llm.mode == "api" else "离线演示语料"
        return (
            f"本次运行基于{mode_text}筛选 {paper_count} 篇文献，得到 {total_props} 条带原文短句的结构化属性。"
            f"系统提出 {len(gaps)} 个 Research Gap 候选，其中 {verified_gap_count} 个满足多来源、原文短句和定位信息的本地可追溯条件（非科学或新颖性验证）。"
            "未通过核验的候选仅用于提示后续检索方向，不应作为已证实结论；组成优化结果来自整理参考锚点上的代理模型，仍需实验验证。"
        )
