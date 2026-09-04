"""
迭代优化模块 (v3.1 - 整理参考锚点驱动)
实现路线A中的 LLM 驱动搜索过程:
  1. 遗传算法 (GA) - 组成空间全局搜索
  2. 贝叶斯优化 (BO) - 高斯过程代理模型 + EI采集函数
  3. 适应度函数 - 多目标 (电导率最大化 + 熔点最小化 + 表面张力适中)
  4. 物理约束 - Ga-In-Sn三元组成空间, 液态金属相图约束

v3.0 改进:
  - 25个带书目信息的整理参考锚点 (literature_data.py)
  - 锚点明确标记为待原始来源逐项复核
  - 冻结参考快照一致性检查（非实时数据库查询）
  - 使用距离反比加权插值，不叠加不连续的经验修正
  - 移除初始种子注入 (优化器从纯随机起点搜索)
  - 5个评价指标 (样本效率/AUC/多样性/鲁棒性/覆盖率)
"""

import math
import random
import json
import time
from datetime import datetime
from literature_data import get_anchor_list, cross_validate_against_reference_snapshot, get_reference_summary


# ============================================================
# 物理代理模型: 基于文献数据的组成-性能映射
# ============================================================

class CompositionPropertySurrogate:
    """
    基于整理参考锚点的插值代理模型 (v3.1)
    使用待原始来源复核的参考锚点 + 距离反比加权插值。
    书目来源组尚未逐项核验，也未证明彼此独立；不得称为已验证实测数据库。
    """

    def __init__(self, knowledge_cards=None, include_extracted_anchors=False, anchors=None):
        self.data_points = []
        if knowledge_cards and include_extracted_anchors:
            self._build_from_cards(knowledge_cards)

        # v3.1: 使用待原始来源复核的整理参考锚点
        self.prior_knowledge = {}
        self._literature_anchors = list(anchors) if anchors is not None else get_anchor_list()

        # 转换为兼容格式
        for anchor in self._literature_anchors:
            self.prior_knowledge[anchor["label"]] = anchor

        # 预计算锚点列表
        self._anchors = list(self._anchors_with_refs())

        # 添加文献数据点
        for dp in self.data_points:
            if "ga" in dp:
                self._anchors.append({
                    "ga": dp["ga"], "in": dp["in"], "sn": dp["sn"],
                    "conductivity": dp.get("conductivity", 3.0e6),
                    "melting_point": dp.get("melting_point", 20.0),
                    "surface_tension": dp.get("surface_tension", 600.0),
                    "density": dp.get("density", 6.3),
                    "viscosity": dp.get("viscosity", 2.0e-3),
                    "label": f"LLM-extracted ({dp.get('paper_id', '?')})",
                    "reference": "LLM extracted from survey papers",
                    "ref_code": "LLM",
                    "data_type": "extracted",
                })

    def _anchors_with_refs(self):
        """返回带文献引用的锚点列表"""
        return [
            {
                "ga": a["ga"], "in": a["in"], "sn": a["sn"],
                "conductivity": a["conductivity"],
                "melting_point": a["melting_point"],
                "surface_tension": a["surface_tension"],
                "density": a["density"],
                "viscosity": a["viscosity"],
                "label": a.get("label", "?"),
                "reference": a.get("reference", "?"),
                "ref_code": a.get("ref_code", "?"),
                "data_type": a.get("data_type", "curated_unverified"),
                "verification_status": a.get("verification_status", "pending_primary_source_audit"),
            }
            for a in self._literature_anchors
        ]

    def _build_from_cards(self, cards):
        """从知识卡片提取组成-性能数据点"""
        for card in cards:
            point = {"paper_id": card["paper_id"], "source": "literature"}
            for prop in card.get("properties", []):
                pname = prop.get("property", "")
                val = prop.get("value", 0)
                if not isinstance(val, (int, float)):
                    continue
                if pname == "electrical conductivity":
                    point["conductivity"] = val
                elif pname == "melting point":
                    point["melting_point"] = val
                elif pname == "surface tension":
                    point["surface_tension"] = val
                elif pname == "density":
                    point["density"] = val
                elif pname == "viscosity":
                    point["viscosity"] = val

            materials = card.get("materials_identified", [])
            text = " ".join(m.lower() for m in materials) + " " + card.get("title", "").lower()
            if "egain" in text or "ga-in" in text:
                point.update({"ga": 75.5, "in": 24.5, "sn": 0.0, "alloy": "EGaIn"})
            elif "galinstan" in text:
                point.update({"ga": 68.5, "in": 21.5, "sn": 10.0, "alloy": "Galinstan"})
            elif "gallium" in text and "alloy" not in text:
                point.update({"ga": 100.0, "in": 0.0, "sn": 0.0, "alloy": "Pure Ga"})
            else:
                continue

            if "conductivity" in point or "melting_point" in point:
                self.data_points.append(point)

    def predict(self, ga, in_pct, sn_pct):
        """预测给定组成的性能 (距离反比加权 + 非线性物理修正)"""
        total = ga + in_pct + sn_pct
        if total > 0:
            ga, in_pct, sn_pct = ga / total * 100, in_pct / total * 100, sn_pct / total * 100

        # 距离反比加权插值
        weights = []
        for anchor in self._anchors:
            dist = math.sqrt(
                (ga - anchor["ga"]) ** 2 +
                (in_pct - anchor["in"]) ** 2 +
                (sn_pct - anchor["sn"]) ** 2
            )
            if dist < 0.1:
                return {
                    "conductivity": anchor["conductivity"],
                    "melting_point": anchor["melting_point"],
                    "surface_tension": anchor["surface_tension"],
                    "density": anchor["density"],
                    "viscosity": anchor["viscosity"],
                    "confidence": 1.0,
                }
            weights.append(1.0 / (dist ** 2))

        total_weight = sum(weights)
        result = {}
        for prop in ["conductivity", "melting_point", "surface_tension", "density", "viscosity"]:
            weighted_sum = sum(w * a[prop] for w, a in zip(weights, self._anchors))
            result[prop] = weighted_sum / total_weight

        # 置信度
        min_dist = min(math.sqrt(
            (ga - a["ga"]) ** 2 + (in_pct - a["in"]) ** 2 + (sn_pct - a["sn"]) ** 2
        ) for a in self._anchors)
        result["confidence"] = max(0.1, 1.0 - min_dist / 60.0)

        return result

    def fitness(self, ga, in_pct, sn_pct, weights=None):
        """多目标适应度函数"""
        if weights is None:
            weights = {"conductivity": 0.4, "melting_point": 0.3, "surface_tension": 0.2, "stability": 0.1}

        pred = self.predict(ga, in_pct, sn_pct)

        cond_score = min(pred["conductivity"] / 3.7e6, 1.0)
        mp_score = max(0, 1.0 - (pred["melting_point"] + 30) / 80.0)
        st = pred["surface_tension"]
        st_score = 1.0 - abs(st - 550) / 300.0
        st_score = max(0, min(1, st_score))
        stability = pred["confidence"]

        total = (
            weights["conductivity"] * cond_score +
            weights["melting_point"] * mp_score +
            weights["surface_tension"] * st_score +
            weights["stability"] * stability
        )

        return {
            "fitness": round(total, 4),
            "conductivity": round(pred["conductivity"], 0),
            "melting_point": round(pred["melting_point"], 1),
            "surface_tension": round(pred["surface_tension"], 1),
            "confidence": round(pred["confidence"], 3),
            "composition": {"ga": round(ga, 1), "in": round(in_pct, 1), "sn": round(sn_pct, 1)},
        }

    def grid_scan(self, resolution=5):
        """扫描代理模型搜索空间，用于计算网格参考最优值 (支持浮点步长)。"""
        best_fitness = -1
        best_result = None
        all_results = []

        # 生成浮点序列
        ga_vals = []
        g = 50.0
        while g <= 100.0 + 1e-9:
            ga_vals.append(round(g, 2))
            g += resolution

        for ga in ga_vals:
            in_vals = []
            i = 0.0
            while i <= (100.0 - ga) + 1e-9:
                in_vals.append(round(i, 2))
                i += resolution
            for in_pct in in_vals:
                sn_pct = 100.0 - ga - in_pct
                if sn_pct < -1e-9:
                    continue
                sn_pct = max(0.0, sn_pct)
                r = self.fitness(ga, in_pct, sn_pct)
                all_results.append(r)
                if r["fitness"] > best_fitness:
                    best_fitness = r["fitness"]
                    best_result = r

        # 同时扫描所有已知锚点 (确保非整数组成点不被遗漏)
        for anchor in self._anchors:
            r = self.fitness(anchor["ga"], anchor["in"], anchor["sn"])
            all_results.append(r)
            if r["fitness"] > best_fitness:
                best_fitness = r["fitness"]
                best_result = r

        return best_result, all_results


# ============================================================
# 遗传算法 (GA)
# ============================================================

class GeneticAlgorithm:
    def __init__(self, surrogate, pop_size=20, generations=15, mutation_rate=0.15, seed=42):
        self.surrogate = surrogate
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)
        self.history = []
        self.evaluation_count = 0
        self.explored_compositions = []

    def _random_individual(self):
        ga = self.rng.uniform(50, 95)
        remaining = 100 - ga
        in_pct = self.rng.uniform(0, remaining)
        sn_pct = remaining - in_pct
        return (ga, in_pct, sn_pct)

    def _initialize_population(self):
        """v2.0: 纯随机初始化 (不注入已知锚点)"""
        population = []
        while len(population) < self.pop_size:
            population.append(self._random_individual())
        return population

    def _evaluate(self, individual):
        self.evaluation_count += 1
        ga, in_pct, sn_pct = individual
        result = self.surrogate.fitness(ga, in_pct, sn_pct)
        self.explored_compositions.append(result["composition"])
        return result

    def _tournament_selection(self, population, fitnesses, k=3):
        selected = self.rng.sample(range(len(population)), min(k, len(population)))
        best = max(selected, key=lambda i: fitnesses[i]["fitness"])
        return population[best]

    def _blx_crossover(self, parent1, parent2, alpha=0.5):
        c1, c2 = [], []
        for p1, p2 in zip(parent1, parent2):
            lo = min(p1, p2) - alpha * abs(p1 - p2)
            hi = max(p1, p2) + alpha * abs(p1 - p2)
            c1.append(self.rng.uniform(lo, hi))
            c2.append(self.rng.uniform(lo, hi))
        return self._normalize(c1), self._normalize(c2)

    def _normalize(self, individual):
        individual = [max(0, x) for x in individual]
        total = sum(individual)
        if total > 0:
            individual = [x / total * 100 for x in individual]
        else:
            individual = [33.3, 33.3, 33.4]
        return tuple(individual)

    def _mutate(self, individual):
        mutated = list(individual)
        for i in range(len(mutated)):
            if self.rng.random() < self.mutation_rate:
                mutated[i] += self.rng.gauss(0, 5)
        return self._normalize(mutated)

    def run(self):
        start_time = time.time()
        population = self._initialize_population()
        fitnesses = [self._evaluate(ind) for ind in population]

        for gen in range(self.generations):
            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i]["fitness"])
            best = fitnesses[best_idx]
            avg_fitness = sum(f["fitness"] for f in fitnesses) / len(fitnesses)

            self.history.append({
                "generation": gen + 1,
                "best_fitness": best["fitness"],
                "avg_fitness": round(avg_fitness, 4),
                "best_composition": best["composition"],
                "best_conductivity": best["conductivity"],
                "best_melting_point": best["melting_point"],
                "evaluations": self.evaluation_count,
            })

            new_population = []
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i]["fitness"], reverse=True)[:2]
            for idx in elite_indices:
                new_population.append(population[idx])

            while len(new_population) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                c1, c2 = self._blx_crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < self.pop_size:
                    new_population.append(self._mutate(c2))

            population = new_population
            fitnesses = [self._evaluate(ind) for ind in population]

        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i]["fitness"])
        best_solution = fitnesses[best_idx]
        elapsed = time.time() - start_time

        return {
            "method": "Genetic Algorithm (GA)",
            "best_fitness": best_solution["fitness"],
            "best_composition": best_solution["composition"],
            "best_properties": {
                "conductivity": best_solution["conductivity"],
                "melting_point": best_solution["melting_point"],
                "surface_tension": best_solution["surface_tension"],
                "confidence": best_solution["confidence"],
            },
            "generations": self.generations,
            "population_size": self.pop_size,
            "total_evaluations": self.evaluation_count,
            "elapsed_time": round(elapsed, 3),
            "convergence_history": self.history,
            "explored_compositions": self.explored_compositions,
            "parameters": {
                "mutation_rate": self.mutation_rate,
                "crossover": "BLX-alpha",
                "selection": "tournament (k=3)",
                "elite_preservation": 2,
                "initialization": "pure random (v2.0)",
            },
        }


# ============================================================
# 贝叶斯优化 (BO)
# ============================================================

class BayesianOptimizer:
    def __init__(self, surrogate, n_iterations=15, n_initial=5, seed=42):
        self.surrogate = surrogate
        self.n_iterations = n_iterations
        self.n_initial = n_initial
        self.rng = random.Random(seed)
        self.history = []
        self.evaluation_count = 0
        self.gp_X = []
        self.gp_y = []
        self.explored_compositions = []

    def _sample_composition(self):
        ga = self.rng.uniform(50, 95)
        remaining = 100 - ga
        in_pct = self.rng.uniform(0, remaining)
        sn_pct = remaining - in_pct
        return (ga, in_pct, sn_pct)

    def _evaluate(self, x):
        self.evaluation_count += 1
        ga, in_pct, sn_pct = x
        result = self.surrogate.fitness(ga, in_pct, sn_pct)
        self.gp_X.append(list(x))
        self.gp_y.append(result["fitness"])
        self.explored_compositions.append(result["composition"])
        return result

    def _rbf_kernel(self, x1, x2, length_scale=20.0):
        dist_sq = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-dist_sq / (2 * length_scale ** 2))

    def _gp_predict(self, x):
        if not self.gp_X:
            return 0.5, 1.0
        n = len(self.gp_X)
        K = [[self._rbf_kernel(self.gp_X[i], self.gp_X[j]) for j in range(n)] for i in range(n)]
        noise = 1e-4
        for i in range(n):
            K[i][i] += noise
        K_s = [self._rbf_kernel(x, self.gp_X[i]) for i in range(n)]
        try:
            alpha = self._solve_linear(K, self.gp_y)
            mean = sum(K_s[i] * alpha[i] for i in range(n))
            K_ss = self._rbf_kernel(x, x) + noise
            v = self._solve_linear(K, K_s)
            variance = K_ss - sum(K_s[i] * v[i] for i in range(n))
            variance = max(1e-6, variance)
        except Exception:
            mean = sum(self.gp_y) / len(self.gp_y) if self.gp_y else 0.5
            variance = 1.0
        return mean, variance



    def _solve_linear(self, A, b):
        n = len(A)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for i in range(n):
            max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
            M[i], M[max_row] = M[max_row], M[i]
            if abs(M[i][i]) < 1e-10:
                M[i][i] = 1e-10
            for j in range(i + 1, n):
                factor = M[j][i] / M[i][i]
                for k in range(i, n + 1):
                    M[j][k] -= factor * M[i][k]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = M[i][n]
            for j in range(i + 1, n):
                x[i] -= M[i][j] * x[j]
            x[i] /= M[i][i]
        return x

    def _expected_improvement(self, x, best_y):
        mean, variance = self._gp_predict(x)
        std = math.sqrt(variance)
        if std < 1e-6:
            return 0.0
        improvement = mean - best_y
        z = improvement / std
        phi_z = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        cdf_z = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        ei = improvement * cdf_z + std * phi_z
        return max(0, ei)

    def _acquire_next(self, best_y, n_candidates=200):
        best_x = None
        best_ei = -1
        for _ in range(n_candidates):
            x = self._sample_composition()
            ei = self._expected_improvement(x, best_y)
            if ei > best_ei:
                best_ei = ei
                best_x = x
        return best_x, best_ei

    def run(self):
        start_time = time.time()

        # v2.0: 纯随机初始采样 (不注入已知锚点)
        initial_points = [self._sample_composition() for _ in range(self.n_initial)]

        for x in initial_points:
            result = self._evaluate(x)
            self.history.append({
                "iteration": len(self.history) + 1,
                "phase": "initial",
                "fitness": result["fitness"],
                "composition": result["composition"],
                "conductivity": result["conductivity"],
                "melting_point": result["melting_point"],
                "evaluations": self.evaluation_count,
                "acquisition_value": None,
            })

        for i in range(self.n_iterations):
            best_y = max(self.gp_y)
            next_x, ei = self._acquire_next(best_y)
            result = self._evaluate(next_x)
            self.history.append({
                "iteration": len(self.history) + 1,
                "phase": "BO",
                "fitness": result["fitness"],
                "composition": result["composition"],
                "conductivity": result["conductivity"],
                "melting_point": result["melting_point"],
                "evaluations": self.evaluation_count,
                "acquisition_value": round(ei, 6),
            })

        best_idx = max(range(len(self.gp_y)), key=lambda i: self.gp_y[i])
        best_composition = tuple(self.gp_X[best_idx])
        best_fitness = self.gp_y[best_idx]
        best_result = self.surrogate.fitness(*best_composition)
        elapsed = time.time() - start_time

        return {
            "method": "Bayesian Optimization (BO)",
            "best_fitness": round(best_fitness, 4),
            "best_composition": best_result["composition"],
            "best_properties": {
                "conductivity": best_result["conductivity"],
                "melting_point": best_result["melting_point"],
                "surface_tension": best_result["surface_tension"],
                "confidence": best_result["confidence"],
            },
            "iterations": self.n_iterations,
            "initial_samples": self.n_initial,
            "total_evaluations": self.evaluation_count,
            "elapsed_time": round(elapsed, 3),
            "convergence_history": self.history,
            "explored_compositions": self.explored_compositions,
            "parameters": {
                "surrogate_model": "Gaussian Process (RBF kernel)",
                "acquisition_function": "Expected Improvement (EI)",
                "candidate_pool_size": 200,
                "initialization": "pure random (v2.0)",
            },
        }


# ============================================================
# 随机搜索基线
# ============================================================

class RandomSearch:
    def __init__(self, surrogate, n_iterations=20, seed=42):
        self.surrogate = surrogate
        self.n_iterations = n_iterations
        self.rng = random.Random(seed)
        self.history = []
        self.evaluation_count = 0
        self.explored_compositions = []

    def run(self):
        start_time = time.time()
        best_fitness = -1
        best_result = None

        # v2.0: 纯随机 (不注入已知锚点)
        all_points = [self._sample() for _ in range(self.n_iterations)]

        for i, x in enumerate(all_points):
            self.evaluation_count += 1
            result = self.surrogate.fitness(*x)
            self.explored_compositions.append(result["composition"])
            if result["fitness"] > best_fitness:
                best_fitness = result["fitness"]
                best_result = result

            self.history.append({
                "iteration": i + 1,
                "fitness": result["fitness"],
                "composition": result["composition"],
                "conductivity": result["conductivity"],
                "melting_point": result["melting_point"],
                "evaluations": self.evaluation_count,
            })

        elapsed = time.time() - start_time
        return {
            "method": "Random Search (Baseline)",
            "best_fitness": best_result["fitness"],
            "best_composition": best_result["composition"],
            "best_properties": {
                "conductivity": best_result["conductivity"],
                "melting_point": best_result["melting_point"],
                "surface_tension": best_result["surface_tension"],
                "confidence": best_result["confidence"],
            },
            "iterations": self.n_iterations,
            "total_evaluations": self.evaluation_count,
            "elapsed_time": round(elapsed, 3),
            "convergence_history": self.history,
            "explored_compositions": self.explored_compositions,
            "parameters": {"strategy": "uniform random (v2.0)"},
        }

    def _sample(self):
        ga = self.rng.uniform(50, 95)
        remaining = 100 - ga
        in_pct = self.rng.uniform(0, remaining)
        sn_pct = remaining - in_pct
        return (ga, in_pct, sn_pct)


# ============================================================
# 证据感知的稳健 Pareto 发现
# ============================================================

def _mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _composition_grid(resolution):
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    points = []
    ga = 50.0
    while ga <= 100.0 + 1e-9:
        indium = 0.0
        while indium <= 100.0 - ga + 1e-9:
            points.append((round(ga, 6), round(indium, 6), round(100.0 - ga - indium, 6)))
            indium += resolution
        ga += resolution
    return points


def _dominates(left, right):
    """四目标 Pareto 支配：高电导、低熔点、表面张力接近550、稳健适应度高。"""
    left_values = (
        left["ensemble_properties"]["conductivity_mean"],
        -left["ensemble_properties"]["melting_point_mean"],
        -abs(left["ensemble_properties"]["surface_tension_mean"] - 550.0),
        left["risk_adjusted_fitness"],
    )
    right_values = (
        right["ensemble_properties"]["conductivity_mean"],
        -right["ensemble_properties"]["melting_point_mean"],
        -abs(right["ensemble_properties"]["surface_tension_mean"] - 550.0),
        right["risk_adjusted_fitness"],
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _counterfactual_composition(composition, sn_delta):
    new_sn = min(45.0, max(0.0, composition["sn"] + sn_delta))
    remaining = 100.0 - new_sn
    ga_in_total = composition["ga"] + composition["in"]
    if ga_in_total <= 0:
        return {"ga": remaining, "in": 0.0, "sn": new_sn}
    return {
        "ga": remaining * composition["ga"] / ga_in_total,
        "in": remaining * composition["in"] / ga_in_total,
        "sn": new_sn,
    }


def run_evidence_robust_discovery(surrogate, resolution=2.5, risk_penalty=5.0, top_k=5, sn_step=5.0):
    """
    以文献来源为分组执行 leave-one-source-out，寻找对单一来源不敏感的 Pareto 候选。

    这是文献证据稳健性分析，不是实验验证。输出中的关系均标记为可证伪候选假说。
    """
    anchors = list(surrogate._literature_anchors)
    sources = sorted({anchor.get("ref_code", "unknown") for anchor in anchors})
    if len(sources) < 3:
        raise ValueError("at least three independent source groups are required")

    omitted_models = []
    for source in sources:
        subset = [anchor for anchor in anchors if anchor.get("ref_code", "unknown") != source]
        omitted_models.append((source, CompositionPropertySurrogate(anchors=subset)))

    records = []
    for ga, indium, tin in _composition_grid(resolution):
        fitness_by_source = {}
        properties = {"conductivity": [], "melting_point": [], "surface_tension": []}
        for source, model in omitted_models:
            result = model.fitness(ga, indium, tin)
            fitness_by_source[source] = result["fitness"]
            for name in properties:
                properties[name].append(result[name])

        fitness_values = list(fitness_by_source.values())
        fitness_mean, fitness_std = _mean_std(fitness_values)
        ensemble_properties = {}
        for name, values in properties.items():
            mean, std = _mean_std(values)
            ensemble_properties[f"{name}_mean"] = round(mean, 4)
            ensemble_properties[f"{name}_std"] = round(std, 4)

        records.append({
            "composition": {"ga": ga, "in": indium, "sn": tin},
            "fitness_mean": round(fitness_mean, 6),
            "fitness_std": round(fitness_std, 6),
            "fitness_worst_case": round(min(fitness_values), 6),
            "risk_adjusted_fitness": round(fitness_mean - risk_penalty * fitness_std, 6),
            "ensemble_properties": ensemble_properties,
            "fitness_by_omitted_source": fitness_by_source,
        })

    best_by_source = {
        source: max(record["fitness_by_omitted_source"][source] for record in records)
        for source in sources
    }
    for record in records:
        retained = sum(
            record["fitness_by_omitted_source"][source] >= 0.95 * best_by_source[source]
            for source in sources
        )
        record["near_optimal_source_fraction"] = round(retained / len(sources), 4)

    pareto = [
        candidate for candidate in records
        if not any(_dominates(other, candidate) for other in records if other is not candidate)
    ]
    pareto.sort(
        key=lambda item: (item["risk_adjusted_fitness"], item["fitness_worst_case"]),
        reverse=True,
    )

    ranked = sorted(
        records,
        key=lambda item: (
            item["risk_adjusted_fitness"],
            item["near_optimal_source_fraction"],
            item["fitness_worst_case"],
        ),
        reverse=True,
    )
    robust_candidate = ranked[0]

    full_grid = []
    for ga, indium, tin in _composition_grid(resolution):
        full_grid.append(surrogate.fitness(ga, indium, tin))
    naive = max(full_grid, key=lambda item: item["fitness"])
    naive_record = min(
        records,
        key=lambda item: sum(
            abs(item["composition"][name] - naive["composition"][name])
            for name in ("ga", "in", "sn")
        ),
    )

    counterfactuals = []
    base_composition = robust_candidate["composition"]
    if sn_step <= 0:
        raise ValueError("sn_step must be positive")
    for delta in (-sn_step, sn_step):
        changed = _counterfactual_composition(base_composition, delta)
        mp_deltas = []
        conductivity_deltas = []
        for _, model in omitted_models:
            before = model.predict(base_composition["ga"], base_composition["in"], base_composition["sn"])
            after = model.predict(changed["ga"], changed["in"], changed["sn"])
            mp_deltas.append(after["melting_point"] - before["melting_point"])
            conductivity_deltas.append(after["conductivity"] - before["conductivity"])
        mp_mean, mp_std = _mean_std(mp_deltas)
        cond_mean, cond_std = _mean_std(conductivity_deltas)
        expected_mp_sign = -1 if mp_mean < 0 else 1
        expected_cond_sign = -1 if cond_mean < 0 else 1
        mp_consistency = sum((value < 0) == (expected_mp_sign < 0) for value in mp_deltas) / len(mp_deltas)
        cond_consistency = sum((value < 0) == (expected_cond_sign < 0) for value in conductivity_deltas) / len(conductivity_deltas)
        mp_signal_ratio = abs(mp_mean) / mp_std if mp_std > 0 else float("inf")
        cond_signal_ratio = abs(cond_mean) / cond_std if cond_std > 0 else float("inf")
        robust_effect = (
            min(mp_consistency, cond_consistency) >= 0.75
            and mp_signal_ratio >= 1.0
            and cond_signal_ratio >= 1.0
        )
        counterfactuals.append({
            "change": f"Sn {delta:+.1f} wt%, Ga/In按原比例守恒调整",
            "from_composition": base_composition,
            "to_composition": {name: round(value, 3) for name, value in changed.items()},
            "predicted_delta": {
                "melting_point_mean_C": round(mp_mean, 4),
                "melting_point_std_C": round(mp_std, 4),
                "conductivity_mean_S_per_m": round(cond_mean, 2),
                "conductivity_std_S_per_m": round(cond_std, 2),
            },
            "sign_consistency": {
                "melting_point": round(mp_consistency, 4),
                "conductivity": round(cond_consistency, 4),
            },
            "effect_to_spread_ratio": {
                "melting_point": round(mp_signal_ratio, 4),
                "conductivity": round(cond_signal_ratio, 4),
            },
            "hypothesis_status": "source_robust_candidate" if robust_effect else "source_sensitive_or_negligible",
            "falsification_rule": "若新增独立来源或后续实验给出的效应方向与预测相反，则否证该候选假说。",
        })

    worst_source = min(
        robust_candidate["fitness_by_omitted_source"],
        key=robust_candidate["fitness_by_omitted_source"].get,
    )
    return {
        "method": "Evidence-Robust Counterfactual Pareto Discovery (ERCPD)",
        "claim_level": "computational_hypothesis_not_experimental_validation",
        "anchor_status": "pending_primary_source_audit",
        "parameters": {
            "resolution_wt_pct": resolution,
            "risk_penalty": risk_penalty,
            "sn_counterfactual_step_wt_pct": sn_step,
            "source_groups": len(sources),
            "grid_candidates": len(records),
        },
        "source_groups": sources,
        "pareto_front_size": len(pareto),
        "robust_candidates": pareto[:top_k],
        "best_risk_adjusted_candidate": robust_candidate,
        "naive_full_surrogate_candidate": {
            "composition": naive["composition"],
            "fitness": naive["fitness"],
            "leave_one_source_out": naive_record,
        },
        "robustness_tradeoff_vs_naive": {
            "worst_case_fitness_delta": round(
                robust_candidate["fitness_worst_case"] - naive_record["fitness_worst_case"], 6
            ),
            "fitness_std_reduction": round(
                naive_record["fitness_std"] - robust_candidate["fitness_std"], 6
            ),
            "mean_fitness_delta": round(
                robust_candidate["fitness_mean"] - naive_record["fitness_mean"], 6
            ),
        },
        "most_influential_omitted_source_for_best_candidate": worst_source,
        "counterfactual_tests": counterfactuals,
        "interpretation_limits": [
            "来源留一法衡量对整理文献来源的敏感性，不等同于实验误差或真实预测区间。",
            "Pareto候选来自当前插值代理模型，只能作为下一轮文献检索或实验的可证伪假说。",
            "参考锚点尚待原始来源逐项复核，复核失败的锚点必须移除后重跑。",
        ],
    }


def run_ercpd_parameter_ablation(surrogate):
    """ERCPD 的风险惩罚与网格分辨率消融，只保留紧凑摘要。"""
    configurations = [
        ("no_risk_penalty", 2.5, 0.0),
        ("low_risk_penalty", 2.5, 1.0),
        ("default", 2.5, 5.0),
        ("high_risk_penalty", 2.5, 10.0),
        ("coarse_grid", 5.0, 5.0),
    ]
    rows = []
    for name, resolution, penalty in configurations:
        result = run_evidence_robust_discovery(
            surrogate,
            resolution=resolution,
            risk_penalty=penalty,
            top_k=1,
        )
        candidate = result["best_risk_adjusted_candidate"]
        rows.append({
            "configuration": name,
            "resolution_wt_pct": resolution,
            "risk_penalty": penalty,
            "composition": candidate["composition"],
            "fitness_mean": candidate["fitness_mean"],
            "fitness_std": candidate["fitness_std"],
            "fitness_worst_case": candidate["fitness_worst_case"],
            "risk_adjusted_fitness": candidate["risk_adjusted_fitness"],
            "near_optimal_source_fraction": candidate["near_optimal_source_fraction"],
            "pareto_front_size": result["pareto_front_size"],
        })
    return {
        "method": "ERCPD parameter ablation",
        "rows": rows,
        "interpretation": "比较风险偏好与网格分辨率对候选组成及来源波动的影响。",
    }


# ============================================================
# 评价指标计算
# ============================================================

def compute_evaluation_metrics(result, surrogate_grid_best_fitness):
    """计算5个额外评价指标"""
    history = result.get("convergence_history", [])
    explored = result.get("explored_compositions", [])

    # 1. 样本效率: 到达95%全局最优的评估次数
    threshold = 0.95 * surrogate_grid_best_fitness
    sample_efficiency = result["total_evaluations"]
    for h in history:
        fitness_val = h.get("best_fitness", h.get("fitness", 0))
        if fitness_val >= threshold:
            sample_efficiency = h.get("evaluations", result["total_evaluations"])
            break

    # 2. 收敛速度AUC: 归一化收敛曲线下面积
    if history:
        fitnesses = [h.get("best_fitness", h.get("fitness", 0)) for h in history]
        normalized = [f / surrogate_grid_best_fitness if surrogate_grid_best_fitness > 0 else 0 for f in fitnesses]
        auc = sum(normalized) / len(normalized)
    else:
        auc = 0.0

    # 3. 解多样性: 探索组成的标准差 (Ga维度)
    if explored:
        ga_values = [c.get("ga", 0) for c in explored]
        in_values = [c.get("in", 0) for c in explored]
        sn_values = [c.get("sn", 0) for c in explored]
        diversity = (math.sqrt(sum((x - sum(ga_values)/len(ga_values))**2 for x in ga_values) / len(ga_values)) +
                     math.sqrt(sum((x - sum(in_values)/len(in_values))**2 for x in in_values) / len(in_values)) +
                     math.sqrt(sum((x - sum(sn_values)/len(sn_values))**2 for x in sn_values) / len(sn_values))) / 3
    else:
        diversity = 0.0

    # 4. 探索覆盖率: 访问的不同网格单元数 / 总网格单元数
    grid_size = 10  # 10%网格
    visited_cells = set()
    for c in explored:
        ga_cell = min(10, max(5, int(c.get("ga", 0) / grid_size + 0.5)))
        in_cell = min(10 - ga_cell, max(0, int(c.get("in", 0) / grid_size + 0.5)))
        cell = (ga_cell, in_cell)
        visited_cells.add(cell)
    # Ga+In+Sn=100 的三元单纯形在10%节点下共有 6+5+...+1=21 个节点。
    total_cells = sum(11 - ga_cell for ga_cell in range(5, 11))
    coverage = len(visited_cells) / total_cells

    # 5. 最优解质量比: best_fitness / global_optimum_fitness
    optimality_gap = result["best_fitness"] / surrogate_grid_best_fitness if surrogate_grid_best_fitness > 0 else 0

    return {
        "sample_efficiency": sample_efficiency,
        "convergence_auc": round(auc, 4),
        "solution_diversity": round(diversity, 2),
        "exploration_coverage": round(coverage, 4),
        "optimality_gap": round(optimality_gap, 4),
    }


def run_multi_seed_robustness(surrogate, method_name, n_seeds=5):
    """多种子鲁棒性测试"""
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    seeds = [42, 123, 456, 789, 1024]
    rng = random.Random(2026)
    while len(seeds) < n_seeds:
        seeds.append(rng.randrange(1, 2**31))
    seeds = seeds[:n_seeds]
    results = []

    for seed in seeds:
        if method_name == "ga":
            opt = GeneticAlgorithm(surrogate, pop_size=20, generations=15, mutation_rate=0.15, seed=seed)
        elif method_name == "bo":
            opt = BayesianOptimizer(surrogate, n_iterations=15, n_initial=5, seed=seed)
        elif method_name == "random":
            opt = RandomSearch(surrogate, n_iterations=20, seed=seed)
        else:
            continue
        results.append(opt.run())

    fitnesses = [r["best_fitness"] for r in results]
    mean_fit = sum(fitnesses) / len(fitnesses)
    variance = sum((f - mean_fit) ** 2 for f in fitnesses) / len(fitnesses)
    std = math.sqrt(variance)

    return {
        "mean_fitness": round(mean_fit, 4),
        "std_fitness": round(std, 4),
        "cv": round(std / mean_fit * 100, 2) if mean_fit > 0 else 0,
        "min_fitness": round(min(fitnesses), 4),
        "max_fitness": round(max(fitnesses), 4),
        "all_fitnesses": [round(f, 4) for f in fitnesses],
    }


# ============================================================
# 消融实验框架 (v2.0)
# ============================================================

def run_ablation_study(surrogate, knowledge_cards=None):
    """
    v2.0 增强版消融实验
    新增: 22个锚点, 多峰景观, 5个评价指标, 多种子鲁棒性
    """
    print("\n" + "=" * 70)
    print("  消融实验 v3.1: 整理参考锚点驱动（待原始来源逐项复核）")
    print("=" * 70)

    # 同一代理模型上的网格参考值，不代表材料空间的真实全局最优。
    print("\n[0] 计算代理模型网格参考最优 (resolution=0.5)...")
    global_opt, all_grid = surrogate.grid_scan(resolution=0.5)
    global_opt_fitness = global_opt["fitness"]
    print(f"  代理模型网格最优: fitness={global_opt_fitness}, "
          f"Ga={global_opt['composition']['ga']}% In={global_opt['composition']['in']}% Sn={global_opt['composition']['sn']}%")
    # 找前5个局部最优
    sorted_grid = sorted(all_grid, key=lambda x: x["fitness"], reverse=True)
    top5 = []
    seen = set()
    for r in sorted_grid:
        key = (round(r["composition"]["ga"]), round(r["composition"]["in"]), round(r["composition"]["sn"]))
        if key not in seen:
            seen.add(key)
            top5.append(r)
        if len(top5) >= 5:
            break
    print(f"  Top-5 组成:")
    for i, r in enumerate(top5):
        print(f"    {i+1}. fitness={r['fitness']}, Ga={r['composition']['ga']}% In={r['composition']['in']}% Sn={r['composition']['sn']}%")

    results = {}

    # A) 基线: 无优化 (EGaIn)
    print("\n[A] 基线 (无优化, EGaIn)...")
    baseline_fitness = surrogate.fitness(75.5, 24.5, 0.0)
    results["baseline"] = {
        "method": "No Optimization (Baseline)",
        "best_fitness": baseline_fitness["fitness"],
        "best_composition": baseline_fitness["composition"],
        "best_properties": {
            "conductivity": baseline_fitness["conductivity"],
            "melting_point": baseline_fitness["melting_point"],
            "surface_tension": baseline_fitness["surface_tension"],
            "confidence": baseline_fitness["confidence"],
        },
        "iterations": 0,
        "total_evaluations": 0,
        "elapsed_time": 0.0,
        "convergence_history": [],
        "explored_compositions": [],
    }
    print(f"  适应度: {baseline_fitness['fitness']}")

    # B) 随机搜索
    print("\n[B] 随机搜索 (20次, 纯随机)...")
    rs = RandomSearch(surrogate, n_iterations=20, seed=42)
    results["random_search"] = rs.run()
    print(f"  适应度: {results['random_search']['best_fitness']}, 耗时: {results['random_search']['elapsed_time']}s")

    # C) 遗传算法
    print("\n[C] 遗传算法 (15代, 种群20, 纯随机初始化)...")
    ga = GeneticAlgorithm(surrogate, pop_size=20, generations=15, mutation_rate=0.15, seed=42)
    results["ga"] = ga.run()
    print(f"  适应度: {results['ga']['best_fitness']}, 耗时: {results['ga']['elapsed_time']}s")

    # D) 贝叶斯优化
    print("\n[D] 贝叶斯优化 (15次, 纯随机初始化)...")
    bo = BayesianOptimizer(surrogate, n_iterations=15, n_initial=5, seed=42)
    results["bo"] = bo.run()
    print(f"  适应度: {results['bo']['best_fitness']}, 耗时: {results['bo']['elapsed_time']}s")

    # E) GA + BO 混合
    print("\n[E] GA+BO 混合 (GA 10代 + BO 10次精修, 纯随机初始化)...")
    ga_hybrid = GeneticAlgorithm(surrogate, pop_size=15, generations=10, mutation_rate=0.15, seed=42)
    ga_result = ga_hybrid.run()
    bo_hybrid = BayesianOptimizer(surrogate, n_iterations=10, n_initial=5, seed=123)
    ga_best = ga_result["best_composition"]
    bo_hybrid.gp_X = [[ga_best["ga"], ga_best["in"], ga_best["sn"]]]
    bo_hybrid.gp_y = [ga_result["best_fitness"]]
    bo_hybrid.evaluation_count = 1
    bo_result = bo_hybrid.run()

    results["ga_bo_hybrid"] = {
        "method": "GA + BO Hybrid",
        "best_fitness": bo_result["best_fitness"],
        "best_composition": bo_result["best_composition"],
        "best_properties": bo_result["best_properties"],
        "iterations": ga_result["generations"] + bo_result["iterations"],
        "total_evaluations": ga_result["total_evaluations"] + bo_result["total_evaluations"],
        "elapsed_time": round(ga_result["elapsed_time"] + bo_result["elapsed_time"], 3),
        "convergence_history": ga_result["convergence_history"] + [
            {**h, "phase": "BO_refinement"} for h in bo_result["convergence_history"]
        ],
        "explored_compositions": ga_result.get("explored_compositions", []) + bo_result.get("explored_compositions", []),
        "ga_stage": {"best_fitness": ga_result["best_fitness"], "generations": ga_result["generations"]},
        "bo_stage": {"best_fitness": bo_result["best_fitness"], "iterations": bo_result["iterations"]},
    }
    print(f"  适应度: {results['ga_bo_hybrid']['best_fitness']}, 耗时: {results['ga_bo_hybrid']['elapsed_time']}s")

    # === 计算评价指标 ===
    print("\n" + "=" * 70)
    print("  评价指标分析 (5个维度)")
    print("=" * 70)

    for key, r in results.items():
        if key == "baseline":
            r["metrics"] = {
                "sample_efficiency": 0,
                "convergence_auc": 0.0,
                "solution_diversity": 0.0,
                "exploration_coverage": 0.0,
                "optimality_gap": round(r["best_fitness"] / global_opt_fitness, 4) if global_opt_fitness > 0 else 0,
            }
        else:
            r["metrics"] = compute_evaluation_metrics(r, global_opt_fitness)

    # === 多种子鲁棒性 ===
    print("\n[鲁棒性] 多种子测试 (5个种子)...")
    robustness = {}
    for method_key, method_name in [("random", "random_search"), ("ga", "ga"), ("bo", "bo")]:
        robustness[method_name] = run_multi_seed_robustness(surrogate, method_key, n_seeds=5)
        r = robustness[method_name]
        print(f"  {method_name}: mean={r['mean_fitness']} std={r['std_fitness']} CV={r['cv']}%")
    results["_robustness"] = robustness

    # === 汇总 ===
    print("\n" + "=" * 70)
    print("  消融实验结果汇总 (v3.1)")
    print("=" * 70)
    print(f"{'方法':<28} {'适应度':<8} {'评估':<6} {'耗时':<8} {'样本效率':<8} {'AUC':<6} {'多样性':<8} {'覆盖率':<6} {'最优比':<6}")
    print("-" * 100)
    for key, r in results.items():
        if key.startswith("_"):
            continue
        m = r.get("metrics", {})
        print(f"{r['method']:<28} {r['best_fitness']:<8} {r['total_evaluations']:<6} "
              f"{r['elapsed_time']:<8} {m.get('sample_efficiency', 0):<8} "
              f"{m.get('convergence_auc', 0):<6} {m.get('solution_diversity', 0):<8} "
              f"{m.get('exploration_coverage', 0):<6} {m.get('optimality_gap', 0):<6}")
    print("=" * 70)

    # 推荐实验方案
    recommendations = _generate_experiment_recommendations(results, surrogate)
    results["recommendations"] = recommendations

    # === 冻结参考快照一致性检查 ===
    print("\n[一致性检查] 与代码内冻结参考快照比较（无实时数据库查询）...")
    cv_results = _run_reference_snapshot_validation(knowledge_cards, surrogate)
    results["_cross_validation"] = cv_results

    # === 文献引用信息 ===
    results["_literature_references"] = get_reference_summary()
    results["_anchor_count"] = len(surrogate._anchors)
    results["_data_model"] = "curated_reference_anchors_pending_primary_source_audit (v3.1)"

    results["_surrogate_grid_optimum"] = {
        "interpretation": "best point on this surrogate grid; not a physical global optimum",
        "fitness": global_opt_fitness,
        "composition": global_opt["composition"],
        "top5_local_optima": [
            {"fitness": r["fitness"], "composition": r["composition"]} for r in top5
        ],
    }

    return results


def _generate_experiment_recommendations(ablation_results, surrogate):
    all_solutions = []
    for key, r in ablation_results.items():
        if key in ("recommendations", "_robustness", "_surrogate_grid_optimum"):
            continue
        comp = r.get("best_composition", {})
        if isinstance(comp, dict):
            all_solutions.append({
                "method": r["method"],
                "ga": comp.get("ga", 0),
                "in": comp.get("in", 0),
                "sn": comp.get("sn", 0),
                "fitness": r["best_fitness"],
                "properties": r.get("best_properties", {}),
                "metrics": r.get("metrics", {}),
            })

    all_solutions.sort(key=lambda x: x["fitness"], reverse=True)
    recommendations = []
    seen_comps = set()
    for sol in all_solutions:
        comp_key = (round(sol["ga"], 0), round(sol["in"], 0), round(sol["sn"], 0))
        if comp_key not in seen_comps:
            seen_comps.add(comp_key)
            recommendations.append({
                "rank": len(recommendations) + 1,
                "method": sol["method"],
                "composition": {"ga": sol["ga"], "in": sol["in"], "sn": sol["sn"]},
                "predicted_properties": sol["properties"],
                "fitness": sol["fitness"],
                "rationale": _explain_recommendation(sol),
            })
        if len(recommendations) >= 3:
            break
    return recommendations


def _explain_recommendation(sol):
    ga = sol["ga"]
    sn = sol["sn"]
    props = sol.get("properties", {})
    reasons = []
    if ga > 74:
        reasons.append("高Ga含量保证优异电导率和自然氧化皮形成")
    elif ga > 65:
        reasons.append("适中Ga含量平衡电导率与低熔点需求")
    if sn > 5:
        reasons.append(f"添加{sn:.1f}% Sn显著降低熔点, 改善低温性能")
    if "melting_point" in props and props["melting_point"] < 20:
        reasons.append(f"预测熔点{props['melting_point']:.1f}C, 满足室温液态要求")
    if "conductivity" in props and props["conductivity"] > 3e6:
        reasons.append(f"预测电导率{props['conductivity']:.0f} S/m, 导电性能优异")
    return "; ".join(reasons) if reasons else "综合适应度最优"


def _run_reference_snapshot_validation(knowledge_cards, surrogate):
    """
    与代码内冻结参考快照比较；不查询外部数据库，也不宣称完成来源核验。
    """
    # 从知识卡片中收集所有抽取的属性
    extracted_props = []
    if knowledge_cards:
        for card in knowledge_cards:
            materials = card.get("materials_identified", [])
            material_name = materials[0] if materials else "liquid metal"
            for prop in card.get("properties", []):
                extracted_props.append({
                    "material": material_name,
                    "property": prop.get("property", ""),
                    "value": prop.get("value", 0),
                    "unit": prop.get("unit", ""),
                    "paper_id": card.get("paper_id", "?"),
                })

    # 执行交叉验证
    cv_results = cross_validate_against_reference_snapshot(extracted_props)

    # 汇总统计
    total = len(cv_results)
    matches = sum(1 for r in cv_results if r["status"] == "match")
    close = sum(1 for r in cv_results if r["status"] == "close")
    mismatches = sum(1 for r in cv_results if r["status"] == "mismatch")

    if total > 0:
        avg_deviation = sum(r["deviation_pct"] for r in cv_results) / total
    else:
        avg_deviation = 0

    summary = {
        "total_properties_validated": total,
        "matches": matches,
        "close_matches": close,
        "mismatches": mismatches,
        "match_rate_pct": round(matches / total * 100, 1) if total > 0 else 0,
        "average_deviation_pct": round(avg_deviation, 2),
        "validation_mode": "frozen_reference_snapshot",
        "live_database_query": False,
        "reference_sources": sorted(set(r["reference_source_id"] for r in cv_results)),
        "limitations": "仅用于发现明显单位或抽取异常；所有锚点仍需回到原始来源逐项核验。",
        "validation_details": cv_results,
        "surrogate_anchors": len(surrogate._anchors),
        "anchor_sources": sorted(set(a.get("ref_code", "?") for a in surrogate._anchors)),
    }

    print(f"  验证属性数: {total}")
    print(f"  匹配（按属性容差）: {matches}")
    print(f"  接近（按属性容差）: {close}")
    print(f"  不匹配（按属性容差）: {mismatches}")
    print("  熔点: 绝对差<1°C/<5°C；其他属性: 偏差<5%/<15%，仅作示意检查")
    print(f"  平均偏差: {avg_deviation:.2f}%")
    print(f"  代理模型锚点数: {len(surrogate._anchors)} (按 {len(summary['anchor_sources'])} 组待核验书目分组，未证明独立性)")

    return summary
