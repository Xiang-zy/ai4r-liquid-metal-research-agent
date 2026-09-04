"""Equal objective-call budget and common domain; no claim of statistical superiority."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer import CompositionPropertySurrogate, RandomSearch, GeneticAlgorithm, BayesianOptimizer


class BoundedGA(GeneticAlgorithm):
    def _normalize(self, individual):
        ga, indium, tin = super()._normalize(individual)
        ga = min(95.0, max(50.0, ga))
        total = indium + tin
        indium = (100 - ga) * indium / total if total else (100 - ga) / 2
        return ga, indium, 100 - ga - indium


def benchmark(seeds=(42, 123, 456, 789, 1024)):
    rows = []
    for seed in seeds:
        surrogate = CompositionPropertySurrogate()
        methods = {
            "random20": RandomSearch(surrogate, n_iterations=20, seed=seed),
            "ga20": BoundedGA(surrogate, pop_size=10, generations=1, seed=seed),
            "bo20": BayesianOptimizer(surrogate, n_initial=5, n_iterations=15, seed=seed),
        }
        for name, method in methods.items():
            result = method.run()
            assert result["total_evaluations"] == 20
            assert all(49.9 <= p["ga"] <= 95.1 and abs(sum(p.values()) - 100) <= 0.11
                       for p in result["explored_compositions"])
            rows.append({"method": name, "seed": seed, "result": result})
    summary = {}
    for method in ("random20", "ga20", "bo20"):
        values = [row["result"]["best_fitness"] for row in rows if row["method"] == method]
        summary[method] = {"mean": statistics.mean(values), "population_std": statistics.pstdev(values), "values": values}
    return {"budget": 20, "seeds": list(seeds), "domain": "50<=Ga<=95; In,Sn>=0; sum=100",
            "objective": "same unverified-anchor surrogate fitness", "summary": summary, "runs": rows,
            "limitations": "Objective-call budgets, not wall time, matched. GA uses a bounded variant and 10 initial+10 offspring evaluations (elites are recounted). BO uses 5 initial+15 adaptive evaluations. Seeds are fixed, not a significance test; no universal or experimental superiority claim."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    result = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
