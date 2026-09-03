import unittest

from unittest.mock import patch

from agents import (
    EvidenceVerificationAgent,
    KnowledgeExtractionAgent,
    LLMClient,
    StructurePropertyAgent,
)
from literature_data import cross_validate_against_reference_snapshot
from optimizer import (
    CompositionPropertySurrogate,
    compute_evaluation_metrics,
    run_evidence_robust_discovery,
    run_ercpd_parameter_ablation,
    run_multi_seed_robustness,
)


class ReferenceSnapshotTests(unittest.TestCase):
    def test_density_units_are_normalized_before_comparison(self):
        result = cross_validate_against_reference_snapshot([
            {"material": "EGaIn", "property": "density", "value": 6280, "unit": "kg/m3"}
        ])
        self.assertEqual(result[0]["normalized_value"], 6.28)
        self.assertEqual(result[0]["status"], "match")
        self.assertEqual(result[0]["validation_mode"], "frozen_reference_snapshot")

    def test_unknown_units_are_not_guessed(self):
        result = cross_validate_against_reference_snapshot([
            {"material": "EGaIn", "property": "density", "value": 6280, "unit": "unknown"}
        ])
        self.assertEqual(result, [])


class SurrogateMetricTests(unittest.TestCase):
    def test_evidence_robust_discovery_is_reproducible_and_conservative(self):
        first = run_evidence_robust_discovery(CompositionPropertySurrogate(), resolution=2.5)
        second = run_evidence_robust_discovery(CompositionPropertySurrogate(), resolution=2.5)
        self.assertEqual(first, second)
        self.assertEqual(first["parameters"]["source_groups"], 12)
        self.assertEqual(first["parameters"]["grid_candidates"], 231)
        self.assertGreater(first["pareto_front_size"], 0)
        self.assertGreater(first["robustness_tradeoff_vs_naive"]["fitness_std_reduction"], 0)
        self.assertEqual(first["claim_level"], "computational_hypothesis_not_experimental_validation")

    def test_counterfactuals_preserve_composition_and_flag_weak_effects(self):
        result = run_evidence_robust_discovery(CompositionPropertySurrogate(), resolution=2.5)
        tests = result["counterfactual_tests"]
        self.assertEqual(len(tests), 2)
        for item in tests:
            self.assertAlmostEqual(sum(item["to_composition"].values()), 100.0, places=3)
            self.assertIn(
                item["hypothesis_status"],
                {"source_robust_candidate", "source_sensitive_or_negligible"},
            )
        self.assertIn("source_sensitive", tests[0]["hypothesis_status"])
        self.assertEqual(tests[1]["hypothesis_status"], "source_robust_candidate")

    def test_ercpd_ablation_exposes_risk_performance_tradeoff(self):
        result = run_ercpd_parameter_ablation(CompositionPropertySurrogate())
        self.assertEqual(len(result["rows"]), 5)
        rows = {row["configuration"]: row for row in result["rows"]}
        self.assertGreater(
            rows["no_risk_penalty"]["fitness_std"],
            rows["default"]["fitness_std"],
        )
        self.assertGreater(
            rows["no_risk_penalty"]["fitness_mean"],
            rows["default"]["fitness_mean"],
        )

    def test_predictions_are_continuous_near_anchor(self):
        surrogate = CompositionPropertySurrogate()
        at_anchor = surrogate.predict(75.5, 24.5, 0)
        nearby = surrogate.predict(75.51, 24.49, 0)
        self.assertLess(abs(at_anchor["conductivity"] - nearby["conductivity"]), 1000)
        self.assertLess(abs(at_anchor["melting_point"] - nearby["melting_point"]), 0.1)

    def test_simplex_coverage_is_bounded(self):
        explored = []
        for ga in range(50, 101, 10):
            for indium in range(0, 101 - ga, 10):
                explored.append({"ga": ga, "in": indium, "sn": 100 - ga - indium})
        metrics = compute_evaluation_metrics({
            "convergence_history": [], "explored_compositions": explored,
            "best_fitness": 1.0, "total_evaluations": len(explored),
        }, 1.0)
        self.assertEqual(metrics["exploration_coverage"], 1.0)

    def test_requested_seed_count_is_used(self):
        result = run_multi_seed_robustness(CompositionPropertySurrogate(), "random", n_seeds=2)
        self.assertEqual(len(result["all_fitnesses"]), 2)


class EvidenceVerificationTests(unittest.TestCase):
    def test_cycles_are_not_misread_as_celsius(self):
        extractor = KnowledgeExtractionAgent(LLMClient())
        card = extractor._extract_fallback(
            {"id": "P1", "title": "EGaIn test", "abstract": "", "chunk": ""},
            "The device survived 3000 cycles and was tested at 25 °C.",
        )
        temperatures = [p["value"] for p in card["properties"] if p["property"] == "melting point"]
        self.assertEqual(temperatures, [25.0])

    def test_untraceable_evidence_is_not_verified(self):
        verifier = EvidenceVerificationAgent(LLMClient())
        result = verifier.run([{
            "id": "GAP-001", "title": "test", "evidence": [
                {"paper_id": "P1", "quote": "a", "quote_verified": False},
                {"paper_id": "P2", "quote": "b", "quote_verified": False},
            ],
        }], [])
        self.assertEqual(result[0]["verification_status"], "verified_with_notes")
        self.assertEqual(result[0]["traceable_evidence_count"], 0)


class LLMSearchIntegrationTests(unittest.TestCase):
    def test_llm_guidance_changes_search_parameters_with_validation(self):
        llm = LLMClient()
        llm.mode = "api"
        agent = StructurePropertyAgent(llm)
        properties = [{"paper_id": f"P{i}", "property": "melting point"} for i in range(5)]
        response = '{"risk_penalty": 8, "sn_counterfactual_step_wt_pct": 10, "focus": "conflicts"}'
        with patch.object(llm, "chat", return_value=response):
            guidance = agent._search_guidance(properties, [])
        self.assertEqual(guidance["risk_penalty"], 8.0)
        self.assertEqual(guidance["sn_counterfactual_step_wt_pct"], 10.0)
        self.assertEqual(guidance["method"], "MiniMax-M3_guided")

    def test_invalid_llm_guidance_falls_back_deterministically(self):
        llm = LLMClient()
        llm.mode = "api"
        agent = StructurePropertyAgent(llm)
        properties = [{"paper_id": f"P{i}"} for i in range(5)]
        with patch.object(llm, "chat", return_value="not json"):
            guidance = agent._search_guidance(properties, [])
        self.assertEqual(guidance["method"], "deterministic_default")
        self.assertEqual(guidance["risk_penalty"], 5.0)


if __name__ == "__main__":
    unittest.main()
