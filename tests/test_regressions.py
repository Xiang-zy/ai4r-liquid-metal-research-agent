"""5.3.1 scientific-integrity, resilience, privacy and CLI regression tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from agents import LLMClient, KnowledgeExtractionAgent, KnowledgeFusionAgent, EvidenceVerificationAgent
from literature_data import cross_validate_against_reference_snapshot, get_anchor_list
from sciverse_client import SciverseClient
from optimizer import BayesianOptimizer, CompositionPropertySurrogate
from test_api_compatibility import FakeResponse


class ComparisonRegressionTests(unittest.TestCase):
    def compare(self, material="Galinstan", value=3.1e6, unit="S/m", prop="electrical conductivity"):
        return cross_validate_against_reference_snapshot([{"material": material, "value": value, "unit": unit, "property": prop}])

    def test_galinstan_is_not_egain(self):
        result = self.compare()[0]
        self.assertEqual(result["reference_material"], "Galinstan")
        self.assertEqual(result["status"], "match")

    def test_composites_and_ambiguous_alloys_are_not_bulk_references(self):
        for name in ("Galinstan-elastomer composite", "EGaIn oxide skin", "Ga-In", "Ga-In-Sn", "EGaIn microchannel sensor"):
            with self.subTest(name=name):
                self.assertEqual(self.compare(name), [])

    def test_nonfinite_boolean_missing_values_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, None):
            with self.subTest(value=value):
                self.assertEqual(self.compare(value=value), [])

    def test_kelvin_conversion_and_temperature_difference(self):
        result = self.compare(value=254.15, unit="K", prop="melting point")[0]
        self.assertEqual(result["normalized_value"], -19)
        self.assertEqual(result["absolute_difference"], 0)
        self.assertEqual(result["deviation_basis"], "absolute_temperature_K")

    def test_conductivity_prefixes(self):
        for value, unit in ((3.1, "MS/m"), (3100, "kS/m"), (3.1e9, "mS/m")):
            self.assertEqual(self.compare(value=value, unit=unit)[0]["status"], "match")

    def test_anchor_public_interface_never_claims_measured(self):
        self.assertEqual(len(get_anchor_list()), 25)
        self.assertTrue(all(a["data_type"] == "curated_unverified" for a in get_anchor_list()))


class ExtractionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.agent = KnowledgeExtractionAgent(LLMClient())
        self.paper = {"id": "P1", "title": "Test", "doi": "10.test/one", "abstract": ""}

    def extract(self, text):
        return self.agent._extract_fallback(dict(self.paper), text)

    def test_scientific_notation_forms(self):
        for token in ("3.4e6", "3.4 x 10^6", "3.4 × 10⁶".replace("⁶", "^6"), "3400000"):
            props = self.extract(f"EGaIn electrical conductivity {token} S/m.")["properties"]
            self.assertEqual(props[0]["value"], 3.4e6)

    def test_test_temperature_is_not_melting_point(self):
        self.assertEqual(self.extract("EGaIn tested at 25 °C for 3000 cycles.")["properties"], [])

    def test_contextual_melting_point_is_retained(self):
        props = self.extract("Galinstan melting point is -19 °C.")["properties"]
        self.assertEqual([(p["property"], p["value"]) for p in props], [("melting point", -19)])

    def test_composition_and_recovery_percent_are_not_strain(self):
        self.assertEqual(self.extract("EGaIn has 75.5% Ga, 24.5% In, and 95% recovery.")["properties"], [])

    def test_contextual_strain_is_retained(self):
        props = self.extract("EGaIn composite supports 300% strain.")["properties"]
        self.assertEqual(props[0]["property"], "max strain")
        self.assertIn("not bulk alloy", props[0]["material"])

    def test_material_order_deterministic_and_ambiguous_not_guessed(self):
        card = self.extract("EGaIn and Galinstan have conductivity 3e6 S/m.")
        self.assertEqual(card["materials_identified"], ["EGaIn", "Galinstan"])
        self.assertIn("ambiguous", card["properties"][0]["material"])

    def test_wrong_number_in_valid_quote_is_rejected(self):
        self.agent.llm.mode = "api"
        data = {"properties": [{"property": "electrical conductivity", "value": 999,
                 "unit": "S/m", "evidence_quote": "EGaIn conductivity 3e6 S/m."}]}
        with patch.object(self.agent.llm, "chat", return_value=json.dumps(data)):
            card = self.agent._extract_with_llm(dict(self.paper), data["properties"][0]["evidence_quote"])
        self.assertEqual(card["properties"], [])

    def test_llm_nonfinite_and_boolean_values_rejected(self):
        self.agent.llm.mode = "api"
        for value in (True, "NaN", "Infinity"):
            data = {"properties": [{"property": "density", "value": value, "evidence_quote": "density 6.3 g/cm3"}]}
            with patch.object(self.agent.llm, "chat", return_value=json.dumps(data)):
                self.assertEqual(self.agent._extract_with_llm(dict(self.paper), "density 6.3 g/cm3")["properties"], [])

    def test_source_snapshot_and_locator_are_saved(self):
        card = self.extract("Galinstan melting point is -19 °C.")
        e = card["properties"][0]["evidence"]
        self.assertTrue(e["quote_verified"])
        self.assertIn(e["quote"], card["source_text"])
        self.assertTrue(e["locator"].startswith("gathered_text:"))
        self.assertIsNone(e["page_no"])


class FusionRegressionTests(unittest.TestCase):
    def fuse(self, materials=("EGaIn", "EGaIn"), units=("S/m", "MS/m"), values=(3e6, 3), conditions=("25 °C", "25 °C")):
        cards = [{"paper_id": f"P{i}", "title": "T", "year": 2026, "properties": [{"material": m,
                  "property": "electrical conductivity", "value": v, "unit": u, "conditions": c}]}
                 for i, (m, u, v, c) in enumerate(zip(materials, units, values, conditions))]
        return KnowledgeFusionAgent(LLMClient()).run(cards)[0]

    def test_normalization_precedes_pooling(self):
        result = self.fuse()
        self.assertEqual(result["consistency"], "high")
        self.assertEqual(result["variation_coefficient"], 0)

    def test_different_materials_not_pooled(self):
        self.assertEqual(self.fuse(materials=("EGaIn", "Galinstan"))["consistency"], "not_comparable")

    def test_unknown_or_mismatched_conditions_not_pooled(self):
        for conditions in (("from text", "from text"), ("25 °C", "50 °C")):
            self.assertEqual(self.fuse(conditions=conditions)["consistency"], "not_comparable")

    def test_unknown_units_not_pooled_with_known(self):
        self.assertIsNone(self.fuse(units=("S/m", "unknown"))["variation_coefficient"])

    def test_cv_is_standard_deviation_not_range(self):
        result = self.fuse(units=("S/m", "S/m"), values=(1, 3))
        self.assertEqual(result["variation_coefficient"], 50)


class EvidenceRegressionTests(unittest.TestCase):
    def verify(self, same_doi=False, synthetic=False, forge=False):
        cards = []
        evidence = []
        extractor = KnowledgeExtractionAgent(LLMClient())
        for i in range(2):
            paper = {"id": f"P{i}", "title": "T", "doi": f"10.test/{0 if same_doi else i}",
                     "data_origin": "synthetic_test_fixture" if synthetic else "retrieved_unverified"}
            card = extractor._extract_fallback(paper, "EGaIn melting point 15.7 °C.")
            cards.append(card)
            item = dict(card["properties"][0]["evidence"])
            if forge:
                item["quote"] = "invented"
            evidence.append(item)
        return EvidenceVerificationAgent(LLMClient()).run([{"id": "G1", "title": "T", "evidence": evidence}], cards)[0]

    def test_two_traceable_sources_are_labeled_with_limited_scope(self):
        result = self.verify()
        self.assertEqual(result["verification_status"], "verified")
        self.assertIn("not_gap_novelty", result["verification_scope"])

    def test_duplicate_doi_not_two_sources(self):
        self.assertNotEqual(self.verify(same_doi=True)["verification_status"], "verified")

    def test_synthetic_input_never_scientifically_verified(self):
        self.assertEqual(self.verify(synthetic=True)["verification_status"], "weak")

    def test_forged_quote_boolean_does_not_establish_traceability(self):
        self.assertEqual(self.verify(forge=True)["traceable_evidence_count"], 0)


class ClientResilienceTests(unittest.TestCase):
    def client(self, cache):
        c = SciverseClient("test-key", max_retries=0, cache_dir=cache)
        c.min_interval_seconds = 0
        return c

    def test_cache_is_scoped_to_account_and_endpoint(self):
        with tempfile.TemporaryDirectory() as cache:
            a = self.client(cache)
            b = SciverseClient("different-key", cache_dir=cache)
            c = SciverseClient("test-key", base_url="https://example.invalid", cache_dir=cache)
            paths = {x._cache_path("GET", "/content", None) for x in (a, b, c)}
            self.assertEqual(len(paths), 3)

    def test_error_response_body_never_logged(self):
        with tempfile.TemporaryDirectory() as cache:
            client = self.client(cache)
            error = urllib.error.HTTPError("https://example.invalid", 401, "bad", {}, io.BytesIO(b"sensitive-response-body"))
            output = io.StringIO()
            with patch("urllib.request.urlopen", side_effect=error), contextlib.redirect_stdout(output):
                self.assertEqual(client.agentic_search("q"), [])
            self.assertNotIn("sensitive-response-body", output.getvalue())
            self.assertEqual(client.failed_call_count, 1)

    def test_malformed_response_does_not_crash(self):
        with tempfile.TemporaryDirectory() as cache:
            client = self.client(cache)
            with patch("urllib.request.urlopen", return_value=FakeResponse([])):
                self.assertEqual(client.agentic_search("q"), [])
            self.assertEqual(client.failed_call_count, 1)

    def test_nonadvancing_pagination_stops(self):
        with tempfile.TemporaryDirectory() as cache:
            client = self.client(cache)
            with patch.object(client, "get_content", return_value={"text": "x", "more": True, "next_offset": 0}) as mock:
                self.assertEqual(client.get_content_multi("doc", num_chunks=10), ["x"])
            self.assertEqual(mock.call_count, 1)

    def test_llm_truncated_or_malformed_responses_fail_closed(self):
        for payload in ({"choices": []}, {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}):
            with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "MINIMAX_MAX_RETRIES": "0"}, clear=True):
                client = LLMClient()
                with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
                    self.assertIsNone(client.chat("test"))
                self.assertEqual(client.failed_call_count, 1)


class GaussianProcessRegressionTests(unittest.TestCase):
    def test_predictive_mean_uses_observed_targets(self):
        model = BayesianOptimizer(CompositionPropertySurrogate())
        model.gp_X = [[60, 20, 20], [90, 5, 5]]
        model.gp_y = [0.1, 0.9]
        low, _ = model._gp_predict(model.gp_X[0])
        high, _ = model._gp_predict(model.gp_X[1])
        self.assertAlmostEqual(low, 0.1, places=3)
        self.assertAlmostEqual(high, 0.9, places=3)

    def test_scaling_targets_scales_mean_not_variance(self):
        model = BayesianOptimizer(CompositionPropertySurrogate())
        model.gp_X = [[60, 20, 20], [90, 5, 5]]
        model.gp_y = [0.1, 0.9]
        mean, variance = model._gp_predict([75, 12.5, 12.5])
        model.gp_y = [0.2, 1.8]
        scaled, second_variance = model._gp_predict([75, 12.5, 12.5])
        self.assertAlmostEqual(scaled, 2 * mean)
        self.assertAlmostEqual(variance, second_variance)


class EqualBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "scripts/equal_budget_benchmark.py"
        spec = importlib.util.spec_from_file_location("equal_budget", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.result = module.benchmark(seeds=(42,))

    def test_each_method_has_exactly_twenty_objective_calls(self):
        self.assertEqual(len(self.result["runs"]), 3)
        self.assertTrue(all(r["result"]["total_evaluations"] == 20 for r in self.result["runs"]))

    def test_all_methods_obey_common_composition_domain(self):
        for row in self.result["runs"]:
            for composition in row["result"]["explored_compositions"]:
                self.assertTrue(49.9 <= composition["ga"] <= 95.1)
                self.assertLessEqual(abs(sum(composition.values())-100), 0.11)


class CLIRegressionTests(unittest.TestCase):
    def cli(self, *args):
        root = Path(__file__).resolve().parents[1]
        env = {k: v for k, v in os.environ.items() if k not in {"MINIMAX_API_KEY", "SCIVERSE_API_KEY"}}
        return subprocess.run([sys.executable, str(root / "run.py"), *args], capture_output=True, text=True, env=env, timeout=20)

    def test_contradictory_modes_rejected(self):
        self.assertNotEqual(self.cli("--offline", "--strict").returncode, 0)

    def test_nonpositive_paper_count_rejected(self):
        self.assertNotEqual(self.cli("--offline", "--max-papers", "0").returncode, 0)

    def test_existing_result_directory_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "pipeline_results.json"
            marker.write_text("keep")
            self.assertNotEqual(self.cli("--offline", "--output-dir", directory).returncode, 0)
            self.assertEqual(marker.read_text(), "keep")

    def test_strict_missing_credentials_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertNotEqual(self.cli("--strict", "--output-dir", directory).returncode, 0)

    def test_strict_rejects_service_failure_even_when_extraction_succeeded(self):
        from run import Pipeline
        pipeline = Pipeline(offline=True)
        pipeline.offline, pipeline.strict = False, True
        pipeline.llm.failed_call_count = 1
        returns = {"planner": {}, "retriever": [{"id": "P1"}],
                   "filter": {"selected_papers": [{"id": "P1"}]},
                   "extractor": [{"properties": [], "extraction_mode": "llm"}],
                   "fusion": [], "gap": [], "verifier": [], "route_a": {}, "reporter": {}}
        with contextlib.ExitStack() as stack:
            for key, value in returns.items():
                stack.enter_context(patch.object(pipeline.agents[key], "run", return_value=value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, "1次服务失败"):
                pipeline.run()


class EndToEndRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="lm agent validation ")
        cls.output = Path(cls.directory.name)
        cls.query = '<img src=x onerror="alert(1)">'
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONHASHSEED="0")
        cls.process = subprocess.run([sys.executable, str(root / "run.py"), "--offline", "--max-papers", "5",
                                      "--query", cls.query, "--output-dir", str(cls.output)],
                                     capture_output=True, text=True, env=env, timeout=40)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_complete_cli_succeeds_and_writes_all_outputs(self):
        self.assertEqual(self.process.returncode, 0, self.process.stderr)
        for name in ("pipeline_results.json", "knowledge_cards.json", "research_gaps.json", "run_manifest.json",
                     "route_a_analysis.json", "ablation_study.json", "survey_report.html"):
            self.assertTrue((self.output / name).is_file(), name)

    def test_html_escapes_untrusted_input(self):
        html = (self.output / "survey_report.html").read_text()
        self.assertNotIn(self.query, html)
        self.assertIn("&lt;img", html)
        self.assertNotIn("已验证(有备注)", html)

    def test_manifest_records_zero_offline_calls_and_hashes(self):
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["application_version"], "5.4.0")
        self.assertEqual(manifest["stats"]["llm_calls"], 0)
        self.assertEqual(manifest["stats"]["sciverse_calls"], 0)
        self.assertEqual(len(manifest["source_files_sha256"]), 7)

    def test_gap_schema_and_computational_claim_guard(self):
        results = json.loads((self.output / "pipeline_results.json").read_text())
        self.assertTrue(all("description" in g for g in results["gaps"]))
        self.assertTrue(all(v["verification_status"] != "verified" for v in results["verifications"]))
        self.assertEqual(results["route_a"]["evidence_robust_discovery"]["claim_level"],
                         "computational_hypothesis_not_experimental_validation")


if __name__ == "__main__":
    unittest.main()
