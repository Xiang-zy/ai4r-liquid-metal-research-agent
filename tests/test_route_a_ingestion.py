import copy
import hashlib
import unittest
import json
from pathlib import Path
from route_a_data import prepare_records, observed_relations, normalize_value, number_in_quote, exploratory_comparisons
from optimizer import CompositionPropertySurrogate, run_evidence_robust_discovery


def card(value=12.0, composition=None, prop='melting_point', unit='C', identity='doc-a', condition='ambient pressure'):
    composition = composition or {'ga': 75.0, 'in': 25.0, 'sn': 0.0}
    text = f"Alloy contains {composition['ga']} wt% Ga, {composition['in']} wt% In, {composition['sn']} wt% Sn. Its melting point is {value} C."
    evidence = {'quote': text, 'source_text_sha256': hashlib.sha256(text.encode()).hexdigest()}
    return {'paper_id': identity, 'title': identity, 'source': {'doc_id': identity, 'data_origin': 'retrieved_unverified'},
            'source_text': text, 'properties': [{'material': 'EGaIn', 'property': prop, 'value': value, 'unit': unit,
                'composition': composition, 'composition_basis': 'wt%', 'composition_evidence': evidence,
                'evidence': evidence, 'conditions': condition}]}


class IngestionTests(unittest.TestCase):
    def test_underscore_alias_and_explicit_composition(self):
        d = prepare_records([card()])
        self.assertEqual(d['records'][0]['property'], 'melting point')
        self.assertEqual(len(d['anchors']), 1)

    def test_partial_property_does_not_fabricate_other_properties(self):
        d = prepare_records([card()])
        self.assertNotIn('conductivity', d['anchors'][0])
        prior = CompositionPropertySurrogate()
        updated = CompositionPropertySurrogate([card()])
        self.assertNotEqual(prior.predict(75, 25, 0)['melting_point'], updated.predict(75, 25, 0)['melting_point'])
        self.assertEqual(prior.predict(75, 25, 0)['conductivity'], updated.predict(75, 25, 0)['conductivity'])

    def test_ercpd_uses_extracted_source(self):
        s = CompositionPropertySurrogate([card()])
        r = run_evidence_robust_discovery(s, resolution=10)
        self.assertEqual(r['extracted_property_anchors'], 1)
        self.assertIn('doc:doc-a', r['source_groups'])
        self.assertEqual(r['source_omission_counts']['doc:doc-a'], 1)

    def test_quote_hash_and_numeric_value_must_match(self):
        for field, value in [('value', 99), ('evidence', {'quote': 'fabricated'}), ('evidence', {'quote': card()['source_text'], 'source_text_sha256': 'bad'})]:
            c = card(); c['properties'][0][field] = value
            self.assertEqual(prepare_records([c])['records'], [])

    def test_composite_cannot_enter_bulk_model(self):
        c = card(); c['properties'][0]['material'] = 'EGaIn polymer composite'
        self.assertEqual(prepare_records([c])['anchors'], [])

    def test_no_name_based_recipe_inference(self):
        c = card(); c['properties'][0].pop('composition')
        self.assertEqual(prepare_records([c])['anchors'], [])

    def test_bad_composition_and_atomic_fraction_rejected(self):
        for key, value in [('composition_basis', 'at%'), ('composition', {'ga': 90, 'in': 90, 'sn': 0}), ('composition', {'ga': True, 'in': 99, 'sn': 0})]:
            c = card(); c['properties'][0][key] = value
            self.assertEqual(prepare_records([c])['anchors'], [])

    def test_same_doi_not_independent_records(self):
        a = card(); b = copy.deepcopy(a); b['paper_id'] = 'another'
        a['doi'] = 'https://doi.org/10.1000/ABC'; b['doi'] = '10.1000/abc'
        d = prepare_records([a,b])
        self.assertEqual(len(d['records']), 1)

    def test_units_respect_prefix_case(self):
        self.assertEqual(normalize_value('electrical conductivity', 3, 'MS/m'), 3e6)
        self.assertEqual(normalize_value('electrical conductivity', 3, 'mS/m'), .003)
        self.assertEqual(normalize_value('viscosity', 2.5, 'cP'), .0025)
        self.assertIsNone(normalize_value('electrical conductivity', 12, 'orders of magnitude relative to water'))

    def test_latex_scientific_notation(self):
        self.assertTrue(number_in_quote(3.4e6, r'$3.4\times10^{6}~\mathrm{S/m}$'))
        self.assertFalse(number_in_quote(True, '1'))

    def test_page_counts_and_synthetic_data_not_accepted(self):
        c = card(prop='page count'); self.assertFalse(prepare_records([c])['records'])
        c = card(); c['source']['data_origin'] = 'synthetic_test_fixture'
        self.assertFalse(prepare_records([c])['anchors'])

    def test_trends_require_distinct_compositions_and_matching_conditions(self):
        cards = [card(10+i*2, {'ga': 70+i*2, 'in': 30-i*2, 'sn': 0}) for i in range(3)]
        rows = prepare_records(cards)['records']
        relations, trends = observed_relations(rows)
        self.assertEqual(len(trends), 2)
        self.assertTrue(all(t['confidence'] == 'descriptive_only' for t in trends))
        for i,r in enumerate(rows): r['conditions'] = str(i)
        self.assertFalse(observed_relations(rows)[0])

    def test_unstated_mass_basis_is_not_guessed(self):
        c = card(); text = c['source_text'].replace('wt%', '%')
        c['source_text'] = text
        c['properties'][0]['evidence'] = {'quote': text}
        c['properties'][0]['composition_evidence'] = {'quote': text}
        self.assertFalse(prepare_records([c])['anchors'])

    def test_same_value_at_distinct_compositions_is_retained(self):
        a = card(); b = card(composition={'ga': 77, 'in': 23, 'sn': 0})
        self.assertEqual(len(prepare_records([a,b])['anchors']), 2)

    def test_more_complete_record_wins_over_old_duplicate(self):
        c = card(); old = copy.deepcopy(c['properties'][0]); old.pop('composition')
        old.pop('composition_evidence'); c['properties'].insert(0, old)
        d = prepare_records([c])
        self.assertEqual(len(d['records']), 1)
        self.assertEqual(len(d['anchors']), 1)

    def test_repeated_unknown_condition_not_double_counted(self):
        c = card(); duplicate = copy.deepcopy(c['properties'][0]); duplicate['conditions'] = ''
        c['properties'].append(duplicate)
        self.assertEqual(len(prepare_records([c])['anchors']), 1)

    def test_cross_source_comparison_is_not_observed_trend(self):
        rows = prepare_records([card(), card(14, {'ga':77, 'in':23, 'sn':0}, identity='doc-b')])['records']
        self.assertEqual(len(exploratory_comparisons(rows)), 1)
        self.assertEqual(observed_relations(rows), ([], []))

    def test_refinement_uses_real_line_spans_and_no_calls_in_tests(self):
        from scripts.replay_cards import refine
        c = card()
        class FakeLLM:
            model = 'unit-test'; total_tokens = 0
            def chat(self, *args, **kwargs):
                return json.dumps({'properties': [dict(c['properties'][0], evidence_span=0, composition_span=0)]})
        d = refine(c, FakeLLM())
        self.assertEqual(d['properties'][-1]['evidence']['quote'], c['source_text'])
        self.assertEqual(len(prepare_records([d])['anchors']), 1)

    def test_public_export_never_contains_private_text_or_quote_fields(self):
        from scripts.export_numeric_observations import export
        d = export([card()])
        def visit(value):
            if isinstance(value, dict):
                self.assertFalse({'source_text', 'quote', 'evidence', 'composition_evidence'} & value.keys())
                for v in value.values(): visit(v)
            elif isinstance(value, list):
                for v in value: visit(v)
        visit(d)
        self.assertNotIn(card()['source_text'], json.dumps(d))

    def test_published_numeric_replay_matches_signature(self):
        from scripts.reproduce_online import reproduce, signature
        root = Path(__file__).resolve().parents[1] / 'data'
        data = json.loads((root/'online_observations.json').read_text())
        baseline = json.loads((root/'submitted_baseline.json').read_text())
        expected = json.loads((root/'online_expected.json').read_text())
        actual = signature(reproduce(data, baseline))
        for k, v in actual.items(): self.assertEqual(v, expected[k], k)


if __name__ == '__main__': unittest.main()
