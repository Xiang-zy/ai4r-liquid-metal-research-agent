#!/usr/bin/env python3
"""Replay the published numeric snapshot without API keys or private text."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer import CompositionPropertySurrogate, run_evidence_robust_discovery
from route_a_data import observed_relations, exploratory_trends, exploratory_comparisons


def reproduce(dataset, reference):
    # This is an explicitly published numeric input, not fabricated knowledge cards
    # passed through the saved-quote verification gate.
    combined = reference['anchors'] + dataset['anchors']
    baseline = CompositionPropertySurrogate(anchors=reference['anchors'])
    updated = CompositionPropertySurrogate(anchors=combined)
    robust = run_evidence_robust_discovery(updated, **reference['parameters'])
    observed, trends = observed_relations(dataset['records'])
    predictions = []
    for comp in sorted({tuple(a[k] for k in ('ga', 'in', 'sn')) for a in dataset['anchors']}):
        predictions.append({'composition': dict(zip(('ga', 'in', 'sn'), comp)),
                            'baseline': baseline.predict(*comp), 'updated': updated.predict(*comp)})
    return {'observed_relationships': observed, 'observed_trends': trends,
            'exploratory_comparisons': exploratory_comparisons(dataset['records']),
            'exploratory_trends': exploratory_trends(dataset['records']),
            'evidence_robust_discovery': robust, 'prediction_changes': predictions,
            'scope': '5.4.0 numeric replay, distinct from the submitted fixed-anchor baseline.'}


def signature(result):
    robust = result['evidence_robust_discovery']
    return {'candidate': robust['best_risk_adjusted_candidate']['composition'],
            'pareto_front_size': robust['pareto_front_size'],
            'extracted_property_anchors': robust['extracted_property_anchors'],
            'source_omission_counts': robust['source_omission_counts'],
            'counterfactual_tests': robust['counterfactual_tests'],
            'prediction_changes': result['prediction_changes'],
            'exploratory_comparisons': result['exploratory_comparisons'],
            'observed_relationships': result['observed_relationships'],
            'observed_trends': result['observed_trends']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): parser.error('Choose an empty output directory')
    root = Path(__file__).resolve().parents[1]
    data_path = root / 'data/online_observations.json'
    reference_path = root / 'data/submitted_baseline.json'
    dataset = json.loads(data_path.read_text())
    result = reproduce(dataset, json.loads(reference_path.read_text()))
    expected = json.loads((root / 'data/online_expected.json').read_text())
    current = signature(result)
    checks = {k: v == expected[k] for k,v in current.items()}
    checks['numeric_input_sha256'] = hashlib.sha256(data_path.read_bytes()).hexdigest() == expected['numeric_input_sha256']
    checks['baseline_input_sha256'] = hashlib.sha256(reference_path.read_bytes()).hexdigest() == expected['baseline_input_sha256']
    verification = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (args.output/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__': raise SystemExit(main())
