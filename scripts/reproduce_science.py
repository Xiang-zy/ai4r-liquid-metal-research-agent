#!/usr/bin/env python3
"""Recompute the submitted numerical baseline without API keys or synthetic cards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer import CompositionPropertySurrogate, run_evidence_robust_discovery
from equal_budget_benchmark import benchmark

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): parser.error('Choose an empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root/'data/submitted_baseline.json').read_text())
    surrogate = CompositionPropertySurrogate(anchors=reference['anchors'])
    robust = run_evidence_robust_discovery(surrogate, **reference['parameters'])
    equal = benchmark(anchors=reference['anchors'])
    current = {'candidate': robust['best_risk_adjusted_candidate']['composition'],
               'pareto_front_size': robust['pareto_front_size'],
               'source_std': robust['best_risk_adjusted_candidate']['fitness_std'],
               'source_mean': robust['best_risk_adjusted_candidate']['fitness_mean'],
               'counterfactual_tests': robust['counterfactual_tests'],
               'benchmark_means': {k:v['mean'] for k,v in equal['summary'].items()}}
    expected = reference['expected']
    checks = {key: value == expected[key] for key,value in current.items()}
    for name, value in [('ercpd.json', robust), ('equal_budget.json', equal),
                        ('verification.json', {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
                         'input_sha256': hashlib.sha256((root/'data/submitted_baseline.json').read_bytes()).hexdigest(),
                         'scope': 'Submitted 5.3.1 fixed-anchor numerical baseline; not replay of private text/API responses.'})]:
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1

if __name__ == '__main__': raise SystemExit(main())
