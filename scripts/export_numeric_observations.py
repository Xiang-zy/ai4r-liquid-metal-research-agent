#!/usr/bin/env python3
"""Export a whitelist of derived facts, never source text, quotes or API responses.

Review the output before publication. Saved-text verification is not an audit of
the primary paper. Repeated review citations need not be independent experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from route_a_data import prepare_records


def export(cards):
    ingestion = prepare_records(cards)
    records = []
    for row in ingestion['records']:
        record = {k: row[k] for k in ('paper_id', 'source_id', 'material', 'property', 'value',
                                     'unit', 'conditions', 'composition', 'property_index')}
        record['provenance_hashes'] = {
            'source_text_sha256': row['evidence'].get('source_text_sha256', ''),
            'property_quote_sha256': hashlib.sha256(row['evidence']['quote'].encode()).hexdigest(),
            'composition_quote_sha256': hashlib.sha256(row['composition_evidence'].get('quote', '').encode()).hexdigest()
                if row['composition'] else None,
        }
        records.append(record)
    allowed = ('ga', 'in', 'sn', 'conductivity', 'melting_point', 'surface_tension', 'density',
               'viscosity', 'paper_id', 'ref_code', 'label', 'data_type', 'conditions')
    anchors = [{k: a[k] for k in allowed if k in a} for a in ingestion['anchors']]
    sources = [{k: c.get(k, '') for k in ('paper_id', 'title', 'doi', 'journal', 'year')}
               for c in cards if c['paper_id'] in {r['paper_id'] for r in records}]
    return {'schema_version': 1, 'dataset_version': '5.4.0',
            'scope': 'Derived numeric observations from a saved 50-card batch and five targeted refinements; no source text.',
            'verification_scope': 'Values and explicit mass compositions checked against saved text, not primary-source audited.',
            'summary': ingestion['summary'], 'sources': sources, 'records': records, 'anchors': anchors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Refusing to overwrite an existing numeric snapshot')
    dataset = export(json.loads(args.cards.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dataset['summary']))


if __name__ == '__main__': main()
