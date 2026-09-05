"""Evidence-bound numeric records shared by extraction, Route A and the surrogate.

Missing properties remain missing. Composition belongs to a material/sample,
never to an entire paper. Local quote verification is not primary-source audit.
"""
import hashlib
import math
import re
from collections import Counter, defaultdict

PROPERTIES = {
    'electrical conductivity': ('conductivity', 'S/m'),
    'melting point': ('melting_point', 'C'),
    'surface tension': ('surface_tension', 'mN/m'),
    'density': ('density', 'g/cm3'),
    'viscosity': ('viscosity', 'Pa s'),
    'thermal conductivity': ('thermal_conductivity', 'W/m/K'),
    'max strain': ('max_strain', '%'),
    'gauge factor': ('gauge_factor', '1'),
}

def canonical_property(name):
    name = re.sub(r'[_\s]+', ' ', str(name).lower()).strip()
    aliases = {
        'conductivity': 'electrical conductivity', 'conductivity (electrical)': 'electrical conductivity',
        'melting temperature': 'melting point', 'dynamic viscosity': 'viscosity',
        'gauge factor (gf)': 'gauge factor', 'maximum strain': 'max strain',
        'maximum stretchable strain': 'max strain', 'maximum tensile strain (stretchability)': 'max strain',
        'maximum stretching range': 'max strain', 'stretchability': 'max strain',
        'fracture strain (maximum)': 'max strain', 'tensile deformation': 'max strain',
    }
    return aliases.get(name, name)

def normalized_text(text):
    return re.sub(r'\s+', ' ', str(text)).strip().casefold()

def math_text(text):
    text = str(text).replace('−', '-').replace('⁻', '-').replace('³', '3').replace('²', '2')
    text = re.sub(r'\\(?:mathrm|text|operatorname)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:times|cdot)', '×', text)
    text = re.sub(r'\^\{([^{}]*)\}', r'^\1', text)
    text = re.sub(r'\\(?:circ|approx|sim)', ' ', text)
    return text.replace('\\%', '%').replace('$', '').replace('{', '').replace('}', '')

def number_in_quote(value, quote):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    pattern = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[×x*]\s*10\s*\^?\s*[-+]?\d+|[eE][-+]?\d+)?'
    for m in re.finditer(pattern, math_text(quote)):
        try:
            number = re.sub(r'\s+', '', m.group())
            number = re.sub(r'[×x*]10\^?', 'e', number)
            if math.isclose(float(number), value, rel_tol=1e-9, abs_tol=1e-12):
                return True
        except (ValueError, OverflowError):
            pass
    return False

def normalize_value(prop, value, unit):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    original_unit = re.sub(r'\s+', '', math_text(unit)).replace('·', '').replace('^', '')
    u = original_unit.lower().replace('°', '')
    u = re.sub(r'\s+', '', u)
    maps = {
        'electrical conductivity': {'s/m': 1, 'sm-1': 1, 's/cm': 100, 'scm-1': 100},
        'surface tension': {'mn/m': 1, 'mnm-1': 1, 'n/m': 1000, 'nm-1': 1000},
        'density': {'g/cm3': 1, 'gcm-3': 1, 'kg/m3': .001, 'kgm-3': .001},
        'viscosity': {'pas': 1, 'pa.s': 1, 'mpas': .001, 'mpa.s': .001, 'cp': .001},
        'thermal conductivity': {'w/m/k': 1, 'wm-1k-1': 1, 'w/(mk)': 1},
        'max strain': {'%': 1, 'percent': 1},
        'gauge factor': {'1': 1, '': 1, 'dimensionless': 1},
    }
    if prop == 'electrical conductivity' and original_unit in {'MS/m', 'mS/m', 'kS/m'}:
        result = value * {'MS/m': 1e6, 'mS/m': 1e-3, 'kS/m': 1e3}[original_unit]
    elif prop == 'melting point':
        result = value if u in {'c', 'celsius'} else value - 273.15 if u in {'k', 'kelvin'} else None
    else:
        factor = maps.get(prop, {}).get(u)
        result = value * factor if factor is not None else None
    if result is not None and (not math.isfinite(result) or (prop != 'melting point' and result < 0)):
        return None
    return result

def material_key(material):
    text = normalized_text(material)
    # Composite and device measurements never enter the bulk Ga/In/Sn model.
    if re.search(r'composite|embedded|elastomer|polymer|hydrogel|droplet|particle|microchannel|device|film|/cu|/|@|\+', text):
        return None
    if 'galinstan' in text:
        return 'galinstan'
    if text in {'ga in sn alloy', 'gain sn', 'gainsn', 'ga-in-sn alloy'}:
        return 'ga-in-sn'
    if 'egain' in text or 'eutectic gallium' in text or 'ga-in' in text:
        return 'egain'
    if text in {'ga', 'gallium', 'pure ga', 'pure gallium'}:
        return 'gallium'
    return None

def source_id(card):
    source = card.get('source', {})
    doi = card.get('doi') or source.get('doi')
    if doi:
        return 'doi:' + re.sub(r'^https?://(?:dx\.)?doi.org/', '', str(doi).strip().lower())
    return 'doc:' + str(source.get('doc_id') or hashlib.sha256(normalized_text(card.get('title', '')).encode()).hexdigest())

def verified_quote(card, evidence, value=None):
    text = card.get('source_text', '')
    quote = evidence.get('quote', '') if isinstance(evidence, dict) else ''
    digest = hashlib.sha256(text.encode()).hexdigest()
    if not quote or normalized_text(quote) not in normalized_text(text):
        return False
    if evidence.get('source_text_sha256') not in (None, '', digest):
        return False
    return value is None or number_in_quote(value, quote)

def composition_for(card, prop):
    """Require explicit wt% evidence; do not infer canonical alloy recipes."""
    composition = prop.get('composition')
    evidence = prop.get('composition_evidence', {})
    if composition is not None:
        if not isinstance(composition, dict) or set(composition) != {'ga', 'in', 'sn'}:
            return None
        vals = list(composition.values())
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in vals):
            return None
        if abs(sum(vals) - 100) > .15 or prop.get('composition_basis') != 'wt%':
            return None
        if not verified_quote(card, evidence) or not all(number_in_quote(v, evidence['quote']) for v in vals if v):
            return None
        basis_text = math_text(evidence['quote']).lower()
        if not re.search(r'wt\s*%|weight|mass|质量', basis_text):
            return None
        return dict(composition)
    key = material_key(prop.get('material', ''))
    found = {}
    fragments = []
    for p in card.get('properties', []):
        if not key or material_key(p.get('material', '')) != key:
            continue
        name = canonical_property(p.get('property_original', p.get('property', '')))
        if 'composition' not in name and 'content' not in name:
            continue
        if str(p.get('unit', '')).lower().replace(' ', '') not in {'wt%', 'weight%'}:
            continue
        if not verified_quote(card, p.get('evidence', {}), p.get('value')):
            continue
        if not re.search(r'wt\s*%|weight|mass|质量', math_text(p['evidence']['quote']).lower()):
            continue
        name += ' ' + normalized_text(p.get('conditions', ''))
        matches = [el for el, pat in [('ga', r'\bga\b|gallium'), ('in', r'\bin\b|indium'), ('sn', r'\bsn\b|tin')] if re.search(pat, name)]
        if len(matches) == 1:
            el = matches[0]
            if el in found and found[el] != p['value']:
                return None
            found[el] = p['value']
            fragments.append(p['evidence']['quote'])
    if key == 'egain' and set(found) == {'ga', 'in'}:
        found['sn'] = 0.0
    if set(found) == {'ga', 'in', 'sn'} and all(0 <= v <= 100 for v in found.values()) and abs(sum(found.values()) - 100) <= .15:
        # Separate fragments must belong to one saved paragraph, not unrelated samples.
        if any(all(normalized_text(q) in normalized_text(line) for q in fragments)
               for line in card.get('source_text', '').splitlines()):
            return found
    return None

def composition_evidence_for(card, prop, composition):
    if prop.get('composition_evidence', {}).get('quote'):
        return prop['composition_evidence']
    if composition:
        for line in card.get('source_text', '').splitlines():
            if (re.search(r'wt\s*%|weight|mass|质量', math_text(line).lower())
                and all(number_in_quote(v, line) for v in composition.values() if v)):
                return {'quote': line, 'source_text_sha256': hashlib.sha256(card['source_text'].encode()).hexdigest(),
                        'verification_scope': 'substring_in_saved_source_text_only'}
    return {}

def prepare_records(cards):
    accepted, rejected, anchors, seen = [], [], [], set()
    for card in cards:
        synthetic = card.get('source', {}).get('data_origin') in {'synthetic_test_fixture', 'historical_demo_fixture'}
        # Prefer an explicit composition binding over an older, less complete duplicate.
        indexed = sorted(enumerate(card.get('properties', [])),
                         key=lambda item: (bool(composition_for(card, item[1])),
                                           bool(item[1].get('composition_evidence', {}).get('quote')),
                                           bool(item[1].get('conditions'))), reverse=True)
        for i, prop in indexed:
            name = canonical_property(prop.get('property', ''))
            reason = None
            if synthetic: reason = 'demonstration_input'
            elif name not in PROPERTIES: reason = 'unsupported_or_nonmaterial_quantity'
            elif not verified_quote(card, prop.get('evidence', {}), prop.get('value')): reason = 'unverified_value_or_quote'
            value = normalize_value(name, prop.get('value'), prop.get('unit', '')) if reason is None else None
            if reason is None and value is None: reason = 'incompatible_unit_or_value'
            if reason:
                rejected.append({'paper_id': card.get('paper_id'), 'property_index': i, 'reason': reason})
                continue
            composition = composition_for(card, prop)
            rec = {'paper_id': card.get('paper_id'), 'source_id': source_id(card), 'material': str(prop.get('material', '')),
                   'property': name, 'value': value, 'unit': PROPERTIES[name][1], 'conditions': str(prop.get('conditions', '')),
                   'composition': composition, 'evidence': prop.get('evidence', {}),
                   'composition_evidence': composition_evidence_for(card, prop, composition), 'property_index': i}
            identity = (rec['source_id'], material_key(rec['material']) or normalized_text(rec['material']), name, value, rec['unit'])
            condition = normalized_text(rec['conditions'])
            comp_key = tuple(composition[k] for k in ('ga', 'in', 'sn')) if composition else None
            if any(old[:5] == identity and (not condition or not old[5] or old[5] == condition)
                   and (comp_key is None or old[6] is None or old[6] == comp_key) for old in seen):
                rejected.append({'paper_id': card.get('paper_id'), 'property_index': i, 'reason': 'duplicate_record'})
                continue
            seen.add(identity + (condition, comp_key))
            accepted.append(rec)
            attr = PROPERTIES[name][0]
            if attr not in {'conductivity', 'melting_point', 'surface_tension', 'density', 'viscosity'}:
                continue
            if not composition or not material_key(rec['material']):
                continue
            anchors.append({**composition, attr: value, 'paper_id': rec['paper_id'], 'ref_code': rec['source_id'],
                            'label': f"extracted:{rec['source_id']}:{i}", 'reference': card.get('title', ''),
                            'data_type': 'extracted', 'verification_status': 'local_text_verified',
                            'conditions': rec['conditions'], 'evidence': rec['evidence'],
                            'composition_evidence': rec['composition_evidence']})
    return {'records': accepted, 'anchors': anchors, 'rejections': rejected,
            'summary': {'input_cards': len(cards), 'raw_properties': sum(len(c.get('properties', [])) for c in cards),
                        'accepted_properties': len(accepted), 'extracted_property_anchors': len(anchors),
                        'rejected_by_reason': dict(Counter(r['reason'] for r in rejected))}}

def observed_relations(records):
    """Compare explicit compositions within a source and identical conditions.

    Two compositions support a descriptive contrast, never a regression/causal claim.
    Three or more give an OLS trend with sample count and fitted range.
    """
    groups = defaultdict(list)
    for r in records:
        if r['composition'] and material_key(r['material']):
            groups[(r['source_id'], r['property'], r['unit'], normalized_text(r['conditions']))].append(r)
    relations, trends = [], []
    for (_, prop, unit, condition), rows in sorted(groups.items()):
        if not condition:
            continue
        unique = {tuple(r['composition'][x] for x in ('ga', 'in', 'sn')) for r in rows}
        if len(unique) < 2:
            continue
        for component in ('ga', 'in', 'sn'):
            pairs = sorted((r['composition'][component], r['value']) for r in rows)
            if len({x for x, y in pairs}) < 2: continue
            xs, ys = zip(*pairs)
            xm, ym = sum(xs)/len(xs), sum(ys)/len(ys)
            slope = sum((x-xm)*(y-ym) for x,y in pairs)/sum((x-xm)**2 for x in xs)
            rel = {'component': component, 'property': prop, 'relationship': 'observed_within_source_association',
                   'description': f'{component} 与 {prop} 的同来源描述性关联', 'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat',
                   'slope': slope, 'slope_unit': f'{unit}/wt%', 'sample_count': len(rows), 'distinct_compositions': len(unique),
                   'conditions': condition, 'supporting_papers': sorted({r['paper_id'] for r in rows}),
                   'source_ids': sorted({r['source_id'] for r in rows}), 'evidence_indices': [r['property_index'] for r in rows],
                   'confidence': 'descriptive_only', 'scope': 'Components co-vary; not an isolated causal effect.'}
            relations.append(rel)
            if len(unique) >= 3:
                trends.append({**rel, 'range_wt_pct': [min(xs), max(xs)], 'method': 'OLS_descriptive'})
    return relations, trends


def exploratory_trends(records):
    """Composition-level descriptive cross-source associations with explicit confounding.

    They guide a later controlled test and never become training observations.
    Equal weight per composition prevents repeated review values dominating the fit.
    """
    groups = defaultdict(list)
    for r in records:
        if r.get('composition') and material_key(r.get('material')) and r['property'] in PROPERTIES:
            groups[(r['property'], r['unit'])].append(r)
    results = []
    for (prop, unit), rows in sorted(groups.items()):
        if len({r['source_id'] for r in rows}) < 3:
            continue
        by_composition = defaultdict(list)
        for r in rows:
            by_composition[tuple(r['composition'][k] for k in ('ga','in','sn'))].append(r)
        if len(by_composition) < 3:
            continue
        points = []
        for comp, items in sorted(by_composition.items()):
            source_values = defaultdict(list)
            for r in items: source_values[r['source_id']].append(r['value'])
            means = [sum(v)/len(v) for v in source_values.values()]
            points.append((comp, sum(means)/len(means)))
        for i, component in enumerate(('ga','in','sn')):
            pairs = [(comp[i], y) for comp,y in points]
            if len({x for x,y in pairs}) < 3: continue
            xm = sum(x for x,y in pairs)/len(pairs)
            ym = sum(y for x,y in pairs)/len(pairs)
            xx = sum((x-xm)**2 for x,y in pairs)
            yy = sum((y-ym)**2 for x,y in pairs)
            xy = sum((x-xm)*(y-ym) for x,y in pairs)
            results.append({'trend_name': f'{component} / {prop} 跨来源探索性关联',
                'component': component, 'property': prop, 'slope': xy/xx, 'slope_unit': f'{unit}/wt%',
                'r_squared': xy**2/(xx*yy) if yy else 0.0, 'distinct_compositions': len(points),
                'source_group_count': len({r['source_id'] for r in rows}),
                'source_ids': sorted({r['source_id'] for r in rows}),
                'observations': [{'composition':dict(zip(('ga','in','sn'),comp)), 'value':y} for comp,y in points],
                'claim_level': 'exploratory_cross_source_association',
                'description': '不同来源组成均值的描述性拟合，条件未统一且其他元素同步变化。',
                'implication': '用于定向补充证据，不作为因果规律、实测验证或新增训练标签。'})
    return results


def exploratory_comparisons(records):
    """Two-composition cross-source contrasts, not fitted trends or causal effects."""
    groups = defaultdict(list)
    for r in records:
        if r.get('composition') and material_key(r.get('material')):
            groups[(r['property'], r['unit'])].append(r)
    results = []
    for (prop, unit), rows in sorted(groups.items()):
        by_comp = defaultdict(list)
        for r in rows:
            by_comp[tuple(r['composition'][k] for k in ('ga', 'in', 'sn'))].append(r)
        if len(by_comp) < 2 or len({r['source_id'] for r in rows}) < 2:
            continue
        observations = []
        for comp, items in sorted(by_comp.items()):
            per_source = defaultdict(list)
            for r in items: per_source[r['source_id']].append(r['value'])
            means = [sum(v)/len(v) for v in per_source.values()]
            observations.append({'composition': dict(zip(('ga', 'in', 'sn'), comp)),
                                 'value': sum(means)/len(means),
                                 'source_ids': sorted(per_source),
                                 'conditions': sorted({r['conditions'] for r in items})})
        results.append({'property': prop, 'unit': unit, 'observations': observations,
                        'claim_level': 'cross_source_descriptive_comparison',
                        'description': '跨来源配比—物性对照；不控制测试条件，不估计单元素因果效应。'})
    return results
