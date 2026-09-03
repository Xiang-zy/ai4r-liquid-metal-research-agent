"""Synthetic test records, NOT papers, experimental data, or publisher excerpts.

This publication snapshot intentionally replaces the historical narrative fixtures.
Numbers are arbitrary test inputs; none is attributed to an actual publication.
The separately frozen competition package retains its original inputs and results.
"""

_SAMPLES = [
    ("EGaIn", 3.0e6, 16.0, 6.2),
    ("Galinstan", 2.8e6, -18.0, 6.3),
    ("EGaIn", 3.1e6, 17.0, 6.1),
    ("Galinstan", 2.9e6, -17.0, 6.4),
    ("EGaIn", 3.2e6, 18.0, 6.3),
]

PAPERS = [
    {
        "id": f"SYN{i:03d}",
        "title": f"SYNTHETIC TEST ONLY: {material} liquid metal fixture {i}",
        "authors": [],
        "journal": "Synthetic test fixture; not a publication",
        "year": 2026,
        "doi": "",
        "abstract": (
            "SYNTHETIC TEST ONLY. These arbitrary values exercise software parsing. "
            "No experiment was performed and no publication is cited. "
            f"Material: {material}. Electrical conductivity: {conductivity:g} S/m. "
            f"Melting point: {melting:g} °C. Density: {density:g} g/cm3. "
            "These values are not scientific evidence."
        ),
        "data_type": "synthetic_test_fixture",
    }
    for i, (material, conductivity, melting, density) in enumerate(_SAMPLES, 1)
]
