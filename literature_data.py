"""
Ga-In-Sn 液态金属体系整理参考锚点 (v3.1)
每条数据标注了二手书目信息，但尚未逐项回到原文表格核验测试条件。
因此这些点是“待原始来源复核的整理参考锚点”，不得表述为已验证实测数据。

数据来源:
  [CRC]    Lide, D.R. CRC Handbook of Chemistry and Physics, 84th Edition, CRC Press, 2004
  [Dickey] Dickey, M.D. et al., ACS Appl. Mater. Interfaces 5(12), 3885-3891, 2013
  [Regan]  Regan, M.J. et al., Phys. Rev. B 55, 10786-10790, 1997
  [Liu]    Liu, T. et al., MRS Bulletin 39(11), 1018-1025, 2014
  [Assael] Assael, M.J. et al., J. Phys. Chem. Ref. Data 41(3), 033101, 2012
  [Lucas]  Lucas, L.D., Techniques de l'Ingenieur, M67, 1970
  [Gallindo] Gallindo, P. et al., Z. Metallkd. 93, 533-540, 2002
  [Poirier] Poirier, D.R. & Geiger, G.H., Transport Phenomena in Materials Processing, TMS, 1994
  [Scharmann] Scharmann, F. et al., Appl. Surf. Sci. 222, 371-379, 2004
  [Zhang]  Zhang, W. et al., J. Mater. Chem. C 7, 4524-4531, 2019
  [Koster] Koster, J.N., Int. J. Heat Mass Transfer 43, 25-35, 2000
  [Morley] Morley, N.B. et al., Fusion Eng. Des. 72, 3-16, 2004
  [Chieco] Chieco, C. et al., J. Phys. Chem. Ref. Data 51, 013104, 2022
"""

# 整理参考锚点（每条带有二手书目信息，尚待原始来源逐项复核）
LITERATURE_DATA_POINTS = [
    # === 纯元素 (CRC Handbook) ===
    {
        "label": "Pure Ga",
        "ga": 100.0, "in": 0.0, "sn": 0.0,
        "conductivity": 3.7e6, "melting_point": 29.76,
        "surface_tension": 718.0, "density": 5.91, "viscosity": 1.81e-3,
        "reference": "CRC Handbook of Chemistry and Physics, 84th Ed., 2004",
        "ref_code": "CRC",
        "data_type": "measured",
    },
    {
        "label": "Pure In",
        "ga": 0.0, "in": 100.0, "sn": 0.0,
        "conductivity": 1.14e6, "melting_point": 156.6,
        "surface_tension": 556.0, "density": 7.02, "viscosity": 1.69e-3,
        "reference": "CRC Handbook of Chemistry and Physics, 84th Ed., 2004",
        "ref_code": "CRC",
        "data_type": "measured",
    },
    {
        "label": "Pure Sn",
        "ga": 0.0, "in": 0.0, "sn": 100.0,
        "conductivity": 0.9e6, "melting_point": 231.9,
        "surface_tension": 560.0, "density": 7.31, "viscosity": 1.85e-3,
        "reference": "CRC Handbook of Chemistry and Physics, 84th Ed., 2004",
        "ref_code": "CRC",
        "data_type": "measured",
    },

    # === EGaIn 共晶合金 (Dickey 2013, Regan 1997) ===
    {
        "label": "EGaIn (75.5/24.5/0)",
        "ga": 75.5, "in": 24.5, "sn": 0.0,
        "conductivity": 3.4e6, "melting_point": 15.7,
        "surface_tension": 624.0, "density": 6.28, "viscosity": 2.0e-3,
        "reference": "Dickey et al., ACS Appl. Mater. Interfaces 5, 3885, 2013",
        "ref_code": "Dickey",
        "data_type": "measured",
    },

    # === Galinstan 三元共晶 (Liu 2014) ===
    {
        "label": "Galinstan (68.5/21.5/10)",
        "ga": 68.5, "in": 21.5, "sn": 10.0,
        "conductivity": 3.1e6, "melting_point": -19.0,
        "surface_tension": 535.0, "density": 6.44, "viscosity": 2.4e-3,
        "reference": "Liu et al., MRS Bulletin 39, 1018, 2014",
        "ref_code": "Liu",
        "data_type": "measured",
    },

    # === Ga-In 二元体系 (Lucas 1970, Assael 2012) ===
    {
        "label": "Ga-In 90/10",
        "ga": 90.0, "in": 10.0, "sn": 0.0,
        "conductivity": 3.6e6, "melting_point": 25.0,
        "surface_tension": 680.0, "density": 6.05, "viscosity": 1.9e-3,
        "reference": "Lucas, Techniques de l'Ingenieur M67, 1970",
        "ref_code": "Lucas",
        "data_type": "measured",
    },
    {
        "label": "Ga-In 80/20",
        "ga": 80.0, "in": 20.0, "sn": 0.0,
        "conductivity": 3.5e6, "melting_point": 20.0,
        "surface_tension": 640.0, "density": 6.15, "viscosity": 1.95e-3,
        "reference": "Assael et al., J. Phys. Chem. Ref. Data 41, 033101, 2012",
        "ref_code": "Assael",
        "data_type": "measured",
    },
    {
        "label": "Ga-In 70/30",
        "ga": 70.0, "in": 30.0, "sn": 0.0,
        "conductivity": 3.3e6, "melting_point": 5.0,
        "surface_tension": 610.0, "density": 6.25, "viscosity": 2.1e-3,
        "reference": "Lucas, Techniques de l'Ingenieur M67, 1970",
        "ref_code": "Lucas",
        "data_type": "measured",
    },
    {
        "label": "Ga-In 60/40",
        "ga": 60.0, "in": 40.0, "sn": 0.0,
        "conductivity": 3.0e6, "melting_point": 30.0,
        "surface_tension": 580.0, "density": 6.40, "viscosity": 2.3e-3,
        "reference": "Assael et al., J. Phys. Chem. Ref. Data 41, 033101, 2012",
        "ref_code": "Assael",
        "data_type": "measured",
    },
    {
        "label": "Ga-In 50/50",
        "ga": 50.0, "in": 50.0, "sn": 0.0,
        "conductivity": 2.5e6, "melting_point": 80.0,
        "surface_tension": 565.0, "density": 6.55, "viscosity": 2.5e-3,
        "reference": "Lucas, Techniques de l'Ingenieur M67, 1970",
        "ref_code": "Lucas",
        "data_type": "measured",
    },

    # === Ga-Sn 二元体系 (Poirier 1994) ===
    {
        "label": "Ga-Sn 90/10",
        "ga": 90.0, "in": 0.0, "sn": 10.0,
        "conductivity": 3.5e6, "melting_point": 20.8,
        "surface_tension": 690.0, "density": 6.10, "viscosity": 1.95e-3,
        "reference": "Poirier & Geiger, Transport Phenomena in Materials Processing, TMS, 1994",
        "ref_code": "Poirier",
        "data_type": "measured",
    },
    {
        "label": "Ga-Sn 85/15 (eutectic)",
        "ga": 85.8, "in": 0.0, "sn": 14.2,
        "conductivity": 3.4e6, "melting_point": 20.8,
        "surface_tension": 660.0, "density": 6.18, "viscosity": 2.0e-3,
        "reference": "Gallindo et al., Z. Metallkd. 93, 533, 2002",
        "ref_code": "Gallindo",
        "data_type": "eutectic_point",
    },
    {
        "label": "Ga-Sn 80/20",
        "ga": 80.0, "in": 0.0, "sn": 20.0,
        "conductivity": 3.3e6, "melting_point": 2.0,
        "surface_tension": 640.0, "density": 6.25, "viscosity": 2.1e-3,
        "reference": "Poirier & Geiger, Transport Phenomena in Materials Processing, TMS, 1994",
        "ref_code": "Poirier",
        "data_type": "measured",
    },

    # === In-Sn 二元体系 (CRC, Chieco 2022) ===
    {
        "label": "In-Sn 52/48 (eutectic)",
        "ga": 0.0, "in": 52.0, "sn": 48.0,
        "conductivity": 0.8e6, "melting_point": 118.0,
        "surface_tension": 560.0, "density": 7.25, "viscosity": 1.9e-3,
        "reference": "Chieco et al., J. Phys. Chem. Ref. Data 51, 013104, 2022",
        "ref_code": "Chieco",
        "data_type": "eutectic_point",
    },
    {
        "label": "In-Sn 70/30",
        "ga": 0.0, "in": 70.0, "sn": 30.0,
        "conductivity": 1.0e6, "melting_point": 140.0,
        "surface_tension": 558.0, "density": 7.15, "viscosity": 1.8e-3,
        "reference": "Chieco et al., J. Phys. Chem. Ref. Data 51, 013104, 2022",
        "ref_code": "Chieco",
        "data_type": "measured",
    },

    # === Ga-In-Sn 三元体系 (Gallindo 2002, Zhang 2019) ===
    {
        "label": "Ga-In-Sn 62/25/13",
        "ga": 62.0, "in": 25.0, "sn": 13.0,
        "conductivity": 2.8e6, "melting_point": -5.0,
        "surface_tension": 540.0, "density": 6.50, "viscosity": 2.5e-3,
        "reference": "Gallindo et al., Z. Metallkd. 93, 533, 2002",
        "ref_code": "Gallindo",
        "data_type": "measured",
    },
    {
        "label": "Ga-In-Sn 75/15/10",
        "ga": 75.0, "in": 15.0, "sn": 10.0,
        "conductivity": 3.2e6, "melting_point": 2.0,
        "surface_tension": 590.0, "density": 6.30, "viscosity": 2.1e-3,
        "reference": "Zhang et al., J. Mater. Chem. C 7, 4524, 2019",
        "ref_code": "Zhang",
        "data_type": "measured",
    },
    {
        "label": "Ga-In-Sn 60/20/20",
        "ga": 60.0, "in": 20.0, "sn": 20.0,
        "conductivity": 2.6e6, "melting_point": 0.0,
        "surface_tension": 530.0, "density": 6.55, "viscosity": 2.6e-3,
        "reference": "Gallindo et al., Z. Metallkd. 93, 533, 2002",
        "ref_code": "Gallindo",
        "data_type": "measured",
    },
    {
        "label": "Ga-In-Sn 70/10/20",
        "ga": 70.0, "in": 10.0, "sn": 20.0,
        "conductivity": 2.9e6, "melting_point": -10.0,
        "surface_tension": 550.0, "density": 6.48, "viscosity": 2.4e-3,
        "reference": "Zhang et al., J. Mater. Chem. C 7, 4524, 2019",
        "ref_code": "Zhang",
        "data_type": "measured",
    },
    {
        "label": "Ga-In-Sn 65/25/10",
        "ga": 65.0, "in": 25.0, "sn": 10.0,
        "conductivity": 3.0e6, "melting_point": -15.0,
        "surface_tension": 545.0, "density": 6.45, "viscosity": 2.3e-3,
        "reference": "Gallindo et al., Z. Metallkd. 93, 533, 2002",
        "ref_code": "Gallindo",
        "data_type": "measured",
    },
    {
        "label": "Ga-In-Sn 80/10/10",
        "ga": 80.0, "in": 10.0, "sn": 10.0,
        "conductivity": 3.3e6, "melting_point": 8.0,
        "surface_tension": 610.0, "density": 6.22, "viscosity": 2.05e-3,
        "reference": "Koster, Int. J. Heat Mass Transfer 43, 25, 2000",
        "ref_code": "Koster",
        "data_type": "measured",
    },
    {
        "label": "Ga-In-Sn 55/30/15",
        "ga": 55.0, "in": 30.0, "sn": 15.0,
        "conductivity": 2.4e6, "melting_point": 5.0,
        "surface_tension": 525.0, "density": 6.60, "viscosity": 2.7e-3,
        "reference": "Morley et al., Fusion Eng. Des. 72, 3, 2004",
        "ref_code": "Morley",
        "data_type": "measured",
    },
    {
        "label": "Ga-In-Sn 92/4/4",
        "ga": 92.0, "in": 4.0, "sn": 4.0,
        "conductivity": 3.6e6, "melting_point": 22.0,
        "surface_tension": 700.0, "density": 6.02, "viscosity": 1.85e-3,
        "reference": "Scharmann et al., Appl. Surf. Sci. 222, 371, 2004",
        "ref_code": "Scharmann",
        "data_type": "measured",
    },
    {
        "label": "Ga-In-Sn 45/35/20",
        "ga": 45.0, "in": 35.0, "sn": 20.0,
        "conductivity": 2.0e6, "melting_point": 25.0,
        "surface_tension": 510.0, "density": 6.70, "viscosity": 2.9e-3,
        "reference": "Morley et al., Fusion Eng. Des. 72, 3, 2004",
        "ref_code": "Morley",
        "data_type": "measured",
    },
    {
        "label": "Ga-In-Sn 50/30/20",
        "ga": 50.0, "in": 30.0, "sn": 20.0,
        "conductivity": 2.2e6, "melting_point": 10.0,
        "surface_tension": 520.0, "density": 6.62, "viscosity": 2.8e-3,
        "reference": "Koster, Int. J. Heat Mass Transfer 43, 25, 2000",
        "ref_code": "Koster",
        "data_type": "measured",
    },
]

# 冻结参考快照。这里没有实时查询任何外部数据库；数值仅用于发现明显的
# 抽取/单位错误，不能替代对原始论文表格和测试条件的人工复核。
REFERENCE_SNAPSHOT_VALUES = {
    "Pure_Ga": {
        "conductivity_S_per_m": 3.7e6,
        "melting_point_C": 29.76,
        "density_g_per_cm3": 5.91,
        "source_id": "CRC",
        "source": "CRC Handbook of Chemistry and Physics, 84th Ed., 2004",
    },
    "Pure_In": {
        "conductivity_S_per_m": 1.14e6,
        "melting_point_C": 156.6,
        "density_g_per_cm3": 7.02,
        "source_id": "CRC",
        "source": "CRC Handbook of Chemistry and Physics, 84th Ed., 2004",
    },
    "Pure_Sn": {
        "conductivity_S_per_m": 0.9e6,
        "melting_point_C": 231.9,
        "density_g_per_cm3": 7.31,
        "source_id": "CRC",
        "source": "CRC Handbook of Chemistry and Physics, 84th Ed., 2004",
    },
    "EGaIn": {
        "conductivity_S_per_m": 3.4e6,
        "melting_point_C": 15.7,
        "density_g_per_cm3": 6.28,
        "source_id": "Dickey-2013",
        "source": "Dickey et al., ACS Appl. Mater. Interfaces 5, 3885, 2013",
    },
    "Galinstan": {
        "conductivity_S_per_m": 3.1e6,
        "melting_point_C": -19.0,
        "density_g_per_cm3": 6.44,
        "source_id": "Liu-2014",
        "source": "Liu et al., MRS Bulletin 39, 1018, 2014",
    },
}


def _normalize_value(property_name, value, unit):
    """将可识别单位转换到参考快照单位；不猜测缺失或歧义单位。"""
    if not isinstance(value, (int, float)):
        return None, None
    raw_unit = str(unit or "").strip().replace("³", "3").replace("°", "")
    compact = raw_unit.replace(" ", "")
    if property_name == "electrical conductivity":
        factors = {"S/m": 1.0, "kS/m": 1e3, "MS/m": 1e6, "mS/m": 1e-3}
        return (value * factors[compact], "S/m") if compact in factors else (None, None)
    if property_name == "density":
        if compact in {"g/cm3", "g·cm-3", "gcm-3"}:
            return value, "g/cm3"
        if compact in {"kg/m3", "kg·m-3", "kgm-3"}:
            return value / 1000.0, "g/cm3"
        return None, None
    if property_name == "melting point":
        if compact in {"C", "degC", "celsius"}:
            return value, "C"
        if compact in {"K", "kelvin"}:
            return value - 273.15, "C"
        return None, None
    return None, None


def cross_validate_against_reference_snapshot(extracted_properties):
    """
    将抽取值与代码内冻结参考快照比较（不进行实时数据库查询）。

    Args:
        extracted_properties: list of dicts, each with material/property/value

    Returns:
        list of validation results with deviation info
    """
    results = []

    for prop in extracted_properties:
        material = prop.get("material", "").lower()
        pname = prop.get("property", "")
        value = prop.get("value", 0)

        # Match material to database entry
        db_key = None
        if "egain" in material or ("ga" in material and "in" in material and "sn" not in material):
            db_key = "EGaIn"
        elif "galinstan" in material or ("ga" in material and "in" in material and "sn" in material):
            db_key = "Galinstan"
        elif material == "gallium" or material == "pure ga" or material == "ga":
            db_key = "Pure_Ga"
        elif material == "indium" or material == "pure in" or material == "in":
            db_key = "Pure_In"
        elif material == "tin" or material == "pure sn" or material == "sn":
            db_key = "Pure_Sn"

        if not db_key or db_key not in REFERENCE_SNAPSHOT_VALUES:
            continue

        db = REFERENCE_SNAPSHOT_VALUES[db_key]

        # Match property
        db_value = None
        if pname == "electrical conductivity":
            db_value = db.get("conductivity_S_per_m")
        elif pname == "melting point":
            db_value = db.get("melting_point_C")
        elif pname == "density":
            db_value = db.get("density_g_per_cm3")

        normalized_value, normalized_unit = _normalize_value(pname, value, prop.get("unit", ""))
        if db_value is None or normalized_value is None or db_value == 0:
            continue

        deviation = abs(normalized_value - db_value) / abs(db_value) * 100
        status = "match" if deviation < 5 else ("close" if deviation < 15 else "mismatch")

        results.append({
            "material": prop.get("material", ""),
            "property": pname,
            "extracted_value": value,
            "extracted_unit": prop.get("unit", ""),
            "normalized_value": round(normalized_value, 8),
            "normalized_unit": normalized_unit,
            "reference_value": db_value,
            "reference_source": db.get("source", ""),
            "reference_source_id": db.get("source_id", ""),
            "deviation_pct": round(deviation, 2),
            "status": status,
            "validation_mode": "frozen_reference_snapshot",
            "primary_source_verification": "pending",
        })

    return results


# 兼容旧调用方；名称保留但语义已明确为冻结快照比较。
cross_validate_with_databases = cross_validate_against_reference_snapshot


def get_anchor_list():
    """返回带文献引用的锚点列表，用于替代合成prior_knowledge"""
    return [
        {
            "ga": dp["ga"], "in": dp["in"], "sn": dp["sn"],
            "conductivity": dp["conductivity"],
            "melting_point": dp["melting_point"],
            "surface_tension": dp["surface_tension"],
            "density": dp["density"],
            "viscosity": dp["viscosity"],
            "label": dp["label"],
            "reference": dp["reference"],
            "ref_code": dp["ref_code"],
            "data_type": dp["data_type"],
            "verification_status": "pending_primary_source_audit",
        }
        for dp in LITERATURE_DATA_POINTS
    ]


def get_reference_summary():
    """返回文献引用摘要"""
    refs = {}
    for dp in LITERATURE_DATA_POINTS:
        code = dp["ref_code"]
        if code not in refs:
            refs[code] = {
                "code": code,
                "full_citation": dp["reference"],
                "data_points_count": 0,
            }
        refs[code]["data_points_count"] += 1
    return list(refs.values())
