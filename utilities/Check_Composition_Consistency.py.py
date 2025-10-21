# -*- coding: utf-8 -*-
"""
Normalize 'nominal composition' to molar fractions for HEAs, cross-check
against 'name' and 'measured composition', apply a pass-if-either rule,
and produce additional ML-friendly fields.

Usage:
  python normalize_nominal_compositions.py \
      --input /path/to/check_as_cast_bcc.csv \
      --output /path/to/check_as_cast_bcc_cleaned.csv \
      [--l1-thr 0.15] [--cos-thr 0.95] [--eps 1e-8]

Outputs (columns appended to the original CSV):
  - nominal_parsed                 : raw parse of nominal composition (JSON)
  - nominal_molar_fractions        : normalized to molar fractions (JSON; sum~1)
  - name_inferred_equal_molar      : equal-molar inference from 'name' (JSON)
  - measured_parsed                : raw parse of measured composition (JSON)
  - measured_molar_fractions       : normalized measured (JSON; sum~1)
  - l1_nom_vs_name, l1_nom_vs_measured
  - cos_nom_vs_name, cos_nom_vs_measured
  - consistency_check              : OK / inconsistent: name,measured / unknown
  - name_without_zero              : drop elements with ~0 nominal fraction
  - cleanned composition           : molar-ratio string like "A1.0B1.0C1.0"

Notes:
  - Detects inputs like 'Al0.25NbTiMoV', 'Nb25Mo25Ta25W25', 'Al:Nb:Ti=1:1:2',
    'Al 20 at.%', 'Al=20 wt%'. If total~100 without explicit unit, treats as at%.
  - If explicitly wt% (or ambiguous 100 with assume_weight_if_ambiguous=True),
    converts to mole via w_i / atomic_weight_i and renormalizes to fractions.
  - Pass rule: row is OK if nominal agrees with EITHER name OR measured within
    thresholds; only inconsistent if BOTH fail (when present).
"""

import re
import math
import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

# -------------------- Config --------------------
ATOMIC_WEIGHTS = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Y": 88.906,
    "Zr": 91.224, "Nb": 92.906, "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42,
    "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60,
    "La": 138.91, "Ce": 140.12, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Pb": 207.2
}

EL_RE = re.compile(r"([A-Z][a-z]?)(-?\d*\.?\d*(?:e[+-]?\d+)?)")

# -------------------- Parsing Helpers --------------------
def clean_text(s: str) -> str:
    """ Clean non-standard characters"""
    s = s.replace(",", " ").replace(";", " ").replace("|", " ")
    s = s.replace("＝", "=").replace("–", "-").replace("—", "-")
    s = s.replace("：", ":")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def try_parse_colon_equal_form(s: str):
    """ 'Al:Nb:Ti=1:1:2' or 'Al Nb Ti = 1 1 2' """
    if "=" not in s:
        return None
    left, right = s.split("=", 1)
    left = clean_text(left)
    right = clean_text(right)
    left_elems = re.findall(r"[A-Z][a-z]?", left)
    if not left_elems:
        return None
    right_nums = re.findall(r"-?\d*\.?\d+(?:e[+-]?\d+)?", right)
    if len(right_nums) != len(left_elems):
        return None
    vals = list(map(float, right_nums))
    return dict(zip(left_elems, vals))

def parse_pairs_with_units(s: str):
    """
    'Al 20 at.%', 'Al=20 wt%', 'Al 20, Nb 30 at%'
    Returns (dict, unit_hint) where unit_hint: {'at','wt',None}
    """
    unit_hint = None
    pairs = {}
    s2 = s.replace("at.%", "at%").replace("at. %", "at%").replace("wt.%", "wt%").replace("wt. %", "wt%")
    tokens = re.split(r"[,\s]+", s2)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[A-Z][a-z]?", tok):
            el = tok
            val = None
            u = None
            if i+1 < len(tokens):
                nxt = tokens[i+1]
                m = re.fullmatch(r"(-?\d*\.?\d+(?:e[+-]?\d+)?)([a-z%]*)", nxt, re.I)
                if m:
                    val = float(m.group(1))
                    u = m.group(2).lower() if m.group(2) else None
                    i += 2
                else:
                    if nxt == "=" and i+2 < len(tokens):
                        m2 = re.fullmatch(r"(-?\d*\.?\d+(?:e[+-]?\d+)?)([a-z%]*)", tokens[i+2], re.I)
                        if m2:
                            val = float(m2.group(1))
                            u = m2.group(2).lower() if m2.group(2) else None
                            i += 3
                        else:
                            i += 1
                    else:
                        i += 1
            else:
                i += 1
            if val is not None:
                pairs[el] = pairs.get(el, 0.0) + val
                if u:
                    if "wt" in u:
                        unit_hint = unit_hint or "wt"
                    if "at" in u:
                        unit_hint = unit_hint or "at"
        else:
            i += 1
    return (pairs if pairs else None, unit_hint)

def parse_concatenated_formula(s: str):
    """
    'Al0.25NbTiMoV', 'Nb25Mo25Ta25W25', 'FeCoCrNi'
    """
    res = defaultdict(float)
    s2 = s.replace("%", " ")
    for el, num in EL_RE.findall(s2):
        if num.strip() in ("", "+", "-"):
            val = 1.0
        else:
            try:
                val = float(num)
            except Exception:
                val = 1.0
        res[el] += val
    return dict(res) if res else None

def parse_any(s: str):
    """Try multiple parsers. Returns (dict, {'unit_hint': ...})."""
    if not isinstance(s, str) or not s.strip():
        return None, {'unit_hint': None}
    s_clean = clean_text(s)
    d = try_parse_colon_equal_form(s_clean)
    if d:
        return d, {'unit_hint': None}
    d2, unit_hint = parse_pairs_with_units(s_clean)
    if d2:
        return d2, {'unit_hint': unit_hint}
    d3 = parse_concatenated_formula(s_clean)
    if d3:
        total = sum(d3.values())
        # If totals ~100 with no explicit unit, interpret as atomic percent
        if abs(total - 100.0) < 1e-3:
            return d3, {'unit_hint': 'at'}
        return d3, {'unit_hint': None}
    return None, {'unit_hint': None}

# -------------------- Normalization --------------------
def normalize_to_molar_fractions(d, unit_hint=None, assume_weight_if_ambiguous=False):
    """
    Convert dict of element->value (ratio, fraction, at%, or wt%) to molar fractions summing to 1.
    """
    if not d:
        return None
    total = sum(d.values())
    is_fraction = math.isfinite(total) and (abs(total - 1.0) < 1e-3) # Change this for different sensitivity
    is_percent = math.isfinite(total) and (abs(total - 100.0) < 1e-1) # Change this for different sensitivity

    if unit_hint == "wt" or (assume_weight_if_ambiguous and is_percent and unit_hint is None):
        # wt% -> moles via w_i / M_i
        moles = {}
        for el, w in d.items():
            aw = ATOMIC_WEIGHTS.get(el)
            moles[el] = (w / aw) if aw else w  # fallback: treat as ratio if missing mass
        tot_m = sum(moles.values())
        if tot_m <= 0:
            return None
        return {el: val / tot_m for el, val in moles.items()}

    # at% or generic % -> fractions
    if is_percent or unit_hint == "at":
        if total == 0:
            return None
        return {el: val / total for el, val in d.items()}

    # Already fractions
    if is_fraction and total > 0:
        return {el: val / total for el, val in d.items()}

    # Generic ratios -> normalize
    if total > 0:
        return {el: val / total for el, val in d.items()}

    return None

# -------------------- Comparisons --------------------
def align_vectors(keys, d1, d2):
    vec1, vec2 = [], []
    for k in keys:
        vec1.append(d1.get(k, 0.0))
        vec2.append(d2.get(k, 0.0))
    return vec1, vec2

def l1_distance(d1, d2):
    keys = sorted(set(d1.keys()) | set(d2.keys()))
    v1, v2 = align_vectors(keys, d1, d2)
    return sum(abs(a - b) for a, b in zip(v1, v2))

def cosine_similarity(d1, d2):
    keys = sorted(set(d1.keys()) | set(d2.keys()))
    v1, v2 = align_vectors(keys, d1, d2)
    dot = sum(a*b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1*n2)

# -------------------- Name inference & formatting --------------------
def infer_equal_from_name(name_str):
    """
    If 'name' is like 'AlNbTiMoV' -> equal molar;
    if 'AlNbTi0.5MoV' -> use given ratios.
    """
    if not isinstance(name_str, str):
        return None
    name_str = clean_text(name_str)
    parsed = parse_concatenated_formula(name_str)
    if not parsed:
        return None
    if any(abs(v - 1.0) > 1e-9 for v in parsed.values()):
        d = parsed
    else:
        d = {el: 1.0 for el in parsed.keys()}
    tot = sum(d.values())
    if tot <= 0:
        return None
    return {el: v / tot for el, v in d.items()}

EL_ORDER_RE = re.compile(r"[A-Z][a-z]?")

def element_order_from_name(name):
    return EL_ORDER_RE.findall(name) if isinstance(name, str) else []

def build_ratio_string(frac_dict, order_hint=None, eps=1e-8, decimals=1):
    """
    Build "A1.0B1.0C1.0" from molar fractions by scaling so min nonzero = 1.0.
    Follows 'order_hint' if provided; otherwise alphabetic.
    """
    if not isinstance(frac_dict, dict) or not frac_dict:
        return None
    # Remove ~zeros
    filt = {el: v for el, v in frac_dict.items() if v is not None and v > eps}
    if not filt:
        return None
    # Scale so min = 1.0
    mn = min(filt.values())
    scaled = {el: (v / mn) for el, v in filt.items()}
    # Ordering
    if order_hint:
        ordered = [(el, scaled[el]) for el in order_hint if el in scaled]
        rest = sorted([(el, val) for el, val in scaled.items() if el not in set(order_hint)], key=lambda x: x[0])
        ordered.extend(rest)
    else:
        ordered = sorted(scaled.items(), key=lambda x: x[0])
    # Format
    fmt = f"{{:.{decimals}f}}"
    parts = [f"{el}{fmt.format(val)}" for el, val in ordered]
    return "".join(parts)

# -------------------- Pipeline --------------------
def parse_and_normalize_series(series):
    parsed = []
    unit_hints = []
    normalized = []
    for s in series:
        d, meta = parse_any(s) if isinstance(s, str) else (None, {'unit_hint': None})
        parsed.append(d)
        unit_hints.append(meta.get('unit_hint'))
        normalized.append(normalize_to_molar_fractions(d, unit_hint=meta.get('unit_hint'), assume_weight_if_ambiguous=False))
    return parsed, unit_hints, normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", required=True, help="Path to output cleaned CSV")
    ap.add_argument("--l1-thr", type=float, default=0.15, help="L1 threshold for match")
    ap.add_argument("--cos-thr", type=float, default=0.95, help="Cosine similarity threshold for match")
    ap.add_argument("--eps", type=float, default=1e-8, help="Zero cutoff for pruning elements")
    ap.add_argument("--ratio-decimals", type=int, default=1, help="Decimals in 'cleanned composition'")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Column detection (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    nom_col = cols_lower.get("nominal composition") or cols_lower.get("nominal_composition") or "nominal composition"
    name_col = cols_lower.get("name") or "name"
    meas_col = cols_lower.get("measured composition") or cols_lower.get("measured_composition") or "measured composition"

    # Parse & normalize
    nom_parsed, nom_unit_hint, nom_norm = parse_and_normalize_series(df[nom_col])
    name_inferred = [infer_equal_from_name(x) for x in df[name_col]] if name_col in df else [None]*len(df)
    meas_parsed, meas_unit_hint, meas_norm = parse_and_normalize_series(df[meas_col]) if meas_col in df else ([None]*len(df), [None]*len(df), [None]*len(df))

    # Consistency: pass-if-either rule
    l1_nom_name, l1_nom_meas, cos_nom_name, cos_nom_meas, consistency = [], [], [], [], []
    for i in range(len(df)):
        nom = nom_norm[i] or {}
        nm = name_inferred[i] or {}
        meas = meas_norm[i] or {}

        l1nn = l1_distance(nom, nm) if nm else None
        l1nm = l1_distance(nom, meas) if meas else None
        cosnn = cosine_similarity(nom, nm) if nm else None
        cosnm = cosine_similarity(nom, meas) if meas else None

        l1_nom_name.append(l1nn)
        l1_nom_meas.append(l1nm)
        cos_nom_name.append(cosnn)
        cos_nom_meas.append(cosnm)

        fail_name = (bool(nm) and ((l1nn is not None and l1nn > args.l1_thr) or (cosnn is not None and cosnn < args.cos_thr)))
        fail_meas = (bool(meas) and ((l1nm is not None and l1nm > args.l1_thr) or (cosnm is not None and cosnm < args.cos_thr)))

        if (nm or meas):
            if fail_name and fail_meas:
                consistency.append("inconsistent: name,measured")
            else:
                consistency.append("OK")
        else:
            consistency.append("unknown")

    # Build output dataframe
    def dict_to_sorted_json(d):
        return json.dumps(d, sort_keys=True) if isinstance(d, dict) else None

    out = df.copy()
    out["nominal_parsed"] = [dict_to_sorted_json(x) for x in nom_parsed]
    out["nominal_molar_fractions"] = [dict_to_sorted_json(x) for x in nom_norm]
    out["name_inferred_equal_molar"] = [dict_to_sorted_json(x) for x in name_inferred]
    out["measured_parsed"] = [dict_to_sorted_json(x) for x in meas_parsed]
    out["measured_molar_fractions"] = [dict_to_sorted_json(x) for x in meas_norm]

    out["l1_nom_vs_name"] = l1_nom_name
    out["l1_nom_vs_measured"] = l1_nom_meas
    out["cos_nom_vs_name"] = cos_nom_name
    out["cos_nom_vs_measured"] = cos_nom_meas
    out["consistency_check"] = consistency

    # --- Zero-prune name & make cleanned composition ---
    name_without_zero = []
    cleanned_strings = []
    for i in range(len(out)):
        name_str = out.at[i, name_col] if name_col in out.columns else None
        order = element_order_from_name(name_str) if isinstance(name_str, str) else []
        nom_frac = nom_norm[i] if isinstance(nom_norm[i], dict) else None

        # name_without_zero: keep only elements with nominal fraction > eps
        if isinstance(nom_frac, dict) and order:
            kept = [el for el in order if (el in nom_frac and nom_frac[el] is not None and nom_frac[el] > args.eps)]
            name_without_zero.append("".join(kept) if kept else "")
        else:
            name_without_zero.append(name_str if isinstance(name_str, str) else "")

        # cleanned composition: scale to smallest=1.0, format with N decimals
        cleanned_strings.append(
            build_ratio_string(nom_frac, order_hint=order, eps=args.eps, decimals=args.ratio_decimals)
            if isinstance(nom_frac, dict) else None
        )

    out["name_without_zero"] = name_without_zero
    out["cleanned composition"] = cleanned_strings  # intentionally spelled as requested

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # Summary
    n_ok = sum(1 for x in consistency if x == "OK")
    n_incons = sum(1 for x in consistency if x.startswith("inconsistent"))
    n_unknown = sum(1 for x in consistency if x == "unknown")
    print(f"[Done] Saved to: {args.output}")
    print(f"  OK: {n_ok} | inconsistent: {n_incons} | unknown: {n_unknown}")

if __name__ == "__main__":
    main()
