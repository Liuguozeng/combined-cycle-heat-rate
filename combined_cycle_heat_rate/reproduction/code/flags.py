from __future__ import annotations
import re
import pandas as pd

def flag_map():
    return dict(re.findall("VALUES \\('([^']+)', '([^']+)', '[NY]'\\)", DICTIONARY.read_text()))

def canonical_flag(value):
    if pd.isna(value) or not str(value).strip():
        return ''
    value = str(value).strip()
    mapping = flag_map()
    if value in mapping:
        return mapping[value]
    if value in mapping.values():
        return value
    raise ValueError('Unknown official measurement flag: ' + repr(value))

def eligible_rows(frame, *, normalized, require_cc=True):
    d = frame.copy()
    if d.duplicated(['Facility ID', 'Unit ID', 'Date', 'Hour']).any():
        raise ValueError('Duplicate unit-hour keys')
    raw = d['Heat Input Measure Indicator'].fillna('').str.strip()
    d['raw_heat_input_flag'] = raw
    values = {v: canonical_flag(v) for v in raw.unique()}
    d['normalized_heat_input_flag'] = raw.map(values)
    flag = d['normalized_heat_input_flag'] if normalized else raw
    for k in ['Operating Time', 'Gross Load (MW)', 'Heat Input (mmBtu)']:
        d[k] = pd.to_numeric(d[k], errors='coerce')
    ok = flag.eq('Measured') & d['Operating Time'].gt(0) & d['Gross Load (MW)'].gt(0) & d['Heat Input (mmBtu)'].gt(0) & d['Primary Fuel Type'].fillna('').str.contains('Pipeline Natural Gas', case=False, regex=False)
    if require_cc:
        ok &= d['Unit Type'].fillna('').str.strip().str.casefold().eq('combined cycle')
    return (d, ok)
