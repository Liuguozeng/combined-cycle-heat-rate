'Strict annual-input adapters; no model fitting or outcome-dependent selection.'
import pandas as pd
import numpy as np


def unique_keys(frame, keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f'Missing or duplicate identities: {keys}')
    return frame


def locate_header(frame, required):
    matches = [i for i, row in frame.iterrows()
               if required <= {str(x).strip() for x in row.dropna()}]
    if len(matches) != 1:
        raise ValueError(f'Expected one header containing {required}; found {matches}')
    return matches[0]


def normalize_ssod(frame, year):
    d = frame.copy()
    d.columns = [c.lower() for c in d.columns]
    raw_date = d['date'].astype(str)
    fmt = '%Y%m%d' if raw_date.str.fullmatch('\\d{8}').all() else '%Y-%m-%d'
    d['date'] = pd.to_datetime(raw_date, format=fmt, errors='raise')
    unique_keys(d, ['date'])
    if not d.date.dt.year.eq(year).all():
        raise ValueError('Weather year mismatch')
    count = pd.to_numeric(d['mean_temperature_measurement_code'], errors='raise')
    if (count.isna() | (count < 0) | (count > 24) | (count % 1 != 0) | count.isin([1, 2, 3])).any():
        raise ValueError('Invalid SSOD temperature observation count')
    temp = pd.to_numeric(d['mean_temperature'], errors='raise')
    # SSOD v2 supplies Celsius and -9999.9 for missing values. Reuse the
    # original endpoint temperature range, not a new bridge-equivalence gate.
    valid = (count > 0) & temp.ne(-9999.9) & temp.between(-60, 60)
    out = pd.DataFrame({'date': d.date, 'temp_c': temp,
                        'temperature_observations': count})
    return out.loc[valid].sort_values('date').reset_index(drop=True)


def annual_boundary_screen(gross, net):
    return net.gt(0) & (gross / net).between(0.90, 1.20)


def linked_capacity(frame):
    d = frame.copy()
    unique_keys(d, ['Plant Code', 'Generator ID'])
    fuels = [f'Energy Source {i}' for i in range(1, 7)]
    ng = d[fuels].fillna('').apply(lambda s: s.astype(str).str.strip()).eq('NG').any(axis=1)
    capacity = pd.to_numeric(d['Nameplate Capacity (MW)'], errors='raise')
    unit = d['Unit Code'].fillna('').astype(str).str.strip()
    keep = (ng & d['Associated with Combined Heat and Power System'].eq('N')
            & d.Status.eq('OP') & d['Prime Mover'].isin(['CT', 'CA'])
            & (capacity > 0) & ~unit.isin(['', 'nan', 'NA']))
    d = d.loc[keep].assign(capacity=capacity[keep], unit_code=unit[keep])
    rows = []
    for fid, group in d.groupby('Plant Code'):
        pairs = [g for _, g in group.groupby('unit_code')
                 if {'CT', 'CA'} <= set(g['Prime Mover'])]
        if not pairs:
            continue
        linked = pd.concat(pairs)
        total = float(linked.capacity.sum())
        if total >= 400:
            rows.append({'facility_id': int(fid), 'qualifying_nameplate_mw': total,
                         'linked_unit_codes': '|'.join(sorted(linked.unit_code.unique())),
                         'qualifying_generators': len(linked)})
    return pd.DataFrame(rows, columns=['facility_id', 'qualifying_nameplate_mw',
                                      'linked_unit_codes', 'qualifying_generators'])
