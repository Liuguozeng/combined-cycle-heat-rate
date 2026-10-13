import io

import json

import re

from pathlib import Path

from zipfile import ZipFile

import numpy as np

import pandas as pd

def workbook(path, member, sheet, required):
    with ZipFile(path) as z:
        names = [n for n in z.namelist() if member in n and n.endswith('.xlsx')]
        if len(names) != 1:
            raise ValueError((member, names))
        content = z.read(names[0])
    top = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=None, nrows=12)
    header = locate_header(top, required)
    frame = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=header)
    frame.columns = [str(c).strip() for c in frame]
    return frame, {'member': names[0], 'sheet': sheet, 'header_excel_row': int(header+1)}

def main():
    (HERE / 'qualification').mkdir(exist_ok=True)
    q = HERE / 'qualification'
    metrics = pd.read_csv(SUP / 'results/FACILITY_MODEL_METRICS.csv')
    ids = sorted(metrics.loc[metrics.year.eq(2024), 'facility_id'].unique())
    assert len(ids) == 135
    cohort = pd.read_csv(ARCH / 'eligibility/PREHOURLY_COHORT.csv', dtype={'selected_station_id':str})
    cohort = cohort.loc[cohort.facility_id.isin(ids)].sort_values('facility_id')
    unique_keys(cohort, ['facility_id'])
    assert len(cohort) == len(ids)
    cohort.to_csv(q / 'FIXED_FOLLOWUP_135.csv', index=False)
    gen, gen_info = workbook(RAW/'eia8602025_final.zip', '3_1_Generator', 'Operable', {'Plant Code','Generator ID'})
    plant, plant_info = workbook(RAW/'eia8602025_final.zip', '2___Plant', 'Plant', {'Plant Code','Latitude','Longitude'})
    gen = gen.loc[gen['Plant Code'].isin(ids)].copy()
    cap = linked_capacity(gen)
    annual, net_info = workbook(RAW/'eia9232025_final.zip', 'Schedules_2_3_4_5', 'Page 1 Generation and Fuel Data', {'Plant Id', 'Net Generation\n(Megawatthours)'})
    annual = annual.loc[annual['Plant Id'].isin(ids)].copy()
    net_column = 'Net Generation\n(Megawatthours)'
    annual[net_column] = pd.to_numeric(annual[net_column], errors='raise')
    net = annual.groupby('Plant Id')[net_column].sum(min_count=1)
    oldnet, old_info = workbook(ARCH/'data/raw/f923_2025.zip', 'Schedules_2_3_4_5', 'Page 1 Generation and Fuel Data', {'Plant Id', net_column})
    oldnet = oldnet.loc[oldnet['Plant Id'].isin(ids)].copy()
    oldnet[net_column] = pd.to_numeric(oldnet[net_column], errors='coerce')
    oldsum = oldnet.groupby('Plant Id')[net_column].sum(min_count=1)
    gross_data = json.loads((ARCH/'data/raw/campd_annual_by_facility_2025.json').read_text())
    gross = pd.DataFrame(gross_data.get('items', gross_data) if isinstance(gross_data, dict) else gross_data)
    gross.to_csv(q/'EPA_ANNUAL_FIELDS_2025.csv', index=False)
    meta = plant.loc[plant['Plant Code'].isin(ids)].rename(columns={'Plant Code':'facility_id'})
    unique_keys(meta, ['facility_id'])
    out = cohort[['facility_id','facility_name','qualifying_nameplate_mw','linked_unit_codes','operator_id','selected_station_id']].merge(cap, on='facility_id', how='left', suffixes=('_2024','_2025'), validate='one_to_one')
    out['net_mwh_2025_final'] = out.facility_id.map(net)
    out['net_mwh_2025_preliminary'] = out.facility_id.map(oldsum)
    out['net_revision_mwh'] = out.net_mwh_2025_final - out.net_mwh_2025_preliminary
    out['gross_mwh_2025'] = out.facility_id.map(gross.set_index('facilityId')['grossLoad'])
    out['gross_net_ratio'] = out.gross_mwh_2025 / out.net_mwh_2025_final
    out['annual_screen_pass'] = annual_boundary_screen(out.gross_mwh_2025, out.net_mwh_2025_final) & out.qualifying_nameplate_mw_2025.notna()
    for src, dst in [('Utility ID','operator_id_2025'),('Utility Name','operator_name_2025'),('Plant Name','facility_name_2025'),('State','state_2025')]:
        out[dst] = out.facility_id.map(meta.set_index('facility_id')[src])
    out.to_csv(q/'EIA_EPA_FINAL_2025_QUALIFICATION.csv', index=False)
    gen.to_csv(q/'EIA860_TARGET_GENERATORS_2025.csv', index=False)
    annual.to_csv(q/'EIA923_TARGET_ROWS_2025.csv', index=False)
