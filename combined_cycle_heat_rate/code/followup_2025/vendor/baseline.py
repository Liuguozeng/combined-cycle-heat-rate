from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def aggregate_facility_hours(rows: pd.DataFrame, nameplate_mw: float, include_substitute: bool=False, min_load_fraction: float=0.2) -> pd.DataFrame:
    data = rows.copy()
    numeric = ['Operating Time', 'Gross Load (MW)', 'Heat Input (mmBtu)', 'CO2 Mass (short tons)']
    for column in numeric:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors='coerce')
    indicator = data['Heat Input Measure Indicator'].fillna('').str.strip()
    allowed = indicator.eq('Measured')
    if include_substitute:
        allowed |= indicator.eq('Measured and Substitute')
    fuel_ok = data['Primary Fuel Type'].fillna('').str.contains('Pipeline Natural Gas', case=False, regex=False)
    valid = data['Operating Time'].gt(0) & data['Gross Load (MW)'].gt(0) & data['Heat Input (mmBtu)'].gt(0) & allowed & fuel_ok
    data = data.loc[valid].copy()
    if data.empty:
        return pd.DataFrame()
    data['date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['hour'] = pd.to_numeric(data['Hour'], errors='coerce').astype('Int64')
    data['gross_mwh'] = data['Gross Load (MW)'] * data['Operating Time']
    grouped = data.groupby(['date', 'hour'], as_index=False, observed=True).agg(gross_mwh=('gross_mwh', 'sum'), heat_input_mmbtu=('Heat Input (mmBtu)', 'sum'), co2_short_tons=('CO2 Mass (short tons)', 'sum'), qualifying_unit_rows=('Unit ID', 'size')).dropna(subset=['date', 'hour'])
    grouped['facility_load_fraction'] = grouped['gross_mwh'] / float(nameplate_mw)
    grouped['gross_heat_rate_mmbtu_per_mwh'] = grouped['heat_input_mmbtu'] / grouped['gross_mwh']
    return grouped.loc[grouped['facility_load_fraction'].between(min_load_fraction, 1.1)].reset_index(drop=True)

def aggregate_facility_days(hourly: pd.DataFrame, weather: pd.DataFrame, facility_id: int, meta: dict) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()
    daily = hourly.groupby('date', as_index=False).agg(gross_mwh=('gross_mwh', 'sum'), heat_input_mmbtu=('heat_input_mmbtu', 'sum'), co2_short_tons=('co2_short_tons', 'sum'), qualifying_hours=('hour', 'nunique')).loc[lambda frame: frame['qualifying_hours'].ge(6)].copy()
    daily['gross_heat_rate_mmbtu_per_mwh'] = daily['heat_input_mmbtu'] / daily['gross_mwh']
    daily['facility_load_fraction'] = daily['gross_mwh'] / (float(meta['nameplate_mw']) * daily['qualifying_hours'])
    daily['co2_rate_short_ton_per_mwh'] = daily['co2_short_tons'] / daily['gross_mwh']
    daily = daily.merge(weather, on='date', how='inner', validate='one_to_one')
    daily['facility_id'] = int(facility_id)
    daily['facility_name'] = meta['name']
    daily['operator_id'] = int(meta['operator_id'])
    daily['cooling_type'] = meta['cooling_type']
    daily['station_distance_km'] = float(meta['station_distance_km'])
    daily['day_of_year'] = daily['date'].dt.dayofyear
    daily['doy_sin'] = np.sin(2 * np.pi * daily['day_of_year'] / 366.0)
    daily['doy_cos'] = np.cos(2 * np.pi * daily['day_of_year'] / 366.0)
    daily['month'] = daily['date'].dt.month
    daily['week'] = daily['date'].dt.to_period('W-SUN').astype(str)
    for cooling in ['recirculating', 'once_through', 'dry_cooling']:
        daily[f'temp_x_{cooling}'] = daily['ambient_temperature_c'] * daily['cooling_type'].eq(cooling).astype(float)
    return daily.sort_values('date').reset_index(drop=True)

def load_weather(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path, usecols=['DATE', 'TEMP'])
    weather['date'] = pd.to_datetime(weather['DATE'], errors='coerce')
    temp_f = pd.to_numeric(weather['TEMP'], errors='coerce')
    weather['ambient_temperature_c'] = (temp_f - 32.0) * 5.0 / 9.0
    weather = weather.loc[weather['date'].notna() & weather['ambient_temperature_c'].between(-60, 60), ['date', 'ambient_temperature_c']]
    return weather.drop_duplicates('date', keep='last').reset_index(drop=True)

def predict_model(model_name: str, model, data: pd.DataFrame) -> np.ndarray:
    if model_name == 'M0':
        return np.full(len(data), float(model), dtype=float)
    return np.asarray(model.predict(data), dtype=float)
