from __future__ import annotations
import numpy as np
import pandas as pd
YEARS = [2022, 2023, 2024]
YEAR_PAIRS = [(2022, 2023), (2023, 2024), (2022, 2024)]
MODELS = ['M1', 'M2']

def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method='average').corr(right.rank(method='average')))

def bias_persistence_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model in sorted(frame['model'].unique()):
        model_frame = frame.loc[frame['model'].eq(model)]
        pivot = model_frame.pivot(index='facility_id', columns='year', values='signed_bias_mmbtu_per_mwh')
        for year_a, year_b in YEAR_PAIRS:
            pair = pivot[[year_a, year_b]].dropna()
            rows.append({'model': model, 'year_a': year_a, 'year_b': year_b, 'n_facilities': int(len(pair)), 'spearman_rho': _spearman(pair[year_a], pair[year_b])})
    return pd.DataFrame(rows, columns=['model', 'year_a', 'year_b', 'n_facilities', 'spearman_rho'])

def coverage_categories(values: pd.Series) -> pd.Series:
    return pd.Series(np.select([values.lt(0.85), values.gt(0.95)], ['under', 'over'], default='nominal'), index=values.index, dtype='object')

def coverage_recurrence_rows(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working['coverage_category'] = coverage_categories(working['coverage_90'])
    rows: list[dict] = []
    for (facility_id, model), group in working.groupby(['facility_id', 'model'], sort=True):
        by_year = group.set_index('year').sort_index()
        if by_year.index.tolist() != YEARS:
            raise ValueError(f'Incomplete recurrence trajectory for facility {facility_id}, {model}')
        categories = {year: str(by_year.loc[year, 'coverage_category']) for year in YEARS}
        coverages = {year: float(by_year.loc[year, 'coverage_90']) for year in YEARS}
        counts = {category: sum((value == category for value in categories.values())) for category in ['under', 'nominal', 'over']}
        rows.append({'facility_id': facility_id, 'operator_id_2024': by_year.loc[2024, 'operator_id'], 'state': by_year.loc[2024, 'state'], 'model': model, 'coverage_2022': coverages[2022], 'coverage_2023': coverages[2023], 'coverage_2024': coverages[2024], 'category_2022': categories[2022], 'category_2023': categories[2023], 'category_2024': categories[2024], 'coverage_trajectory': '>'.join((categories[year] for year in YEARS)), 'transition_2022_2023': f'{categories[2022]}>{categories[2023]}', 'transition_2023_2024': f'{categories[2023]}>{categories[2024]}', 'undercovered_years': counts['under'], 'nominal_years': counts['nominal'], 'overcovered_years': counts['over'], 'undercovered_in_at_least_two_years': counts['under'] >= 2, 'overcovered_in_at_least_two_years': counts['over'] >= 2})
    return pd.DataFrame(rows)

def persistence_summary(identity: dict, bias_rows: pd.DataFrame, recurrence_rows: pd.DataFrame) -> dict:
    relation_summaries: dict[str, dict] = {}
    for model in MODELS:
        model_bias = bias_rows.loc[bias_rows['model'].eq(model)]
        model_recurrence = recurrence_rows.loc[recurrence_rows['model'].eq(model)]
        relation_summaries[model] = {'bias_rank_correlations': [{'year_pair': f'{int(row.year_a)}-{int(row.year_b)}', 'n_facilities': int(row.n_facilities), 'spearman_rho': float(row.spearman_rho)} for row in model_bias.itertuples(index=False)], 'coverage_recurrence': {'facilities': int(model_recurrence['facility_id'].nunique()), 'undercovered_in_at_least_two_years': int(model_recurrence['undercovered_in_at_least_two_years'].sum()), 'overcovered_in_at_least_two_years': int(model_recurrence['overcovered_in_at_least_two_years'].sum()), 'trajectory_counts': {str(key): int(value) for key, value in model_recurrence['coverage_trajectory'].value_counts().sort_index().items()}, 'transition_2022_2023_counts': {str(key): int(value) for key, value in model_recurrence['transition_2022_2023'].value_counts().sort_index().items()}, 'transition_2023_2024_counts': {str(key): int(value) for key, value in model_recurrence['transition_2023_2024'].value_counts().sort_index().items()}}}
    return {'analysis_id': 'COMMON_PANEL_PERSISTENCE', 'classification': 'protocol-locked post hoc common-panel persistence analysis', 'identity': identity, 'relations': relation_summaries, 'interpretation_rule': 'Interpret rank persistence and coverage recurrence descriptively. Weak or mixed persistence is reported as unstable facility ranking; no rescue analysis is permitted.'}
