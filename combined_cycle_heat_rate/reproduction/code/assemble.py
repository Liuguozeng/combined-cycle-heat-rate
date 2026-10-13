from __future__ import annotations
import pandas as pd

def assemble(m, all_facility, full_rows, common_rows):
    facility = pd.concat(all_facility, ignore_index=True)
    full = pd.DataFrame(full_rows).sort_values(['subset_type', 'subset_id', 'year', 'model'])
    common = pd.DataFrame(common_rows).sort_values(['subset_type', 'subset_id', 'year', 'model'])
    loo = full.loc[full.subset_type.eq('leave_one_out')].copy()
    loo['passes_coverage'] = loo.facility_equal_conditional_coverage.gt(0.95)
    loo['retains_low_tail'] = loo.facilities_below_0_85.ge(1)
    loo['retains_high_tail'] = loo.facilities_above_0_95.ge(1)
    loo['passes_all'] = loo[['passes_coverage', 'retains_low_tail', 'retains_high_tail']].all(axis=1)
    lto = full.loc[full.subset_type.eq('leave_two_out')]
    mapping = {'subsets': ('subset_id', 'nunique')}
    for col in ['facility_equal_conditional_coverage', 'facility_equal_signed_bias_mmbtu_per_mwh', 'median_facility_mae_mmbtu_per_mwh', 'facility_equal_mean_winkler_score_mmbtu_per_mwh']:
        for prefix, agg in [('minimum', 'min'), ('maximum', 'max')]:
            mapping[prefix + '_' + col] = (col, agg)
    stress = lto.groupby(['year', 'model'], as_index=False, observed=True).agg(**mapping).sort_values(['year', 'model'])
    return {'SOURCE_COMPOSITION_FACILITY_METRICS.csv': facility, 'SOURCE_COMPOSITION_FULL_COHORT_SUMMARY.csv': full, 'SOURCE_COMPOSITION_COMMON_PANEL_SUMMARY.csv': common, 'SOURCE_COMPOSITION_LOO_GATE.csv': loo, 'SOURCE_COMPOSITION_LTO_STRESS_ENVELOPE.csv': stress}
