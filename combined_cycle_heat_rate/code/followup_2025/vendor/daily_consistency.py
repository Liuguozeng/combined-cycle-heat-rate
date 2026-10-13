'Bounded daily-product checks; no hourly selection or model logic.'
import pandas as pd
from decimal import Decimal

DAILY_RULE = '20260926-v2-daily-consistency'
FIELDS = ['ssod_temp_c','ssod_min_c','ssod_max_c','ssod_min_code','ssod_max_code',
          'ssod_daily_inconsistent','ssod_consistency_status','ssod_consistency_reason',
          'ssod_order_registered_only']


def separated_above(left,right):
    # SSOD temperatures are printed to tenths. Each rounded field spans +/-0.05 C.
    return pd.Series([False if pd.isna(a) or pd.isna(b) else
        Decimal(str(a))-Decimal(str(b)) > Decimal('0.1')
        for a,b in zip(left,right)],index=left.index)


def inspect_ssod(frame):
    out=pd.DataFrame(index=frame.index)
    for original,target in [('mean_temperature','ssod_temp_c'),
                            ('min_temperature','ssod_min_c'),
                            ('max_temperature','ssod_max_c')]:
        values=frame[original].astype('string').str.strip()
        numeric=pd.to_numeric(values.mask(values.eq('')),errors='raise').astype(float)
        out[target]=numeric.replace(-9999.9,float('nan'))
    for key in ('min','max'):
        out[f'ssod_{key}_code']=frame[f'{key}_temperature_Measurement_Code'].fillna('').astype(str).str.strip()
    reasons=pd.Series('',index=frame.index,dtype=object)
    orders=[
        ('min_above_max',separated_above(out.ssod_min_c,out.ssod_max_c),
         out.ssod_min_code.eq('*') & out.ssod_max_code.eq('*')),
        ('min_above_mean',separated_above(out.ssod_min_c,out.ssod_temp_c),out.ssod_min_code.eq('*')),
        ('mean_above_max',separated_above(out.ssod_temp_c,out.ssod_max_c),out.ssod_max_code.eq('*')),
    ]
    tests=[
        ('hourly_derived_min_out_of_range',out.ssod_min_c.notna() &
         out.ssod_min_code.eq('*') & ~out.ssod_min_c.between(-60,60)),
        ('hourly_derived_max_out_of_range',out.ssod_max_c.notna() &
         out.ssod_max_code.eq('*') & ~out.ssod_max_c.between(-60,60)),
    ]+[(reason,mask & comparable) for reason,mask,comparable in orders]
    for reason,mask in tests:
        reasons.loc[mask]=reasons.loc[mask].map(lambda text:(text+';' if text else '')+reason)
    out['ssod_daily_inconsistent']=reasons.ne('')
    out['ssod_consistency_status']=reasons.ne('').map({True:'inconsistent',False:'no_detected_contradiction'})
    out['ssod_consistency_reason']=reasons
    registered=pd.Series('',index=frame.index,dtype=object)
    for reason,mask,comparable in orders:
        flagged=mask & ~comparable
        registered.loc[flagged]=registered.loc[flagged].map(
            lambda text:(text+';' if text else '')+reason+'_period_not_established')
    out['ssod_order_registered_only']=registered
    return out


def apply_daily_consistency(daily):
    out=daily.copy()
    rejected=out.daily_method.eq('official_ssod_unchanged') & out.ssod_daily_inconsistent.eq(True)
    out['ssod_consistency_rejected']=rejected
    out.loc[rejected,'weather_usable']=False
    out.loc[rejected,'temp_c']=float('nan')
    return out
