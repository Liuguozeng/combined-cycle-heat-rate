import math
import unittest
import pandas as pd
from weather import check_observation, select_hours, daily_from_hours, quality

def row(value=10.8, report='METAR KSBD 020540Z 09005KT 10SM CLR 11/M02 A3013 RMK AO2 T01081024', **kw):
    out={'STATION':'USW00023122','DATE':'2025-01-02T05:40:00','temperature':value,
         'temperature_Quality_Code':'5','temperature_Source_Code':'343','temperature_Source_Station_ID':'999999-23122',
         'temperature_Report_Type':'FM15','REM':report,'REM_Source_Code':'343',
         'REM_Source_Station_ID':'999999-23122','REM_Report_Type':'FM15'}
    out.update(kw)
    return out

class WeatherTests(unittest.TestCase):
    def test_normal_tenths_unchanged(self):
        r=check_observation(row(),'USW00023122','KSBD')
        self.assertEqual((r['selected_temp_c'],r['action']),(10.8,'unchanged'))
    def test_four_known_are_generic_fixtures(self):
        for date,old,integer in [('2025-04-18T14:54:00',101,10),('2025-05-16T22:53:00',270,27),('2025-05-28T20:54:00',263,26),('2025-07-07T15:55:00',214,21)]:
            d=pd.Timestamp(date)
            r=check_observation(row(old,f'METAR KSBD {d:%d%H%M}Z 00000KT 7SM CLR {integer:02}/07 A2999 RMK',DATE=date),'USW00023122','KSBD')
            self.assertEqual(r['selected_temp_c'],integer)
            self.assertEqual(r['precision_c'],1.)
            self.assertEqual(r['action'],'corrected_integer_report')
    def test_additional_four_2024_fixtures(self):
        cases=[('2024-07-05T14:54:00',247,24),('2024-07-26T16:49:00',325,32),('2024-10-27T17:51:00',258,25),('2024-11-07T20:50:00',252,25)]
        for stamp,old,integer in cases:
            d=pd.Timestamp(stamp)
            r=check_observation(row(old,f'METAR KSBD {d:%d%H%M}Z 00000KT 7SM CLR {integer:02}/07 A2999 RMK',DATE=stamp),'USW00023122','KSBD')
            self.assertEqual((r['selected_temp_c'],r['precision_c']),(integer,1.))
    def test_other_station_t_group_not_hardcoded(self):
        r=check_observation(row(101,'METAR KXYZ 020540Z 00000KT 10SM CLR M02/M04 A3013 RMK T10171040',STATION='USI0000KXYZ'),'USI0000KXYZ','KXYZ')
        self.assertEqual(r['selected_temp_c'],-1.7)
        self.assertEqual(r['precision_c'],.1)
    def test_t_group_priority(self):
        r=check_observation(row(108),'USW00023122','KSBD')
        self.assertEqual(r['selected_temp_c'],10.8)
    def test_t_group_without_dewpoint(self):
        r=check_observation(row(108,'METAR KSBD 020540Z 00000KT 10SM CLR 11// A3013 RMK T0108'),'USW00023122','KSBD')
        self.assertEqual(r['selected_temp_c'],10.8)
        self.assertEqual(r['precision_c'],.1)
    def test_compatible_integer_not_precision_upgrade(self):
        for original,body,tgroup in [(2.,'02','T00171010'),(3.,'03','T00331010'),(-2.,'M02','T10171030')]:
            r=check_observation(row(original,f'METAR KSBD 020540Z 00000KT 10SM CLR {body}/M03 A3013 RMK {tgroup}'),'USW00023122','KSBD')
            self.assertEqual(r['action'],'unchanged')
            self.assertEqual(r['selected_temp_c'],original)
            self.assertEqual(r['match_status'],'precision_compatible_integer_field')
    def test_incompatible_integer_still_corrected(self):
        r=check_observation(row(20.,'METAR KSBD 020540Z 00000KT 10SM CLR 02/M01 A3013 RMK T00171010'),'USW00023122','KSBD')
        self.assertEqual(r['selected_temp_c'],1.7)
        self.assertEqual(r['action'],'corrected_t_group')
    def test_integer_does_not_destroy_valid_tenths(self):
        r=check_observation(row(10.8,'METAR KSBD 020540Z 00000KT 10SM CLR 11/M02 A3013 RMK'),'USW00023122','KSBD')
        self.assertEqual(r['selected_temp_c'],10.8)
        self.assertEqual(r['action'],'unchanged')
    def test_idempotent(self):
        x=row(108)
        first=check_observation(x,'USW00023122','KSBD')
        x['temperature']=first['selected_temp_c']
        second=check_observation(x,'USW00023122','KSBD')
        self.assertEqual(second['action'],'unchanged')
        self.assertEqual(first['selected_temp_c'],second['selected_temp_c'])
    def test_wrong_source_no_correction(self):
        r=check_observation(row(108,REM_Source_Code='223'),'USW00023122','KSBD')
        self.assertTrue(math.isnan(r['selected_temp_c']))
        self.assertEqual(r['match_status'],'source_mismatch')
    def test_wrong_time_no_correction(self):
        r=check_observation(row(108,DATE='2025-01-03T05:40:00'),'USW00023122','KSBD')
        self.assertEqual(r['match_status'],'utc_mismatch')
        self.assertFalse(r['valid'])
    def test_wrong_station_no_correction(self):
        r=check_observation(row(108),'USW00023122','KLAX')
        self.assertEqual(r['match_status'],'station_mismatch')
        self.assertFalse(r['valid'])
    def test_source_specific_quality(self):
        self.assertTrue(quality('343','5')[0])
        self.assertFalse(quality('223','5')[0])
        self.assertFalse(quality('343','3')[0])
        self.assertFalse(quality('343','a')[0])
        self.assertFalse(quality('999','5')[0])
        self.assertFalse(quality('343','?')[0])
        self.assertFalse(quality('999','')[0])
    def test_bad_qc_not_rescued_by_message(self):
        r=check_observation(row(108,temperature_Quality_Code='3'),'USW00023122','KSBD')
        self.assertFalse(r['valid'])
    def test_conflicting_t_groups_unresolved(self):
        r=check_observation(row(10.8,report='METAR KSBD 020540Z 00000KT 10SM CLR 11/M02 A3013 RMK T01081024 T01091024'),'USW00023122','KSBD')
        self.assertFalse(r['valid'])
    def test_missing_no_zero(self):
        self.assertFalse(check_observation(row(float('nan')),'USW00023122','KSBD')['valid'])
    def test_negative_integer(self):
        r=check_observation(row(-102,'METAR KSBD 020540Z 00000KT 10SM CLR M10/M12 A3013 RMK'),'USW00023122','KSBD')
        self.assertEqual(r['selected_temp_c'],-10)
    def test_whole_row_latest_and_routine_priority(self):
        rows=[check_observation(row(DATE=t),'USW00023122','KSBD') for t in ['2025-01-02T05:40:00','2025-01-02T05:50:00']]
        rows[1].update(selected_temp_c=12.,valid=True,action='unchanged',report_type='FM12')
        sel,stats=select_hours(pd.DataFrame(rows))
        self.assertEqual(sel.iloc[0].selected_temp_c,10.8)
        self.assertEqual(sel.iloc[0].DATE,'2025-01-02T05:40:00')
    def test_duplicate_conflicts_cannot_win(self):
        a=check_observation(row(),'USW00023122','KSBD'); b=dict(a,selected_temp_c=11.)
        sel,stats=select_hours(pd.DataFrame([a,b]))
        self.assertEqual(len(sel),0)
        self.assertEqual(stats['conflicting_identity_rows'],2)
    def test_exact_duplicate_collapse(self):
        a=check_observation(row(),'USW00023122','KSBD')
        sel,stats=select_hours(pd.DataFrame([a,a]))
        self.assertEqual(len(sel),1)
    def test_six_hour_coverage_and_date_boundary(self):
        rows=[]
        for stamp in ['2024-12-31T23:59:00','2025-01-01T00:01:00','2025-01-01T06:01:00','2025-01-01T12:01:00','2025-01-01T18:01:00']:
            d=pd.Timestamp(stamp)
            rows.append(check_observation(row(10.,f'METAR KSBD {d:%d%H%M}Z 00000KT 10SM CLR 10/07 A3000',DATE=stamp),'USW00023122','KSBD'))
        sel,_=select_hours(pd.DataFrame(rows)); daily=daily_from_hours(sel)
        self.assertFalse(daily.loc['2024-12-31','weather_usable'])
        self.assertTrue(daily.loc['2025-01-01','weather_usable'])
        self.assertEqual(daily.loc['2025-01-01','temp_c'],10.)

if __name__=='__main__': unittest.main(verbosity=2)
