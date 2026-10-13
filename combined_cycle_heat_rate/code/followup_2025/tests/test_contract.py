import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portable


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.root = Path(self.temp.name)
        self.rows = []
        self.inputs = []
        for key, role in [('epa', 'epa_hourly'), ('model', 'frozen_model'),
                          ('expected', 'expected')]:
            p = self.root / (key + '.txt')
            p.write_text(key)
            self.rows.append(dict(id=key, role=role, sha256=portable.digest(p)))
            self.inputs.append(dict(id=key, path=p.name))
        self.contract = dict(paper='combined_cycle_heat_rate', schema_version=1, inputs=self.rows)
        self.manifest = dict(paper='combined_cycle_heat_rate', schema_version=1, inputs=self.inputs)
        self.path = self.root / 'input.json'

    def tearDown(self):
        self.temp.cleanup()

    def resolve(self):
        self.path.write_text(json.dumps(self.manifest))
        return portable.resolve_inputs(self.path, self.contract)

    def test_explicit_manifest_success(self):
        self.assertEqual(set(self.resolve()), {'epa', 'model', 'expected'})

    def test_missing_lawful_input_is_not_executed(self):
        self.manifest['inputs'] = self.inputs[1:]
        with self.assertRaisesRegex(FileNotFoundError, 'NOT_EXECUTED.*epa'):
            self.resolve()

    def test_wrong_cache_hash_fails(self):
        (self.root / 'model.txt').write_text('changed cache')
        with self.assertRaisesRegex(ValueError, 'hash.*model'):
            self.resolve()

    def test_duplicate_manifest_identity_fails(self):
        self.manifest['inputs'].append(copy.deepcopy(self.inputs[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.resolve()

    def test_cross_paper_fails(self):
        self.manifest['paper'] = 'return_temperature_comparability'
        with self.assertRaisesRegex(ValueError, 'Cross-paper'):
            self.resolve()

    def test_missing_expected_fails(self):
        (self.root / 'expected.txt').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'NOT_EXECUTED.*expected'):
            self.resolve()

    def test_manifest_cannot_override_pinned_hash(self):
        self.inputs[1]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'manifest hash'):
            self.resolve()

    def test_unknown_input_fails(self):
        self.manifest['inputs'].append(dict(id='undeclared', path='epa.txt'))
        with self.assertRaisesRegex(ValueError, 'Undeclared'):
            self.resolve()

    def test_output_cannot_cover_an_input(self):
        resolved = self.resolve()
        with self.assertRaises(ValueError):
            portable.validate_output(self.root, resolved)


class FrozenMethodTests(unittest.TestCase):
    def test_weather_reference_signature_excludes_only_execution_timestamp(self):
        import pandas as pd
        frame = pd.DataFrame({'id': [1], 'value': [10.0], 'scan_utc': ['first']})
        same = frame.assign(scan_utc='second')
        changed = frame.assign(value=10.1)
        self.assertEqual(portable.table_signature(frame), portable.table_signature(same))
        self.assertNotEqual(portable.table_signature(frame), portable.table_signature(changed))

    def test_wrong_weather_year_fails(self):
        import pandas as pd
        inp = portable.load_vendor('inputs')
        annual = portable.load_vendor('annual', {'unique_keys': inp.unique_keys})
        with self.assertRaisesRegex(ValueError, 'Weather year mismatch'):
            annual.validate_daily_weather(pd.DataFrame({'date': ['2025-01-01'], 'temp_c': [20.]}), 2024)

    def test_result_numeric_contradiction_fails(self):
        import pandas as pd
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            p = Path(tmp)
            pd.DataFrame({'id': [1], 'value': [1.0]}).to_csv(p / 'old.csv', index=False)
            pd.DataFrame({'id': [1], 'value': [1.01]}).to_csv(p / 'new.csv', index=False)
            with self.assertRaises(AssertionError):
                portable.compare_tables(p / 'old.csv', p / 'new.csv', 'fixture', 1e-12, ['id'])

    def test_unknown_measurement_code_fails(self):
        dictionary = Path(__file__).resolve().parents[3] / 'data/followup_2025/reference/dictionary.sql'
        flags = portable.load_vendor('flags', {'DICTIONARY': dictionary})
        self.assertEqual(flags.canonical_flag('MEASURE'), 'Measured')
        with self.assertRaisesRegex(ValueError, 'Unknown official measurement'):
            flags.canonical_flag('MEASURE_UNKNOWN')

    def test_duplicate_unit_hour_fails(self):
        import pandas as pd
        flags = portable.load_vendor('flags')
        frame = pd.DataFrame({'Facility ID': [1, 1], 'Unit ID': ['A', 'A'],
                              'Date': ['2025-01-01'] * 2, 'Hour': [0, 0]})
        with self.assertRaisesRegex(ValueError, 'Duplicate unit-hour'):
            flags.eligible_rows(frame, normalized=True)

    def test_frozen_resampling_constants(self):
        national = portable.load_vendor('national')
        panel = portable.load_vendor('panel')
        self.assertEqual((national.N_BOOT, national.SEED), (5000, 20260827))
        self.assertEqual((panel.BOOTSTRAPS, panel.SEED_HEAT_RATE), (5000, 20260831))

    def test_old_year_guard_retained(self):
        import pandas as pd
        replay = portable.load_vendor('replay')
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            p = Path(tmp)
            (p / 'INVENTORY_RESULT.json').write_text('{"proceed":true}')
            pd.DataFrame([dict(role='external', year=2025, facility_id=1,
                               path='not-read', source_sha256='not-read')]).to_csv(p / 'ACTUAL_HITS.csv', index=False)
            replay.OUT = p
            replay.CHECK = p / 'check'
            replay.RESULTS = p / 'results'
            replay.DATA = p / 'data'
            with self.assertRaisesRegex(RuntimeError, 'year-specific evaluator'):
                replay.primary()


class IsolationTests(unittest.TestCase):
    def test_network_undeclared_read_and_external_write_are_blocked(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            output = root / 'output'
            output.mkdir()
            secret = root / 'undeclared.txt'
            secret.write_text('fixture')
            code = "import pathlib, socket, sys\nsys.path.insert(0, sys.argv[1])\nimport portable\noutput, secret = map(pathlib.Path, sys.argv[2:4])\ncounts = portable.install_guard(output, {})\nfor action in [lambda: secret.read_text(), lambda: secret.write_text('changed'), lambda: socket.socket()]:\n    try:\n        action()\n    except PermissionError:\n        pass\n    else:\n        raise AssertionError('guard did not block')\nprint(counts['denied_events'])\n"
            result = subprocess.run([sys.executable, '-B', '-c', code, str(portable.HERE), str(output), str(secret)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), '3')
            self.assertEqual(secret.read_text(), 'fixture')


if __name__ == '__main__':
    unittest.main()
