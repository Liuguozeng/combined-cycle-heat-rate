'Portable preparation and payload-contract negatives; no network or fits.'
import json
from pathlib import Path
import tempfile
import unittest

import package_tools as p


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'input.csv').write_text('a,b\n1,2\n')
        self.example = {'files': {'x': {'path': '{PACKAGE_ROOT}/input.csv', 'sha256': p.sha(self.root / 'input.csv')}}}

    def test_internal_preparation(self):
        got = p.prepare(self.root, self.example)
        self.assertEqual(got['files']['x']['path'], str((self.root / 'input.csv').resolve()))
        self.assertEqual(got['archives'], [])

    def test_external_explicit_same_hash(self):
        external = {'files': {'x': {'path': str(self.root / 'input.csv'), 'sha256': self.example['files']['x']['sha256']}}}
        self.assertEqual(p.prepare(self.root, self.example, external)['files']['x']['preparation'], 'explicit_external_manifest')
        external['files']['x']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            p.prepare(self.root, self.example, external)

    def test_no_missing_fallback(self):
        (self.root / 'input.csv').unlink()
        with self.assertRaises(ValueError):
            p.prepare(self.root, self.example)

    def test_wrong_input_and_unknown_role(self):
        (self.root / 'input.csv').write_text('wrong')
        with self.assertRaises(ValueError):
            p.prepare(self.root, self.example)
        with self.assertRaises(ValueError):
            p.prepare(self.root, self.example, {'files': {'unknown': {}}})

    def test_traversal_absolute_placeholder_and_symlink(self):
        for name in ('../input.csv', str(self.root / 'input.csv'), 'x\\y'):
            with self.assertRaises(ValueError):
                p.internal_path(self.root, name)
        self.example['files']['x']['path'] = '{HISTORY}/input.csv'
        with self.assertRaises(ValueError):
            p.prepare(self.root, self.example)
        (self.root / 'escape').symlink_to(self.root.parent)
        with self.assertRaises(ValueError):
            p.internal_path(self.root, 'escape/outside')

    def test_hash_manifest_missing_modified_and_extra(self):
        entry = {'path': 'input.csv', 'sha256': p.sha(self.root / 'input.csv'), 'bytes': (self.root / 'input.csv').stat().st_size}
        p.write_json(self.root / 'SHA256_MANIFEST.json', {'files': [entry]})
        self.assertEqual(p.verify_payload(self.root), 1)
        (self.root / 'extra').write_text('x')
        with self.assertRaises(ValueError):
            p.verify_payload(self.root)
        (self.root / 'extra').unlink()
        (self.root / 'input.csv').write_text('corrupt')
        with self.assertRaises(ValueError):
            p.verify_payload(self.root)

    def test_privacy_and_dependency_gate(self):
        self.assertTrue(p.private_markers(b'/' + b'Users/person/private'))
        self.assertTrue(p.private_markers(b'\\/' + b'Volumes\\/disk'))
        self.assertFalse(p.private_markers(b'https://www.epa.gov/'))
        (self.root / 'dependencies').mkdir()
        (self.root / 'dependencies/x.py').write_text('x=1')
        with self.assertRaises(ValueError):
            p.privacy_scan(self.root)


if __name__ == '__main__':
    unittest.main()
