'Run contract and retained weather tests with an explicit writable directory.'
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True, type=Path)
    args = parser.parse_args()
    target = args.work_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(target)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE / 'vendor'))
    suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
    with (target / 'tests.log').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  test_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / 'tests').glob('test_*.py')})
    (target / 'TEST_STATUS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
