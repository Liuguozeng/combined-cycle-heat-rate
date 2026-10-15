'Explicit, local release validation and current figure regeneration.'
import argparse
import hashlib
import json
import os
import re
import runpy
import sys
from pathlib import Path

def invoke(entry, args):
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *map(str, args)]
    runpy.run_path(str(entry), run_name='__main__')

def verify(root):
    declared = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        h, name = line.split('  ', 1)
        p = root / name
        if name in declared or not p.resolve().is_relative_to(root) or not p.is_file():
            raise ValueError('Missing, duplicate or invalid member: ' + name)
        if hashlib.sha256(p.read_bytes()).hexdigest() != h:
            raise ValueError('Changed member: ' + name)
        declared[name] = h
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts}
    if actual != set(declared):
        raise ValueError('Undeclared or missing package member')
    return {'identity_verified': len(actual), 'raw_replay': 'NOT_EXECUTED_REQUIRES_LEGAL_INPUTS'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify', 'figures', 'fixed', 'historical-mean'])
    p.add_argument('--package-root', type=Path, required=True)
    p.add_argument('--input-manifest', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    root = a.package_root.resolve()
    identity = verify(root)
    contract = json.loads((root / 'RELEASE_INTERFACE.json').read_text())
    if contract['paper'] != 'combined_cycle_heat_rate':
        raise ValueError('Cross-paper release root')
    if a.command == 'verify':
        print(json.dumps(identity, indent=2))
        return
    if a.command == 'historical-mean':
        if a.input_manifest is None:
            raise ValueError('historical-mean requires an explicit --input-manifest')
        manifest = a.input_manifest.resolve()
        if not manifest.is_relative_to(root) or not manifest.is_file():
            raise ValueError('Mean-control manifest must belong to this verified release')
        registered = root / 'historical_mean/input_manifest.json'
        cfg = json.loads(manifest.read_text())
        expected = json.loads(registered.read_text())
        if cfg != expected:
            raise ValueError('Wrong registered historical-mean branch or specification')
        for role, item in cfg['files'].items():
            input_path = (manifest.parent / item['path']).resolve()
            if not input_path.is_relative_to(root) or not input_path.is_file():
                raise ValueError('Input escapes the release: ' + role)
        if a.output is None or a.output.exists():
            raise ValueError('An explicit new output directory is required')
        invoke(root / 'historical_mean/code/historical_mean_control.py',
               ['--input-manifest', manifest, '--output', a.output])
        return
    if a.output is None or a.output.exists():
        raise ValueError('An explicit new empty output directory is required')
    a.output.mkdir(parents=True)
    if a.command == 'fixed':
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        transfer = root / 'historical_transfer'
        manifest = a.output / 'registered_input_manifest.json'
        invoke(transfer / 'code/package_tools.py', ['--package-root', transfer, '--output', manifest])
        invoke(transfer / 'code/check_prepared.py', ['--package-root', transfer, '--input-manifest', manifest,
               '--analysis-spec', transfer / 'analysis_spec.json', '--output', a.output / 'FIXED_MODEL_CHECK.json'])
        return
    retained = a.output / 'retained'
    invoke(root / 'draw_current_figures.py', ['--package-root', root, '--output', retained])
    newer = a.output / 'comparison'
    invoke(root / 'historical_transfer/code/render_historical_transfer.py', ['--results', root / 'historical_transfer/results', '--mean-summary', root / 'historical_mean/results/SUMMARY_MEAN_CONTROL.csv', '--output', newer])
    old_main = 6
    outputs = []
    for path in newer.glob('*'):
        if path.suffix not in ('.pdf', '.png', '.svg'):
            continue
        name = re.sub('combined_cycle_heat_rate_Figure_' + str(old_main), 'Figure_4', path.name)
        name = re.sub(r'^Figure_2(?=[_.])', 'Figure_4', name)
        name = re.sub('combined_cycle_heat_rate_Figure_S', 'Figure_S', name)
        target = a.output / name
        target.write_bytes(path.read_bytes())
        outputs.append(name)
    if not any(n.startswith('Figure_4.') for n in outputs):
        raise ValueError('Historical-assignment Figure 4 was not generated')
    (a.output / 'REPRODUCTION_RECEIPT.json').write_text(json.dumps({
        'identity': identity, 'core_outputs': outputs, 'retained_directory': 'retained',
        'raw_replay': 'NOT_EXECUTED', 'model_fits': 0}, indent=2) + '\n')

if __name__ == '__main__':
    main()
