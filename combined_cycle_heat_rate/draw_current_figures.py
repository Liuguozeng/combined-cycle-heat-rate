'Render retained evidence using explicit package inputs and current display numbers.'
import argparse
import json
import runpy
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, output = a.package_root.resolve(), a.output.resolve()
    if output.exists():
        raise ValueError('An empty new output directory is required')
    contract = json.loads((root / 'MANUSCRIPT_INTERFACE.json').read_text())
    stage = output / 'stable_evidence_ids'
    if contract['paper'] != 'combined_cycle_heat_rate':
        raise ValueError('Wrong manuscript interface for this figure entry point')
    entry = root / 'plotting/plot_results.py'
    if not entry.exists():
        raise ValueError('Public Code.zip is required for primary Figures 3/4')
    argv = [str(entry), '--output-root', str(stage)]
    sys.path.insert(0, str(entry.parent))
    sys.argv = argv
    runpy.run_path(str(entry), run_name='__main__')
    if contract['paper'] == 'combined_cycle_heat_rate':
        entry = root / 'selected_presentation/scripts/build_mixed_evidence_figures.py'
        sys.path.insert(0, str(entry.parent))
        sys.argv = [str(entry), '--output-root', str(stage), '--figures', 'combined_cycle_heat_rate:4', 'combined_cycle_heat_rate:5', '--followup-input-manifest', str(root / 'selected_presentation/presentation_inputs.json'), '--heat_rate-fig5-variant', 'followup_first', '--emit-retrospective-si']
        runpy.run_path(str(entry), run_name='__main__')
    import re
    outputs = []
    for path in stage.rglob('*'):
        if not path.is_file() or path.suffix not in ('.pdf', '.png', '.svg'):
            continue
        match = re.search('(?:Figure_|fig)([2-5])(?=[_.])', path.name)
        if not match:
            continue
        old = match.group(1)
        new = contract['stable_evidence_to_display'][old]
        name = path.name[:match.start(1)] + str(new) + path.name[match.end(1):]
        target = output / name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError('Conflicting display identity: ' + name)
        target.write_bytes(path.read_bytes())
        outputs.append({'stable_evidence_id': old, 'display_figure': new, 'file': name})
    if contract['paper'] == 'combined_cycle_heat_rate':
        entry = root / 'selected_presentation/scripts/compact_layout.py'
        sys.path.insert(0, str(entry.parent))
        sys.argv = [str(entry), '--input-manifest', str(root / 'selected_presentation/presentation_inputs.json'),
                    '--output', str(output / 'selected_compact_A')]
        runpy.run_path(str(entry), run_name='__main__')
        for prior in outputs:
            if str(prior['display_figure']) in ('5', '6'):
                (output / prior['file']).unlink()
        for number in (5, 6):
            for suffix in ('pdf', 'png'):
                source = output / 'selected_compact_A' / f'Figure_{number}.{suffix}'
                target = output / f'Figure_{number}.{suffix}'
                target.write_bytes(source.read_bytes())
        outputs = [r for r in outputs if str(r['display_figure']) not in ('5', '6')]
        for number in (5, 6):
            outputs.append({'display_figure': number, 'file': f'Figure_{number}.pdf',
                            'layout': 'selected compact A; all main evidence retained'})
    if not outputs:
        raise ValueError('No retained result figures were rendered')
    (output / 'DISPLAY_RECEIPT.json').write_text(json.dumps({'outputs': outputs, 'interface': contract}, indent=2) + '\n')
    print(json.dumps(outputs, indent=2))

if __name__ == '__main__':
    main()
