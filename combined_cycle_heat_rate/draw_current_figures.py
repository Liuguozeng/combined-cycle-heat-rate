"""Render heat-rate figures using explicit source and display identities."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


def invoke(entry, arguments):
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *map(str, arguments)]
    runpy.run_path(str(entry), run_name='__main__')


def copy_figure(source, destination, records, identity):
    for suffix in ['pdf', 'png']:
        path = source.with_suffix('.' + suffix)
        if not path.is_file():
            raise ValueError('Missing rendered figure: ' + str(path))
        target = destination.with_suffix('.' + suffix)
        target.write_bytes(path.read_bytes())
        records.append(dict(source_identity=identity, display_figure=destination.stem, file=target.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.package_root.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('An empty new output directory is required')
    contract = json.loads((root / 'MANUSCRIPT_INTERFACE.json').read_text())
    if contract['paper'] != 'combined_cycle_heat_rate':
        raise ValueError('Wrong manuscript interface for this figure entry point')
    primary = output / 'primary_evidence'
    invoke(root / 'plotting/plot_results.py', ['--output-root', primary])
    compact = output / 'compact_evidence'
    invoke(root / 'selected_presentation/scripts/compact_layout.py',
           ['--input-manifest', root / 'selected_presentation/presentation_inputs.json', '--output', compact])
    records = []
    for source, display in [(primary / 'Figure_2/Figure_2', 'Figure_S3'),
                            (primary / 'Figure_3/Figure_3', 'Figure_2'),
                            (compact / 'Figure_6', 'Figure_3'),
                            (compact / 'Figure_5', 'Figure_5')]:
        copy_figure(source, output / display, records, str(source.relative_to(output)))
    from followup_presentation import FollowupInputs, render_retrospective_si
    inputs = FollowupInputs(root / 'selected_presentation/presentation_inputs.json')
    holder = SimpleNamespace(output=output / 'supplementary_evidence', records=[])
    render_retrospective_si(holder, inputs)
    retrospective = holder.output / 'combined_cycle_heat_rate/FIGS1_2025_CORE_RETROSPECTIVE/Figure_S1_RETROSPECTIVE'
    copy_figure(retrospective, output / 'Figure_S1', records, 'original_three_year_bias_pairs')
    (output / 'DISPLAY_RECEIPT.json').write_text(json.dumps(dict(outputs=records, interface=contract), indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
