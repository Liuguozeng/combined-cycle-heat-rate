'Explicit internal/external input preparation for the bounded evidence package.'
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def internal_path(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in str(p):
        raise ValueError('unsafe internal path')
    result = (root / p).resolve()
    if not result.is_relative_to(root) or result.is_symlink():
        raise ValueError('internal path escapes package')
    return result


def prepare(root, example=None, external=None):
    root = Path(root).resolve()
    example = example or json.loads((root / 'input_manifest.example.json').read_text())
    overrides = (external or {}).get('files', {})
    if set(overrides) - set(example['files']):
        raise ValueError('unknown external input role')
    files = {}
    for role, entry in example['files'].items():
        if role in overrides:
            item = overrides[role]
            p = Path(item['path'])
            if not p.is_absolute():
                raise ValueError('external input must be explicit absolute path')
            if item['sha256'] != entry['sha256']:
                raise ValueError('external input changes registered hash')
            p = p.resolve()
            mode = 'explicit_external_manifest'
        else:
            prefix = '{PACKAGE_ROOT}/'
            if not entry['path'].startswith(prefix):
                raise ValueError('unknown placeholder')
            p = internal_path(root, entry['path'][len(prefix):])
            mode = 'package_internal'
        if not p.is_file() or sha(p) != entry['sha256']:
            raise ValueError('missing or wrong registered input: ' + role)
        files[role] = {'path': str(p), 'sha256': entry['sha256'], 'preparation': mode}
    return {'schema': 1, 'files': files, 'archives': [],
            'scope': 'frozen_model_evaluation_from_registered_derived_inputs'}


def verify_payload(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'SHA256_MANIFEST.json').read_text())
    overlay_path = root / 'PRESENTATION_PATCHES.json'
    overlay = {}
    if overlay_path.exists():
        patches = json.loads(overlay_path.read_text())
        if patches.get('schema') != 'presentation-only-v1':
            raise ValueError('unknown presentation overlay schema')
        for patch in patches['files']:
            # This exception is confined to rendering and validation of its labels.
            # Scientific inputs, analysis code and cached models remain pinned below.
            if patch['path'] not in {'code/render_historical_transfer.py', 'code/package_tools.py'}:
                raise ValueError('scientific payload override forbidden')
            if patch['path'] in overlay:
                raise ValueError('duplicate presentation overlay path')
            overlay[patch['path']] = patch
    listed = set()
    for entry in manifest['files']:
        if entry['path'] in listed:
            raise ValueError('duplicate hash manifest path')
        listed.add(entry['path'])
        p = internal_path(root, entry['path'])
        expected = entry
        if entry['path'] in overlay:
            patch = overlay[entry['path']]
            if patch['baseline_sha256'] != entry['sha256']:
                raise ValueError('presentation overlay baseline differs')
            expected = patch
        if not p.is_file() or p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
            raise ValueError('payload hash failure: ' + entry['path'])
    if not set(overlay).issubset(listed):
        raise ValueError('unregistered presentation override')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    extras = {'SHA256_MANIFEST.json'} | ({'PRESENTATION_PATCHES.json'} if overlay_path.exists() else set())
    if actual != listed | extras:
        raise ValueError('unregistered or missing payload file')
    return len(listed) + len(extras - {'SHA256_MANIFEST.json'})


def private_markers(data):
    # Scan escaped JSON strings as well as literal filesystem paths.
    data = data.replace(b'\\/', b'/')
    return bool(re.search(b'/(?:Users|Volumes|home|mnt)/|file:' + b'//', data))


def privacy_scan(root):
    checked = 0
    for path in Path(root).rglob('*'):
        if not path.is_file():
            continue
        data = gzip.open(path, 'rb').read() if path.suffix == '.gz' else path.read_bytes()
        if private_markers(data):
            raise ValueError('private filesystem path in payload: ' + path.name)
        if path.suffix.lower() in {'.png', '.pdf', '.svg', '.tif', '.tiff'}:
            raise ValueError('figure/binary document excluded from evidence payload')
        if any(x in path.parts for x in ('dependencies', '__pycache__', '.venv')):
            raise ValueError('runtime dependencies excluded from evidence payload')
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--external-manifest', type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    if args.output.resolve().is_relative_to(root):
        raise ValueError('write generated local manifest outside immutable package')
    verify_payload(root)
    external = json.loads(args.external_manifest.read_text()) if args.external_manifest else None
    prepared = prepare(root, external=external)
    write_json(args.output, prepared)
    print(json.dumps({'status': 'pass', 'roles': len(prepared['files']), 'preparation': 'explicit', 'raw_reconstruction': False}))


if __name__ == '__main__':
    main()
