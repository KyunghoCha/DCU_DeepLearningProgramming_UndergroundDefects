"""Reconstruct the recorded split from the course data and included annotation labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def within(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Expected a relative data path: {relative}')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes the data directory: {relative}')
    return path


def prepare(source, output, manifest, annotations):
    if output.resolve().is_relative_to(source.resolve()) or output.resolve().is_relative_to(annotations.resolve()):
        raise ValueError('Output cannot be inside source images or annotation labels.')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists: {output}')
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row['status'] != 'included' for row in rows):
        raise ValueError('Use the recorded included split manifest.')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.prepare-', dir=output.parent))
    counts = dict.fromkeys(('train', 'val', 'test'), 0)
    try:
        seen = set()
        for row in rows:
            split = row['split']
            if split not in counts:
                raise ValueError('Unknown split.')
            image = within(source, row['source_image'])
            label = (within(annotations, 'labels/' + row['origin_id'] + '.txt')
                     if row['manual'].lower() == 'true' else within(source, row['source_label']))
            if digest(image) != row['file_sha256'] or digest(label) != row['label_sha256']:
                raise ValueError(f'Input SHA-256 differs: {row["origin_id"]}')
            destination_image = within(stage, row['output_image'])
            destination_label = within(stage, row['output_label'])
            if (Path(row['output_image']).parent.as_posix() != f'images/{split}' or
                    Path(row['output_label']).parent.as_posix() != f'labels/{split}' or
                    destination_image.stem != destination_label.stem or
                    row['output_image'] in seen):
                raise ValueError('Recorded output paths are inconsistent.')
            seen.add(row['output_image'])
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            destination_image.symlink_to(image.resolve())
            with destination_label.open('x', encoding='utf-8') as stream:
                stream.write(label.read_text(encoding='utf-8-sig').rstrip() + '\n')
            counts[split] += 1
        config = {'path': str(output.resolve()), 'train': 'images/train', 'val': 'images/val',
                  'test': 'images/test', 'names': {0: 'crack', 1: 'efflorescence', 2: 'spall_detach'}}
        (stage / 'data.yaml').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        (stage / 'source.json').write_text(json.dumps({'source_root': str(source.resolve()),
            'split_manifest_sha256': digest(manifest)}, ensure_ascii=False, indent=2) + '\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', '--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/split_manifest.csv')
    parser.add_argument('--annotations', type=Path, default=ROOT / 'annotations')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source.resolve(), args.output.absolute(),
                             args.manifest.resolve(), args.annotations.resolve()), indent=2))
