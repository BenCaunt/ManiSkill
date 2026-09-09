"""Preparation rejects changed external inputs and never publishes failed cooks."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import __main__ as pipeline
from . import cook


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def reference(tmp_path, monkeypatch):
    pack = tmp_path / 'reference'
    pack.mkdir()
    exported = dict(source_commit=pipeline.SOURCE_COMMIT,
                    geometry=[dict(name='bottle', file='body-0.npz')])
    (pack / 'export.json').write_text(json.dumps(exported))
    (pack / 'body-0.npz').write_bytes(b'synthetic geometry input; never loaded')
    for name in pipeline.NOTICES:
        (pack / name).write_text('Synthetic notice: preserve these bytes.\n')
    provenance = {k: 'Synthetic test metadata' for k in ['creator', 'attribution',
        'retrieved_at', 'license', 'license_evidence_url', 'units', 'coordinate_frame', 'transforms', 'physical_basis']}
    provenance.update(source_commit=pipeline.SOURCE_COMMIT, source_urls=['https://example.invalid/source'],
                      files={p.name: digest(p) for p in pack.iterdir()})
    (pack / 'PROVENANCE.json').write_text(json.dumps(provenance))
    monkeypatch.setattr(pipeline, 'EXPORT_SHA256', digest(pack / 'export.json'))
    monkeypatch.setattr(pipeline.structure, 'SOURCE_SHA', digest(pack / 'body-0.npz'))
    return pack


def test_external_input_contract(reference):
    _, record = pipeline.reference(reference)
    assert record['license'] == 'Synthetic test metadata'


@pytest.mark.parametrize('kind', ['geometry', 'export', 'notice', 'revision', 'missing_metadata', 'symlink'])
def test_changed_reference_rejected(reference, kind):
    if kind in ['geometry', 'export', 'notice']:
        name = {'geometry': 'body-0.npz', 'export': 'export.json', 'notice': pipeline.NOTICES[0]}[kind]
        with (reference / name).open('ab') as f:
            f.write(b'changed')
    elif kind == 'symlink':
        path = reference / pipeline.NOTICES[0]
        path.rename(reference / 'elsewhere')
        path.symlink_to(reference / 'elsewhere')
    else:
        path = reference / 'PROVENANCE.json'
        record = json.loads(path.read_text())
        if kind == 'revision':
            record['source_commit'] = '0' * 40
        else:
            record.pop('license')
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        pipeline.reference(reference)


def test_failed_verification_never_creates_pack(reference, tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    verdict = dict(cases={'cpu': {'passed': True}, 'gpu-data': {'passed': False}}, piece_count=384)
    with pytest.raises(ValueError, match='verification did not pass'):
        pipeline.package(reference, output, verdict)
    assert list(output.iterdir()) == []


def test_failed_native_run_leaves_diagnostic_record_without_pack(reference, tmp_path, monkeypatch):
    output = tmp_path / 'failed'
    monkeypatch.setattr(pipeline.structure, 'inspect', lambda p: ({}, {'test': np.ones(1)}))
    def cells(structure, folder):
        folder.mkdir()
        declared = json.loads((Path(pipeline.__file__).parent / 'limits.json').read_text())
        pipeline.write(folder / 'protocol.json', dict(declared, manifest_sha256='a' * 64))
    monkeypatch.setattr(pipeline.cells, 'prepare', cells)
    def failed(*args):
        raise RuntimeError('Synthetic child cook failure')
    monkeypatch.setattr(pipeline.cook, 'run', failed)
    with pytest.raises(RuntimeError, match='child cook failure'):
        pipeline.prepare(reference, tmp_path / 'build', output)
    report = json.loads((output / 'execution.json').read_text())
    assert not report['complete'] and 'child cook failure' in report['error']
    assert not (output / 'pack').exists() and (output / 'structure/report.json').exists()
    with pytest.raises(FileExistsError):
        pipeline.prepare(reference, tmp_path / 'build', output)
    assert json.loads((output / 'execution.json').read_text()) == report


def test_changed_extension_rejected_before_child_import(tmp_path):
    build = tmp_path / 'build'
    build.mkdir()
    extension = build / 'module.so'
    extension.write_bytes(b'not a loadable extension')
    pipeline.write(build / 'build.json', dict(sapien='3.0.3', extension=extension.name, extension_sha256='0' * 64))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='checksum or ABI'):
        cook.run(tmp_path / 'inputs', output, build)
    assert not output.exists()
