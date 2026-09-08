"""Independent native cooked mesh identity and geometry checks; no simulator."""
import json
from pathlib import Path

import numpy as np

from .cooked_extensions import FILENAME, source_digest, validate_build
from .job_archive import file_hash


def load_evidence(root,request,config,cooked_root):
    root,cooked_root=Path(root),Path(cooked_root)
    spec=request['native_cooked_extension']
    if file_hash(root/'cooked_extensions.py')!=request['cooked_extensions_helper_sha256']:
        raise ValueError('Cooked staging helper identity differs')
    directory=root/'cooked-extension'
    record=json.loads((directory/'record.json').read_text())
    manifest=json.loads((directory/'build.json').read_text());validate_build(spec,manifest)
    if (file_hash(directory/FILENAME)!=spec['sha256']
            or file_hash(directory/'build.json')!=record['build_manifest_sha256']
            or any(record[k]!=spec[k] for k in spec)):
        raise ValueError('Archived cooked extension identity differs')
    for name,digest in manifest['source']['files'].items():
        if request['source_files'].get('tools/softbody/native/cooked/'+name)!=digest:
            raise ValueError('Candidate cooked source differs from the compiled binary: '+name)
    def pinned(name):
        path=cooked_root/name
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or not path.resolve().is_relative_to(cooked_root.resolve())
                or name not in config['files'] or file_hash(path)!=config['files'][name]):
            raise ValueError('Unpinned or changed cooked geometry input: '+name)
        return path
    for name in config['files']:pinned(name)
    pack_path=pinned('pack-v1/pack.json')
    if file_hash(pack_path)!=config['pack_manifest_sha256']:
        raise ValueError('Cooked pack manifest differs')
    pack=json.loads(pack_path.read_text());references={}
    for name,digest in pack['files'].items():
        if file_hash(pinned('pack-v1/'+name))!=digest:
            raise ValueError('Cooked pack file differs from its own manifest')
    for name in pack['leaves']:
        path=pinned('output-v1/gpu-data/'+name+'.npz')
        with np.load(path,allow_pickle=False) as arrays:references[name]=dict(arrays)
    return pack,references


def check_model(model,pack,references,count,bounds_limit):
    failures=[];maximum=0.
    if not np.isfinite(bounds_limit) or bounds_limit<0:raise ValueError('Invalid cooked bounds limit')
    if not np.array_equal(model['source_shape_count'],np.full(count,len(pack['leaves']))):
        failures.append('Cooked bottle shape count differs')
    for i in range(count):
        for j,name in enumerate(pack['leaves']):
            prefix=f'cooked/{i}/{j}/';expected=references[name]
            def equal(key,want):
                actual=np.asarray(model[prefix+key])
                if not np.isfinite(actual).all() or not np.array_equal(actual,want):failures.append(prefix+key+' differs')
            for key in ('vertices','planes','polygon_indices','polygon_offsets'):equal(key,expected[key])
            equal('cached_vertices',expected['vertices'])
            triangles=[]
            for start,end in zip(expected['polygon_offsets'][:-1],expected['polygon_offsets'][1:]):
                face=expected['polygon_indices'][start:end]
                triangles.extend(face[[0,k,k+1]] for k in range(1,len(face)-1))
            equal('cached_triangles',triangles);equal('flags',[True,True]);equal('properties',[300,1,1,0])
            equal('scale',[1,1,1]);equal('pose',[0,0,0,1,0,0,0])
            bounds=np.stack([expected['vertices'].min(0),expected['vertices'].max(0)])
            actual=np.asarray(model[prefix+'mesh_aabb'])
            if actual.shape!=(2,3) or not np.isfinite(actual).all():raise ValueError('Invalid native cooked bounds')
            error=float(np.max(abs(actual-bounds)));maximum=max(maximum,error)
            if error>bounds_limit:failures.append(prefix+'bounds differ')
    return dict(failures=failures,max_bounds_error_m=maximum,pieces=len(pack['leaves'])*count)
