"""Run an unchanged physics probe while recording its installed module origins."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys


def origins(site):
    result={}
    for name,module in sorted(sys.modules.items()):
        if name.split('.')[0] not in ('mani_skill','warp_maniskill','warp','sapien303_actor_bridge','sapien303_cooked_bridge'):
            continue
        filename=getattr(module,'__file__',None)
        if filename is None:continue
        path=Path(filename).resolve()
        if not path.is_relative_to(site):
            raise ValueError('Module escaped installed wheel: '+name+' '+str(path))
        result[name]=dict(path=str(path),relative_path=str(path.relative_to(site)),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['coupling','batch'],required=True)
    p.add_argument('probe_args',nargs=argparse.REMAINDER);a=p.parse_args()
    site=a.site.resolve();a.output.mkdir(parents=True,exist_ok=False)
    # Enable before ManiSkill imports, as in the original GPU batch probe.
    import sapien
    if a.mode=='batch':sapien.physx.enable_gpu()
    for name in ('mani_skill.envs.softbody.mpm','sapien303_actor_bridge','sapien303_cooked_bridge'):
        importlib.import_module(name)
    before=origins(site)
    probe=Path('/input/probe.py' if a.mode=='batch' else '/input/tools/softbody/probe_coupling.py')
    args=a.probe_args[1:] if a.probe_args[:1]==['--'] else a.probe_args
    # The original coupling CLI creates its output directory itself.
    destination=a.output if a.mode=='batch' else a.output/'contact'
    sys.argv=[str(probe),'--output',str(destination),*args]
    record=dict(before=before,installed_root=str(site),probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest())
    try:
        runpy.run_path(str(probe),run_name='__main__')
    finally:
        record['after']=origins(site)
        (a.output/'module-origins.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
