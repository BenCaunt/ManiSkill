"""Worker-side installed-wheel validation; no checkout or oracle mounted."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    data=Path('/input');out=Path('/output');site=Path('/work/site')
    request=json.loads((data/'request.json').read_text())
    wheel=data/request['wheel']['filename'];assert digest(wheel)==request['wheel']['sha256']
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(site),str(wheel)],check=True)
    manifest=site/'mani_skill/envs/softbody/native-bundle.json'
    assert digest(manifest)==request['wheel']['bundle_sha256']
    bundle=json.loads(manifest.read_text())
    scratch=Path('/work/staging')
    for kind,helper,manifest_key,directory in [
        ('actor','native_extensions','actor_build','native-extension'),
        ('cooked','cooked_extensions','cooked_build','cooked-extension')]:
        spec=request['native_'+kind+'_extension'];build=scratch/'records'/spec['build']
        if kind=='cooked':build=build/'build'
        build.mkdir(parents=True)
        module='sapien303_'+kind+'_bridge';binary=next(site.glob(module+'*.so'))
        assert digest(binary)==spec['sha256'];shutil.copyfile(binary,build/binary.name)
        write(build/'build.json',bundle[manifest_key])
        helper_path=data/(helper+'.py')
        s=importlib.util.spec_from_file_location(helper,helper_path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        stage=m.stage_actor if kind=='actor' else m.stage
        stage(spec,scratch,out/directory);shutil.copyfile(helper_path,out/helper_path.name)
    shutil.copyfile(out/'native-extension/record.json',out/'native-extension.json')
    shutil.copyfile(out/'native-extension/build.json',out/'extension-build.json')
    shutil.copyfile(site/'mani_skill/utils/sapien303.py',out/'sapien303.py')
    (out/'source').mkdir()
    for path in (site/'mani_skill/envs/softbody').glob('*.py'):shutil.copyfile(path,out/'source'/path.name)
    for name in ('scene.py','sapien_env.py'):shutil.copyfile(site/'mani_skill/envs'/name,out/'source'/name)
    jobs=[]
    for device in ('cpu','cuda'):
        for variant in ('free','rotated','slider','hinge'):
            args=['--device',device,'--steps','100']
            if variant=='rotated':args+=['--rotated']
            elif variant in ('slider','hinge'):args+=['--articulation',variant]
            jobs.append(('coupling-'+device+'-'+variant,'coupling',args))
    for i,case in enumerate(request['cases']):
        jobs.append((case['name'],'batch',['--source',str(site),'--request',str(data/'request.json'),'--case-index',str(i)]))
    results={}
    for name,mode,args in jobs:
        with (out/(name+'.log')).open('wb') as log:
            try:
                proc=subprocess.run([sys.executable,str(data/'installed_wheel_probe.py'),'--site',str(site),
                    '--output',str(out/name),'--mode',mode,'--',*args],stdout=log,stderr=subprocess.STDOUT,timeout=360)
                code=proc.returncode
            except subprocess.TimeoutExpired:code=124
        (out/name).mkdir(exist_ok=True);write(out/name/'execution.json',dict(exit_code=code))
        results[name]=code;write(out/'progress.json',results);print(name,code,flush=True)
    write(out/'suite.json',dict(jobs=results,site=str(site),wheel=request['wheel']))
    return 1 if any(results.values()) else 0


if __name__=='__main__':raise SystemExit(main())
