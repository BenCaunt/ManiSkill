#!/bin/bash
set -euo pipefail
cd /home/ubuntu/softbody
label=${1:?label required}
expected=${2:?input checksum required}
[[ "$label" =~ ^packaging-native-v[0-9]+$ && "$expected" =~ ^[a-f0-9]{64}$ ]] || exit 2
out="$PWD/records/$label"
input="$PWD/probe-inputs/$label"
work="$PWD/probe-caches/$label"
mkdir "$out" "$work"
echo "$expected  $PWD/$label.tgz" | sha256sum --check
reuse=$(python3 - "$label" "$input" <<'PY'
import importlib.util,json,pathlib,sys
p=pathlib.Path('/home/ubuntu/softbody/gpu_coupling_archive.py')
s=importlib.util.spec_from_file_location('archive_helper',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pathlib.Path(sys.argv[2]);m.safe_extract(pathlib.Path('/home/ubuntu/softbody')/(sys.argv[1]+'.tgz'),d,max_bytes=64*1024**2)
r=json.loads((d/'request.json').read_text())
assert m.inventory(d/'source')==r['source_files']
for name,digest in r['build_dependencies'].items():assert m.file_hash(d/'build-dependencies'/name)==digest
reuse=r.get('reuse_bundle')
if reuse:
 import re
 assert re.fullmatch('packaging-native-v[0-9]+',reuse['build'])
 path=pathlib.Path('/home/ubuntu/softbody/records')/reuse['build']/'native/bundle'
 assert m.file_hash(path/'bundle.json')==reuse['manifest_sha256']
 print(path)
PY
)
exec 9>supervisor-jobs/gpu.lock
flock -w 90 9
image_id=$(cat provenance/candidate-image-id.txt)
container="softbody-$label"
trap 'docker rm --force "$container" >/dev/null 2>&1 || true' EXIT
status=0
bundle_args=()
if [[ -n "$reuse" ]]; then bundle_args=(-v "$reuse:/existing-bundle:ro"); fi
timeout --kill-after=20s 1200 docker run --rm --init --name "$container" \
 --gpus device=0 --runtime=nvidia --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges \
 --pids-limit=512 --memory=20g --cpus=8 --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,size=4g --shm-size=1g \
 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e XDG_CACHE_HOME=/work/cache \
 -v "$input:/input:ro" -v "$out:/output" -v "$work:/work" \
 "${bundle_args[@]}" \
 -v "$PWD/probe-inputs/native-bridge-v2/vendor/eigen:/eigen:ro" \
 "$image_id" bash -euc 'cp -a /input/source /work/source
 python -m pip install --no-index --no-deps --target /work/build-deps /input/build-dependencies/*.whl
 cd /work/source
 if [[ -d /existing-bundle ]]; then
   mkdir -p /output/native/dist
   MANISKILL_SOFTBODY_BINARY_DIR=/existing-bundle PYTHONPATH=/work/build-deps python -c "from setuptools.build_meta import build_wheel; print(build_wheel(\"/output/native/dist\"))"
   cp -a /existing-bundle /output/native/bundle
 else
   PYTHONPATH=/work/build-deps python tools/softbody/native/build_wheel.py --eigen /eigen --output /output/native
 fi' \
 > "$out/run.log" 2>&1 || status=$?
docker rm --force "$container" >/dev/null 2>&1 || true
flock -u 9
cp "$input/request.json" "$out/"
cp "$0" "$out/runner.sh"
python3 - "$out" "$status" "$image_id" "$expected" <<'PY'
import hashlib,json,pathlib,sys,tarfile
out=pathlib.Path(sys.argv[1]);record=dict(exit_code=int(sys.argv[2]),image_id=sys.argv[3],input_sha256=sys.argv[4])
if not record['exit_code']:
 wheels=list((out/'native/dist').glob('*.whl'));assert len(wheels)==1
 (out/'native/result.json').write_text(json.dumps(dict(wheel=wheels[0].name,
  sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
  bundle_sha256=hashlib.sha256((out/'native/bundle/bundle.json').read_bytes()).hexdigest()),indent=2)+'\n')
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
archive=out.parent/(out.name+'.tgz')
with tarfile.open(archive,'w:gz') as z:
 for p in sorted(out.rglob('*')):
  if p.is_file():z.add(p,arcname=p.relative_to(out),recursive=False)
record.update(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
print(json.dumps(record,indent=2))
PY
tail -n 20 "$out/run.log"
exit "$status"
