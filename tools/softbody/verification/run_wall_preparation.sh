#!/bin/bash
set -euo pipefail
cd /home/ubuntu/softbody
label=${1:?label required}
expected=${2:?archive checksum required}
[[ "$label" =~ ^wall-preparation-v[0-9]+$ && "$expected" =~ ^[a-f0-9]{64}$ ]] || exit 2
out="$PWD/records/$label"
input="$PWD/probe-inputs/$label"
work="$PWD/probe-caches/$label"
mkdir "$out" "$work"
echo "$expected  $PWD/$label.tgz" | sha256sum --check
python3 - "$label" "$input" <<'PY'
import importlib.util,json,pathlib,sys
p=pathlib.Path('/home/ubuntu/softbody/gpu_coupling_archive.py')
s=importlib.util.spec_from_file_location('archive_helper',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pathlib.Path(sys.argv[2]);m.safe_extract(pathlib.Path('/home/ubuntu/softbody')/(sys.argv[1]+'.tgz'),d,max_bytes=32*1024**2)
r=json.loads((d/'request.json').read_text())
assert m.inventory(d/'source')==r['source_files']
for name,digest in r['dependencies'].items():assert m.file_hash(d/'dependencies'/name)==digest
assert pathlib.Path('/home/ubuntu/softbody/provenance/candidate-image-id.txt').read_text().strip()==r['image_id']
PY
exec 9>supervisor-jobs/gpu.lock
flock -w 90 9
image_id=$(cat provenance/candidate-image-id.txt)
container="softbody-$label"
trap 'docker rm --force "$container" >/dev/null 2>&1 || true' EXIT
status=0
timeout --kill-after=15s 300 docker run --rm --init --name "$container" \
 --gpus device=0 --runtime=nvidia --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges \
 --pids-limit=256 --memory=8g --cpus=8 --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,size=2g \
 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work/deps:/input/source \
 -v "$input:/input:ro" -v "$out:/output" -v "$work:/work" \
 -v "$PWD/legacy-data/pour-v1:/reference:ro" \
 -v "$PWD/probe-inputs/native-bridge-v2/vendor/eigen:/eigen:ro" \
 -v "$PWD/probe-inputs/wall-physx-linux-release:/physx:ro" \
 "$image_id" bash -euc 'python -m pip install --no-index --no-deps --target /work/deps /input/dependencies/*.whl
 python /input/source/tools/softbody/native/polyhedron/build.py --eigen /eigen --physx /physx --output /output/build
 python -m tools.softbody.wall_preparation --reference-pack /reference --build /output/build --output /output/prepared' \
 > "$out/run.log" 2>&1 || status=$?
docker rm --force "$container" >/dev/null 2>&1 || true
flock -u 9
cp "$input/request.json" "$out/"
cp "$0" "$out/runner.sh"
cp -a "$input/source" "$out/source"
python3 - "$out" "$status" "$image_id" "$expected" <<'PY'
import hashlib,json,pathlib,sys,tarfile
out=pathlib.Path(sys.argv[1]);record=dict(exit_code=int(sys.argv[2]),image_id=sys.argv[3],input_sha256=sys.argv[4])
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
archive=out.parent/(out.name+'.tgz')
with tarfile.open(archive,'w:gz') as z:
 for p in sorted(out.rglob('*')):
  if p.is_file():z.add(p,arcname=p.relative_to(out),recursive=False)
record.update(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)
print(json.dumps(record,indent=2))
PY
exit "$status"
