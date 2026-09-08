#!/bin/bash
set -euo pipefail
cd /home/ubuntu/softbody
label=${1:?label required}
expected=${2:?input checksum required}
[[ "$label" =~ ^packaging-installed-v[0-9]+$ && "$expected" =~ ^[a-f0-9]{64}$ ]] || exit 2
out="$PWD/records/$label"
input="$PWD/probe-inputs/$label"
work="$PWD/probe-caches/$label"
mkdir "$out" "$work"
echo "$expected  $PWD/$label.tgz" | sha256sum --check
python3 - "$label" "$input" <<'PY'
import importlib.util,json,pathlib,sys
p=pathlib.Path('/home/ubuntu/softbody/gpu_coupling_archive.py')
s=importlib.util.spec_from_file_location('archive_helper',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pathlib.Path(sys.argv[2]);m.safe_extract(pathlib.Path('/home/ubuntu/softbody')/(sys.argv[1]+'.tgz'),d,max_bytes=64*1024**2)
r=json.loads((d/'request.json').read_text())
actual=m.inventory(d);actual.pop('request.json')
assert actual==r['input_files']
assert not (d/'source').exists()
PY
exec 9>supervisor-jobs/gpu.lock
flock -w 90 9
image_id=$(cat provenance/candidate-image-id.txt)
version=105.1-physx-5.3.1.patch0
library="$PWD/provenance/physx-gpu/$version/files/libPhysXGpu_64.so"
echo "4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706  $library" | sha256sum --check
container="softbody-$label"
trap 'docker rm --force "$container" >/dev/null 2>&1 || true' EXIT
status=0
timeout --kill-after=20s 2400 docker run --rm --init --name "$container" \
 --gpus device=0 --runtime=nvidia --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges \
 --pids-limit=512 --memory=20g --cpus=8 --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,size=4g --shm-size=1g \
 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work/site \
 -e XDG_CACHE_HOME=/work/cache -e MPLCONFIGDIR=/work/matplotlib \
 -e MANISKILL_LEGACY_ASSET_DIR=/legacy-assets -e MANISKILL_LEGACY_MPM_DATA=/legacy-data \
 -v "$input:/input:ro" -v "$out:/output" -v "$work:/work" \
 -v "$PWD/checkouts/ManiSkill2/mani_skill2/assets:/legacy-assets:ro" \
 -v "$PWD/legacy-data/pour-v1:/legacy-data:ro" \
 -v "$PWD/probe-inputs/cooked-probe-v5/pack:/cooked-pack:ro" \
 -v "$library:/tmp/.sapien/physx/$version/libPhysXGpu_64.so:ro" \
 "$image_id" python /input/installed_wheel_suite.py > "$out/run.log" 2>&1 || status=$?
docker rm --force "$container" >/dev/null 2>&1 || true
flock -u 9
cp "$input/request.json" "$input/probe.py" "$input/installed_wheel_probe.py" "$input/installed_wheel_suite.py" "$out/"
cp "$0" "$out/runner.sh"
python3 - "$out" "$status" "$image_id" "$expected" <<'PY'
import hashlib,json,pathlib,sys,tarfile
out=pathlib.Path(sys.argv[1]);record=dict(exit_code=int(sys.argv[2]),image_id=sys.argv[3],input_sha256=sys.argv[4])
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
