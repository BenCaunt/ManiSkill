#!/bin/bash
set -euo pipefail
cd /home/ubuntu/softbody
label=${1:?label required}
expected=${2:?archive SHA256 required}
[[ "$label" =~ ^gpu-batch-v[0-9]+$ && "$expected" =~ ^[a-f0-9]{64}$ ]] || exit 2
out="$PWD/records/$label"
input="$PWD/probe-inputs/$label"
cache="$PWD/probe-caches/$label"
mkdir "$out"
echo "$expected  $PWD/$label.tgz" | sha256sum --check
has_extension=$(python3 - "$label" "$input" <<'PY'
import importlib.util,json,pathlib,sys
p=pathlib.Path('/home/ubuntu/softbody/gpu_coupling_archive.py')
s=importlib.util.spec_from_file_location('archive_helper',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pathlib.Path(sys.argv[2]);m.safe_extract(pathlib.Path('/home/ubuntu/softbody')/(sys.argv[1]+'.tgz'),d,max_bytes=64*1024**2)
r=json.loads((d/'request.json').read_text())
assert m.inventory(d/'source')==r['source_files'] and m.file_hash(d/'probe.py')==r['probe_sha256']
for name,digest in r.get('render_performance_helpers',{}).items():
 assert name in ('render_update_probe.py','baseline_particle_visuals.py')
 assert m.file_hash(d/name)==digest
if r.get('native_actor_extension') is not None:
 assert m.file_hash(d/'native_extensions.py')==r['native_extensions_helper_sha256']
if r.get('native_cooked_extension') is not None:
 assert m.file_hash(d/'cooked_extensions.py')==r['cooked_extensions_helper_sha256']
 config=json.loads((d/'source/tools/softbody/native/cooked/source.json').read_text())
 for name,digest in config['files'].items():assert m.file_hash(d/'source/tools/softbody/native/cooked'/name)==digest
print('yes' if r.get('native_actor_extension') is not None else 'no')
PY
)
exec 9>supervisor-jobs/gpu.lock
flock -w 90 9
image_id=$(cat provenance/candidate-image-id.txt)
version=105.1-physx-5.3.1.patch0
library="$PWD/provenance/physx-gpu/$version/files/libPhysXGpu_64.so"
warp_library="$PWD/probe-builds/gpu-coupling-v4/warp_maniskill/warp/bin/warp.so"
echo "4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706  $library" | sha256sum --check
echo "ed1c35995381c224f9ab359ad4fbe3cc48c0ff5205aa695f08deb2e73cac2e56  $warp_library" | sha256sum --check
extension_args=()
if [[ "$has_extension" == yes ]]; then
  python3 "$input/native_extensions.py" "$input/request.json" "$PWD" "$out/native-extension"
  cp "$input/native_extensions.py" "$out/native_extensions.py"
fi
if [[ -f "$out/native-extension/record.json" ]]; then
  extension_args=(-v "$out/native-extension:/native-extension:ro" -e PYTHONPATH=/native-extension)
  cp "$out/native-extension/build.json" "$out/extension-build.json"
  cp "$out/native-extension/record.json" "$out/native-extension.json"
  cp "$input/source/mani_skill/utils/sapien303.py" "$out/sapien303.py"
fi
has_cooked=$(python3 -c 'import json,sys;print("yes" if json.load(open(sys.argv[1])).get("native_cooked_extension") else "no")' "$input/request.json")
if [[ "$has_cooked" == yes ]]; then
  python3 "$input/cooked_extensions.py" "$input/request.json" "$PWD" "$out/cooked-extension"
  cp "$input/cooked_extensions.py" "$out/cooked_extensions.py"
  extension_args+=(-v "$out/cooked-extension:/cooked-extension:ro" -e PYTHONPATH=/native-extension:/cooked-extension)
fi
container="softbody-$label"
trap 'docker rm --force "$container" >/dev/null 2>&1 || true' EXIT
status=0
case_list=$(python3 - "$input/request.json" <<'PY_CASES'
import json,re,sys
request=json.load(open(sys.argv[1]));cases=request['cases']
assert cases and len(cases)==len({case['name'] for case in cases})
for index,case in enumerate(cases):
 assert re.fullmatch('[A-Za-z0-9_-]+',case['name']) and re.fullmatch('[a-z_]+',case['control_mode'])
 assert case['task'] in ('Fill','Excavate','Hang','Pour','Write','Pinch') and case['backend'] in ('cpu','gpu')
 print(index,case['name'],case['task'],case['backend'],case['control_mode'])
PY_CASES
)
while read -r case_index case_name task backend control_mode; do
  last_log="$out/$case_name/run.log"
  extra=()
  case "$task" in
    Hang) extra+=(-v "$PWD/records/hang-asset-pack-v2/pack:/legacy-data:ro") ;;
    Pour)
      extra+=(-v "$PWD/legacy-data/pour-v1:/legacy-data:ro")
      if [[ "$has_cooked" == yes ]]; then
        extra+=(-v "$PWD/probe-inputs/cooked-probe-v5/pack:/cooked-pack:ro")
      fi ;;
    Write) extra+=(-v "$PWD/records/write-asset-pack-v2/pack:/legacy-data:ro" -v "$PWD/records/write-asset-pack-v2/levels:/levels:ro") ;;
    Pinch) extra+=(-v "$PWD/records/pinch-asset-pack-v2/pack:/legacy-data:ro" -v "$PWD/records/pinch-asset-pack-v2/pack/levels:/levels:ro") ;;
  esac
    mkdir -p "$out/$case_name" "$cache/$case_name"
    run_status=0
    timeout --kill-after=15s 360 docker run --rm --init --name "$container" \
      --gpus device=0 --runtime=nvidia --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges \
      --pids-limit=512 --memory=20g --cpus=8 --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,size=4g --shm-size=1g \
      "${extension_args[@]}" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e XDG_CACHE_HOME=/cache -e MPLCONFIGDIR=/cache/matplotlib \
      -e MANISKILL_LEGACY_ASSET_DIR=/legacy-assets -e MANISKILL_LEGACY_MPM_DATA=/legacy-data \
      -v "$input:/input:ro" -v "$out/$case_name:/output" -v "$cache/$case_name:/cache" \
      -v "$PWD/checkouts/ManiSkill2/mani_skill2/assets:/legacy-assets:ro" "${extra[@]}" \
      -v "$warp_library:/input/source/warp_maniskill/warp/bin/warp.so:ro" \
      -v "$library:/tmp/.sapien/physx/$version/libPhysXGpu_64.so:ro" \
      "$image_id" python /input/probe.py --source /input/source --request /input/request.json --case-index "$case_index" --output /output \
      > "$out/$case_name/run.log" 2>&1 || run_status=$?
    docker rm --force "$container" >/dev/null 2>&1 || true
    printf '{"exit_code":%s}\n' "$run_status" > "$out/$case_name/execution.json"
    printf '%s exit %s\n' "$case_name" "$run_status"
    if [[ "$run_status" != 0 ]]; then status=1; fi
done <<< "$case_list"
flock -u 9
cp "$input/request.json" "$input/probe.py" "$out/"
for helper in render_update_probe.py baseline_particle_visuals.py; do
  if [[ -f "$input/$helper" ]]; then cp "$input/$helper" "$out/"; fi
done
mkdir "$out/source"
cp "$input/source/mani_skill/envs/softbody/"*.py "$out/source/"
cp "$input/source/mani_skill/envs/scene.py" "$out/source/scene.py"
cp "$input/source/mani_skill/envs/sapien_env.py" "$out/source/sapien_env.py"
python3 - "$out" "$status" "$image_id" "$expected" <<'PY'
import hashlib,json,pathlib,sys,tarfile
out=pathlib.Path(sys.argv[1]);record={'exit_code':int(sys.argv[2]),'image_id':sys.argv[3],'input_sha256':sys.argv[4]}
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
archive=out.parent/(out.name+'.tgz')
with tarfile.open(archive,'w:gz') as z:
 for p in sorted(out.rglob('*')):
  if p.is_file():z.add(p,arcname=p.relative_to(out),recursive=False)
record.update(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
print(json.dumps(record,indent=2))
PY
tail -n 35 "$last_log"
exit "$status"
