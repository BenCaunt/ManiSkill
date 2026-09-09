# Reproduce the original Pour bottle walls

The offline preparation command turns the pinned ManiSkill 2 numeric Pour export
into 384 convex wall pieces and independently checks them before publishing a
runtime pack. It preserves original scale, mass, inertia, center of mass and
source notices. The hollow walls intentionally replace the reference's closed
rigid hull; this preparation result does not establish physical parity.

The input remains external research data under its original terms. Generate it
with [export_pour_assets.py](export_pour_assets.py) in the pinned ManiSkill 2
environment, or use that exact previously exported pack. Preparation checks the
export SHA-256 `54d4c40bdfe3184788d5e1f1841c8806611cff1249f0aad8b5c9a150ab982710`,
original body arrays, provenance and notices. It never downloads assets, reads
reference pickles or adds restricted geometry to the shared asset catalog.

## Build and prepare

Use SAPIEN 3.0.3, its matching SDK headers, Eigen 3.4.0 and Torch headers. The
preparation code also needs NumPy, SciPy and Shapely. Recorded successful
environments use NumPy 1.23.5 / SciPy 1.10.1 on Linux and NumPy 2.2.6 / SciPy
1.18.1 on macOS, both with Shapely 2.1.2. Native libraries, headers and source
checksums are pinned in [the cooker configuration](native/polyhedron/source.json).

Linux additionally requires the static PhysX release specified by
[SAPIEN's pinned build configuration](https://github.com/haosulab/SAPIEN/blob/731622eac5b140b320076c8a1b6eb4b553c3ccd4/cmake/physx5.cmake):
[linux-release.zip](https://github.com/sapien-sim/physx-precompiled/releases/download/105.1-physx-5.3.1.patch0/linux-release.zip),
SHA-256 `3e9f0b74bb1319700fe0d0618e664bedef0db154a42b69baf129977d65b4b2c4`.
Its MD5 also matches the upstream build pin. Extract it to a local directory;
the build command checks each linked Cooking, Common and Extensions library.
The Linux wheel exports the foundation interface but lacks the cooking and
memory-stream entry points used by this helper. The build therefore links the
matching static libraries and verifies an actual module import in a bounded
child process. This helper only prepares collision data; it does not replace
the simulation's physics library.

From this checkout:

```sh
python tools/softbody/native/polyhedron/build.py \
  --eigen /path/to/eigen-3.4.0 --physx /path/to/extracted-physx \
  --output /new/cooker-build

python -m tools.softbody.wall_preparation \
  --reference-pack /path/to/pour-numeric-pack \
  --build /new/cooker-build --output /new/preparation
```

Omit `--physx` on macOS. Build and output directories must be new. Native CPU
and GPU-data cooking each run in a bounded child; no GPU simulation occurs in
this step. A failed child leaves its log and execution record. A failing
independent geometry verdict produces no published `pack` directory.

The final output contains `structure`, `inputs`, native readbacks, `verdict.json`,
the complete execution provenance, and `pack`. The pack retains original notices
and metadata, alongside derived geometry and original physical parameters.
Keep the full preparation record for independent verification:

```sh
python tools/softbody/verify_wall_pack.py \
  --prepared /new/preparation --reference /path/to/pour-numeric-pack \
  --pack-sha256 SHA256_FROM_PREPARATION --output /new/independent-verdict.json
```

This verifier imports neither SAPIEN nor the preparation's partitioning code. It
checks source ring connectivity and caps, polygon coverage and overlap, native
vertices and planes, volume, and a continuous cylindrical cavity. It also checks
that the packed bytes came from those verified native readbacks and that original
physical parameters and notices were retained. The acceptance limits are fixed
in [limits.json](wall_preparation/limits.json); editing an output protocol to
weaken them is rejected.

## Use the generated pack

Build the separate [native cooked-mesh loader](native/cooked/README.md), then
pass the generated pack path and its explicit manifest checksum:

```python
import gymnasium as gym
import mani_skill.envs.softbody.pour

env = gym.make(
    "Pour-v0", legacy_mpm_data_dir="/path/to/pour-numeric-pack",
    bottle_collision_dir="/new/preparation/pack",
    bottle_collision_sha256="SHA256_FROM_PREPARATION",
    sim_backend="physx_cuda", num_envs=2,
)
```

The runtime checks the supplied manifest, every file, all 384 pieces, the original
source geometry identity, and unchanged mass/COM/inertia. The checksum is an
explicit input; the runtime never trusts whatever manifest it finds in a folder.
Omitting it retains the earlier pinned pack by default.

## Verification results

Fresh local and Linux builds pass both cooking modes and the independent pack
verifier: 390 packed files, a maximum wall-union error bound of 0.173020 micrometers
under the original 1 micrometer limit, and a minimum continuous cavity clearance
of 11.256295 mm against the 6 mm requirement. All 384 native vertex and plane
arrays match the earlier pack exactly. Six Linux polygon lists have equivalent
cyclic starting indices; raw serialized blobs are not identical across platforms.
Two local runs from different working directories produce the same pack checksum.

The 19 independent geometry controls and ten preparation failure controls pass,
along with 16 existing cooked-loader and Pour batch tests. Four additional
corruptions of real results are rejected even after updating their manifest
checksums: changed packed blob, changed mass, weakened geometry gate and changed
source notice. The first two Linux attempts failed on missing native symbols;
their archives and logs remain part of the recorded evidence.

Three uninstrumented GPU Pour cases load the freshly generated Linux pack:
one environment and two two-environment controller modes, including partial reset,
particle-count changes and reconfiguration. They complete and retain exactly the
previous 22 failures: 14 ordinary-camera visibility gates and eight exact selected
bottle-pose gates. All ten independent initial occlusion checks pass. No numeric
gate is relaxed. This is a setup/lifecycle regression; the earlier full 299-control
demonstration used the previous pack and is not rerun here.

The [result manifest](verification/wall-preparation-results.json) links protocols,
source inventories, independent verdicts and retained failed builds. The supplied
remote runner uses the existing pinned Lambda layout and image. It installs the
checksum-pinned preparation dependency into an isolated directory and runs with
network access disabled. Raw geometry, compiled dependencies and result archives
remain external.

Full task parity, broad platform/dependency compatibility and the previously
missing original Pinch/Write levels remain unfinished.
