# Pinned SAPIEN 3.0.3 cooked convex loader

Pour uses the externally supplied 384-piece bottle wall pack with manifest
SHA-256 `57ca9d867a663651257e8318df3ef70aca1fae40127ef29d25944461b3db43bc`.
Put it at `LEGACY_MPM_DATA/collision-cooked`, set `MANISKILL_BOTTLE_COLLISION_DIR`,
or pass `bottle_collision_dir` (highest priority).
The runtime verifies every blob and provenance file before native construction.
The pack retains original restricted asset notices and is not shipped here.
The [offline preparation command](../../wall-preparation.md) now reproduces and
independently verifies a new pack from the pinned numeric reference export.
Pass its generated hash with `bottle_collision_sha256`; omitting that argument
retains the earlier default above. Original source and mass-property checks apply
to both paths.

Build the adapter with the pinned SAPIEN 3.0.3 wheel, Torch headers and Eigen 3.4.0:

```sh
python tools/softbody/native/cooked/build.py --eigen /path/to/eigen-3.4.0 --output /new/build
export PYTHONPATH=/new/build:$PYTHONPATH
```

The source manifest pins the native wheel, original SDK header and all factory
inputs. The SDK overlay adds one static factory method without changing fields
or virtual methods. This compatibility build is experimental; the supplied
SAPIEN source patch is the route to a full wheel build, which remains unverified.
The adapter constructs ordinary native convex shapes with consistent owned meshes
and cached bounds. Batch rows own separate shapes sharing immutable native meshes.
The clone helper preserves density and other properties omitted by stock 3.0.3.

Isolated Linux CPU/GPU geometry, lifetime, clone and drop checks pass on the pinned
pack. [Pour integration](../../gpu-pour-batching.md) additionally passes its
geometry/model checks and a full recorded demonstration, while strict selected
pose and ordinary camera gates remain failures. These results do not establish
Pour parity. No mass, inertia, COM, scale or fluid SDF changes are
part of this loader. The wall geometry intentionally replaces the original closed
rigid convex hull, as documented in the external pack.

`mesh.h` retains the upstream Apache-2.0 notice from SAPIEN commit
731622eac5b140b320076c8a1b6eb4b553c3ccd4. The added factory/adapter is Apache-2.0.
The pybind11 conduit headers retain their license and pinned provenance in `vendor`.
