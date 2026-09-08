# Experimental soft-body distributions

The fork now packages its bundled MPM runtime, native build sources and their
licenses. An ordinary wheel is source-only. A separate Linux build command
produces a wheel containing the compiled Warp runtime, selected-actor adapter
and cooked-mesh adapter. Benchmark assets and exported collision packs remain
external, with their original licenses; installing the wheel does not download
them or establish reference-physics parity.

## Source wheel

Build from a clean checkout or unpacked source distribution with setuptools:

```sh
python -c 'from setuptools.build_meta import build_sdist, build_wheel; build_sdist("dist"); build_wheel("dist")'
python -m pip install 'dist/mani_skill-3.0.1-py3-none-any.whl[softbody]'
python -m warp_maniskill.build_lib
```

Use a writable virtual environment. The last command compiles the bundled Warp
library into that installation. CPU builds require a C++ compiler; CUDA builds
require CUDA 11.8. This path has been exercised on macOS arm64 with a CPU build
and physical contact probe. It does not provide the Linux GPU adapters.

The `softbody` extra pins SAPIEN 3.0.3. Avoid importing a different Warp package
before the bundled runtime. The build rejects an already imported foreign Warp.
Native headers, CUDA/C++ sources, vendored interoperability headers, source
provenance and license notices are included. Interpreter caches, object files,
and native libraries left in the checkout are excluded from source wheels.
A previous native build in the same build directory must be removed or the
source-only build fails explicitly.

## Linux native wheel

The tested build environment is Linux x86_64, Python 3.10, SAPIEN 3.0.3, Torch
2.5.1, CUDA 11.8, GCC 11, setuptools 84.0.0 and Eigen 3.4.0. Supply the pinned
Eigen headers described in [the actor adapter documentation](native/README.md).
Dependencies and toolchains must already be installed; the build command makes
no downloads. Use a writable checkout and a new output directory outside it:

```sh
python tools/softbody/native/build_wheel.py \
  --eigen /path/to/eigen-3.4.0 --output /path/to/new-native-build
python -m pip install /path/to/new-native-build/dist/mani_skill-3.0.1-cp310-cp310-linux_x86_64.whl
```

The native wheel unconditionally requires SAPIEN 3.0.3, including when installed
without the extra. Its platform and Python ABI tags describe the building
interpreter; no manylinux portability is claimed. The wheel carries exactly
three native libraries and `mani_skill/envs/softbody/native-bundle.json`, which
records source hashes, library hashes, commands and native ABI provenance.
The adapters use a relative `sapien.libs` loader path.

`MANISKILL_SOFTBODY_BINARY_DIR` is an advanced rebuild input: it points to that
bundle directory. Packaging rejects an incomplete bundle, changed source,
wrong Python/platform, symlinked files, checksum mismatches or non-x86_64 ELF
libraries. It is not an implicit scan of build outputs. The native package is
experimental and has not been published to PyPI.

To inspect an installed tree without importing its packages, run
`python -I -S tools/softbody/inspect_installation.py --site-packages /path/to/site-packages`
from this checkout. The standard-library-only tool prints JSON with installation
type, expected files, SHA-256 results, recorded native ABI/build metadata,
inspecting-interpreter ABI and license provenance. Source-only installs normally
exit 0; `--require-native` makes them exit 1. Missing/corrupt bundles also exit 1.
An ABI mismatch with the inspecting interpreter is reported, so a foreign tree
can still be inspected. Success establishes only static file integrity, not
dependency/loader compatibility, CUDA availability or physics support/parity.
Source-only checks cover key source/license files; locally compiled Warp is
unverified. Use a trusted tree that is not being modified during inspection;
reads are bounded to 1 MiB for JSON, 256 MiB per file and 1 GiB in total.
The native manifest includes build-tree inputs as well as installed files. The
inspector reports the upstream robot-authoring template and Warp changelog in
`build_only_sources`, with their recorded hashes and exclusion reasons; neither
is wheel package data. All other declared source and native files are checked.
The initial diagnostic incorrectly reported those two omissions as missing
runtime files. The corrected diagnostic passes on the actual native wheel and
source installation, with 13 focused tests. This correction does not rebuild or
change the tested wheels.

## Evidence and limits

The source wheel and a wheel rebuilt from its sdist each pass an independent
481-file code/source/license audit. A deliberate stale `.so`, `.o` and `.pyc`
injection was excluded. An installation outside the source checkout compiled
its own CPU library and passed a 100-step contact test on macOS. The Linux
native wheel passed its 502-file inventory, ELF, ABI and dependency audit. The
first Linux build failed the unconditional SAPIEN dependency gate; that failed
record is retained alongside the corrected build.

The separate installed-runtime suite uses a network-disabled container with
only the wheel, unchanged diagnostics and external assets mounted. It records
all imported ManiSkill/Warp/adapter file origins, runs eight CPU/CUDA analytic
contact variants and repeats the four Pour GPU cases under their original
numeric limits. Packaging does not relax the existing selected-pose or camera
visibility failures. All twelve processes completed. Independent checks passed all eight analytic
contacts and all twelve installed module-origin records. The Pour verdict
retained exactly the earlier 24 failures: 16 default-view particle-visibility
checks and eight exact selected-bottle-pose checks. No new failures appeared.
All twelve separate initial visibility interventions passed. Exact result hashes
and measured impulses are in [the result manifest](verification/packaging-results.json).
The install used `--no-deps` in the pinned image; dependency resolution on a
fresh OS/CUDA stack and native compilation from a Linux sdist remain unverified.

The package follows setuptools' [package-data rules](https://setuptools.pypa.io/en/latest/userguide/datafiles.html)
and uses the Python packaging specification's
[platform compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/).
The bundled solver's [separate license](../../warp_maniskill/LICENSE.md),
including its non-commercial research/evaluation terms, remains applicable.
