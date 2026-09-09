# Local coding and remote verification

This directory contains the local Codex supervisor, Linux job manager, capture
code and independent numeric evaluator used for the experimental port. The
supervisor writes a durable checkpoint before remote submission. After a local
process interruption, `--resume` recovers the same job and the original budget.
An implementation agent's successful exit cannot override a failing comparison.

The [live recovery record](../supervisor-recovery.md) documents a real local
SIGKILL while the existing Lambda worker was replaying Fill. The recovery snapshot
passed 130 local tests and reproduced that run's saved comparison exactly,
including both physics failures. The [source manifest](source-manifest.json)
identifies the bytes used during the live test and the subsequent transport fix.
The current package passes 214 tests and adds durable reference-demo and scaling jobs;
the [multi-episode Excavate record](../excavate-diversity.md) supplies the live
reference and CPU/GPU replay evidence for those additions.
The [GPU scaling study](../gpu-scaling.md) records all-row reset isolation,
short synchronized timing and observed memory at 2, 8 and 32 environments.

## Requirements and separation

The tested local setup uses macOS, Python 3.12, NumPy, Git, SSH/SCP and an
authenticated Codex CLI with named filesystem permissions. The CLI options are
documented in OpenAI's [non-interactive guide](https://learn.chatgpt.com/docs/non-interactive-mode).
The Linux worker already has Docker, the NVIDIA container runtime, a pinned
candidate image and the external reference assets. `gpu_job.py` records the
tested `/home/ubuntu/softbody` layout. This code uses an existing lease; it does
not provision a machine or install the independent provider termination timer.
The scaling verifier also imports SciPy and Pillow through its shared numeric
check modules. Local test execution requires h5py and pytest.

Copy this entire directory to a trusted location **outside** the candidate Git
checkout. Keep the reference traces, reset/action fixtures, calibrated protocols,
lease record and output there as well. Freeze these inputs before starting.
Run from the trusted copy, setting `harness` to that copy. Running the supervisor
from inside the writable candidate defeats this separation and is unsupported.

The coding child has write access to the candidate and scratch space, read access
to declared toolchains, and no shell network access. A real sandbox canary must
confirm a candidate write and denied verification read before coding starts.
`readable_context` supplies only separate public source or installed dependencies;
it cannot include the reference outcomes, fixtures, protocol, harness or output.
If a virtual environment's Python is a symlink, include its resolved runtime
prefix as well as the virtual environment. Select `model` and `reasoning`
explicitly when particular coding settings are required.

Only candidate source, trusted capture modules and frozen reset/actions enter
the replay container. Original Git history, reference futures, acceptance
thresholds and credentials stay outside it. The Linux manager builds Warp in a
separate copy, then runs a non-root, network-disabled replay with read-only
source and isolated output. A shared lock serializes GPU work. Do not run the
older manual GPU scripts concurrently.

## Run and recover

Create a JSON configuration with absolute paths for `candidate`, `harness`,
`output`, `lease` and `python_executable`; the image must be an immutable
`sha256:` ID. Set a concrete `milestone` and positive `max_iterations`,
`max_stalled_iterations`, `iteration_timeout_s`, `worker_timeout_s`,
`wall_budget_s` and `gpu_budget_s`. The last field bounds worker elapsed time,
including queue/build/replay, rather than the provider bill. Idle instance time
still costs money and is bounded by the separate lease termination deadline.

Each item in `cases` needs a unique `id`, `split: "development"`, and absolute
`reference`, `fixture` and `protocol` paths. The protocol must be calibrated from
independent reference repeats and match the fixture and reference hashes.
Reference self-comparison must pass before coding starts. The lease supplies
`state: "active"`, `ip`, `ssh_private_key` and `terminate_at_epoch`; SSH host
records live in `lambda-known-hosts` beside the private key. Credentials and
machine-specific configuration are intentionally outside this source snapshot.

An optional `candidate_sim_backend` on each case selects `physx_cpu` or
`physx_cuda`. The loop freezes this choice in its configuration and forwards it
to the immutable replay job. Give CPU and GPU cases distinct IDs when comparing
both against the same fixture. A choice conflicting with fixture metadata is
rejected before coding starts. Backend selection cannot override a failed
independent comparison.

```sh
PYTHONPATH=/absolute/trusted/supervisor PYTHONDONTWRITEBYTECODE=1 \
  /absolute/python -P -m softbody_lab.loop /absolute/config.json

# After an interruption, use the identical config and existing output:
PYTHONPATH=/absolute/trusted/supervisor PYTHONDONTWRITEBYTECODE=1 \
  /absolute/python -P -m softbody_lab.loop /absolute/config.json --resume
```

Resume preserves the original absolute deadline; disconnected time counts.
Completed coding is not rerun. An uncertain worker launch retains its job ID
and reservation until observed. Collected worker time is charged once. A pinned
result archive can be recovered offline after lease expiry. Changed source,
trusted data, budgets or job identity stop recovery. Concurrent supervisors on
one output are rejected. An interrupted coding process without a verified
successful exit requires inspection; recovery cannot infer that it stopped.

Recovery is explicit. Keep the laptop awake for local coding and supervision;
this command does not install an automatic restart service. The tested SIGKILL
case does not establish laptop power-loss or interrupted-coding recovery.

For an individual existing job, `softbody_lab.remote_replay` provides `status`,
`resume`, `wait` and `collect`, each taking `--output /absolute/existing/job`.
Never replace a job merely because an SSH observation failed. Successful SSH
with malformed or empty JSON is also an observation failure and retries the
same handle.

Individual `remote_replay submit` and `capture` commands accept
`--candidate-sim-backend physx_cpu` or `physx_cuda`. This chooses the candidate's
PhysX execution backend without changing the MPM device. The Excavate backend
diagnostic uses CUDA MPM in both cases. The frozen physical
fixture and controls remain identical. The immutable worker request pins this
choice, and capture checks and records the actual backend after reset. A
conflicting backend already declared in a fixture is rejected. Omitting the
flag retains the previous default. This option is candidate-only and does not
change the reference simulator or define an acceptance tolerance.

For a GPU replay, the manager copies the existing PhysX GPU library from
`provenance/physx-gpu/105.1-physx-5.3.1.patch0/files/libPhysXGpu_64.so` under the
worker root. It verifies the pinned SHA-256 before and after copying, then
mounts that copy read-only in SAPIEN's cache. Missing or changed binaries fail
before building or running the candidate. No runtime download is required.

## Generate additional reference fixtures

`remote_replay.prepare_reference_demo(inputs, output, lease, image,
dependencies={"warp": WARP_SHA256, "sdf": SDF_SHA256})` uses the existing clean
ManiSkill 2 checkout pinned at `493be36121a9dd06071a57172274babe617b789f`.
Submit, observe, resume and collect the returned job exactly like a candidate
job. The shared GPU lock, timeout, container isolation and interruption handling
apply to both roles. The reference image must also be an immutable image ID.

The input directory contains `input.json`, `initial.npy` and `actions.npy`, as
exported by `demo_inputs.export_episode` from a checksummed official dataset.
Only those three files are uploaded; later demonstration states and outcomes
are excluded. Exporting from HDF5 additionally requires h5py. The supported
worker cache layouts currently cover Fill, Excavate, Hang and Pour, with explicit
Warp-library and task-SDF checksums. The manager copies and verifies both
dependencies before mounting them read-only. It cannot modify the reference
checkout or choose candidate code as the reference implementation.

On success, collection verifies the reference role, clean source commit, image,
submitted input archive and native initial-state checksum. The generated
`collected/output/fixture` must match `collected/output/trace` before candidate
use. Reference and candidate run separately; the port receives the portable
initialization, material values and controls. Keep the full reference trace in
the trusted verification directory. Reference task failure is valid evidence
and must not be filtered out of a predeclared study.

## Capture scaling and reset isolation

`remote_replay.prepare_scaling(candidate, output, lease, image, case,
timeout_s=2400, harness=trusted_copy, native_actor_extension=native_spec)`
prepares a candidate-only GPU diagnostic. Use the existing submit, observe,
resume and collect APIs on its durable output directory. It does not run the
coding loop or select an acceptance tolerance. The case has exactly four fields:
`env_id` (`Fill-v0` or `Excavate-v0`), `num_envs` (2, 8 or 32), a list of distinct
integer `seeds` of that length, and `timed_controls` (1–20). The native spec pins
an already built adapter's build directory, library checksum and C++ checksum.

Freeze the case, runtime inventory and trusted harness before submission.
The container receives source and trusted capture code; it receives no reference
trajectory or future outcomes. Capture uses the original task particle recipes,
CUDA MPM, GPU PhysX, real joint drive controls and reset APIs. Ten snapshots
cover initial state, stepping, reversed partial checkpoint restoration, fresh
partial reset, reversed full flat-state restoration and scene reconstruction.
Every environment's actual particles, complete exposed state, robot/coupling
native body rows, model inputs, random streams and elapsed counters are checked.
Native row capture must include all robot links as well as coupling bodies.

After collection, run the independent verifier from the trusted copy:

```sh
PYTHONPATH=/absolute/trusted/supervisor PYTHONDONTWRITEBYTECODE=1 \
  /absolute/python -P -m softbody_lab.scaling_checks \
  /absolute/existing/job /absolute/new-verdict.json
```

The CLI revalidates the checksummed result archive before evaluating snapshots.
It exits nonzero on any strict difference; a completed GPU process is not a
physics pass. Keep failed verdicts and original job handles. Step timing includes
observations, excludes rendering and snapshot I/O, and synchronizes both GPU
runtimes. Device usage is sampled at snapshot boundaries, so the reported maximum
is an observed value, not a continuously measured peak. Torch allocator peaks
and process high-water RSS have separate meanings. These short controls do not
establish full-task success, reference parity or large-scale training throughput.

## Local verification

From the trusted copy, with NumPy, SciPy, Pillow, h5py and pytest installed:

```sh
PYTHONPATH=/absolute/trusted/supervisor PYTHONDONTWRITEBYTECODE=1 \
  /absolute/python -P -m pytest -c /dev/null -q -p no:cacheprovider tests
```

The tests cover process interruption, uncertainty, budget/accounting failures,
changed inputs, sandbox configuration, unsafe archives, offline collection and
forged numeric outcomes. They simulate remote failures; the separate live-run
record supplies the actual GPU evidence. The first standalone test run found
three missing-import failures in outcome tests. Including the unchanged outcome
audit modules fixed the package; no test or acceptance threshold was removed.
