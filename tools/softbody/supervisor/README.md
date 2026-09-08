# Local coding and remote verification

This directory contains the local Codex supervisor, Linux job manager, capture
code and independent numeric evaluator used for the experimental port. The
supervisor writes a durable checkpoint before remote submission. After a local
process interruption, `--resume` recovers the same job and the original budget.
An implementation agent's successful exit cannot override a failing comparison.

The [live recovery record](../supervisor-recovery.md) documents a real local
SIGKILL while the existing Lambda worker was replaying Fill. The packaged code
passes 130 local tests and reproduces that run's saved comparison exactly,
including both physics failures. The [source manifest](source-manifest.json)
identifies the bytes used during the live test and the subsequent transport fix.

## Requirements and separation

The tested local setup uses macOS, Python 3.12, NumPy, Git, SSH/SCP and an
authenticated Codex CLI with named filesystem permissions. The CLI options are
documented in OpenAI's [non-interactive guide](https://learn.chatgpt.com/docs/non-interactive-mode).
The Linux worker already has Docker, the NVIDIA container runtime, a pinned
candidate image and the external reference assets. `gpu_job.py` records the
tested `/home/ubuntu/softbody` layout. This code uses an existing lease; it does
not provision a machine or install the independent provider termination timer.

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

## Local verification

From the trusted copy, with NumPy and pytest installed:

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
