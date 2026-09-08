# Live supervisor recovery

The local supervisor recovered a real remote Fill replay after its process was
killed. The recovered comparison retains both strict initial-pose failures.
This establishes process recovery for the tested job, not reference physics
parity or completion of the soft-body port.

The run used the published runtime at `71696fd5b0e73983ff28293478877271b10e6610`,
an independent writable coding checkout and a frozen external harness. One
bounded coding run produced the installed-package diagnostic in 473.47 seconds.
The sandbox canary confirmed candidate writes and denied verification reads.
No runtime source changed. A single NVIDIA A10 worker rebuilt Warp and replayed
the existing portable Fill fixture in 66.37 seconds.

While Docker reported the actual replay running, the observer sent SIGKILL to
local supervisor PID 70747. The remote manager remained live as PID 215422 with
the same process birth identifier. A new local process invoked the production
`--resume` path, collected job `c0b96d26adb44e3681c6e115eee780cf`, and finished
with the expected `iteration_budget_exhausted` result.

The recovery checks confirmed one coding run and one new remote job, unchanged
coding log and exit record, the original loop and worker deadlines, the same
remote PID/birth/start, complete result collection and exactly one 66.368353 s
worker charge. No reservation remained. A separate post-run audit verified the
frozen harness, reference/fixture/protocol inventory, source fingerprint and
result archive, then reproduced the complete physics verdict. The final worker
inventory had no running containers, and the independent lease timer was active.

The physics comparison still fails:

| Gate | Measured error | Original limit |
| --- | ---: | ---: |
| Initial derived position | 3.5762786865234375e-7 m | 1e-8 m |
| Initial quaternion component | 1.7881393432617188e-7 | 1e-8 |

All trajectory limits in this short fixture passed. Those results do not replace
the longer task demonstrations or their stricter failing comparisons.

## Retained failures and scope

The first external observer queried the manager during its initial upload. SSH
exited successfully with empty output, and the observer stopped on a JSON parse
error. The supervisor and its remote job continued. A continuation observer
verified those same identities before injecting the crash. Thus this is a
successful live crash/resume test with a corrected observer, not a flawless
end-to-end fault-driver run. Both observer logs and source hashes are retained.

The host transport was subsequently changed to treat malformed/empty JSON as an
observation failure. Three regression cases verify that polling retries the same
job; the combined supervisor/remote-job suite passes 64 tests. The live test used
the earlier frozen transport. The [standalone supervisor](supervisor/README.md)
includes the later fix and passes 130 local tests; its evaluator reproduces the
saved live verdict exactly. A new live run of the corrected transport has not
been performed.

The coding child's first diagnostic passed its 12 synthetic tests but reported
two false missing-file errors against the actual native wheel. Those files are
build inputs excluded from installed package data. The corrected inspector
reports their hashes separately, checks 495 installed native/source files,
recognizes the source-only installation, and passes 13 focused tests. The
foreign Linux ABI is reported without loading native code on macOS. No physics
or wheel-format changes were made.

This experiment covers a local supervisor process crash during remote replay
followed by explicit resume. It does not test laptop reboot/power loss, a crash
during coding, automatic restart, multi-iteration improvement or held-out
acceptance. Original Pinch/Write benchmark level archives also remain unavailable
from their pinned source in the latest retrieval attempt; authored diagnostic
goals do not substitute for those benchmark levels.

Exact checks, measurements, source identities and artifact hashes are in the
[recovery result manifest](verification/supervisor-recovery-results.json).
Local raw evidence is retained under
`artifacts/softbody/supervisor-live-resume-v1` in the development workspace.
