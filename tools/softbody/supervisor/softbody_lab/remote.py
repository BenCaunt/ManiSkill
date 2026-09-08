"""SSH access to the single Lambda worker recorded in a local lease."""

import argparse
import ipaddress
import json
import shlex
import subprocess
import time
from pathlib import Path


def connection(lease_path):
    lease = json.loads(lease_path.read_text())
    if time.time() >= lease["terminate_at_epoch"]:
        raise ValueError("GPU lease has expired")
    address = str(ipaddress.ip_address(lease["ip"]))
    if lease["state"] not in ("active", "booting"):
        raise ValueError("GPU is not running or booting; refresh lease status")
    known_hosts = Path(lease["ssh_private_key"]).parent / "lambda-known-hosts"
    options = ["-i", lease["ssh_private_key"], "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
               "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}",
               "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
    return options, "ubuntu@" + address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease", type=Path, default=Path("artifacts/softbody/lambda/lease.json"))
    parser.add_argument("--timeout", type=int, default=120)
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("run")
    run.add_argument("script", type=Path, help="Local shell script; transmitted over SSH stdin")
    run.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments passed to the remote shell script")
    copy = subs.add_parser("put")
    copy.add_argument("source", type=Path)
    copy.add_argument("destination", help="Absolute path on worker")
    fetch = subs.add_parser("get")
    fetch.add_argument("source", help="Absolute path on worker")
    fetch.add_argument("destination", type=Path)
    args = parser.parse_args()
    options, target = connection(args.lease)
    if args.command == "run":
        remote_command = shlex.join(["bash", "-seu", "--", *args.arguments])
        result = subprocess.run(["ssh", *options, target, remote_command],
                                input=args.script.read_bytes(), timeout=args.timeout)
    else:
        remote_path = args.destination if args.command == "put" else args.source
        if not remote_path.startswith("/") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-" for c in remote_path):
            raise ValueError("Remote path must be an absolute path without shell metacharacters")
        paths = [str(args.source.resolve()), target + ":" + remote_path] if args.command == "put" else [target + ":" + remote_path, str(args.destination.resolve())]
        result = subprocess.run(["scp", *options, "--", *paths], timeout=args.timeout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
