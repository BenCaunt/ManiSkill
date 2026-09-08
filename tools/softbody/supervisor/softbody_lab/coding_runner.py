"""Codex command profile for local implementation, separate from verification.

Uses the installed CLI's named filesystem permissions. No bypass flags and no
permission escalation: denied commands fail back to the implementation agent.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess


def config_args(candidate, *, readable=()):
    candidate = Path(candidate).resolve()
    filesystem = {':root': 'deny', ':minimal': 'read', ':tmpdir': 'write',
                  ':slash_tmp': 'deny', str(candidate): 'write'}
    # macOS's minimal profile covers system tools, but not Homebrew or a
    # separately installed Node runtime. These are read-only toolchains.
    if Path('/opt/homebrew').is_dir():
        filesystem['/opt/homebrew'] = 'read'
    if Path('/Library/Developer/CommandLineTools').is_dir():
        filesystem['/Library/Developer/CommandLineTools'] = 'read'
    node = shutil.which('node')
    if node:
        filesystem[str(Path(node).resolve().parent.parent)] = 'read'
    for path in readable:
        path = Path(path).resolve()
        if path == candidate or path in candidate.parents or candidate in path.parents:
            raise ValueError('Read-only context must be separate from implementation checkout')
        filesystem[str(path)] = 'read'
    settings = {'default_permissions': 'softbody', 'approval_policy': 'never', 'web_search': 'disabled',
                'permissions': {'softbody': {'extends': ':workspace', 'filesystem': filesystem,
                                             'network': {'enabled': False}}}}
    def toml(value):
        if isinstance(value, dict):
            return '{'+', '.join(json.dumps(k)+' = '+toml(v) for k,v in value.items())+'}'
        return json.dumps(value)
    result = []
    for key, value in settings.items():
        result += ['-c', key+'='+toml(value)]
    return result


def command(candidate, *, readable=(), model=None, reasoning=None):
    result = ['codex', 'exec', '--ignore-user-config', '--strict-config', '--ephemeral', '--json',
              '--cd', str(Path(candidate).resolve()), *config_args(candidate, readable=readable)]
    if model:
        result += ['--model', model]
    if reasoning:
        result += ['-c', 'model_reasoning_effort='+json.dumps(reasoning)]
    return [*result, '-']


def environment(candidate):
    # Keep account auth in its normal local store, but do not forward cloud or
    # API credentials from the supervisor's process environment to child tools.
    keep = ('PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'LANG', 'LC_ALL', 'CODEX_HOME')
    result = {key: os.environ[key] for key in keep if key in os.environ}
    git_tools = Path('/Library/Developer/CommandLineTools/usr/bin')
    if (git_tools/'git').is_file():
        result['PATH'] = str(git_tools)+os.pathsep+result.get('PATH', '/usr/bin:/bin')
    result.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null', GIT_TERMINAL_PROMPT='0')
    temporary = Path(candidate).resolve()/'.softbody-tmp'
    temporary.mkdir(exist_ok=True)
    result['TMPDIR'] = str(temporary)
    return result


def permission_probe(candidate, forbidden, output, *, readable=(), python_executable=None, runtime_modules=()):
    """Exercise the real CLI sandbox using an innocuous canary, not secrets."""
    candidate, forbidden = Path(candidate).resolve(), Path(forbidden).resolve()
    output = Path(output)
    canary = forbidden/'permission-canary.txt'
    canary.write_text('verification data must be inaccessible to the coding worker\n')
    code = '''import importlib,json,pathlib,subprocess,sys
candidate=pathlib.Path(sys.argv[1]); forbidden=pathlib.Path(sys.argv[2])
allowed=candidate/'.softbody-permission-probe'
allowed.write_text('temporary sandbox write')
assert allowed.read_text()=='temporary sandbox write'
allowed.unlink()
try: forbidden.read_bytes()
except PermissionError: denied=True
else: denied=False
assert denied, 'Coding worker could read verification data'
modules=json.loads(sys.argv[3])
for module in modules: importlib.import_module(module)
git_result=subprocess.run(['git','--version'],capture_output=True,text=True)
if git_result.returncode: raise RuntimeError('Git preflight: '+git_result.stderr)
git=git_result.stdout.strip()
print(json.dumps({'candidate_write':True,'verification_read_denied':denied,
                 'python_executable':sys.executable,'runtime_modules':modules,'git_version':git}))
'''
    executable = str(python_executable or shutil.which('python3'))
    args = ['codex', 'sandbox', '--cd', str(candidate), '-P', 'softbody',
            *config_args(candidate, readable=readable), '--', executable, '-c', code,
            str(candidate), str(canary), json.dumps(list(runtime_modules))]
    result = subprocess.run(args, env=environment(candidate), capture_output=True, text=True, timeout=45)
    output.write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('Coding permission probe failed; inspect '+str(output))
    record = json.loads(result.stdout)
    if (record.get('candidate_write') is not True or record.get('verification_read_denied') is not True
            or record.get('runtime_modules') != list(runtime_modules) or not record.get('git_version')):
        raise RuntimeError('Unexpected coding permission probe result')
    return record
