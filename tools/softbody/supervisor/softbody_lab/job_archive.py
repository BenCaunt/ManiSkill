"""Bounded, numeric/data-only job transport; never follow archive links."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_members(archive, *, max_bytes=4*1024**3, max_members=100000):
    members = archive.getmembers()
    if len(members) > max_members or sum(m.size for m in members) > max_bytes:
        raise ValueError('Archive exceeds declared extraction budget')
    names = set()
    for member in members:
        name = PurePosixPath(member.name)
        if (not member.isfile() or name.is_absolute() or '..' in name.parts
                or '\\' in member.name or member.name != name.as_posix()
                or member.name in names or not name.parts):
            raise ValueError('Archive contains a link, duplicate or unsafe path')
        names.add(member.name)
    if any(parent.as_posix() in names for name in names for parent in PurePosixPath(name).parents):
        raise ValueError('Archive file shadows a parent directory')
    return members


def archive_inventory(path, *, max_bytes=4*1024**3, max_members=100000):
    """Read hashes directly from a bounded archive without trusting extracted files."""
    result = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in checked_members(archive, max_bytes=max_bytes, max_members=max_members):
            digest = hashlib.sha256()
            with archive.extractfile(member) as source:
                while chunk := source.read(1024**2):
                    digest.update(chunk)
            result[member.name] = digest.hexdigest()
    return result


def safe_extract(path, destination, *, max_bytes=4*1024**3, max_members=100000):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Archive destination must be new')
    with tarfile.open(path, 'r:gz') as archive:
        members = checked_members(archive, max_bytes=max_bytes, max_members=max_members)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.'+destination.name+'.extract-', dir=destination.parent))
        try:
            for member in members:
                target = temporary/member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as output:
                    while chunk := source.read(1024**2):
                        output.write(chunk)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            if destination.exists() or destination.is_symlink():
                raise ValueError('Archive destination appeared during extraction')
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Job trees cannot contain symlinks')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_hash(path)
        elif not path.is_dir():
            raise ValueError('Job trees can contain only regular files')
    return result


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)
