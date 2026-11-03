"""Transaction-local SHA-256 reuse with metadata guards and a full final reread."""
from contextlib import contextmanager
from pathlib import Path


def signature(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class DigestTransaction:
    def __init__(self, reader, min_bytes=128 * 1024**2):
        self.reader, self.min_bytes = reader, min_bytes
        self.entries = {}
        self.hits = 0
        self.sealed = False

    def digest(self, path):
        if self.sealed:
            raise ValueError('digest transaction already sealed')
        path = Path(path).absolute()
        before = signature(path)
        if before[2] < self.min_bytes:
            return self.reader(path)
        if str(path) in self.entries:
            stored = self.entries[str(path)]
            if before != stored['signature']:
                raise ValueError('large digest input changed inside transaction')
            self.hits += 1
            return stored['sha256']
        value = self.reader(path)
        if signature(path) != before:
            raise ValueError('large digest input changed while hashing')
        self.entries[str(path)] = dict(signature=before, sha256=value, bytes=before[2])
        return value

    def seal(self):
        if self.sealed:
            raise ValueError('digest transaction already sealed')
        for name, entry in self.entries.items():
            if signature(name) != entry['signature']:
                raise ValueError('large digest input changed before final reread')
            value = self.reader(name)
            if value != entry['sha256'] or signature(name) != entry['signature']:
                raise ValueError('large digest input changed during final reread')
        self.sealed = True


@contextmanager
def guarded_digests(bindings, min_bytes=128 * 1024**2):
    """Temporarily bind equivalent file-digest functions; restore on all exits.

    Small metadata/code files still use the original hash implementation on
    every request. Each large file is fully hashed on first use and reread in
    full before success; device/inode/size/mtime/ctime guard every reuse.
    """
    original = [(owner, name, getattr(owner, name)) for owner, name in bindings]
    memo = DigestTransaction(original[0][2], min_bytes=min_bytes)
    try:
        for owner, name, _ in original:
            setattr(owner, name, memo.digest)
        yield memo
        memo.seal()
    finally:
        for owner, name, value in original:
            setattr(owner, name, value)
