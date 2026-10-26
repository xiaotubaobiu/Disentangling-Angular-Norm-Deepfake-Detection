"""Finite throttled copy of closed D evidence. Never removes or replaces a source."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone

R = Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
COLD = Path('/Dataset/deepfake_detector_v2_cold_20261023/staged_copies')
PROGRESS = R / 'reports/D_closed_cold_copy_progress_20261024T1510.json'
MANIFEST = R / 'reports/D_closed_cold_copy_manifest_20261024T1510.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


state = dict(pid=os.getpid(), started_utc=now(), status='STARTING', source_originals_retained=True,
             SSD_released_bytes=0, actual_relocation_completed=False, bandwidth_limit_KiB_s=16384)


def update(status, **fields):
    state.update(status=status, utc=now(), **fields)
    save(PROGRESS, state)


with (R / 'D_closed_cold_copy.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert not MANIFEST.exists()
        assert shutil.which('rsync')
        config = json.loads((R / 'configs/execution.json').read_text())
        assert not Path('/proc/3942476').exists()
        sources = []
        for run, task in config['runs'].items():
            assert run.startswith('D_') and (R / 'runs' / run / 'complete.json').is_file()
            folder = R / 'runs' / run
            sources.append(folder / 'state.pt')
            sources.extend(sorted(folder.glob('epoch_*_methods.npy')))
        allowed_qualification_roots = {
            'clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0', 'archive',
            'pipeline_resume_clip_vit_b16', 'pipeline_resume_dinov2_vit_b14',
            'pipeline_resume_efficientnet_b0',
        }
        for p in (R / 'qualification').rglob('*.pt'):
            relative = p.relative_to(R / 'qualification')
            if relative.parts[0] in allowed_qualification_roots:
                sources.append(p)
        sources = sorted(set(sources))
        total = sum(p.stat().st_size for p in sources)
        assert shutil.disk_usage('/Dataset').free > total + 512 * 1024**3
        entries = []
        update('HASHING_CLOSED_SOURCE_INVENTORY', file_count=len(sources), total_bytes=total, hashed_files=0)
        for index, source in enumerate(sources):
            assert source.is_file() and not source.is_symlink()
            assert source.resolve().is_relative_to(R.resolve())
            relative = source.relative_to(R)
            assert source.name not in ('final.pt', 'source_best.pt')
            stat = source.stat()
            digest = sha(source)
            target = COLD / relative
            assert target.parent.resolve().is_relative_to(COLD.resolve())
            if target.exists():
                assert target.is_file() and not target.is_symlink() and sha(target) == digest
            entries.append(dict(relative=str(relative), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                source_inode=stat.st_ino, sha256=digest, staged_path=str(target)))
            update('HASHING_CLOSED_SOURCE_INVENTORY', hashed_files=index + 1, current_file=str(relative))
        manifest = dict(utc=now(), status='SOURCE_SHA_INVENTORY_COMPLETE_COPY_PENDING', entries=entries,
                        total_bytes=total, file_count=len(entries), source_originals_retained=True,
                        excluded=['all final.pt', 'all source_best.pt', 'all H files', 'live caches/manifests/code'])
        save(MANIFEST, manifest)
        filelist = R / 'reports/D_closed_cold_copy_files_20261024T1510.nul'
        filelist.write_bytes(b''.join(e['relative'].encode() + b'\0' for e in entries))
        COLD.mkdir(parents=True, exist_ok=True)
        log = R / 'reports/D_closed_cold_copy_rsync_20261024T1510.log'
        command = ['/usr/bin/rsync', '-rt', '--checksum', '--relative', '--from0',
                   '--files-from=' + str(filelist), '--bwlimit=16384', '--partial-dir=.rsync-partial',
                   '--info=progress2', str(R) + '/', str(COLD) + '/']
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            update('RSYNC_COPY_ONLY_RUNNING', rsync_pid=child.pid, command=command,
                   manifest_sha256=sha(MANIFEST), copy_log=str(log), estimated_transfer_seconds=total/(16*1024**2))
            rc = child.wait()
        assert rc == 0, f'rsync returned {rc}; sources untouched; retain partial staging for inspection'
        update('VERIFYING_BOTH_ORIGINAL_AND_STAGED_COPY_SHA', rsync_returncode=rc, verified_files=0)
        for index, entry in enumerate(entries):
            source, target = R / entry['relative'], Path(entry['staged_path'])
            stat = source.stat()
            assert not source.is_symlink() and stat.st_ino == entry['source_inode']
            assert stat.st_size == entry['bytes'] and stat.st_mtime_ns == entry['mtime_ns']
            assert sha(source) == sha(target) == entry['sha256']
            update('VERIFYING_BOTH_ORIGINAL_AND_STAGED_COPY_SHA', verified_files=index + 1,
                   current_file=entry['relative'])
        manifest.update(status='ALL_CLOSED_D_COPIES_SHA_VERIFIED_SOURCE_ORIGINALS_RETAINED',
                        verified_utc=now(), SSD_released_bytes=0, actual_relocation_completed=False)
        save(MANIFEST, manifest)
        update('ALL_CLOSED_D_COPIES_SHA_VERIFIED_SOURCE_ORIGINALS_RETAINED',
               verified_files=len(entries), manifest_sha256=sha(MANIFEST), ended_utc=now())
        print(json.dumps(state, ensure_ascii=False), flush=True)
    except BaseException:
        update('REVIEW_REQUIRED_SOURCES_NOT_REMOVED', traceback=traceback.format_exc(), ended_utc=now())
        traceback.print_exc()
        raise
