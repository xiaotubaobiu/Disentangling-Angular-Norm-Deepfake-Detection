"""Atomic PNG publication for persistent H workers; pixel geometry is unchanged."""
import os
from pathlib import Path

import cv2

_native_imwrite=cv2.imwrite


def atomic_imwrite(filename,image,params=None):
    path=Path(filename)
    temporary=path.with_name(f'.{path.stem}.{os.getpid()}.pending{path.suffix}')
    try:
        ok=_native_imwrite(str(temporary),image,params or [])
        if ok: os.replace(temporary,path)
        return ok
    finally:
        temporary.unlink(missing_ok=True)
