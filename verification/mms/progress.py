"""Small flushed progress/timing helpers for long verification operations."""

from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import count
import sys
from threading import Event, Lock, Thread
from time import perf_counter

_ids = count(1)
_id_lock = Lock()


def log(message):
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


@contextmanager
def operation(stage, heartbeat_s=10.0, detail=""):
    """Log timed operation boundaries and optionally emit a daemon heartbeat."""
    with _id_lock:
        stage_id = next(_ids)
    started = perf_counter()
    suffix = f" | {detail}" if detail else ""
    log(f"START | id={stage_id} | stage={stage}{suffix}")
    stop = Event()

    def beat():
        while not stop.wait(heartbeat_s):
            elapsed = perf_counter() - started
            log(f"HEARTBEAT | id={stage_id} | stage={stage} | elapsed={elapsed:.1f}s | operation still running")

    thread = None
    if heartbeat_s and heartbeat_s > 0:
        thread = Thread(target=beat, name=f"mms-heartbeat-{stage_id}", daemon=True)
        thread.start()
    try:
        yield
    except BaseException as exc:
        elapsed = perf_counter() - started
        log(f"FAIL | id={stage_id} | stage={stage} | elapsed={elapsed:.3f}s | {type(exc).__name__}: {exc}")
        raise
    else:
        elapsed = perf_counter() - started
        log(f"DONE | id={stage_id} | stage={stage} | elapsed={elapsed:.3f}s")
    finally:
        stop.set()
        if thread is not None:
            thread.join()


def flush_streams():
    sys.stdout.flush()
    sys.stderr.flush()
