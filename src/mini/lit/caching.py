"""
A cache for the expensive calls a document makes: figures, fits, anything slow.

``@memo`` keys a call the way :mod:`mini.memo` keys a task — the function's qualified name plus a fingerprint of its inputs is the identity, and a fingerprint of its source (and the project code it references, transitively) is the validity evidence — so editing the plot function re-renders the figure, and editing the prose around it does not. The value is pickled under ``.mini/lit-cache/`` (or ``$MINI_CACHE_DIR/lit-cache/``); a hit is served from memory within a process and from disk across processes, which is what makes a fresh ``render`` of a figure-heavy report take well under a second.

A memoized function that writes assets through the current :class:`~mini.reports.Publisher` (a ``themed`` figure writes two PNGs) has those files recorded with its value, and the hit is honoured only while they exist. The cache keeps a copy of those files too, so a hit in a fresh output directory (another checkout, another container sharing the cache) restores them rather than redrawing; with neither copy, the call re-draws, and a stale cache can never point at a missing image.

Inputs need a stable encoding. Plain data, dataclasses, and NumPy arrays are handled (an array is hashed by its bytes); an object whose ``repr`` carries a memory address makes the call miss every time, and :mod:`mini.memo` logs a warning when that happens. Digesting a large input costs time on every call (about half a second for a 10 MB metrics dict), so a results object assembled from published artifacts can define ``__memo_key__()`` returning those artifacts' hashes, and is then keyed by them instead.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import os
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar, overload

from mini.memo import _builtin_name, _is_project_source, _value_json, reachable_values, task_key_parts
from mini.reports import current_publisher

__all__ = ["memo", "cache_dir", "set_cache_dir"]

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_cache_dir: Path | None = None
_hot: dict[tuple[str, str], Any] = {}


def cache_dir() -> Path:
    if _cache_dir is None:
        from mini.runs import cache_root

        return cache_root() / "lit-cache"
    return _cache_dir


def set_cache_dir(path: Path | str | None) -> None:
    global _cache_dir
    _cache_dir = None if path is None else Path(path)
    _hot.clear()


@functools.cache
def _values_fp(fn: Callable) -> str:
    """Evidence on top of the task fingerprint: every plain value *fn* reads, arrays included.

    Two things the task fingerprint leaves out matter here. A report reads its design through ``import experiment as ex`` and then ``ex.GATE``, which that fingerprint does not see; and it keeps its data in module-level arrays, which that fingerprint skips for want of a JSON encoding (:func:`mini.memo.reachable_values` says why both stay out of task records). Here a spurious miss redraws a figure, so both go in, arrays hashed by content like the inputs are: editing a gate re-renders what quotes it, and a figure that reads a global array re-draws when the array changes. What still cannot be encoded — a model, a store — is warned about once, since a figure that reads it would otherwise be served stale.
    """
    tracked, untracked = reachable_values(fn, lambda v: _value_json(_prepare(v), default=_builtin_name))
    # Data the figure could be stale against: a container, or an instance of the project's own classes (a
    # loaded model, a results bundle). A logger or a store handle read from a helper is neither.
    untracked = {
        n: v for n, v in untracked.items() if isinstance(v, (dict, list, tuple, set)) or _is_project_source(type(v))
    }
    if untracked:
        names = ", ".join(f"{n} ({type(v).__name__})" for n, v in sorted(untracked.items()))
        log.warning(
            "lit.caching: %s reads %s, which the cache cannot fingerprint; a hit would outlive a change to it. "
            "Pass it as an argument instead.",
            getattr(fn, "__qualname__", fn),
            names,
        )
    blob = "\n".join(f"{k}={v}" for k, v in sorted(tracked.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _prepare(o: Any) -> Any:
    """Replace the inputs :func:`mini.memo.task_key_parts` cannot encode stably with digests it can.

    An object with a ``__memo_key__()`` method is encoded as what it returns and never walked: a results object built from published artifacts answers with their hashes, which key its content the way an ``Artifact`` argument keys a task's, for a few bytes instead of a digest of every array and run it holds.
    """
    if (key := getattr(o, "__memo_key__", None)) is not None and callable(key):
        return ["memo_key", type(o).__qualname__, _prepare(key())]
    mod = type(o).__module__
    if mod.startswith("numpy") and hasattr(o, "tobytes"):
        return ["ndarray", str(o.dtype), list(o.shape), hashlib.sha256(o.tobytes()).hexdigest()[:16]]
    if mod.startswith("pandas") and hasattr(o, "to_numpy"):
        import pandas as pd

        h = pd.util.hash_pandas_object(o, index=True).to_numpy()
        return ["pandas", type(o).__name__, hashlib.sha256(h.tobytes()).hexdigest()[:16]]
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {"__dataclass__": type(o).__qualname__} | {
            f.name: _prepare(getattr(o, f.name)) for f in dataclasses.fields(o)
        }
    if isinstance(o, dict):
        return {str(k): _prepare(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_prepare(x) for x in o]
    return o


@overload
def memo(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def memo(*, version: str | None = ...) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def memo(fn: Callable[P, R] | None = None, /, *, version: str | None = None) -> Any:
    """Memoize *fn* on disk, keyed by its source and inputs (see the module docstring).

    ``version=`` is an explicit invalidation lever for a change the source fingerprint cannot see (new data under an unchanged ref name).
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key, parts = task_key_parts(fn, (_prepare(args), _prepare(kwargs)), version)
            evidence = f"{parts['code_fp']}:{_values_fp(fn)}:{parts.get('version', '')}"
            pub = current_publisher()
            asset_dir = pub.asset_dir if pub is not None else None

            if (hit := _hot.get((key, evidence))) is not None and _assets_present(hit["assets"], asset_dir):
                return hit["value"]
            path = cache_dir() / f"{key}.pkl"
            if path.exists():
                try:
                    rec = pickle.loads(path.read_bytes())
                except Exception:
                    rec = None
                if (
                    rec
                    and rec.get("evidence") == evidence
                    and (
                        _assets_present(rec["assets"], asset_dir)
                        or _restore_assets(rec["assets"], _asset_copies(key, evidence), asset_dir)
                    )
                ):
                    _hot[key, evidence] = rec
                    return rec["value"]

            mark = len(pub.log) if pub is not None else 0
            value = fn(*args, **kwargs)
            assets = pub.log[mark:] if pub is not None else []
            rec = {"evidence": evidence, "value": value, "assets": assets, "deps": parts["deps"]}
            _hot[key, evidence] = rec
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if asset_dir is not None:
                    _keep_assets(assets, asset_dir, _asset_copies(key, evidence))
                tmp = path.with_suffix(_tmp_suffix())  # two renders may cache the same key at once
                tmp.write_bytes(pickle.dumps(rec))
                tmp.replace(path)
            except Exception as e:  # an unpicklable value still returns; it just isn't cached across processes
                log.warning("lit.caching: %s not cached to disk: %s", getattr(fn, "__qualname__", fn), e)
            return value

        return wrapper

    return decorate(fn) if fn is not None else decorate


def _assets_present(names: list[str], asset_dir: Path | None) -> bool:
    if not names:
        return True
    return asset_dir is not None and all((asset_dir / n).exists() for n in names)


def _tmp_suffix() -> str:
    # Unique per writer: the cache may be shared by several containers, whose pids can coincide.
    return f".{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp"


def _asset_copies(key: str, evidence: str) -> Path:
    """Where the cache keeps the assets of one record: by key *and* evidence, so a copy never outlives the value it was drawn with."""
    return cache_dir() / "assets" / f"{key}-{hashlib.sha256(evidence.encode()).hexdigest()[:12]}"


def _keep_assets(names: list[str], asset_dir: Path, copies: Path) -> None:
    for n in names:
        (copies / n).parent.mkdir(parents=True, exist_ok=True)
        tmp = copies / f"{n}{_tmp_suffix()}"
        shutil.copyfile(asset_dir / n, tmp)
        tmp.replace(copies / n)


def _restore_assets(names: list[str], copies: Path, asset_dir: Path | None) -> bool:
    """Copy a record's assets from the cache into *asset_dir*, returning whether all of them were there."""
    if asset_dir is None or not all((copies / n).is_file() for n in names):
        return False
    for n in names:
        (asset_dir / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(copies / n, asset_dir / n)
    return True
