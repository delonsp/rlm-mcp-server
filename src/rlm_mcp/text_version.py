"""Full content versions, cached by immutable string identity with a memory cap."""
from collections import OrderedDict
import hashlib
import sys
from threading import Lock

_cache: OrderedDict[int, tuple[str, str, int]] = OrderedDict()
_lock = Lock()
_MAX_BYTES = 64 * 1024 * 1024
_MAX_ENTRIES = 16


def forget_content_version(text: str) -> None:
    """Release a discarded source instead of retaining it until cache eviction."""
    with _lock:
        _cache.pop(id(text), None)


def clear_content_versions() -> None:
    with _lock:
        _cache.clear()


def content_version(text: str) -> str:
    """Hash every character once; never mistake equal-size sampled text for equal content."""
    key = id(text)
    with _lock:
        found = _cache.get(key)
        if found is not None and found[0] is text:
            _cache.move_to_end(key)
            return found[1]
        version = hashlib.sha256(text.encode('utf-8', 'surrogatepass')).hexdigest()
        size = sys.getsizeof(text)
        if size <= _MAX_BYTES:
            _cache[key] = (text, version, size)
            while len(_cache) > _MAX_ENTRIES or sum(v[2] for v in _cache.values()) > _MAX_BYTES:
                _cache.popitem(last=False)
        return version
