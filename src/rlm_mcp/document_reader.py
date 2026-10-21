"""Exact, version-bound passage reading with bounded per-REPL caches."""
from array import array
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import re
import sys
from .text_version import content_version

_PAGE_MARKER = re.compile(r"^--- P[áa]gina (\d+) ---[ \t\r]*$", re.MULTILINE)
_CACHE_MAX_BYTES = 64 * 1024 * 1024
_CACHE_MAX_DOCS = 8


@dataclass
class TextView:
    text: str
    version: str
    starts: array
    page_starts: list[int]
    page_numbers: list[int]
    size_bytes: int


def _view(repl, name: str, text: str) -> TextView:
    cache = getattr(repl, '_document_views', None)
    if cache is None:
        cache = repl._document_views = OrderedDict()
    previous = cache.get(name)
    if previous is not None and previous.text is text:
        cache.move_to_end(name)
        return previous
    starts = array('Q', [0])
    starts.extend(m.end() for m in re.finditer('\n', text))
    markers = [(m.start(), int(m[1])) for m in _PAGE_MARKER.finditer(text)]
    size = (sys.getsizeof(text) + sys.getsizeof(starts)
            + sum(sys.getsizeof(p) + sys.getsizeof(n) + 16 for p, n in markers))
    view = TextView(text, content_version(text),
                    starts, [p for p, _ in markers], [n for _, n in markers], size)
    cache.pop(name, None)
    if size <= _CACHE_MAX_BYTES:
        cache[name] = view
        while len(cache) > _CACHE_MAX_DOCS or sum(v.size_bytes for v in cache.values()) > _CACHE_MAX_BYTES:
            cache.popitem(last=False)
    return view


def _integer(args: dict, key: str, default: int, low: int, high: int) -> int:
    value = args.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"'{key}' deve ser inteiro entre {low} e {high}.")
    return value


def read_passage(repl, args: dict) -> dict:
    """Read an immutable text snapshot; continuation must match its SHA-256."""
    name = args['var_name']
    max_chars = _integer(args, 'max_chars', 12000, 1, 20000)
    line_count = _integer(args, 'line_count', 40, 1, 200)
    with repl._execute_lock:
        text = repl.variables.get(name)
        if not isinstance(text, str):
            raise ValueError(f"Variável '{name}' não existe ou não é texto.")
        view = _view(repl, name, text)
        meta = repl.variable_metadata.get(name)
        source = meta.source if meta else 'unknown'

    expected = args.get('expected_version')
    if expected is not None and expected != view.version:
        raise ValueError('A versão da fonte mudou. Refaz a busca/leitura antes de continuar a citação.')
    if 'offset' in args:
        if 'start_line' in args:
            raise ValueError('Use offset ou start_line, não ambos.')
        start = _integer(args, 'offset', 0, 0, len(text))
        if start and not expected:
            raise ValueError('Informe expected_version ao continuar a leitura por offset.')
        first = bisect_right(view.starts, start) - 1
    else:
        first = _integer(args, 'start_line', 1, 1, len(view.starts)) - 1
        start = view.starts[first]
    last = first + line_count
    range_end = view.starts[last] if last < len(view.starts) else len(text)
    end = min(range_end, start + max_chars)
    pages = []
    if end > start and view.page_starts:
        page_idx = max(0, bisect_right(view.page_starts, start) - 1)
        while page_idx < len(view.page_starts) and view.page_starts[page_idx] < end:
            number = view.page_numbers[page_idx]
            if number not in pages:
                pages.append(number)
            page_idx += 1
    return {
        'var_name': name, 'source': source, 'version': view.version,
        'line_start': first + 1,
        'line_end': bisect_right(view.starts, end - 1) if end > start else first + 1,
        'pages': pages, 'offset': start, 'end_offset': end, 'total_chars': len(text),
        'text': text[start:end],
        'next': {'var_name': name, 'offset': end, 'expected_version': view.version}
                if end < len(text) else None,
    }
