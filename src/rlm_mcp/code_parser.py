"""
Tree-sitter based code parser for RLM MCP Server.

Parses source code files into structured representations (functions, classes,
methods, imports) enabling structural search across code loaded into the REPL.

Supported languages: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++.
Gracefully degrades if a language grammar is not installed.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("rlm-mcp.code-parser")


@dataclass
class CodeSymbol:
    """A symbol extracted from source code."""
    name: str
    kind: str  # function, class, method, import, variable
    line_start: int
    line_end: int
    signature: str
    docstring: Optional[str] = None
    parent: Optional[str] = None


@dataclass
class CodeStructure:
    """Parsed structure of a source code file."""
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    total_lines: int = 0

    def search(
        self,
        query: Optional[str] = None,
        kind: Optional[str] = None,
        include_source: bool = False,
        source_code: Optional[str] = None,
    ) -> list[dict]:
        """Search symbols by name query and/or kind filter.

        Args:
            query: Substring to match in symbol name (case-insensitive)
            kind: Filter by symbol kind (function, class, method, import, variable)
            include_source: If True and source_code provided, include source lines
            source_code: The original source code text

        Returns:
            List of matching symbol dicts
        """
        results = []
        source_lines = source_code.split("\n") if source_code else None

        for sym in self.symbols:
            if kind and sym.kind != kind:
                continue
            if query and query.lower() not in sym.name.lower():
                continue

            entry = {
                "name": sym.name,
                "kind": sym.kind,
                "line_start": sym.line_start,
                "line_end": sym.line_end,
                "signature": sym.signature,
                "parent": sym.parent,
            }
            if sym.docstring:
                entry["docstring"] = sym.docstring

            if include_source and source_lines:
                start = max(0, sym.line_start - 1)
                end = min(len(source_lines), sym.line_end)
                entry["source"] = "\n".join(source_lines[start:end])

            results.append(entry)

        return results


# Extension -> language mapping
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

SUPPORTED_LANGUAGES = {"python", "javascript", "typescript", "go", "rust", "java", "c", "cpp"}

# Grammar module mapping: language -> pip package import name
_GRAMMAR_MODULES: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
}

# Cache for loaded Language objects
_language_cache: dict[str, object] = {}


def _get_language(lang: str):
    """Load a tree-sitter Language object, with caching.

    Returns None if the grammar is not installed.
    """
    if lang in _language_cache:
        return _language_cache[lang]

    module_name = _GRAMMAR_MODULES.get(lang)
    if not module_name:
        return None

    try:
        import importlib
        from tree_sitter import Language

        mod = importlib.import_module(module_name)
        # tree-sitter-typescript exposes typescript() and tsx()
        if lang == "typescript" and hasattr(mod, "language_typescript"):
            language_obj = Language(mod.language_typescript())
        elif hasattr(mod, "language"):
            language_obj = Language(mod.language())
        else:
            logger.warning(f"Grammar module {module_name} has no language() function")
            return None

        _language_cache[lang] = language_obj
        return language_obj
    except ImportError:
        logger.info(f"Grammar not installed for {lang} (pip install {module_name})")
        return None
    except Exception as e:
        logger.warning(f"Failed to load grammar for {lang}: {e}")
        return None


def detect_language(filename: str, content: Optional[str] = None) -> Optional[str]:
    """Detect programming language from filename extension.

    Args:
        filename: File name or path
        content: Optional content for heuristic detection (shebang, etc.)

    Returns:
        Language name or None if not detected
    """
    import os
    _, ext = os.path.splitext(filename.lower())

    if ext in EXTENSION_MAP:
        return EXTENSION_MAP[ext]

    # Shebang detection for extensionless files
    if content:
        first_line = content.split("\n", 1)[0].strip()
        if first_line.startswith("#!"):
            if "python" in first_line:
                return "python"
            if "node" in first_line:
                return "javascript"

    return None


def _extract_text(node, source_bytes: bytes) -> str:
    """Extract text from a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_docstring(node, source_bytes: bytes, language: str) -> Optional[str]:
    """Extract docstring from a function/class node if present."""
    if language == "python":
        # Python: first child expression_statement with a string
        body = None
        for child in node.children:
            if child.type == "block":
                body = child
                break
        if body and body.child_count > 0:
            first_stmt = body.children[0]
            if first_stmt.type == "expression_statement" and first_stmt.child_count > 0:
                expr = first_stmt.children[0]
                if expr.type == "string":
                    doc = _extract_text(expr, source_bytes)
                    # Strip triple quotes
                    for q in ('"""', "'''"):
                        if doc.startswith(q) and doc.endswith(q):
                            return doc[3:-3].strip()
                    return doc.strip("\"'").strip()

    elif language in ("javascript", "typescript"):
        # JS/TS: look for preceding comment node
        prev = node.prev_named_sibling
        if prev and prev.type == "comment":
            text = _extract_text(prev, source_bytes)
            if text.startswith("/**"):
                return text.strip("/* \n").strip()

    return None


def _parse_python(root, source_bytes: bytes) -> CodeStructure:
    """Extract symbols from a Python AST."""
    structure = CodeStructure(language="python")
    source_text = source_bytes.decode("utf-8", errors="replace")
    structure.total_lines = source_text.count("\n") + 1

    def _walk(node, parent_name: Optional[str] = None):
        if node.type == "import_statement" or node.type == "import_from_statement":
            text = _extract_text(node, source_bytes).strip()
            structure.imports.append(text)
            structure.symbols.append(CodeSymbol(
                name=text,
                kind="import",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=text,
                parent=parent_name,
            ))

        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            params_node = node.child_by_field_name("parameters")
            return_node = node.child_by_field_name("return_type")

            fname = _extract_text(name_node, source_bytes) if name_node else "?"
            params = _extract_text(params_node, source_bytes) if params_node else "()"
            ret = f" -> {_extract_text(return_node, source_bytes)}" if return_node else ""
            sig = f"def {fname}{params}{ret}"

            kind = "method" if parent_name else "function"
            docstring = _find_docstring(node, source_bytes, "python")

            structure.symbols.append(CodeSymbol(
                name=fname,
                kind=kind,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                docstring=docstring,
                parent=parent_name,
            ))

            # Recurse into function body for nested defs
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        _walk(stmt, parent_name=fname)

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            cname = _extract_text(name_node, source_bytes) if name_node else "?"

            # Get superclasses
            superclasses = node.child_by_field_name("superclasses")
            bases = f"({_extract_text(superclasses, source_bytes)})" if superclasses else ""
            sig = f"class {cname}{bases}"

            docstring = _find_docstring(node, source_bytes, "python")

            structure.symbols.append(CodeSymbol(
                name=cname,
                kind="class",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                docstring=docstring,
                parent=parent_name,
            ))

            # Recurse into class body for methods
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        _walk(stmt, parent_name=cname)

        elif node.type == "decorated_definition":
            # Handle @decorator before function/class
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    _walk(child, parent_name)

        elif node.type == "expression_statement":
            # Top-level assignments like: x = 10
            for child in node.children:
                if child.type == "assignment":
                    left = child.child_by_field_name("left")
                    if left and left.type == "identifier" and parent_name is None:
                        vname = _extract_text(left, source_bytes)
                        # Skip private/dunder
                        if not vname.startswith("_"):
                            structure.symbols.append(CodeSymbol(
                                name=vname,
                                kind="variable",
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                signature=_extract_text(child, source_bytes),
                                parent=parent_name,
                            ))

    for child in root.children:
        _walk(child)

    return structure


def _parse_generic(root, source_bytes: bytes, language: str) -> CodeStructure:
    """Generic symbol extraction for JS/TS/Go/Rust/Java/C/C++.

    Uses common node type patterns across languages.
    """
    structure = CodeStructure(language=language)
    source_text = source_bytes.decode("utf-8", errors="replace")
    structure.total_lines = source_text.count("\n") + 1

    # Node types for functions across languages
    function_types = {
        "function_declaration", "function_definition", "method_definition",
        "method_declaration", "arrow_function", "function_item",  # Rust
        "function_expression",
    }
    class_types = {
        "class_declaration", "class_definition", "struct_item",  # Rust
        "interface_declaration", "type_alias_declaration",
        "struct_specifier",  # C/C++
    }
    import_types = {
        "import_statement", "import_declaration", "use_declaration",  # Rust
        "include_directive",  # C/C++ #include
    }

    def _walk(node, parent_name: Optional[str] = None):
        ntype = node.type

        if ntype in import_types:
            text = _extract_text(node, source_bytes).strip()
            structure.imports.append(text)
            structure.symbols.append(CodeSymbol(
                name=text,
                kind="import",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=text,
                parent=parent_name,
            ))

        elif ntype in function_types:
            name_node = node.child_by_field_name("name")
            fname = _extract_text(name_node, source_bytes) if name_node else "?"

            # Build signature from first line
            full_text = _extract_text(node, source_bytes)
            first_line = full_text.split("\n", 1)[0].strip()
            # Truncate very long signatures
            sig = first_line[:200] + "..." if len(first_line) > 200 else first_line

            kind = "method" if parent_name else "function"
            docstring = _find_docstring(node, source_bytes, language)

            structure.symbols.append(CodeSymbol(
                name=fname,
                kind=kind,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                docstring=docstring,
                parent=parent_name,
            ))

        elif ntype in class_types:
            name_node = node.child_by_field_name("name")
            cname = _extract_text(name_node, source_bytes) if name_node else "?"

            full_text = _extract_text(node, source_bytes)
            first_line = full_text.split("\n", 1)[0].strip()
            sig = first_line[:200] + "..." if len(first_line) > 200 else first_line

            docstring = _find_docstring(node, source_bytes, language)

            structure.symbols.append(CodeSymbol(
                name=cname,
                kind="class",
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                docstring=docstring,
                parent=parent_name,
            ))

            # Recurse into class body for methods
            for child in node.children:
                if child.type in ("class_body", "declaration_list", "block"):
                    for stmt in child.children:
                        _walk(stmt, parent_name=cname)
            return  # Don't recurse children again

        # Recurse into all children
        for child in node.children:
            _walk(child, parent_name)

    for child in root.children:
        _walk(child)

    return structure


def parse(code: str, language: str) -> Optional[CodeStructure]:
    """Parse source code into a CodeStructure.

    Args:
        code: Source code text
        language: Language name (must be in SUPPORTED_LANGUAGES)

    Returns:
        CodeStructure or None if parsing failed (grammar not installed)
    """
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language: {language}")
        return None

    lang_obj = _get_language(language)
    if not lang_obj:
        logger.info(f"No grammar available for {language}, cannot parse")
        return None

    try:
        from tree_sitter import Parser

        parser = Parser(lang_obj)
        source_bytes = code.encode("utf-8")
        tree = parser.parse(source_bytes)

        if language == "python":
            return _parse_python(tree.root_node, source_bytes)
        else:
            return _parse_generic(tree.root_node, source_bytes, language)

    except ImportError:
        logger.warning("tree-sitter package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to parse {language} code: {e}")
        return None


def is_available() -> bool:
    """Check if tree-sitter is installed and functional."""
    try:
        import tree_sitter  # noqa: F401
        return True
    except ImportError:
        return False


def available_languages() -> list[str]:
    """Return list of languages with installed grammars."""
    if not is_available():
        return []

    available = []
    for lang in SUPPORTED_LANGUAGES:
        if _get_language(lang) is not None:
            available.append(lang)
    return sorted(available)
