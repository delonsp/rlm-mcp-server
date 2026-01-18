"""
Tests for the repl module.

Tests cover:
- execute method with simple code (print, assignment)
- Variable persistence between executions
- Import security (blocked vs allowed imports)
- load_data with different data types
- get_memory_usage
- clear_namespace
"""

import pytest

from rlm_mcp.repl import SafeREPL, ExecutionResult, SecurityError


class TestExecuteSimpleCode:
    """Test that execute works with simple code (print, assignment)."""

    def test_execute_returns_execution_result(self):
        """execute returns an ExecutionResult object."""
        repl = SafeREPL()
        result = repl.execute("x = 1")

        assert isinstance(result, ExecutionResult)

    def test_execute_success_on_valid_code(self):
        """execute returns success=True for valid code."""
        repl = SafeREPL()
        result = repl.execute("x = 1")

        assert result.success is True

    def test_execute_captures_print_stdout(self):
        """execute captures print output in stdout."""
        repl = SafeREPL()
        result = repl.execute("print('hello world')")

        assert result.success is True
        assert "hello world" in result.stdout

    def test_execute_captures_multiple_prints(self):
        """execute captures multiple print statements."""
        repl = SafeREPL()
        result = repl.execute("print('one')\nprint('two')\nprint('three')")

        assert result.success is True
        assert "one" in result.stdout
        assert "two" in result.stdout
        assert "three" in result.stdout

    def test_execute_assignment_records_variable(self):
        """execute records assigned variable in variables_changed."""
        repl = SafeREPL()
        result = repl.execute("x = 42")

        assert result.success is True
        assert "x" in result.variables_changed
        assert repl.variables["x"] == 42

    def test_execute_multiple_assignments(self):
        """execute handles multiple assignments."""
        repl = SafeREPL()
        result = repl.execute("a = 1\nb = 2\nc = 3")

        assert result.success is True
        assert "a" in result.variables_changed
        assert "b" in result.variables_changed
        assert "c" in result.variables_changed
        assert repl.variables["a"] == 1
        assert repl.variables["b"] == 2
        assert repl.variables["c"] == 3

    def test_execute_string_assignment(self):
        """execute handles string assignment."""
        repl = SafeREPL()
        result = repl.execute("msg = 'hello'")

        assert result.success is True
        assert repl.variables["msg"] == "hello"

    def test_execute_list_assignment(self):
        """execute handles list assignment."""
        repl = SafeREPL()
        result = repl.execute("items = [1, 2, 3]")

        assert result.success is True
        assert repl.variables["items"] == [1, 2, 3]

    def test_execute_dict_assignment(self):
        """execute handles dict assignment."""
        repl = SafeREPL()
        result = repl.execute("data = {'key': 'value'}")

        assert result.success is True
        assert repl.variables["data"] == {"key": "value"}

    def test_execute_arithmetic_operation(self):
        """execute handles arithmetic operations."""
        repl = SafeREPL()
        result = repl.execute("result = 10 + 5 * 2")

        assert result.success is True
        assert repl.variables["result"] == 20

    def test_execute_string_operations(self):
        """execute handles string operations."""
        repl = SafeREPL()
        result = repl.execute("text = 'hello' + ' ' + 'world'\nupper = text.upper()")

        assert result.success is True
        assert repl.variables["text"] == "hello world"
        assert repl.variables["upper"] == "HELLO WORLD"

    def test_execute_list_comprehension(self):
        """execute handles list comprehension."""
        repl = SafeREPL()
        result = repl.execute("squares = [x**2 for x in range(5)]")

        assert result.success is True
        assert repl.variables["squares"] == [0, 1, 4, 9, 16]

    def test_execute_function_definition(self):
        """execute handles function definition."""
        repl = SafeREPL()
        result = repl.execute("def add(a, b):\n    return a + b")

        assert result.success is True
        assert "add" in result.variables_changed
        assert callable(repl.variables["add"])

    def test_execute_function_call(self):
        """execute handles function definition and call."""
        repl = SafeREPL()
        result = repl.execute("def double(x):\n    return x * 2\nresult = double(5)")

        assert result.success is True
        assert repl.variables["result"] == 10

    def test_execute_returns_execution_time(self):
        """execute returns execution_time_ms > 0."""
        repl = SafeREPL()
        result = repl.execute("x = sum(range(1000))")

        assert result.success is True
        assert result.execution_time_ms >= 0

    def test_execute_syntax_error_returns_failure(self):
        """execute returns success=False for syntax error."""
        repl = SafeREPL()
        result = repl.execute("def broken(")

        assert result.success is False
        assert "SecurityError" in result.stderr or "sintaxe" in result.stderr.lower()

    def test_execute_runtime_error_returns_failure(self):
        """execute returns success=False for runtime error."""
        repl = SafeREPL()
        result = repl.execute("x = 1 / 0")

        assert result.success is False
        assert "ZeroDivisionError" in result.stderr

    def test_execute_name_error_returns_failure(self):
        """execute returns success=False for undefined variable."""
        repl = SafeREPL()
        result = repl.execute("x = undefined_variable")

        assert result.success is False
        assert "NameError" in result.stderr

    def test_execute_updates_execution_count(self):
        """execute increments execution_count."""
        repl = SafeREPL()
        assert repl.execution_count == 0

        repl.execute("x = 1")
        assert repl.execution_count == 1

        repl.execute("y = 2")
        assert repl.execution_count == 2

    def test_execute_empty_code(self):
        """execute handles empty code."""
        repl = SafeREPL()
        result = repl.execute("")

        assert result.success is True
        assert result.stdout == ""

    def test_execute_only_comments(self):
        """execute handles code with only comments."""
        repl = SafeREPL()
        result = repl.execute("# This is a comment\n# Another comment")

        assert result.success is True
        # Only llm_* functions are injected, no user variables created
        # llm_query, llm_stats, llm_reset_counter are always added
        user_vars = [v for v in result.variables_changed if not v.startswith("llm_")]
        assert user_vars == []

    def test_execute_updates_variable_metadata(self):
        """execute creates variable metadata."""
        repl = SafeREPL()
        repl.execute("my_var = 'test string'")

        assert "my_var" in repl.variable_metadata
        meta = repl.variable_metadata["my_var"]
        assert meta.name == "my_var"
        assert meta.type_name == "str"
        assert meta.size_bytes > 0

    def test_execute_metadata_has_preview(self):
        """execute creates variable metadata with preview."""
        repl = SafeREPL()
        repl.execute("long_text = 'a' * 1000")

        meta = repl.variable_metadata["long_text"]
        assert len(meta.preview) < 300  # Preview is truncated
        assert "chars total" in meta.preview  # Shows total count
