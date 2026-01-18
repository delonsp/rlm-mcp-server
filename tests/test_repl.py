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


class TestExecutePreservesVariables:
    """Test that execute preserves variables between executions."""

    def test_variable_from_first_execution_available_in_second(self):
        """Variable assigned in first execute is accessible in second."""
        repl = SafeREPL()
        repl.execute("x = 42")
        result = repl.execute("y = x * 2")

        assert result.success is True
        assert repl.variables["y"] == 84

    def test_multiple_variables_persist(self):
        """Multiple variables persist across executions."""
        repl = SafeREPL()
        repl.execute("a = 1")
        repl.execute("b = 2")
        repl.execute("c = 3")
        result = repl.execute("total = a + b + c")

        assert result.success is True
        assert repl.variables["total"] == 6

    def test_variable_can_be_modified_in_subsequent_execution(self):
        """Variable can be modified in a subsequent execution."""
        repl = SafeREPL()
        repl.execute("counter = 0")
        repl.execute("counter = counter + 1")
        repl.execute("counter = counter + 1")

        assert repl.variables["counter"] == 2

    def test_list_variable_persists_and_can_be_modified(self):
        """List variable persists and can be modified."""
        repl = SafeREPL()
        repl.execute("items = []")
        repl.execute("items.append(1)")
        repl.execute("items.append(2)")
        repl.execute("items.append(3)")

        assert repl.variables["items"] == [1, 2, 3]

    def test_dict_variable_persists_and_can_be_modified(self):
        """Dict variable persists and can be modified."""
        repl = SafeREPL()
        repl.execute("data = {}")
        repl.execute("data['name'] = 'Alice'")
        repl.execute("data['age'] = 30")

        assert repl.variables["data"] == {"name": "Alice", "age": 30}

    def test_function_defined_in_first_execution_callable_in_second(self):
        """Function defined in first execute can be called in second."""
        repl = SafeREPL()
        repl.execute("def square(n):\n    return n ** 2")
        result = repl.execute("result = square(5)")

        assert result.success is True
        assert repl.variables["result"] == 25

    def test_class_definition_not_supported_in_sandbox(self):
        """Class definition is not supported in sandbox (__build_class__ not exposed)."""
        repl = SafeREPL()
        # The sandbox doesn't expose __build_class__, which is needed for class definitions.
        # This is a security feature - classes can be used for escaping sandboxes.
        result = repl.execute("class Point:\n    pass")

        assert result.success is False
        assert "__build_class__" in result.stderr

    def test_imported_module_persists(self):
        """Imported module persists and can be used in subsequent execution."""
        repl = SafeREPL()
        # Note: re, json, math, collections, datetime are pre-imported
        # But we can import others from ALLOWED_IMPORTS
        repl.execute("import statistics")
        result = repl.execute("avg = statistics.mean([1, 2, 3, 4, 5])")

        assert result.success is True
        assert repl.variables["avg"] == 3.0

    def test_variables_property_reflects_current_state(self):
        """repl.variables reflects the current state after multiple executions."""
        repl = SafeREPL()
        assert "x" not in repl.variables

        repl.execute("x = 10")
        assert repl.variables["x"] == 10

        repl.execute("x = 20")
        assert repl.variables["x"] == 20

        repl.execute("x = 30")
        assert repl.variables["x"] == 30

    def test_del_in_namespace_does_not_remove_from_repl_variables(self):
        """Using del in code doesn't remove from repl.variables (tracked separately)."""
        repl = SafeREPL()
        repl.execute("x = 100")
        assert repl.variables["x"] == 100

        # del x in the namespace doesn't affect repl.variables
        # because variables are only synced when new/changed, not when deleted
        repl.execute("del x")

        # The variable is still in repl.variables (design behavior)
        assert repl.variables["x"] == 100

        # And it's still accessible in subsequent executions
        result = repl.execute("y = x")
        assert result.success is True
        assert repl.variables["y"] == 100

    def test_variable_metadata_persists(self):
        """Variable metadata persists between executions."""
        repl = SafeREPL()
        repl.execute("data = 'hello world'")

        meta1 = repl.variable_metadata["data"]
        assert meta1.name == "data"
        created_at = meta1.created_at

        # Modify the variable
        repl.execute("data = data + '!'")

        meta2 = repl.variable_metadata["data"]
        assert meta2.name == "data"
        # created_at should be preserved
        assert meta2.created_at == created_at
        # last_accessed should be updated
        assert meta2.last_accessed >= meta1.last_accessed

    def test_failed_execution_does_not_lose_existing_variables(self):
        """Failed execution does not lose previously defined variables."""
        repl = SafeREPL()
        repl.execute("x = 100")
        repl.execute("y = 200")

        # This should fail
        result = repl.execute("z = undefined_var")
        assert result.success is False

        # Existing variables should still be there
        assert repl.variables["x"] == 100
        assert repl.variables["y"] == 200

    def test_partial_execution_preserves_variables_defined_before_error(self):
        """Variables defined before an error are preserved."""
        repl = SafeREPL()
        # a will be defined, then error occurs
        result = repl.execute("a = 1\nb = 2\nc = undefined")

        assert result.success is False
        # Variables defined before the error should be preserved
        assert repl.variables["a"] == 1
        assert repl.variables["b"] == 2
        assert "c" not in repl.variables

    def test_variables_isolated_between_repl_instances(self):
        """Variables are isolated between different SafeREPL instances."""
        repl1 = SafeREPL()
        repl2 = SafeREPL()

        repl1.execute("x = 'instance1'")
        repl2.execute("x = 'instance2'")

        assert repl1.variables["x"] == "instance1"
        assert repl2.variables["x"] == "instance2"

    def test_complex_data_structure_persists(self):
        """Complex nested data structures persist correctly."""
        repl = SafeREPL()
        repl.execute("nested = {'users': [{'name': 'Alice'}, {'name': 'Bob'}]}")
        result = repl.execute("first_user = nested['users'][0]['name']")

        assert result.success is True
        assert repl.variables["first_user"] == "Alice"

    def test_generator_expression_result_persists(self):
        """Result of generator expression persists (when converted to list)."""
        repl = SafeREPL()
        repl.execute("nums = list(range(5))")
        result = repl.execute("doubled = [n * 2 for n in nums]")

        assert result.success is True
        assert repl.variables["doubled"] == [0, 2, 4, 6, 8]

    def test_string_operations_across_executions(self):
        """String operations work across executions."""
        repl = SafeREPL()
        repl.execute("text = 'hello'")
        repl.execute("text = text.upper()")
        repl.execute("text = text + ' WORLD'")

        assert repl.variables["text"] == "HELLO WORLD"

    def test_lambda_persists(self):
        """Lambda function persists and can be used."""
        repl = SafeREPL()
        repl.execute("double = lambda x: x * 2")
        result = repl.execute("result = double(21)")

        assert result.success is True
        assert repl.variables["result"] == 42

    def test_many_executions_preserve_all_variables(self):
        """Many sequential executions preserve all variables."""
        repl = SafeREPL()

        # Create 20 variables
        for i in range(20):
            repl.execute(f"var_{i} = {i}")

        # Verify all exist
        for i in range(20):
            assert repl.variables[f"var_{i}"] == i

        # Use all of them in a calculation
        result = repl.execute("total = sum([var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19])")

        assert result.success is True
        assert repl.variables["total"] == sum(range(20))  # 190

    def test_llm_functions_always_available(self):
        """llm_query, llm_stats, llm_reset_counter are always available."""
        repl = SafeREPL()

        # After any execution, llm functions should be present
        result = repl.execute("x = 1")
        assert result.success is True

        # Check they're callable
        assert callable(repl.variables.get("llm_query"))
        assert callable(repl.variables.get("llm_stats"))
        assert callable(repl.variables.get("llm_reset_counter"))


class TestExecuteBlocksDangerousImports:
    """Test that execute blocks dangerous imports (os, subprocess, socket, etc.)."""

    def test_import_os_is_blocked(self):
        """import os is blocked."""
        repl = SafeREPL()
        result = repl.execute("import os")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "bloqueado" in result.stderr.lower() or "blocked" in result.stderr.lower()

    def test_import_subprocess_is_blocked(self):
        """import subprocess is blocked."""
        repl = SafeREPL()
        result = repl.execute("import subprocess")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "subprocess" in result.stderr

    def test_import_socket_is_blocked(self):
        """import socket is blocked."""
        repl = SafeREPL()
        result = repl.execute("import socket")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "socket" in result.stderr

    def test_import_sys_is_blocked(self):
        """import sys is blocked."""
        repl = SafeREPL()
        result = repl.execute("import sys")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "sys" in result.stderr

    def test_import_shutil_is_blocked(self):
        """import shutil is blocked."""
        repl = SafeREPL()
        result = repl.execute("import shutil")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "shutil" in result.stderr

    def test_import_pathlib_is_blocked(self):
        """import pathlib is blocked."""
        repl = SafeREPL()
        result = repl.execute("import pathlib")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "pathlib" in result.stderr

    def test_import_http_is_blocked(self):
        """import http is blocked."""
        repl = SafeREPL()
        result = repl.execute("import http")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "http" in result.stderr

    def test_import_urllib_is_blocked(self):
        """import urllib is blocked."""
        repl = SafeREPL()
        result = repl.execute("import urllib")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "urllib" in result.stderr

    def test_import_requests_is_blocked(self):
        """import requests is blocked."""
        repl = SafeREPL()
        result = repl.execute("import requests")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "requests" in result.stderr

    def test_import_pickle_is_blocked(self):
        """import pickle is blocked."""
        repl = SafeREPL()
        result = repl.execute("import pickle")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "pickle" in result.stderr

    def test_import_sqlite3_is_blocked(self):
        """import sqlite3 is blocked."""
        repl = SafeREPL()
        result = repl.execute("import sqlite3")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "sqlite3" in result.stderr

    def test_import_multiprocessing_is_blocked(self):
        """import multiprocessing is blocked."""
        repl = SafeREPL()
        result = repl.execute("import multiprocessing")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "multiprocessing" in result.stderr

    def test_import_threading_is_blocked(self):
        """import threading is blocked."""
        repl = SafeREPL()
        result = repl.execute("import threading")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "threading" in result.stderr

    def test_import_ctypes_is_blocked(self):
        """import ctypes is blocked."""
        repl = SafeREPL()
        result = repl.execute("import ctypes")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "ctypes" in result.stderr

    def test_import_importlib_is_blocked(self):
        """import importlib is blocked."""
        repl = SafeREPL()
        result = repl.execute("import importlib")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "importlib" in result.stderr

    def test_import_builtins_is_blocked(self):
        """import builtins is blocked."""
        repl = SafeREPL()
        result = repl.execute("import builtins")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "builtins" in result.stderr

    def test_from_os_import_is_blocked(self):
        """from os import ... is blocked."""
        repl = SafeREPL()
        result = repl.execute("from os import system")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "os" in result.stderr

    def test_from_subprocess_import_is_blocked(self):
        """from subprocess import ... is blocked."""
        repl = SafeREPL()
        result = repl.execute("from subprocess import run")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "subprocess" in result.stderr

    def test_import_os_path_is_blocked(self):
        """import os.path is blocked (base module is os)."""
        repl = SafeREPL()
        result = repl.execute("import os.path")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "os" in result.stderr

    def test_blocked_import_doesnt_modify_namespace(self):
        """Blocked import doesn't add anything to the namespace."""
        repl = SafeREPL()
        result = repl.execute("import os")

        assert result.success is False
        assert "os" not in repl.variables

    def test_error_message_mentions_blocked(self):
        """Error message indicates the import was blocked for security."""
        repl = SafeREPL()
        result = repl.execute("import os")

        assert result.success is False
        # Error message should mention it's a security block
        assert "bloqueado" in result.stderr.lower() or "seguranca" in result.stderr.lower()

    def test_unknown_module_also_blocked(self):
        """Unknown module that is not in whitelist is also blocked."""
        repl = SafeREPL()
        result = repl.execute("import some_unknown_module_xyz")

        assert result.success is False
        assert "SecurityError" in result.stderr
        # Error message should mention it's not allowed
        assert "nao permitido" in result.stderr.lower() or "not permitted" in result.stderr.lower() or "Permitidos" in result.stderr

    def test_blocked_import_in_try_except_caught_but_module_not_loaded(self):
        """Blocked import wrapped in try-except can be caught, but module is never loaded."""
        repl = SafeREPL()
        # User can catch the exception - this is fine behavior
        # The important thing is that the dangerous module is NEVER loaded
        result = repl.execute("""
try:
    import os
    x = 1  # This line never executes
except:
    x = 2  # Exception caught, x set to 2
""")

        # The SecurityError is caught by the user's try-except, so execution succeeds
        assert result.success is True
        # User's exception handler ran
        assert repl.variables["x"] == 2
        # Most importantly: os is NOT in the namespace (the import failed)
        assert "os" not in repl.variables

    def test_multiple_dangerous_imports_all_blocked(self):
        """Multiple dangerous imports are all blocked."""
        repl = SafeREPL()

        for module in ["os", "subprocess", "socket", "sys", "shutil"]:
            result = repl.execute(f"import {module}")
            assert result.success is False, f"Expected {module} to be blocked"
            assert "SecurityError" in result.stderr, f"Expected SecurityError for {module}"
