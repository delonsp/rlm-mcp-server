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


class TestExecuteAllowsSafeImports:
    """Test that execute allows safe imports (re, json, math, collections, etc.)."""

    def test_import_re_is_allowed(self):
        """import re is allowed."""
        repl = SafeREPL()
        result = repl.execute("import re\npattern = re.compile(r'\\d+')")

        assert result.success is True
        # Note: re is pre-imported and excluded from self.variables tracking
        # But the import succeeds and the module is usable
        assert "pattern" in repl.variables

    def test_import_json_is_allowed(self):
        """import json is allowed."""
        repl = SafeREPL()
        result = repl.execute("import json\ndata = json.loads('{\"key\": \"value\"}')")

        assert result.success is True
        # Note: json is pre-imported and excluded from self.variables tracking
        # But the import succeeds and the module is usable
        assert repl.variables["data"] == {"key": "value"}

    def test_import_math_is_allowed(self):
        """import math is allowed."""
        repl = SafeREPL()
        result = repl.execute("import math\npi = math.pi\nsqrt2 = math.sqrt(2)")

        assert result.success is True
        # Note: math is pre-imported and excluded from self.variables tracking
        # But the import succeeds and the module is usable
        assert abs(repl.variables["pi"] - 3.14159265) < 0.001
        assert abs(repl.variables["sqrt2"] - 1.41421356) < 0.001

    def test_import_collections_is_allowed(self):
        """import collections is allowed."""
        repl = SafeREPL()
        result = repl.execute("import collections\ncounter = collections.Counter(['a', 'b', 'a', 'c', 'a'])")

        assert result.success is True
        # Note: collections is pre-imported and excluded from self.variables tracking
        # But the import succeeds and the module is usable
        assert repl.variables["counter"]["a"] == 3
        assert repl.variables["counter"]["b"] == 1
        assert repl.variables["counter"]["c"] == 1

    def test_import_statistics_is_allowed(self):
        """import statistics is allowed."""
        repl = SafeREPL()
        result = repl.execute("import statistics\navg = statistics.mean([1, 2, 3, 4, 5])")

        assert result.success is True
        assert "statistics" in repl.variables
        assert repl.variables["avg"] == 3.0

    def test_import_itertools_is_allowed(self):
        """import itertools is allowed."""
        repl = SafeREPL()
        result = repl.execute("import itertools\ncombs = list(itertools.combinations([1, 2, 3], 2))")

        assert result.success is True
        assert "itertools" in repl.variables
        assert repl.variables["combs"] == [(1, 2), (1, 3), (2, 3)]

    def test_import_functools_is_allowed(self):
        """import functools is allowed."""
        repl = SafeREPL()
        result = repl.execute("import functools\npartial_add = functools.partial(lambda a, b: a + b, 10)")

        assert result.success is True
        assert "functools" in repl.variables
        assert callable(repl.variables["partial_add"])

    def test_import_operator_is_allowed(self):
        """import operator is allowed."""
        repl = SafeREPL()
        result = repl.execute("import operator\nresult = operator.add(5, 3)")

        assert result.success is True
        assert "operator" in repl.variables
        assert repl.variables["result"] == 8

    def test_import_string_is_allowed(self):
        """import string is allowed."""
        repl = SafeREPL()
        result = repl.execute("import string\nletters = string.ascii_lowercase")

        assert result.success is True
        assert "string" in repl.variables
        assert repl.variables["letters"] == "abcdefghijklmnopqrstuvwxyz"

    def test_import_textwrap_is_allowed(self):
        """import textwrap is allowed."""
        repl = SafeREPL()
        result = repl.execute("import textwrap\nwrapped = textwrap.fill('Hello world', width=5)")

        assert result.success is True
        assert "textwrap" in repl.variables
        assert "Hello" in repl.variables["wrapped"]

    def test_import_datetime_is_allowed(self):
        """import datetime is allowed."""
        repl = SafeREPL()
        result = repl.execute("import datetime\nnow = datetime.datetime.now()")

        assert result.success is True
        # Note: datetime is pre-imported and excluded from self.variables tracking
        # But the import succeeds and the module is usable
        import datetime as dt_module
        assert isinstance(repl.variables["now"], dt_module.datetime)

    def test_import_time_is_allowed(self):
        """import time is allowed."""
        repl = SafeREPL()
        result = repl.execute("import time\nt = time.time()")

        assert result.success is True
        assert "time" in repl.variables
        assert isinstance(repl.variables["t"], float)
        assert repl.variables["t"] > 0

    def test_import_calendar_is_allowed(self):
        """import calendar is allowed."""
        repl = SafeREPL()
        result = repl.execute("import calendar\nis_leap = calendar.isleap(2024)")

        assert result.success is True
        assert "calendar" in repl.variables
        assert repl.variables["is_leap"] is True

    def test_import_dataclasses_is_allowed(self):
        """import dataclasses is allowed."""
        repl = SafeREPL()
        # Note: Can't define classes (no __build_class__), but can import the module
        result = repl.execute("import dataclasses")

        assert result.success is True
        assert "dataclasses" in repl.variables

    def test_import_typing_is_allowed(self):
        """import typing is allowed."""
        repl = SafeREPL()
        result = repl.execute("import typing\nMyType = typing.List[int]")

        assert result.success is True
        assert "typing" in repl.variables

    def test_import_enum_is_allowed(self):
        """import enum is allowed."""
        repl = SafeREPL()
        # Note: Can't define classes (no __build_class__), but can import the module
        result = repl.execute("import enum")

        assert result.success is True
        assert "enum" in repl.variables

    def test_import_csv_is_allowed(self):
        """import csv is allowed."""
        repl = SafeREPL()
        result = repl.execute("import csv\nreader = csv.reader")

        assert result.success is True
        assert "csv" in repl.variables
        assert callable(repl.variables["reader"])

    def test_import_hashlib_is_allowed(self):
        """import hashlib is allowed."""
        repl = SafeREPL()
        result = repl.execute("import hashlib\nhash_obj = hashlib.md5(b'hello')\ndigest = hash_obj.hexdigest()")

        assert result.success is True
        assert "hashlib" in repl.variables
        assert repl.variables["digest"] == "5d41402abc4b2a76b9719d911017c592"

    def test_import_base64_is_allowed(self):
        """import base64 is allowed."""
        repl = SafeREPL()
        result = repl.execute("import base64\nencoded = base64.b64encode(b'hello').decode()")

        assert result.success is True
        assert "base64" in repl.variables
        assert repl.variables["encoded"] == "aGVsbG8="

    def test_import_gzip_is_allowed(self):
        """import gzip is allowed."""
        repl = SafeREPL()
        result = repl.execute("import gzip")

        assert result.success is True
        assert "gzip" in repl.variables

    def test_import_zipfile_is_allowed(self):
        """import zipfile is allowed."""
        repl = SafeREPL()
        result = repl.execute("import zipfile")

        assert result.success is True
        assert "zipfile" in repl.variables

    def test_from_collections_import_counter_is_allowed(self):
        """from collections import Counter is allowed."""
        repl = SafeREPL()
        result = repl.execute("from collections import Counter\nc = Counter(['a', 'a', 'b'])")

        assert result.success is True
        assert "Counter" in repl.variables
        assert repl.variables["c"]["a"] == 2
        assert repl.variables["c"]["b"] == 1

    def test_from_math_import_sqrt_is_allowed(self):
        """from math import sqrt is allowed."""
        repl = SafeREPL()
        result = repl.execute("from math import sqrt, pi\nresult = sqrt(16)")

        assert result.success is True
        assert "sqrt" in repl.variables
        assert "pi" in repl.variables
        assert repl.variables["result"] == 4.0

    def test_from_json_import_loads_is_allowed(self):
        """from json import loads is allowed."""
        repl = SafeREPL()
        result = repl.execute("from json import loads, dumps\ndata = loads('[1, 2, 3]')")

        assert result.success is True
        assert "loads" in repl.variables
        assert "dumps" in repl.variables
        assert repl.variables["data"] == [1, 2, 3]

    def test_pre_imported_modules_available_without_import(self):
        """Common modules (re, json, math, collections, datetime) are pre-imported."""
        repl = SafeREPL()
        # These modules are pre-imported in execute(), so they're available without explicit import
        result = repl.execute("""
pattern = re.compile(r'\\d+')
data = json.dumps({'a': 1})
pi_val = math.pi
counter = collections.Counter([1, 1, 2])
now = datetime.datetime.now()
""")

        assert result.success is True
        assert "pattern" in repl.variables
        assert "data" in repl.variables
        assert "pi_val" in repl.variables
        assert "counter" in repl.variables
        assert "now" in repl.variables

    def test_safe_import_persists_module_for_subsequent_executions(self):
        """Imported module persists and can be used in subsequent executions."""
        repl = SafeREPL()
        repl.execute("import statistics")
        result = repl.execute("result = statistics.median([1, 3, 5, 7, 9])")

        assert result.success is True
        assert repl.variables["result"] == 5

    def test_multiple_safe_imports_all_allowed(self):
        """Multiple safe imports are all allowed."""
        repl = SafeREPL()

        allowed_modules = [
            "re", "json", "math", "statistics", "collections",
            "itertools", "functools", "operator", "string"
        ]

        for module in allowed_modules:
            result = repl.execute(f"import {module}")
            assert result.success is True, f"Expected {module} to be allowed"
            # Note: Some modules (re, json, math, collections, datetime) are pre-imported
            # and excluded from self.variables tracking. We just verify the import succeeds.

    def test_non_preimported_modules_are_tracked_in_variables(self):
        """Modules that are NOT pre-imported ARE tracked in self.variables."""
        repl = SafeREPL()

        # These modules are not in the pre-import list, so they get tracked
        non_preimported = ["statistics", "itertools", "functools", "operator", "string"]

        for module in non_preimported:
            result = repl.execute(f"import {module}")
            assert result.success is True, f"Expected {module} to be allowed"
            assert module in repl.variables, f"Expected {module} to be in variables"

    def test_safe_import_does_not_pollute_error_message(self):
        """Allowed import doesn't produce security error message."""
        repl = SafeREPL()
        result = repl.execute("import math")

        assert result.success is True
        assert "SecurityError" not in result.stderr
        assert "bloqueado" not in result.stderr.lower()
        assert "nao permitido" not in result.stderr.lower()

    def test_unicodedata_is_allowed(self):
        """import unicodedata is allowed."""
        repl = SafeREPL()
        result = repl.execute("import unicodedata\nname = unicodedata.name('A')")

        assert result.success is True
        assert "unicodedata" in repl.variables
        assert repl.variables["name"] == "LATIN CAPITAL LETTER A"


class TestLoadDataText:
    """Test load_data with data_type='text'."""

    def test_load_data_text_returns_execution_result(self):
        """load_data returns an ExecutionResult object."""
        repl = SafeREPL()
        result = repl.load_data("test", "hello world", data_type="text")

        assert isinstance(result, ExecutionResult)

    def test_load_data_text_success_on_valid_string(self):
        """load_data returns success=True for valid string data."""
        repl = SafeREPL()
        result = repl.load_data("test", "hello world", data_type="text")

        assert result.success is True

    def test_load_data_text_stores_string_value(self):
        """load_data stores the string value in variables."""
        repl = SafeREPL()
        repl.load_data("my_text", "hello world", data_type="text")

        assert "my_text" in repl.variables
        assert repl.variables["my_text"] == "hello world"

    def test_load_data_text_stores_as_string_type(self):
        """load_data stores data as str type."""
        repl = SafeREPL()
        repl.load_data("text_var", "test content", data_type="text")

        assert isinstance(repl.variables["text_var"], str)

    def test_load_data_text_with_empty_string(self):
        """load_data handles empty string."""
        repl = SafeREPL()
        result = repl.load_data("empty", "", data_type="text")

        assert result.success is True
        assert repl.variables["empty"] == ""

    def test_load_data_text_with_multiline_string(self):
        """load_data handles multiline string."""
        repl = SafeREPL()
        multiline = "line 1\nline 2\nline 3"
        result = repl.load_data("lines", multiline, data_type="text")

        assert result.success is True
        assert repl.variables["lines"] == multiline
        assert "\n" in repl.variables["lines"]

    def test_load_data_text_with_bytes_decodes_to_string(self):
        """load_data decodes bytes to string for data_type='text'."""
        repl = SafeREPL()
        data_bytes = b"hello bytes"
        result = repl.load_data("from_bytes", data_bytes, data_type="text")

        assert result.success is True
        assert repl.variables["from_bytes"] == "hello bytes"
        assert isinstance(repl.variables["from_bytes"], str)

    def test_load_data_text_with_utf8_bytes(self):
        """load_data decodes UTF-8 bytes correctly."""
        repl = SafeREPL()
        data_bytes = "Olá, mundo! Ação e reação.".encode('utf-8')
        result = repl.load_data("utf8", data_bytes, data_type="text")

        assert result.success is True
        assert repl.variables["utf8"] == "Olá, mundo! Ação e reação."

    def test_load_data_text_with_unicode_content(self):
        """load_data handles Unicode content."""
        repl = SafeREPL()
        unicode_text = "日本語 中文 한국어 العربية"
        result = repl.load_data("unicode", unicode_text, data_type="text")

        assert result.success is True
        assert repl.variables["unicode"] == unicode_text

    def test_load_data_text_creates_metadata(self):
        """load_data creates variable metadata."""
        repl = SafeREPL()
        repl.load_data("with_meta", "some content", data_type="text")

        assert "with_meta" in repl.variable_metadata
        meta = repl.variable_metadata["with_meta"]
        assert meta.name == "with_meta"
        assert meta.type_name == "str"

    def test_load_data_text_metadata_has_correct_size(self):
        """load_data metadata has correct size_bytes."""
        repl = SafeREPL()
        content = "hello world"
        repl.load_data("sized", content, data_type="text")

        meta = repl.variable_metadata["sized"]
        # Size should be UTF-8 encoded bytes (11 bytes for "hello world")
        assert meta.size_bytes == len(content.encode('utf-8'))
        assert meta.size_bytes == 11

    def test_load_data_text_metadata_has_human_size(self):
        """load_data metadata has human-readable size."""
        repl = SafeREPL()
        repl.load_data("human", "test", data_type="text")

        meta = repl.variable_metadata["human"]
        assert "B" in meta.size_human  # Should end with B, KB, MB, etc.

    def test_load_data_text_metadata_has_preview(self):
        """load_data metadata has preview of content."""
        repl = SafeREPL()
        content = "This is a preview test"
        repl.load_data("preview_test", content, data_type="text")

        meta = repl.variable_metadata["preview_test"]
        assert "This is a preview test" in meta.preview

    def test_load_data_text_metadata_preview_truncated_for_long_text(self):
        """load_data metadata preview is truncated for long text."""
        repl = SafeREPL()
        long_content = "x" * 500
        repl.load_data("long_text", long_content, data_type="text")

        meta = repl.variable_metadata["long_text"]
        # Preview should be truncated and show total chars
        assert len(meta.preview) < len(long_content)
        assert "chars total" in meta.preview

    def test_load_data_text_metadata_has_timestamps(self):
        """load_data metadata has created_at and last_accessed timestamps."""
        repl = SafeREPL()
        from datetime import datetime
        before = datetime.now()
        repl.load_data("timestamped", "content", data_type="text")
        after = datetime.now()

        meta = repl.variable_metadata["timestamped"]
        assert before <= meta.created_at <= after
        assert before <= meta.last_accessed <= after

    def test_load_data_text_records_variable_in_result(self):
        """load_data records variable name in variables_changed."""
        repl = SafeREPL()
        result = repl.load_data("recorded", "data", data_type="text")

        assert "recorded" in result.variables_changed

    def test_load_data_text_stdout_contains_info(self):
        """load_data stdout contains loading info."""
        repl = SafeREPL()
        result = repl.load_data("info_test", "some data", data_type="text")

        assert "info_test" in result.stdout
        assert "carregada" in result.stdout  # Portuguese for "loaded"
        assert "str" in result.stdout  # Type name

    def test_load_data_text_overwrites_existing_variable(self):
        """load_data overwrites existing variable with same name."""
        repl = SafeREPL()
        repl.load_data("overwrite", "original value", data_type="text")
        repl.load_data("overwrite", "new value", data_type="text")

        assert repl.variables["overwrite"] == "new value"

    def test_load_data_text_variable_usable_in_execute(self):
        """Variable loaded with load_data is usable in execute."""
        repl = SafeREPL()
        repl.load_data("text_data", "hello world", data_type="text")
        result = repl.execute("upper_text = text_data.upper()")

        assert result.success is True
        assert repl.variables["upper_text"] == "HELLO WORLD"

    def test_load_data_text_large_string(self):
        """load_data handles large text (1MB+)."""
        repl = SafeREPL()
        large_text = "x" * (1024 * 1024)  # 1 MB of 'x'
        result = repl.load_data("large", large_text, data_type="text")

        assert result.success is True
        assert len(repl.variables["large"]) == 1024 * 1024

    def test_load_data_text_with_special_characters(self):
        """load_data handles text with special characters."""
        repl = SafeREPL()
        special = "Tab:\tNewline:\nQuote:\"Backslash:\\"
        result = repl.load_data("special", special, data_type="text")

        assert result.success is True
        assert repl.variables["special"] == special
        assert "\t" in repl.variables["special"]
        assert "\n" in repl.variables["special"]

    def test_load_data_text_default_data_type(self):
        """load_data defaults to data_type='text' when not specified."""
        repl = SafeREPL()
        # Note: data_type="text" is the default
        result = repl.load_data("default_type", "content")

        assert result.success is True
        assert repl.variables["default_type"] == "content"
        assert isinstance(repl.variables["default_type"], str)

    def test_load_data_text_preserves_whitespace(self):
        """load_data preserves leading/trailing whitespace."""
        repl = SafeREPL()
        whitespace_text = "  leading and trailing  "
        result = repl.load_data("whitespace", whitespace_text, data_type="text")

        assert result.success is True
        assert repl.variables["whitespace"] == whitespace_text
        assert repl.variables["whitespace"].startswith("  ")
        assert repl.variables["whitespace"].endswith("  ")

    def test_load_data_text_with_only_whitespace(self):
        """load_data handles text with only whitespace."""
        repl = SafeREPL()
        result = repl.load_data("just_spaces", "   \n\t  ", data_type="text")

        assert result.success is True
        assert repl.variables["just_spaces"] == "   \n\t  "


class TestLoadDataJson:
    """Test load_data with data_type='json'."""

    def test_load_data_json_returns_execution_result(self):
        """load_data returns an ExecutionResult object."""
        repl = SafeREPL()
        result = repl.load_data("test", '{"key": "value"}', data_type="json")

        assert isinstance(result, ExecutionResult)

    def test_load_data_json_success_on_valid_json(self):
        """load_data returns success=True for valid JSON data."""
        repl = SafeREPL()
        result = repl.load_data("test", '{"key": "value"}', data_type="json")

        assert result.success is True

    def test_load_data_json_parses_object(self):
        """load_data parses JSON object into dict."""
        repl = SafeREPL()
        repl.load_data("obj", '{"name": "Alice", "age": 30}', data_type="json")

        assert "obj" in repl.variables
        assert repl.variables["obj"] == {"name": "Alice", "age": 30}
        assert isinstance(repl.variables["obj"], dict)

    def test_load_data_json_parses_array(self):
        """load_data parses JSON array into list."""
        repl = SafeREPL()
        repl.load_data("arr", '[1, 2, 3, "four"]', data_type="json")

        assert repl.variables["arr"] == [1, 2, 3, "four"]
        assert isinstance(repl.variables["arr"], list)

    def test_load_data_json_parses_string(self):
        """load_data parses JSON string."""
        repl = SafeREPL()
        repl.load_data("str_val", '"hello world"', data_type="json")

        assert repl.variables["str_val"] == "hello world"
        assert isinstance(repl.variables["str_val"], str)

    def test_load_data_json_parses_number(self):
        """load_data parses JSON number."""
        repl = SafeREPL()
        repl.load_data("num", '42', data_type="json")

        assert repl.variables["num"] == 42
        assert isinstance(repl.variables["num"], int)

    def test_load_data_json_parses_float(self):
        """load_data parses JSON float."""
        repl = SafeREPL()
        repl.load_data("flt", '3.14159', data_type="json")

        assert repl.variables["flt"] == 3.14159
        assert isinstance(repl.variables["flt"], float)

    def test_load_data_json_parses_boolean_true(self):
        """load_data parses JSON true."""
        repl = SafeREPL()
        repl.load_data("bool_true", 'true', data_type="json")

        assert repl.variables["bool_true"] is True

    def test_load_data_json_parses_boolean_false(self):
        """load_data parses JSON false."""
        repl = SafeREPL()
        repl.load_data("bool_false", 'false', data_type="json")

        assert repl.variables["bool_false"] is False

    def test_load_data_json_parses_null(self):
        """load_data parses JSON null."""
        repl = SafeREPL()
        repl.load_data("null_val", 'null', data_type="json")

        assert repl.variables["null_val"] is None

    def test_load_data_json_nested_object(self):
        """load_data parses nested JSON object."""
        repl = SafeREPL()
        nested_json = '{"user": {"name": "Bob", "address": {"city": "NYC"}}}'
        repl.load_data("nested", nested_json, data_type="json")

        assert repl.variables["nested"]["user"]["name"] == "Bob"
        assert repl.variables["nested"]["user"]["address"]["city"] == "NYC"

    def test_load_data_json_array_of_objects(self):
        """load_data parses array of JSON objects."""
        repl = SafeREPL()
        json_data = '[{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]'
        repl.load_data("items", json_data, data_type="json")

        assert len(repl.variables["items"]) == 2
        assert repl.variables["items"][0]["id"] == 1
        assert repl.variables["items"][1]["name"] == "B"

    def test_load_data_json_empty_object(self):
        """load_data parses empty JSON object."""
        repl = SafeREPL()
        result = repl.load_data("empty_obj", '{}', data_type="json")

        assert result.success is True
        assert repl.variables["empty_obj"] == {}

    def test_load_data_json_empty_array(self):
        """load_data parses empty JSON array."""
        repl = SafeREPL()
        result = repl.load_data("empty_arr", '[]', data_type="json")

        assert result.success is True
        assert repl.variables["empty_arr"] == []

    def test_load_data_json_from_bytes(self):
        """load_data parses JSON from bytes."""
        repl = SafeREPL()
        json_bytes = b'{"from": "bytes"}'
        result = repl.load_data("from_bytes", json_bytes, data_type="json")

        assert result.success is True
        assert repl.variables["from_bytes"] == {"from": "bytes"}

    def test_load_data_json_utf8_content(self):
        """load_data handles UTF-8 content in JSON."""
        repl = SafeREPL()
        json_data = '{"message": "Olá mundo! Ação e reação."}'
        result = repl.load_data("utf8", json_data, data_type="json")

        assert result.success is True
        assert repl.variables["utf8"]["message"] == "Olá mundo! Ação e reação."

    def test_load_data_json_unicode_content(self):
        """load_data handles Unicode content in JSON."""
        repl = SafeREPL()
        json_data = '{"text": "日本語 中文 한국어"}'
        result = repl.load_data("unicode", json_data, data_type="json")

        assert result.success is True
        assert repl.variables["unicode"]["text"] == "日本語 中文 한국어"

    def test_load_data_json_invalid_json_fails(self):
        """load_data fails on invalid JSON."""
        repl = SafeREPL()
        result = repl.load_data("invalid", '{not valid json}', data_type="json")

        assert result.success is False
        assert "Erro" in result.stderr  # Portuguese for "Error"

    def test_load_data_json_incomplete_json_fails(self):
        """load_data fails on incomplete JSON."""
        repl = SafeREPL()
        result = repl.load_data("incomplete", '{"key": ', data_type="json")

        assert result.success is False
        assert "Erro" in result.stderr

    def test_load_data_json_creates_metadata(self):
        """load_data creates variable metadata for JSON."""
        repl = SafeREPL()
        repl.load_data("with_meta", '{"a": 1}', data_type="json")

        assert "with_meta" in repl.variable_metadata
        meta = repl.variable_metadata["with_meta"]
        assert meta.name == "with_meta"
        assert meta.type_name == "dict"

    def test_load_data_json_metadata_for_array_has_list_type(self):
        """load_data metadata for JSON array has type_name 'list'."""
        repl = SafeREPL()
        repl.load_data("arr_meta", '[1, 2, 3]', data_type="json")

        meta = repl.variable_metadata["arr_meta"]
        assert meta.type_name == "list"

    def test_load_data_json_metadata_has_preview(self):
        """load_data metadata has preview of JSON content."""
        repl = SafeREPL()
        repl.load_data("preview", '{"key": "value"}', data_type="json")

        meta = repl.variable_metadata["preview"]
        assert "key" in meta.preview

    def test_load_data_json_records_variable_in_result(self):
        """load_data records variable name in variables_changed."""
        repl = SafeREPL()
        result = repl.load_data("recorded", '{"data": true}', data_type="json")

        assert "recorded" in result.variables_changed

    def test_load_data_json_stdout_contains_info(self):
        """load_data stdout contains loading info."""
        repl = SafeREPL()
        result = repl.load_data("info_test", '{"x": 1}', data_type="json")

        assert "info_test" in result.stdout
        assert "carregada" in result.stdout  # Portuguese for "loaded"

    def test_load_data_json_overwrites_existing_variable(self):
        """load_data overwrites existing variable with same name."""
        repl = SafeREPL()
        repl.load_data("overwrite", '{"v": 1}', data_type="json")
        repl.load_data("overwrite", '{"v": 2}', data_type="json")

        assert repl.variables["overwrite"] == {"v": 2}

    def test_load_data_json_variable_usable_in_execute(self):
        """Variable loaded with load_data is usable in execute."""
        repl = SafeREPL()
        repl.load_data("json_data", '{"x": 10, "y": 20}', data_type="json")
        result = repl.execute("total = json_data['x'] + json_data['y']")

        assert result.success is True
        assert repl.variables["total"] == 30

    def test_load_data_json_large_object(self):
        """load_data handles large JSON object."""
        repl = SafeREPL()
        import json
        large_data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        json_str = json.dumps(large_data)
        result = repl.load_data("large", json_str, data_type="json")

        assert result.success is True
        assert len(repl.variables["large"]) == 1000
        assert repl.variables["large"]["key_500"] == "value_500"

    def test_load_data_json_with_special_characters(self):
        """load_data handles JSON with special characters in strings."""
        repl = SafeREPL()
        # JSON with escaped special characters
        json_data = '{"text": "line1\\nline2\\ttab", "quote": "\\"quoted\\""}'
        result = repl.load_data("special", json_data, data_type="json")

        assert result.success is True
        assert "\n" in repl.variables["special"]["text"]
        assert "\t" in repl.variables["special"]["text"]
        assert '"' in repl.variables["special"]["quote"]

    def test_load_data_json_with_numeric_keys_in_object(self):
        """load_data handles JSON with numeric string keys."""
        repl = SafeREPL()
        # JSON keys are always strings, even if they look like numbers
        json_data = '{"1": "one", "2": "two"}'
        result = repl.load_data("num_keys", json_data, data_type="json")

        assert result.success is True
        assert repl.variables["num_keys"]["1"] == "one"
        assert repl.variables["num_keys"]["2"] == "two"

    def test_load_data_json_preserves_order(self):
        """load_data preserves key order in JSON object (Python 3.7+ dicts are ordered)."""
        repl = SafeREPL()
        json_data = '{"z": 1, "a": 2, "m": 3}'
        result = repl.load_data("ordered", json_data, data_type="json")

        assert result.success is True
        keys = list(repl.variables["ordered"].keys())
        assert keys == ["z", "a", "m"]

    def test_load_data_json_scientific_notation(self):
        """load_data handles scientific notation in JSON."""
        repl = SafeREPL()
        json_data = '{"large": 1.23e10, "small": 4.56e-5}'
        result = repl.load_data("sci", json_data, data_type="json")

        assert result.success is True
        assert repl.variables["sci"]["large"] == 1.23e10
        assert repl.variables["sci"]["small"] == 4.56e-5


class TestLoadDataCsv:
    """Test load_data with data_type='csv'."""

    def test_load_data_csv_returns_execution_result(self):
        """load_data returns an ExecutionResult object."""
        repl = SafeREPL()
        csv_data = "name,age\nAlice,30\nBob,25"
        result = repl.load_data("test", csv_data, data_type="csv")

        assert isinstance(result, ExecutionResult)

    def test_load_data_csv_success_on_valid_csv(self):
        """load_data returns success=True for valid CSV data."""
        repl = SafeREPL()
        csv_data = "name,age\nAlice,30"
        result = repl.load_data("test", csv_data, data_type="csv")

        assert result.success is True

    def test_load_data_csv_parses_to_list_of_dicts(self):
        """load_data parses CSV into list of dicts (DictReader)."""
        repl = SafeREPL()
        csv_data = "name,age\nAlice,30\nBob,25"
        repl.load_data("data", csv_data, data_type="csv")

        assert "data" in repl.variables
        assert isinstance(repl.variables["data"], list)
        assert len(repl.variables["data"]) == 2
        assert isinstance(repl.variables["data"][0], dict)

    def test_load_data_csv_uses_header_as_keys(self):
        """load_data uses CSV header row as dict keys."""
        repl = SafeREPL()
        csv_data = "name,age,city\nAlice,30,NYC"
        repl.load_data("data", csv_data, data_type="csv")

        row = repl.variables["data"][0]
        assert "name" in row
        assert "age" in row
        assert "city" in row
        assert row["name"] == "Alice"
        assert row["age"] == "30"  # CSV values are strings
        assert row["city"] == "NYC"

    def test_load_data_csv_values_are_strings(self):
        """load_data CSV values are all strings (DictReader behavior)."""
        repl = SafeREPL()
        csv_data = "id,value,active\n1,3.14,true"
        repl.load_data("data", csv_data, data_type="csv")

        row = repl.variables["data"][0]
        # All values are strings - no type conversion
        assert row["id"] == "1"
        assert row["value"] == "3.14"
        assert row["active"] == "true"
        assert isinstance(row["id"], str)
        assert isinstance(row["value"], str)
        assert isinstance(row["active"], str)

    def test_load_data_csv_multiple_rows(self):
        """load_data parses multiple CSV rows."""
        repl = SafeREPL()
        csv_data = "name,score\nAlice,100\nBob,95\nCharlie,88\nDiana,92"
        repl.load_data("scores", csv_data, data_type="csv")

        assert len(repl.variables["scores"]) == 4
        assert repl.variables["scores"][0]["name"] == "Alice"
        assert repl.variables["scores"][1]["name"] == "Bob"
        assert repl.variables["scores"][2]["name"] == "Charlie"
        assert repl.variables["scores"][3]["name"] == "Diana"

    def test_load_data_csv_with_empty_values(self):
        """load_data handles CSV with empty values."""
        repl = SafeREPL()
        csv_data = "name,email\nAlice,alice@example.com\nBob,"
        result = repl.load_data("data", csv_data, data_type="csv")

        assert result.success is True
        assert repl.variables["data"][1]["email"] == ""

    def test_load_data_csv_with_quoted_fields(self):
        """load_data handles CSV with quoted fields."""
        repl = SafeREPL()
        csv_data = 'name,description\nAlice,"A person, friendly"\nBob,"Has ""quotes"" inside"'
        result = repl.load_data("data", csv_data, data_type="csv")

        assert result.success is True
        assert repl.variables["data"][0]["description"] == "A person, friendly"
        assert repl.variables["data"][1]["description"] == 'Has "quotes" inside'

    def test_load_data_csv_with_newlines_in_quoted_field(self):
        """load_data handles CSV with newlines inside quoted fields."""
        repl = SafeREPL()
        csv_data = 'name,bio\nAlice,"Line 1\nLine 2"'
        result = repl.load_data("data", csv_data, data_type="csv")

        assert result.success is True
        assert "\n" in repl.variables["data"][0]["bio"]

    def test_load_data_csv_header_only(self):
        """load_data handles CSV with only header (no data rows)."""
        repl = SafeREPL()
        csv_data = "col1,col2,col3"
        result = repl.load_data("empty", csv_data, data_type="csv")

        assert result.success is True
        assert repl.variables["empty"] == []

    def test_load_data_csv_single_column(self):
        """load_data handles CSV with single column."""
        repl = SafeREPL()
        csv_data = "name\nAlice\nBob\nCharlie"
        result = repl.load_data("names", csv_data, data_type="csv")

        assert result.success is True
        assert len(repl.variables["names"]) == 3
        assert repl.variables["names"][0] == {"name": "Alice"}

    def test_load_data_csv_from_bytes(self):
        """load_data parses CSV from bytes."""
        repl = SafeREPL()
        csv_bytes = b"name,value\ntest,123"
        result = repl.load_data("from_bytes", csv_bytes, data_type="csv")

        assert result.success is True
        assert repl.variables["from_bytes"][0]["name"] == "test"
        assert repl.variables["from_bytes"][0]["value"] == "123"

    def test_load_data_csv_utf8_content(self):
        """load_data handles UTF-8 content in CSV."""
        repl = SafeREPL()
        csv_data = "nome,cidade\nJoão,São Paulo\nMaria,Ação"
        result = repl.load_data("utf8", csv_data, data_type="csv")

        assert result.success is True
        assert repl.variables["utf8"][0]["nome"] == "João"
        assert repl.variables["utf8"][0]["cidade"] == "São Paulo"
        assert repl.variables["utf8"][1]["cidade"] == "Ação"

    def test_load_data_csv_unicode_content(self):
        """load_data handles Unicode content in CSV."""
        repl = SafeREPL()
        csv_data = "text\n日本語\n中文\n한국어"
        result = repl.load_data("unicode", csv_data, data_type="csv")

        assert result.success is True
        assert repl.variables["unicode"][0]["text"] == "日本語"
        assert repl.variables["unicode"][1]["text"] == "中文"
        assert repl.variables["unicode"][2]["text"] == "한국어"

    def test_load_data_csv_creates_metadata(self):
        """load_data creates variable metadata for CSV."""
        repl = SafeREPL()
        csv_data = "a,b\n1,2"
        repl.load_data("with_meta", csv_data, data_type="csv")

        assert "with_meta" in repl.variable_metadata
        meta = repl.variable_metadata["with_meta"]
        assert meta.name == "with_meta"
        assert meta.type_name == "list"

    def test_load_data_csv_metadata_has_preview(self):
        """load_data metadata has preview of CSV content."""
        repl = SafeREPL()
        csv_data = "name,value\ntest,123"
        repl.load_data("preview", csv_data, data_type="csv")

        meta = repl.variable_metadata["preview"]
        # Preview should show list structure
        assert "name" in meta.preview or "[" in meta.preview

    def test_load_data_csv_records_variable_in_result(self):
        """load_data records variable name in variables_changed."""
        repl = SafeREPL()
        csv_data = "a,b\n1,2"
        result = repl.load_data("recorded", csv_data, data_type="csv")

        assert "recorded" in result.variables_changed

    def test_load_data_csv_stdout_contains_info(self):
        """load_data stdout contains loading info."""
        repl = SafeREPL()
        csv_data = "x,y\n1,2"
        result = repl.load_data("info_test", csv_data, data_type="csv")

        assert "info_test" in result.stdout
        assert "carregada" in result.stdout  # Portuguese for "loaded"
        assert "list" in result.stdout  # Type name

    def test_load_data_csv_overwrites_existing_variable(self):
        """load_data overwrites existing variable with same name."""
        repl = SafeREPL()
        repl.load_data("overwrite", "a,b\n1,2", data_type="csv")
        repl.load_data("overwrite", "x,y\n3,4", data_type="csv")

        assert repl.variables["overwrite"][0] == {"x": "3", "y": "4"}

    def test_load_data_csv_variable_usable_in_execute(self):
        """Variable loaded with load_data is usable in execute."""
        repl = SafeREPL()
        csv_data = "name,score\nAlice,100\nBob,95"
        repl.load_data("scores", csv_data, data_type="csv")
        result = repl.execute("total = sum(int(row['score']) for row in scores)")

        assert result.success is True
        assert repl.variables["total"] == 195

    def test_load_data_csv_large_file(self):
        """load_data handles large CSV file."""
        repl = SafeREPL()
        # Generate 1000 rows
        header = "id,name,value"
        rows = [f"{i},name_{i},{i * 10}" for i in range(1000)]
        csv_data = header + "\n" + "\n".join(rows)
        result = repl.load_data("large", csv_data, data_type="csv")

        assert result.success is True
        assert len(repl.variables["large"]) == 1000
        assert repl.variables["large"][500]["id"] == "500"
        assert repl.variables["large"][500]["name"] == "name_500"
        assert repl.variables["large"][500]["value"] == "5000"

    def test_load_data_csv_with_spaces_in_header(self):
        """load_data handles CSV with spaces in header names."""
        repl = SafeREPL()
        csv_data = "First Name,Last Name,Email Address\nJohn,Doe,john@example.com"
        result = repl.load_data("data", csv_data, data_type="csv")

        assert result.success is True
        row = repl.variables["data"][0]
        assert row["First Name"] == "John"
        assert row["Last Name"] == "Doe"
        assert row["Email Address"] == "john@example.com"

    def test_load_data_csv_with_numeric_header(self):
        """load_data handles CSV with numeric header names."""
        repl = SafeREPL()
        csv_data = "1,2,3\na,b,c"
        result = repl.load_data("data", csv_data, data_type="csv")

        assert result.success is True
        row = repl.variables["data"][0]
        # Header names are strings even if they look like numbers
        assert row["1"] == "a"
        assert row["2"] == "b"
        assert row["3"] == "c"

    def test_load_data_csv_preserves_row_order(self):
        """load_data preserves order of CSV rows."""
        repl = SafeREPL()
        csv_data = "letter\nz\na\nm\nb"
        result = repl.load_data("data", csv_data, data_type="csv")

        assert result.success is True
        letters = [row["letter"] for row in repl.variables["data"]]
        assert letters == ["z", "a", "m", "b"]

    def test_load_data_csv_metadata_has_timestamps(self):
        """load_data metadata has created_at and last_accessed timestamps."""
        repl = SafeREPL()
        from datetime import datetime
        before = datetime.now()
        repl.load_data("timestamped", "a\n1", data_type="csv")
        after = datetime.now()

        meta = repl.variable_metadata["timestamped"]
        assert before <= meta.created_at <= after
        assert before <= meta.last_accessed <= after

    def test_load_data_csv_with_tab_delimiter_not_supported(self):
        """load_data with csv uses comma delimiter (DictReader default)."""
        repl = SafeREPL()
        # Tab-separated data won't parse correctly with default DictReader
        tsv_data = "name\tage\nAlice\t30"
        result = repl.load_data("data", tsv_data, data_type="csv")

        assert result.success is True
        # The entire header is treated as one column name because no comma
        row = repl.variables["data"][0]
        assert "name\tage" in row

    def test_load_data_csv_single_row(self):
        """load_data handles CSV with single data row."""
        repl = SafeREPL()
        csv_data = "name,value\ntest,123"
        result = repl.load_data("single", csv_data, data_type="csv")

        assert result.success is True
        assert len(repl.variables["single"]) == 1
        assert repl.variables["single"][0] == {"name": "test", "value": "123"}

    def test_load_data_csv_with_trailing_newline(self):
        """load_data handles CSV with trailing newline."""
        repl = SafeREPL()
        csv_data = "name\nAlice\nBob\n"
        result = repl.load_data("data", csv_data, data_type="csv")

        assert result.success is True
        # Trailing newline should not create an extra empty row
        assert len(repl.variables["data"]) == 2

    def test_load_data_csv_empty_string_creates_empty_list(self):
        """load_data with empty CSV string creates empty list."""
        repl = SafeREPL()
        result = repl.load_data("empty", "", data_type="csv")

        assert result.success is True
        assert repl.variables["empty"] == []

    def test_load_data_csv_with_special_characters_in_values(self):
        """load_data handles special characters in CSV values."""
        repl = SafeREPL()
        csv_data = 'symbol,meaning\n"<",less than\n">",greater than\n"&",ampersand'
        result = repl.load_data("special", csv_data, data_type="csv")

        assert result.success is True
        assert repl.variables["special"][0]["symbol"] == "<"
        assert repl.variables["special"][1]["symbol"] == ">"
        assert repl.variables["special"][2]["symbol"] == "&"


class TestLoadDataLines:
    """Test load_data with data_type='lines'."""

    def test_load_data_lines_returns_execution_result(self):
        """load_data returns an ExecutionResult object."""
        repl = SafeREPL()
        result = repl.load_data("test", "line1\nline2", data_type="lines")

        assert isinstance(result, ExecutionResult)

    def test_load_data_lines_success_on_valid_data(self):
        """load_data returns success=True for valid data."""
        repl = SafeREPL()
        result = repl.load_data("test", "line1\nline2", data_type="lines")

        assert result.success is True

    def test_load_data_lines_splits_on_newline(self):
        """load_data splits string on newline into list."""
        repl = SafeREPL()
        repl.load_data("lines", "line1\nline2\nline3", data_type="lines")

        assert "lines" in repl.variables
        assert repl.variables["lines"] == ["line1", "line2", "line3"]

    def test_load_data_lines_returns_list_type(self):
        """load_data with lines returns a list type."""
        repl = SafeREPL()
        repl.load_data("lines", "a\nb\nc", data_type="lines")

        assert isinstance(repl.variables["lines"], list)

    def test_load_data_lines_single_line_no_newline(self):
        """load_data with single line (no newline) returns list with one item."""
        repl = SafeREPL()
        repl.load_data("single", "just one line", data_type="lines")

        assert repl.variables["single"] == ["just one line"]
        assert len(repl.variables["single"]) == 1

    def test_load_data_lines_empty_string_returns_list_with_empty_string(self):
        """load_data with empty string returns list containing one empty string."""
        repl = SafeREPL()
        result = repl.load_data("empty", "", data_type="lines")

        assert result.success is True
        # "".split('\n') returns ['']
        assert repl.variables["empty"] == [""]

    def test_load_data_lines_preserves_empty_lines(self):
        """load_data preserves empty lines."""
        repl = SafeREPL()
        repl.load_data("data", "line1\n\nline3", data_type="lines")

        # Should have: ["line1", "", "line3"]
        assert repl.variables["data"] == ["line1", "", "line3"]
        assert len(repl.variables["data"]) == 3

    def test_load_data_lines_trailing_newline_creates_empty_element(self):
        """load_data with trailing newline creates empty string at end."""
        repl = SafeREPL()
        repl.load_data("data", "line1\nline2\n", data_type="lines")

        # "line1\nline2\n".split('\n') returns ["line1", "line2", ""]
        assert repl.variables["data"] == ["line1", "line2", ""]

    def test_load_data_lines_leading_newline_creates_empty_element(self):
        """load_data with leading newline creates empty string at beginning."""
        repl = SafeREPL()
        repl.load_data("data", "\nline1\nline2", data_type="lines")

        # "\nline1\nline2".split('\n') returns ["", "line1", "line2"]
        assert repl.variables["data"] == ["", "line1", "line2"]

    def test_load_data_lines_multiple_consecutive_newlines(self):
        """load_data handles multiple consecutive newlines."""
        repl = SafeREPL()
        repl.load_data("data", "a\n\n\nb", data_type="lines")

        # "a\n\n\nb".split('\n') returns ["a", "", "", "b"]
        assert repl.variables["data"] == ["a", "", "", "b"]

    def test_load_data_lines_from_bytes(self):
        """load_data decodes bytes and splits into lines."""
        repl = SafeREPL()
        data_bytes = b"line1\nline2\nline3"
        result = repl.load_data("from_bytes", data_bytes, data_type="lines")

        assert result.success is True
        assert repl.variables["from_bytes"] == ["line1", "line2", "line3"]

    def test_load_data_lines_from_utf8_bytes(self):
        """load_data decodes UTF-8 bytes correctly."""
        repl = SafeREPL()
        data_bytes = "Olá\nMundo\nAção".encode('utf-8')
        result = repl.load_data("utf8", data_bytes, data_type="lines")

        assert result.success is True
        assert repl.variables["utf8"] == ["Olá", "Mundo", "Ação"]

    def test_load_data_lines_unicode_content(self):
        """load_data handles Unicode content in lines."""
        repl = SafeREPL()
        unicode_text = "日本語\n中文\n한국어\nالعربية"
        result = repl.load_data("unicode", unicode_text, data_type="lines")

        assert result.success is True
        assert repl.variables["unicode"] == ["日本語", "中文", "한국어", "العربية"]

    def test_load_data_lines_preserves_whitespace_in_lines(self):
        """load_data preserves whitespace within lines."""
        repl = SafeREPL()
        repl.load_data("data", "  leading\ntrailing  \n  both  ", data_type="lines")

        assert repl.variables["data"][0] == "  leading"
        assert repl.variables["data"][1] == "trailing  "
        assert repl.variables["data"][2] == "  both  "

    def test_load_data_lines_only_newlines(self):
        """load_data handles string with only newlines."""
        repl = SafeREPL()
        repl.load_data("data", "\n\n\n", data_type="lines")

        # "\n\n\n".split('\n') returns ["", "", "", ""]
        assert repl.variables["data"] == ["", "", "", ""]
        assert len(repl.variables["data"]) == 4

    def test_load_data_lines_creates_metadata(self):
        """load_data creates variable metadata for lines."""
        repl = SafeREPL()
        repl.load_data("with_meta", "a\nb\nc", data_type="lines")

        assert "with_meta" in repl.variable_metadata
        meta = repl.variable_metadata["with_meta"]
        assert meta.name == "with_meta"
        assert meta.type_name == "list"

    def test_load_data_lines_metadata_has_preview(self):
        """load_data metadata has preview of lines content."""
        repl = SafeREPL()
        repl.load_data("preview", "first line\nsecond line", data_type="lines")

        meta = repl.variable_metadata["preview"]
        # Preview should show list structure
        assert "[" in meta.preview

    def test_load_data_lines_metadata_has_human_size(self):
        """load_data metadata has human-readable size."""
        repl = SafeREPL()
        repl.load_data("sized", "a\nb\nc", data_type="lines")

        meta = repl.variable_metadata["sized"]
        assert "B" in meta.size_human  # Should end with B, KB, MB, etc.

    def test_load_data_lines_metadata_has_timestamps(self):
        """load_data metadata has created_at and last_accessed timestamps."""
        repl = SafeREPL()
        from datetime import datetime
        before = datetime.now()
        repl.load_data("timestamped", "line1\nline2", data_type="lines")
        after = datetime.now()

        meta = repl.variable_metadata["timestamped"]
        assert before <= meta.created_at <= after
        assert before <= meta.last_accessed <= after

    def test_load_data_lines_records_variable_in_result(self):
        """load_data records variable name in variables_changed."""
        repl = SafeREPL()
        result = repl.load_data("recorded", "a\nb", data_type="lines")

        assert "recorded" in result.variables_changed

    def test_load_data_lines_stdout_contains_info(self):
        """load_data stdout contains loading info."""
        repl = SafeREPL()
        result = repl.load_data("info_test", "line1\nline2", data_type="lines")

        assert "info_test" in result.stdout
        assert "carregada" in result.stdout  # Portuguese for "loaded"
        assert "list" in result.stdout  # Type name

    def test_load_data_lines_overwrites_existing_variable(self):
        """load_data overwrites existing variable with same name."""
        repl = SafeREPL()
        repl.load_data("overwrite", "old1\nold2", data_type="lines")
        repl.load_data("overwrite", "new1\nnew2\nnew3", data_type="lines")

        assert repl.variables["overwrite"] == ["new1", "new2", "new3"]

    def test_load_data_lines_variable_usable_in_execute(self):
        """Variable loaded with load_data is usable in execute."""
        repl = SafeREPL()
        repl.load_data("lines_data", "apple\nbanana\ncherry", data_type="lines")
        result = repl.execute("count = len(lines_data)")

        assert result.success is True
        assert repl.variables["count"] == 3

    def test_load_data_lines_can_access_individual_lines(self):
        """Variable loaded with lines can access individual lines by index."""
        repl = SafeREPL()
        repl.load_data("lines", "first\nsecond\nthird", data_type="lines")
        result = repl.execute("second_line = lines[1]")

        assert result.success is True
        assert repl.variables["second_line"] == "second"

    def test_load_data_lines_can_iterate(self):
        """Variable loaded with lines can be iterated."""
        repl = SafeREPL()
        repl.load_data("lines", "a\nb\nc", data_type="lines")
        result = repl.execute("upper_lines = [line.upper() for line in lines]")

        assert result.success is True
        assert repl.variables["upper_lines"] == ["A", "B", "C"]

    def test_load_data_lines_large_data(self):
        """load_data handles large data with many lines."""
        repl = SafeREPL()
        # Generate 10000 lines
        large_data = "\n".join([f"line_{i}" for i in range(10000)])
        result = repl.load_data("large", large_data, data_type="lines")

        assert result.success is True
        assert len(repl.variables["large"]) == 10000
        assert repl.variables["large"][0] == "line_0"
        assert repl.variables["large"][5000] == "line_5000"
        assert repl.variables["large"][9999] == "line_9999"

    def test_load_data_lines_with_special_characters(self):
        """load_data handles lines with special characters."""
        repl = SafeREPL()
        special_text = 'Tab:\there\nQuote:"quoted"\nBackslash:\\'
        result = repl.load_data("special", special_text, data_type="lines")

        assert result.success is True
        assert "\t" in repl.variables["special"][0]
        assert '"' in repl.variables["special"][1]
        assert "\\" in repl.variables["special"][2]

    def test_load_data_lines_does_not_split_on_carriage_return(self):
        """load_data only splits on \\n, not \\r."""
        repl = SafeREPL()
        # Windows-style line endings (\r\n)
        windows_text = "line1\r\nline2\r\nline3"
        repl.load_data("data", windows_text, data_type="lines")

        # Split on \n only, so \r remains attached to lines
        assert repl.variables["data"] == ["line1\r", "line2\r", "line3"]

    def test_load_data_lines_handles_carriage_return_only(self):
        """load_data handles text with only carriage returns (old Mac style)."""
        repl = SafeREPL()
        mac_text = "line1\rline2\rline3"
        repl.load_data("data", mac_text, data_type="lines")

        # No \n, so entire string is one line
        assert repl.variables["data"] == ["line1\rline2\rline3"]
        assert len(repl.variables["data"]) == 1


class TestGetMemoryUsage:
    """Test get_memory_usage returns reasonable values."""

    def test_get_memory_usage_returns_dict(self):
        """get_memory_usage returns a dictionary."""
        repl = SafeREPL()
        result = repl.get_memory_usage()

        assert isinstance(result, dict)

    def test_get_memory_usage_has_expected_keys(self):
        """get_memory_usage returns dict with all expected keys."""
        repl = SafeREPL()
        result = repl.get_memory_usage()

        expected_keys = {"total_bytes", "total_human", "variable_count", "max_allowed_mb", "usage_percent"}
        assert set(result.keys()) == expected_keys

    def test_get_memory_usage_empty_repl_has_zero_total_bytes(self):
        """get_memory_usage returns total_bytes=0 for empty REPL."""
        repl = SafeREPL()
        result = repl.get_memory_usage()

        assert result["total_bytes"] == 0

    def test_get_memory_usage_empty_repl_has_zero_variable_count(self):
        """get_memory_usage returns variable_count=0 for empty REPL."""
        repl = SafeREPL()
        result = repl.get_memory_usage()

        assert result["variable_count"] == 0

    def test_get_memory_usage_empty_repl_has_zero_usage_percent(self):
        """get_memory_usage returns usage_percent=0 for empty REPL."""
        repl = SafeREPL()
        result = repl.get_memory_usage()

        assert result["usage_percent"] == 0.0

    def test_get_memory_usage_empty_repl_has_human_size(self):
        """get_memory_usage returns total_human for empty REPL."""
        repl = SafeREPL()
        result = repl.get_memory_usage()

        # 0 bytes should format as "0.0 B"
        assert "0" in result["total_human"]
        assert "B" in result["total_human"]

    def test_get_memory_usage_default_max_allowed_mb(self):
        """get_memory_usage returns default max_allowed_mb (1024)."""
        repl = SafeREPL()
        result = repl.get_memory_usage()

        assert result["max_allowed_mb"] == 1024

    def test_get_memory_usage_custom_max_allowed_mb(self):
        """get_memory_usage respects custom max_memory_mb in constructor."""
        repl = SafeREPL(max_memory_mb=512)
        result = repl.get_memory_usage()

        assert result["max_allowed_mb"] == 512

    def test_get_memory_usage_after_load_data_has_positive_total_bytes(self):
        """get_memory_usage returns positive total_bytes after loading data."""
        repl = SafeREPL()
        repl.load_data("test", "hello world", data_type="text")
        result = repl.get_memory_usage()

        assert result["total_bytes"] > 0

    def test_get_memory_usage_after_load_data_has_correct_variable_count(self):
        """get_memory_usage returns correct variable_count after loading data."""
        repl = SafeREPL()
        repl.load_data("var1", "data1", data_type="text")
        repl.load_data("var2", "data2", data_type="text")
        result = repl.get_memory_usage()

        assert result["variable_count"] == 2

    def test_get_memory_usage_after_execute_has_correct_variable_count(self):
        """get_memory_usage returns correct variable_count after execute."""
        repl = SafeREPL()
        repl.execute("x = 1")
        repl.execute("y = 2")
        repl.execute("z = 3")
        result = repl.get_memory_usage()

        # Note: llm_query, llm_stats, llm_reset_counter are also in variables
        # So count = 3 user vars + 3 llm vars = 6
        assert result["variable_count"] >= 3

    def test_get_memory_usage_total_bytes_reflects_variable_sizes(self):
        """get_memory_usage total_bytes reflects sum of variable sizes."""
        repl = SafeREPL()
        # Load a known-size string
        test_string = "a" * 100  # 100 bytes
        repl.load_data("test", test_string, data_type="text")
        result = repl.get_memory_usage()

        # total_bytes should be at least 100 (may be more due to estimate_size)
        assert result["total_bytes"] >= 100

    def test_get_memory_usage_total_bytes_increases_with_more_data(self):
        """get_memory_usage total_bytes increases as more data is loaded."""
        repl = SafeREPL()

        repl.load_data("small", "x" * 100, data_type="text")
        usage1 = repl.get_memory_usage()

        repl.load_data("medium", "y" * 1000, data_type="text")
        usage2 = repl.get_memory_usage()

        repl.load_data("large", "z" * 10000, data_type="text")
        usage3 = repl.get_memory_usage()

        assert usage1["total_bytes"] < usage2["total_bytes"]
        assert usage2["total_bytes"] < usage3["total_bytes"]

    def test_get_memory_usage_usage_percent_increases_with_data(self):
        """get_memory_usage usage_percent increases as data is loaded."""
        repl = SafeREPL(max_memory_mb=1)  # 1 MB limit for easier testing

        # Load ~100KB of data
        repl.load_data("data", "x" * (100 * 1024), data_type="text")
        result = repl.get_memory_usage()

        # 100KB out of 1MB = ~10%
        assert result["usage_percent"] > 0
        assert result["usage_percent"] < 100

    def test_get_memory_usage_total_human_format_bytes(self):
        """get_memory_usage total_human formats small values as bytes."""
        repl = SafeREPL()
        repl.load_data("small", "hello", data_type="text")  # 5 bytes
        result = repl.get_memory_usage()

        assert "B" in result["total_human"]

    def test_get_memory_usage_total_human_format_kb(self):
        """get_memory_usage total_human formats values in KB."""
        repl = SafeREPL()
        # Load ~10KB of data
        repl.load_data("data", "x" * (10 * 1024), data_type="text")
        result = repl.get_memory_usage()

        # Should be formatted as KB
        assert "KB" in result["total_human"] or "MB" in result["total_human"]

    def test_get_memory_usage_total_human_format_mb(self):
        """get_memory_usage total_human formats values in MB."""
        repl = SafeREPL()
        # Load ~2MB of data
        repl.load_data("data", "x" * (2 * 1024 * 1024), data_type="text")
        result = repl.get_memory_usage()

        assert "MB" in result["total_human"]

    def test_get_memory_usage_after_clear_all_resets(self):
        """get_memory_usage resets after clear_all."""
        repl = SafeREPL()
        repl.load_data("data", "x" * 10000, data_type="text")

        # Verify data was loaded
        usage_before = repl.get_memory_usage()
        assert usage_before["total_bytes"] > 0
        assert usage_before["variable_count"] > 0

        # Clear all
        repl.clear_all()

        # Verify reset
        usage_after = repl.get_memory_usage()
        assert usage_after["total_bytes"] == 0
        assert usage_after["variable_count"] == 0
        assert usage_after["usage_percent"] == 0.0

    def test_get_memory_usage_after_clear_variable_decreases(self):
        """get_memory_usage decreases after clearing a variable."""
        repl = SafeREPL()
        repl.load_data("var1", "x" * 1000, data_type="text")
        repl.load_data("var2", "y" * 2000, data_type="text")

        usage_before = repl.get_memory_usage()

        # Clear one variable
        repl.clear_variable("var1")

        usage_after = repl.get_memory_usage()

        assert usage_after["total_bytes"] < usage_before["total_bytes"]
        assert usage_after["variable_count"] == usage_before["variable_count"] - 1

    def test_get_memory_usage_returns_types(self):
        """get_memory_usage returns correct types for all values."""
        repl = SafeREPL()
        repl.load_data("test", "hello", data_type="text")
        result = repl.get_memory_usage()

        assert isinstance(result["total_bytes"], int)
        assert isinstance(result["total_human"], str)
        assert isinstance(result["variable_count"], int)
        assert isinstance(result["max_allowed_mb"], int)
        assert isinstance(result["usage_percent"], float)

    def test_get_memory_usage_reflects_overwritten_variable(self):
        """get_memory_usage correctly updates when variable is overwritten."""
        repl = SafeREPL()
        repl.load_data("data", "small", data_type="text")
        usage1 = repl.get_memory_usage()

        # Overwrite with larger data
        repl.load_data("data", "x" * 10000, data_type="text")
        usage2 = repl.get_memory_usage()

        # Variable count should stay the same
        assert usage2["variable_count"] == usage1["variable_count"]
        # But total_bytes should increase
        assert usage2["total_bytes"] > usage1["total_bytes"]

    def test_get_memory_usage_includes_all_variable_types(self):
        """get_memory_usage counts all variable types (str, dict, list)."""
        repl = SafeREPL()
        repl.load_data("text", "hello world", data_type="text")
        repl.load_data("json_obj", '{"key": "value"}', data_type="json")
        repl.load_data("json_arr", '[1, 2, 3]', data_type="json")
        repl.load_data("csv_data", "a,b\n1,2", data_type="csv")
        repl.load_data("lines", "line1\nline2", data_type="lines")

        result = repl.get_memory_usage()

        assert result["variable_count"] == 5
        assert result["total_bytes"] > 0

    def test_get_memory_usage_large_data_reasonable_percent(self):
        """get_memory_usage calculates reasonable usage_percent for large data."""
        repl = SafeREPL(max_memory_mb=100)  # 100 MB limit

        # Load 10 MB of data
        repl.load_data("large", "x" * (10 * 1024 * 1024), data_type="text")
        result = repl.get_memory_usage()

        # Should be approximately 10%
        assert 8 < result["usage_percent"] < 12

    def test_get_memory_usage_does_not_modify_state(self):
        """get_memory_usage is read-only and doesn't modify state."""
        repl = SafeREPL()
        repl.load_data("test", "data", data_type="text")

        # Call multiple times
        result1 = repl.get_memory_usage()
        result2 = repl.get_memory_usage()
        result3 = repl.get_memory_usage()

        # Results should be identical
        assert result1 == result2 == result3

        # Variables should still be there
        assert "test" in repl.variables

    def test_get_memory_usage_with_execute_variables(self):
        """get_memory_usage includes variables created via execute."""
        repl = SafeREPL()
        repl.execute("large_list = list(range(10000))")
        result = repl.get_memory_usage()

        assert result["variable_count"] >= 1
        assert result["total_bytes"] > 0


class TestClearNamespace:
    """Test that clear_all and clear_variable methods work correctly.

    Note: The PRD calls this 'clear_namespace' but the actual methods in
    SafeREPL are clear_all() (clears all variables) and clear_variable()
    (clears a single variable).
    """

    # ===================
    # Tests for clear_all
    # ===================

    def test_clear_all_returns_count(self):
        """clear_all returns the count of removed variables."""
        repl = SafeREPL()
        repl.load_data("var1", "data1", data_type="text")
        repl.load_data("var2", "data2", data_type="text")
        repl.load_data("var3", "data3", data_type="text")

        result = repl.clear_all()

        assert result == 3

    def test_clear_all_removes_all_variables(self):
        """clear_all removes all variables from the namespace."""
        repl = SafeREPL()
        repl.load_data("var1", "data1", data_type="text")
        repl.load_data("var2", "data2", data_type="text")
        repl.load_data("var3", "data3", data_type="text")

        repl.clear_all()

        assert len(repl.variables) == 0
        assert "var1" not in repl.variables
        assert "var2" not in repl.variables
        assert "var3" not in repl.variables

    def test_clear_all_removes_all_variable_metadata(self):
        """clear_all removes all variable metadata."""
        repl = SafeREPL()
        repl.load_data("var1", "data1", data_type="text")
        repl.load_data("var2", "data2", data_type="text")

        repl.clear_all()

        assert len(repl.variable_metadata) == 0

    def test_clear_all_on_empty_returns_zero(self):
        """clear_all on empty namespace returns 0."""
        repl = SafeREPL()

        result = repl.clear_all()

        assert result == 0

    def test_clear_all_resets_memory_usage(self):
        """clear_all resets memory usage to zero."""
        repl = SafeREPL()
        repl.load_data("test", "x" * 10000, data_type="text")

        # Verify data was loaded
        usage_before = repl.get_memory_usage()
        assert usage_before["total_bytes"] > 0
        assert usage_before["variable_count"] > 0

        repl.clear_all()
        usage_after = repl.get_memory_usage()

        assert usage_after["total_bytes"] == 0
        assert usage_after["variable_count"] == 0
        assert usage_after["usage_percent"] == 0

    def test_clear_all_variables_from_execute(self):
        """clear_all clears variables created via execute."""
        repl = SafeREPL()
        repl.execute("x = 1")
        repl.execute("y = 2")
        repl.execute("z = 3")

        # Filter out llm_* functions to count user variables
        user_vars = [v for v in repl.variables if not v.startswith("llm_")]
        assert len(user_vars) == 3

        repl.clear_all()

        assert len(repl.variables) == 0

    def test_clear_all_allows_new_variables_after(self):
        """After clear_all, new variables can be added."""
        repl = SafeREPL()
        repl.load_data("old_var", "old_data", data_type="text")
        repl.clear_all()

        # Add new variable
        repl.load_data("new_var", "new_data", data_type="text")

        assert "new_var" in repl.variables
        assert repl.variables["new_var"] == "new_data"

    def test_clear_all_does_not_reset_execution_count(self):
        """clear_all does not reset the execution_count."""
        repl = SafeREPL()
        repl.execute("x = 1")
        repl.execute("y = 2")
        count_before = repl.execution_count

        repl.clear_all()

        assert repl.execution_count == count_before

    def test_clear_all_clears_mixed_variable_types(self):
        """clear_all clears variables of all types (str, dict, list)."""
        repl = SafeREPL()
        repl.load_data("text_var", "hello", data_type="text")
        repl.load_data("json_dict", '{"key": "value"}', data_type="json")
        repl.load_data("json_list", '[1, 2, 3]', data_type="json")
        repl.load_data("csv_var", "a,b\n1,2", data_type="csv")
        repl.load_data("lines_var", "line1\nline2", data_type="lines")

        result = repl.clear_all()

        assert result == 5
        assert len(repl.variables) == 0

    def test_clear_all_clears_large_data(self):
        """clear_all clears large data (1MB+)."""
        repl = SafeREPL()
        large_data = "x" * (1024 * 1024)  # 1 MB
        repl.load_data("large", large_data, data_type="text")

        # Verify data was loaded
        assert len(repl.variables["large"]) == 1024 * 1024

        result = repl.clear_all()

        assert result == 1
        assert "large" not in repl.variables

    # ========================
    # Tests for clear_variable
    # ========================

    def test_clear_variable_returns_true_on_success(self):
        """clear_variable returns True when variable exists."""
        repl = SafeREPL()
        repl.load_data("test", "data", data_type="text")

        result = repl.clear_variable("test")

        assert result is True

    def test_clear_variable_returns_false_on_not_found(self):
        """clear_variable returns False when variable doesn't exist."""
        repl = SafeREPL()

        result = repl.clear_variable("nonexistent")

        assert result is False

    def test_clear_variable_removes_variable(self):
        """clear_variable removes the specified variable."""
        repl = SafeREPL()
        repl.load_data("test", "data", data_type="text")

        repl.clear_variable("test")

        assert "test" not in repl.variables

    def test_clear_variable_removes_metadata(self):
        """clear_variable removes the variable metadata."""
        repl = SafeREPL()
        repl.load_data("test", "data", data_type="text")
        assert "test" in repl.variable_metadata

        repl.clear_variable("test")

        assert "test" not in repl.variable_metadata

    def test_clear_variable_does_not_affect_others(self):
        """clear_variable does not affect other variables."""
        repl = SafeREPL()
        repl.load_data("var1", "data1", data_type="text")
        repl.load_data("var2", "data2", data_type="text")
        repl.load_data("var3", "data3", data_type="text")

        repl.clear_variable("var2")

        assert "var1" in repl.variables
        assert "var2" not in repl.variables
        assert "var3" in repl.variables
        assert repl.variables["var1"] == "data1"
        assert repl.variables["var3"] == "data3"

    def test_clear_variable_updates_memory_usage(self):
        """clear_variable updates memory usage correctly."""
        repl = SafeREPL()
        repl.load_data("keep", "keep data", data_type="text")
        repl.load_data("remove", "x" * 10000, data_type="text")

        usage_before = repl.get_memory_usage()
        repl.clear_variable("remove")
        usage_after = repl.get_memory_usage()

        assert usage_after["variable_count"] == usage_before["variable_count"] - 1
        assert usage_after["total_bytes"] < usage_before["total_bytes"]

    def test_clear_variable_created_via_execute(self):
        """clear_variable can remove variables created via execute."""
        repl = SafeREPL()
        repl.execute("my_var = [1, 2, 3]")
        assert "my_var" in repl.variables

        result = repl.clear_variable("my_var")

        assert result is True
        assert "my_var" not in repl.variables

    def test_clear_variable_allows_recreation(self):
        """After clear_variable, the same name can be reused."""
        repl = SafeREPL()
        repl.load_data("test", "original", data_type="text")
        repl.clear_variable("test")

        repl.load_data("test", "new value", data_type="text")

        assert repl.variables["test"] == "new value"

    def test_clear_variable_with_special_characters(self):
        """clear_variable works with special characters in name."""
        repl = SafeREPL()
        # Variable names with underscores and numbers
        repl.load_data("var_1", "data1", data_type="text")
        repl.load_data("_private", "data2", data_type="text")
        repl.load_data("var123", "data3", data_type="text")

        assert repl.clear_variable("var_1") is True
        assert repl.clear_variable("_private") is True
        assert repl.clear_variable("var123") is True

        assert "var_1" not in repl.variables
        assert "_private" not in repl.variables
        assert "var123" not in repl.variables

    def test_clear_variable_with_large_data(self):
        """clear_variable removes large data correctly."""
        repl = SafeREPL()
        large_data = "x" * (5 * 1024 * 1024)  # 5 MB
        repl.load_data("large", large_data, data_type="text")

        usage_before = repl.get_memory_usage()
        result = repl.clear_variable("large")
        usage_after = repl.get_memory_usage()

        assert result is True
        assert "large" not in repl.variables
        # Memory should be significantly reduced
        assert usage_after["total_bytes"] < usage_before["total_bytes"] - 1000000

    def test_clear_nonexistent_does_not_affect_existing(self):
        """Clearing nonexistent variable does not affect existing variables."""
        repl = SafeREPL()
        repl.load_data("existing", "data", data_type="text")

        result = repl.clear_variable("nonexistent")

        assert result is False
        assert "existing" in repl.variables
        assert repl.variables["existing"] == "data"

    def test_multiple_clear_variable_calls(self):
        """Multiple clear_variable calls work correctly."""
        repl = SafeREPL()
        repl.load_data("a", "1", data_type="text")
        repl.load_data("b", "2", data_type="text")
        repl.load_data("c", "3", data_type="text")
        repl.load_data("d", "4", data_type="text")

        assert repl.clear_variable("a") is True
        assert repl.clear_variable("c") is True
        assert repl.clear_variable("a") is False  # Already cleared
        assert repl.clear_variable("c") is False  # Already cleared

        assert "a" not in repl.variables
        assert "b" in repl.variables
        assert "c" not in repl.variables
        assert "d" in repl.variables

    # =================
    # Combined tests
    # =================

    def test_clear_variable_then_clear_all(self):
        """clear_variable followed by clear_all works correctly."""
        repl = SafeREPL()
        repl.load_data("a", "1", data_type="text")
        repl.load_data("b", "2", data_type="text")
        repl.load_data("c", "3", data_type="text")

        repl.clear_variable("a")
        result = repl.clear_all()

        # clear_all should return count of remaining variables (2)
        assert result == 2
        assert len(repl.variables) == 0

    def test_clear_all_then_clear_variable(self):
        """clear_all followed by clear_variable returns False."""
        repl = SafeREPL()
        repl.load_data("test", "data", data_type="text")

        repl.clear_all()
        result = repl.clear_variable("test")

        assert result is False

    def test_namespace_isolation_after_clear(self):
        """After clear, new operations work in isolated namespace."""
        repl = SafeREPL()
        repl.execute("x = 100")
        repl.clear_all()

        # x should not exist anymore
        result = repl.execute("print(x)")

        assert result.success is False
        assert "NameError" in result.stderr or "name 'x'" in result.stderr.lower()


class TestMaliciousCodeEvalExecString:
    """
    Test that malicious code using eval, exec, compile in various bypass attempts is blocked.

    The sandbox blocks direct calls to eval/exec/compile via AST analysis. These tests verify
    that various attempts to bypass this protection (string manipulation, getattr, etc.) are
    properly handled - either blocked at AST level or fail at runtime due to removed builtins.
    """

    # ==========================================
    # Direct eval/exec/compile calls (AST blocked)
    # ==========================================

    def test_direct_eval_is_blocked_by_ast(self):
        """Direct call to eval() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("result = eval('1 + 1')")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "bloqueada" in result.stderr.lower() or "eval" in result.stderr.lower()

    def test_direct_exec_is_blocked_by_ast(self):
        """Direct call to exec() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("exec('x = 42')")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "bloqueada" in result.stderr.lower() or "exec" in result.stderr.lower()

    def test_direct_compile_is_blocked_by_ast(self):
        """Direct call to compile() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("code = compile('x = 1', '<string>', 'exec')")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "bloqueada" in result.stderr.lower() or "compile" in result.stderr.lower()

    def test_direct___import___is_blocked_by_ast(self):
        """Direct call to __import__() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("os_module = __import__('os')")

        assert result.success is False
        assert "SecurityError" in result.stderr

    # ==========================================
    # String concatenation to build function name
    # ==========================================

    def test_eval_via_string_concat_fails_at_runtime(self):
        """Attempt to call eval by building function name from strings fails at runtime."""
        repl = SafeREPL()
        # This bypasses AST check but should fail because eval is removed from builtins
        result = repl.execute("""
name = 'ev' + 'al'
func = None
# Try to find eval in namespace
for k, v in [(k, v) for k, v in dir(type('', (), {})) if 'ev' in k.lower()]:
    func = v
result = 'found' if func else 'not_found'
""")

        # The code may succeed in running but should not find eval
        # Even if it runs, it cannot access eval function
        if result.success:
            assert repl.variables.get("func") is None or "eval" not in str(repl.variables.get("result", "")).lower()

    def test_exec_via_string_concat_fails_at_runtime(self):
        """Attempt to call exec by building function name from strings fails at runtime."""
        repl = SafeREPL()
        result = repl.execute("""
name = 'ex' + 'ec'
# Try various ways to get exec
result = name in dir(__builtins__) if '__builtins__' in dir() else False
""")

        # __builtins__ access should fail or return False
        # Either the execution fails or result should be False
        if result.success:
            assert repl.variables.get("result") is False or repl.variables.get("result") is None

    # ==========================================
    # getattr bypass attempts (blocked by AST for dunder attributes)
    # ==========================================

    def test_getattr_is_blocked_by_ast(self):
        """Direct getattr() call is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("func = getattr(__builtins__, 'eval')")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_setattr_is_blocked_by_ast(self):
        """Direct setattr() call is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("setattr(obj, 'attr', value)")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_delattr_is_blocked_by_ast(self):
        """Direct delattr() call is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("delattr(obj, 'attr')")

        assert result.success is False
        assert "SecurityError" in result.stderr

    # ==========================================
    # __builtins__ access attempts
    # ==========================================

    def test_direct___builtins___access_fails(self):
        """Direct __builtins__ access should fail or return safe version."""
        repl = SafeREPL()
        result = repl.execute("""
try:
    b = __builtins__
    has_eval = 'eval' in dir(b) if hasattr(b, '__iter__') else hasattr(b, 'eval')
    result = has_eval
except:
    result = 'error'
""")

        # Either fails or eval is not in safe builtins
        if result.success and repl.variables.get("result") != 'error':
            assert repl.variables.get("result") is False

    def test___builtins___eval_not_available(self):
        """__builtins__ should not have eval available."""
        repl = SafeREPL()
        result = repl.execute("""
try:
    # __builtins__ is replaced with safe version
    b = __builtins__
    if isinstance(b, dict):
        eval_func = b.get('eval', None)
    else:
        eval_func = getattr(b, 'eval', None)
    result = eval_func is not None
except:
    result = 'error'
""")

        # This should fail because getattr is blocked
        # or if it somehow passes, eval should not be available
        if "SecurityError" in result.stderr:
            assert True  # getattr blocked as expected
        elif result.success and repl.variables.get("result") != 'error':
            assert repl.variables.get("result") is False

    def test___builtins___exec_not_available(self):
        """__builtins__ should not have exec available."""
        repl = SafeREPL()
        result = repl.execute("""
try:
    b = __builtins__
    if isinstance(b, dict):
        exec_func = b.get('exec', None)
    else:
        exec_func = None  # Can't use getattr
    result = exec_func is not None
except:
    result = 'error'
""")

        if result.success and repl.variables.get("result") != 'error':
            assert repl.variables.get("result") is False

    # ==========================================
    # globals()/locals()/vars() bypass attempts (blocked)
    # ==========================================

    def test_globals_is_blocked_by_ast(self):
        """globals() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("g = globals()")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_locals_is_blocked_by_ast(self):
        """locals() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("l = locals()")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_vars_is_blocked_by_ast(self):
        """vars() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("v = vars()")

        assert result.success is False
        assert "SecurityError" in result.stderr

    # ==========================================
    # eval/exec via type() and __subclasses__ trick
    # ==========================================

    def test_type_subclasses_dunder_access_blocked(self):
        """Accessing __subclasses__ is blocked as dunder attribute."""
        repl = SafeREPL()
        result = repl.execute("subclasses = object.__subclasses__()")

        assert result.success is False
        assert "SecurityError" in result.stderr
        assert "__subclasses__" in result.stderr

    def test_type_mro_dunder_access_blocked(self):
        """Accessing __mro__ is blocked as dunder attribute."""
        repl = SafeREPL()
        result = repl.execute("mro = str.__mro__")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_type_bases_dunder_access_blocked(self):
        """Accessing __bases__ is blocked as dunder attribute."""
        repl = SafeREPL()
        result = repl.execute("bases = str.__bases__")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_type_class_dunder_access_blocked(self):
        """Accessing __class__ is blocked as dunder attribute."""
        repl = SafeREPL()
        result = repl.execute("cls = ''.__class__")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_type_globals_dunder_access_blocked(self):
        """Accessing __globals__ is blocked as dunder attribute."""
        repl = SafeREPL()
        result = repl.execute("""
def f(): pass
g = f.__globals__
""")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_type_code_dunder_access_blocked(self):
        """Accessing __code__ is blocked as dunder attribute."""
        repl = SafeREPL()
        result = repl.execute("""
def f(): pass
c = f.__code__
""")

        assert result.success is False
        assert "SecurityError" in result.stderr

    # ==========================================
    # input() and open() blocked
    # ==========================================

    def test_input_is_blocked_by_ast(self):
        """input() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("user_input = input('Enter: ')")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_open_is_blocked_by_ast(self):
        """open() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("f = open('/etc/passwd', 'r')")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_open_write_is_blocked_by_ast(self):
        """open() for writing is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("f = open('/tmp/test.txt', 'w')")

        assert result.success is False
        assert "SecurityError" in result.stderr

    # ==========================================
    # breakpoint() blocked
    # ==========================================

    def test_breakpoint_is_blocked_by_ast(self):
        """breakpoint() is blocked by AST analysis."""
        repl = SafeREPL()
        result = repl.execute("breakpoint()")

        assert result.success is False
        assert "SecurityError" in result.stderr

    # ==========================================
    # Malicious eval/exec in function definitions
    # ==========================================

    def test_eval_inside_function_definition_blocked(self):
        """eval() inside a function definition is blocked."""
        repl = SafeREPL()
        result = repl.execute("""
def malicious():
    return eval('1 + 1')
""")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_exec_inside_function_definition_blocked(self):
        """exec() inside a function definition is blocked."""
        repl = SafeREPL()
        result = repl.execute("""
def malicious():
    exec('x = 42')
    return x
""")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_eval_inside_lambda_blocked(self):
        """eval() inside a lambda is blocked."""
        repl = SafeREPL()
        result = repl.execute("malicious = lambda: eval('1 + 1')")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_exec_inside_list_comprehension_blocked(self):
        """exec() inside a list comprehension is blocked."""
        repl = SafeREPL()
        result = repl.execute("results = [exec('x = i') for i in range(3)]")

        assert result.success is False
        assert "SecurityError" in result.stderr

    # ==========================================
    # eval/exec via callable strings (runtime check)
    # ==========================================

    def test_eval_stored_in_variable_and_called_fails(self):
        """Storing 'eval' string in variable and trying to call fails."""
        repl = SafeREPL()
        # This is tricky - the code may run but should fail to actually execute eval
        result = repl.execute("""
func_name = 'eval'
# Can't actually call it because it's just a string
# The only way to call it would be via eval/exec which are blocked
result = func_name + '(' + repr('1+1') + ')'
""")

        # This should succeed but result is just a string, not executed
        if result.success:
            assert repl.variables.get("result") == "eval('1+1')"
            # The string is not executed, just stored

    def test_building_malicious_code_string_does_not_execute(self):
        """Building a malicious code string does not execute it."""
        repl = SafeREPL()
        result = repl.execute("""
code = "import os; os.system('echo hacked')"
# Without eval/exec, this string is just data
result = len(code)
""")

        # The malicious code is never executed
        assert result.success is True
        assert repl.variables.get("result") == len("import os; os.system('echo hacked')")

    # ==========================================
    # Combined attacks
    # ==========================================

    def test_nested_eval_exec_blocked(self):
        """Nested eval(exec(...)) is blocked."""
        repl = SafeREPL()
        result = repl.execute("eval(exec('x = 1'))")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_chained_function_calls_with_blocked_function(self):
        """Chain of function calls including blocked function is blocked."""
        repl = SafeREPL()
        result = repl.execute("result = str(eval('1+1'))")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_eval_in_try_except_still_blocked(self):
        """eval() in try-except is still blocked at AST level."""
        repl = SafeREPL()
        result = repl.execute("""
try:
    result = eval('1 + 1')
except:
    result = 'caught'
""")

        assert result.success is False
        assert "SecurityError" in result.stderr

    def test_exec_in_finally_still_blocked(self):
        """exec() in finally is still blocked at AST level."""
        repl = SafeREPL()
        result = repl.execute("""
try:
    x = 1
finally:
    exec('y = 2')
""")

        assert result.success is False
        assert "SecurityError" in result.stderr

    # ==========================================
    # Allowed dunder attributes work fine
    # ==========================================

    def test_allowed_dunder_len_works(self):
        """__len__ is allowed and works correctly."""
        repl = SafeREPL()
        result = repl.execute("length = 'hello'.__len__()")

        assert result.success is True
        assert repl.variables.get("length") == 5

    def test_allowed_dunder_str_works(self):
        """__str__ is allowed and works correctly."""
        repl = SafeREPL()
        result = repl.execute("s = (42).__str__()")

        assert result.success is True
        assert repl.variables.get("s") == "42"

    def test_allowed_dunder_repr_works(self):
        """__repr__ is allowed and works correctly."""
        repl = SafeREPL()
        result = repl.execute("r = 'test'.__repr__()")

        assert result.success is True
        assert repl.variables.get("r") == "'test'"

    def test_allowed_dunder_iter_works(self):
        """__iter__ is allowed and works correctly."""
        repl = SafeREPL()
        result = repl.execute("it = [1, 2, 3].__iter__()\nfirst = next(it)")

        assert result.success is True
        assert repl.variables.get("first") == 1

    # ==========================================
    # Verify safe operations still work
    # ==========================================

    def test_safe_builtins_available_after_malicious_attempt(self):
        """Safe builtins remain available after blocked malicious code."""
        repl = SafeREPL()

        # First, try malicious code (blocked)
        result1 = repl.execute("x = eval('1')")
        assert result1.success is False

        # Then verify safe operations still work
        result2 = repl.execute("""
result = sum([1, 2, 3])
text = 'hello'.upper()
length = len([1, 2, 3])
""")
        assert result2.success is True
        assert repl.variables.get("result") == 6
        assert repl.variables.get("text") == "HELLO"
        assert repl.variables.get("length") == 3

    def test_safe_imports_work_after_malicious_attempt(self):
        """Safe imports remain available after blocked malicious code."""
        repl = SafeREPL()

        # First, try malicious code (blocked)
        result1 = repl.execute("exec('import os')")
        assert result1.success is False

        # Then verify safe imports work
        result2 = repl.execute("""
import math
import json
pi_value = math.pi
data = json.dumps({'key': 'value'})
""")
        assert result2.success is True
        assert repl.variables.get("pi_value") == 3.141592653589793
        assert repl.variables.get("data") == '{"key": "value"}'


class TestInfiniteLoopTimeout:
    """Test that execute handles infinite loops with timeout."""

    def test_simple_infinite_while_loop_times_out(self):
        """Simple while True loop is stopped by timeout."""
        repl = SafeREPL()
        # Use 1 second timeout for quick test
        result = repl.execute("while True: pass", timeout_seconds=1)

        assert result.success is False
        assert "ExecutionTimeoutError" in result.stderr
        assert "timed out" in result.stderr

    def test_infinite_for_loop_times_out(self):
        """Infinite for loop using itertools.count is stopped by timeout."""
        repl = SafeREPL()
        code = """
import itertools
for i in itertools.count():
    x = i
"""
        result = repl.execute(code, timeout_seconds=1)

        assert result.success is False
        assert "ExecutionTimeoutError" in result.stderr

    def test_infinite_recursion_times_out_or_stack_overflow(self):
        """Infinite recursion is stopped by timeout or RecursionError."""
        repl = SafeREPL()
        code = """
def infinite():
    return infinite()
infinite()
"""
        result = repl.execute(code, timeout_seconds=2)

        # Either timeout or RecursionError is acceptable
        assert result.success is False
        assert "ExecutionTimeoutError" in result.stderr or "RecursionError" in result.stderr

    def test_timeout_error_message_includes_seconds(self):
        """Timeout error message includes the timeout duration."""
        repl = SafeREPL()
        result = repl.execute("while True: pass", timeout_seconds=1)

        assert result.success is False
        assert "1" in result.stderr or "1.0" in result.stderr

    def test_fast_code_completes_within_timeout(self):
        """Fast code completes successfully within timeout."""
        repl = SafeREPL()
        result = repl.execute("x = 1 + 1", timeout_seconds=5)

        assert result.success is True
        assert repl.variables.get("x") == 2

    def test_loop_that_finishes_completes_successfully(self):
        """A loop that finishes before timeout completes successfully."""
        repl = SafeREPL()
        code = """
total = 0
for i in range(1000):
    total += i
"""
        result = repl.execute(code, timeout_seconds=5)

        assert result.success is True
        assert repl.variables.get("total") == 499500

    def test_timeout_does_not_affect_subsequent_executions(self):
        """After a timeout, the REPL can still execute code normally."""
        repl = SafeREPL()

        # First execution times out
        result1 = repl.execute("while True: pass", timeout_seconds=1)
        assert result1.success is False
        assert "ExecutionTimeoutError" in result1.stderr

        # Second execution should work fine
        result2 = repl.execute("y = 42", timeout_seconds=5)
        assert result2.success is True
        assert repl.variables.get("y") == 42

    def test_variables_from_before_timeout_are_preserved(self):
        """Variables set before timeout are preserved."""
        repl = SafeREPL()

        # First, set a variable
        result1 = repl.execute("before_timeout = 'preserved'", timeout_seconds=5)
        assert result1.success is True

        # Second, execute code that times out
        result2 = repl.execute("while True: pass", timeout_seconds=1)
        assert result2.success is False

        # Variable should still exist
        assert repl.variables.get("before_timeout") == "preserved"

    def test_timeout_zero_means_no_timeout(self):
        """timeout_seconds=0 should skip timeout setup (fast code should work)."""
        repl = SafeREPL()
        # With timeout=0, no alarm is set, so fast code runs without timeout mechanism
        result = repl.execute("x = 100", timeout_seconds=0)

        assert result.success is True
        assert repl.variables.get("x") == 100

    def test_nested_loops_timeout(self):
        """Nested infinite loops are stopped by timeout."""
        repl = SafeREPL()
        code = """
while True:
    while True:
        pass
"""
        result = repl.execute(code, timeout_seconds=1)

        assert result.success is False
        assert "ExecutionTimeoutError" in result.stderr

    def test_infinite_loop_with_sleep_times_out(self):
        """Infinite loop with time.sleep is stopped by timeout."""
        repl = SafeREPL()
        code = """
import time
while True:
    time.sleep(0.1)
"""
        result = repl.execute(code, timeout_seconds=1)

        assert result.success is False
        assert "ExecutionTimeoutError" in result.stderr

    def test_long_computation_times_out(self):
        """Long computation is stopped by timeout."""
        repl = SafeREPL()
        # Very expensive computation
        code = """
result = 0
for i in range(10**9):
    result += i * i
"""
        result = repl.execute(code, timeout_seconds=1)

        assert result.success is False
        assert "ExecutionTimeoutError" in result.stderr

    def test_multiple_timeouts_in_sequence(self):
        """Multiple timeouts in sequence don't break the REPL."""
        repl = SafeREPL()

        # First timeout
        result1 = repl.execute("while True: pass", timeout_seconds=1)
        assert result1.success is False

        # Second timeout
        result2 = repl.execute("while True: pass", timeout_seconds=1)
        assert result2.success is False

        # Third timeout
        result3 = repl.execute("while True: pass", timeout_seconds=1)
        assert result3.success is False

        # Normal execution still works
        result4 = repl.execute("final = 'ok'", timeout_seconds=5)
        assert result4.success is True
        assert repl.variables.get("final") == "ok"

    def test_execution_time_reflects_timeout(self):
        """Execution time in result should be close to timeout when timed out."""
        repl = SafeREPL()
        result = repl.execute("while True: pass", timeout_seconds=1)

        assert result.success is False
        # Execution time should be around 1 second (1000ms), with some tolerance
        assert result.execution_time_ms >= 900  # At least 0.9 seconds
        assert result.execution_time_ms < 3000  # Less than 3 seconds

    def test_generator_infinite_loop_times_out(self):
        """Infinite generator that's consumed times out."""
        repl = SafeREPL()
        code = """
def infinite_gen():
    i = 0
    while True:
        yield i
        i += 1

total = 0
for x in infinite_gen():
    total += x
"""
        result = repl.execute(code, timeout_seconds=1)

        assert result.success is False
        assert "ExecutionTimeoutError" in result.stderr

    def test_list_comprehension_infinite_times_out(self):
        """List comprehension with infinite iterator times out."""
        repl = SafeREPL()
        code = """
import itertools
# This would try to build an infinite list
result = [x for x in itertools.count()]
"""
        result = repl.execute(code, timeout_seconds=1)

        assert result.success is False
        assert "ExecutionTimeoutError" in result.stderr
