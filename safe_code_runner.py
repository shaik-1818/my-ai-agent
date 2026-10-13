import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "re",
    "statistics",
    "string",
}

ALLOWED_NODES = {
    ast.Add, ast.And, ast.AnnAssign, ast.Assert, ast.Assign, ast.AugAssign,
    ast.BinOp, ast.BoolOp, ast.Break, ast.Call, ast.Compare, ast.comprehension,
    ast.Constant, ast.Continue, ast.Dict, ast.DictComp, ast.Div, ast.Eq,
    ast.ExceptHandler, ast.Expr, ast.FloorDiv, ast.For, ast.FunctionDef,
    ast.GeneratorExp, ast.Gt, ast.GtE, ast.If, ast.IfExp, ast.Import,
    ast.ImportFrom, ast.In, ast.Index, ast.Is, ast.IsNot, ast.keyword,
    ast.Lt, ast.LtE, ast.Mod, ast.Mult, ast.Name, ast.Not, ast.NotEq,
    ast.Or, ast.Pass, ast.Pow, ast.Raise, ast.Return, ast.Set, ast.SetComp,
    ast.Slice, ast.Sub, ast.Subscript, ast.Try, ast.Tuple, ast.UnaryOp,
    ast.USub, ast.UAdd, ast.While, ast.Load, ast.Store, ast.List, ast.ListComp,
    ast.Attribute, ast.alias, ast.arguments, ast.arg, ast.BitAnd, ast.BitOr,
    ast.BitXor, ast.LShift, ast.RShift, ast.Invert, ast.MatMult,
}

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "KeyError": KeyError,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}

MAX_SOURCE_BYTES = 64_000
MAX_CASES = 30
MAX_TEST_DATA_BYTES = 256_000
MAX_EXECUTION_SECONDS = 5
MAX_RESULT_BYTES = 262_144
MAX_TEST_EXECUTIONS = 4

RUNNER_SOURCE = r'''import importlib
import json
import sys

ALLOWED_IMPORTS = {
    "bisect", "collections", "functools", "heapq", "itertools",
    "math", "re", "statistics", "string"
}
MAX_CAPTURE_BYTES = 262144
captured_bytes = 0
stdout_truncated = False


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name not in ALLOWED_IMPORTS:
        raise ImportError("This import is not allowed: " + str(name))
    return importlib.import_module(name)


def limited_range(*args):
    values = range(*args)
    if len(values) > 200000:
        raise ValueError("range exceeds the sandbox iteration limit")
    return values


def safe_print(*values, sep=" ", end="\n"):
    global captured_bytes, stdout_truncated
    text = sep.join(str(value) for value in values) + end
    encoded = text.encode("utf-8", errors="replace")
    remaining = MAX_CAPTURE_BYTES - captured_bytes
    if len(encoded) > remaining:
        encoded = encoded[:max(0, remaining)]
        stdout_truncated = True
    captured_bytes += len(encoded)
    sys.stdout.write(encoded.decode("utf-8", errors="ignore"))
    sys.stdout.flush()


safe_builtins = {
    "__import__": safe_import,
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "Exception": Exception,
    "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "KeyError": KeyError, "len": len,
    "list": list, "map": map, "max": max, "min": min,
    "next": next, "print": safe_print, "range": limited_range,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "TypeError": TypeError, "ValueError": ValueError, "zip": zip
}

source_path, cases_path, result_path, mode = sys.argv[1:]
with open(source_path, "r", encoding="utf-8") as source_file:
    source = source_file.read()
with open(cases_path, "r", encoding="utf-8") as cases_file:
    cases = json.load(cases_file)

results = []
if mode == "exact_code":
    namespace = {"__builtins__": safe_builtins, "__name__": "__main__"}
    exec(compile(source, "<approved-solution>", "exec"), namespace, namespace)
elif mode == "problem_solving":
    for case in cases:
        namespace = {"__builtins__": safe_builtins, "__name__": "__main__"}
        exec(compile(source, "<approved-solution>", "exec"), namespace, namespace)
        solve = namespace.get("solve")
        if not callable(solve):
            raise RuntimeError("Generated code must define solve(input_data).")

        input_data = json.loads(json.dumps(case["input"]))
        try:
            actual = solve(input_data)
            actual_text = json.dumps(actual, ensure_ascii=True, allow_nan=False)
            if len(actual_text.encode("utf-8")) > MAX_CAPTURE_BYTES:
                raise ValueError("solution result exceeds the output size limit")
            actual = json.loads(actual_text)
            expected = case["expected"]
            passed = actual == expected
            results.append({"passed": passed, "actual": actual, "expected": expected})
        except Exception as error:
            results.append({
                "passed": False,
                "error": (type(error).__name__ + ": " + str(error))[:1000],
                "expected": case["expected"]
            })
else:
    raise RuntimeError("Unsupported execution mode.")

with open(result_path, "w", encoding="utf-8") as result_file:
    json.dump({"results": results, "stdout_truncated": stdout_truncated}, result_file)
'''


class _CodePolicy(ast.NodeVisitor):
    forbidden_names = {
        "breakpoint", "compile", "delattr", "eval", "exec", "getattr",
        "globals", "input", "locals", "open", "setattr",
        "vars", "__builtins__", "__import__",
    }

    def __init__(self, mode):
        self.mode = mode
        self.errors = []

    def generic_visit(self, node):
        if type(node) not in ALLOWED_NODES and not isinstance(node, (ast.Module, ast.Load, ast.Store)):
            self.errors.append(f"Unsupported syntax: {type(node).__name__}")
            return
        super().generic_visit(node)

    def visit_Name(self, node):
        allowed_private_name = self.mode == "exact_code" and node.id == "__name__"
        if (node.id.startswith("_") and not allowed_private_name) or node.id in self.forbidden_names:
            self.errors.append(f"Restricted name: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("_"):
            self.errors.append(f"Restricted attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                self.errors.append(f"Import is not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level or not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
            self.errors.append(f"Import is not allowed: {node.module}")
        if any(alias.name.startswith("_") for alias in node.names):
            self.errors.append("Private imports are not allowed")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.decorator_list or node.returns or any(arg.annotation for arg in node.args.args):
            self.errors.append("Decorators and annotations are not allowed")
        if node.args.defaults or node.args.kw_defaults:
            self.errors.append("Default argument expressions are not allowed")
        self.generic_visit(node)


def validate_solution(source, mode="problem_solving"):
    if mode not in {"problem_solving", "exact_code"}:
        return {"success": False, "error": f"Unsupported code mode: {mode}"}

    if not isinstance(source, str) or not source.strip():
        return {"success": False, "error": "Generated source is empty."}

    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return {"success": False, "error": f"Generated source exceeds {MAX_SOURCE_BYTES} bytes."}

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return {
            "success": False,
            "error": f"Generated source has a syntax error: {error.msg} at line {error.lineno}.",
        }

    if mode == "problem_solving":
        for statement in tree.body:
            is_docstring = (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
            if not isinstance(statement, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign)) and not is_docstring:
                return {
                    "success": False,
                    "error": "Only function definitions, safe imports, literal constants, and a module docstring are allowed at module scope.",
                }
            if isinstance(statement, ast.Assign):
                if (
                    any(not isinstance(target, ast.Name) for target in statement.targets)
                    or not _is_literal_expression(statement.value)
                ):
                    return {"success": False, "error": "Module-level assignments must contain literal values only."}

    policy = _CodePolicy(mode)
    policy.visit(tree)
    if policy.errors:
        return {"success": False, "error": "; ".join(dict.fromkeys(policy.errors))}

    if mode == "problem_solving" and not any(
        isinstance(node, ast.FunctionDef) and node.name == "solve"
        for node in tree.body
    ):
        return {"success": False, "error": "Generated code must define solve(input_data)."}

    return {"success": True}


def _is_literal_expression(node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal_expression(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is None or _is_literal_expression(key)
            for key in node.keys
        ) and all(_is_literal_expression(value) for value in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_literal_expression(node.operand)
    return False


def _safe_environment():
    allowed_keys = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed_keys}
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


class ExecutionBudget:
    def __init__(self, max_executions=MAX_TEST_EXECUTIONS):
        self.max_executions = max(0, min(int(max_executions), MAX_TEST_EXECUTIONS))
        self.executions = 0

    @property
    def remaining(self):
        return self.max_executions - self.executions

    def consume(self):
        if self.remaining <= 0:
            return False
        self.executions += 1
        return True


def _execution_result(
    mode="problem_solving",
    success=False,
    stdout="",
    stderr="",
    exit_code=None,
    timeout=False,
    duration_seconds=0.0,
    error=None,
    results=None,
    executed=False,
    stdout_truncated=False,
):
    return {
        "mode": mode,
        "success": success,
        "executed": executed,
        "all_passed": success,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timeout": timeout,
        "duration_seconds": round(duration_seconds, 3),
        "stdout_truncated": stdout_truncated,
        "error": error,
        "results": results or [],
    }


def _output_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _execute_code(
    source,
    mode,
    test_cases,
    approved=False,
    execution_budget=None,
    timeout_seconds=MAX_EXECUTION_SECONDS,
):
    validation = validate_solution(source, mode=mode)
    if not validation["success"]:
        return _execution_result(mode=mode, error=validation["error"])

    if not approved:
        return _execution_result(mode=mode, error="Execution was not approved.")

    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= MAX_EXECUTION_SECONDS:
        return _execution_result(
            mode=mode,
            error=f"Timeout must be greater than zero and no more than {MAX_EXECUTION_SECONDS} seconds."
        )

    if mode == "problem_solving" and (
        not isinstance(test_cases, list) or not test_cases or len(test_cases) > MAX_CASES
    ):
        return _execution_result(
            mode=mode,
            error=f"Provide between 1 and {MAX_CASES} test cases."
        )

    if mode == "problem_solving":
        for case in test_cases:
            if not isinstance(case, dict) or "input" not in case or "expected" not in case:
                return _execution_result(
                    mode=mode,
                    error="Each test case must contain 'input' and 'expected'."
                )

    try:
        serialized_cases = json.dumps(test_cases or [], ensure_ascii=True)
    except (TypeError, ValueError) as error:
        return _execution_result(mode=mode, error=f"Test data must be JSON-compatible: {error}")

    if len(serialized_cases.encode("utf-8")) > MAX_TEST_DATA_BYTES:
        return _execution_result(mode=mode, error=f"Test data exceeds {MAX_TEST_DATA_BYTES} bytes.")

    budget = execution_budget or ExecutionBudget()
    if not budget.consume():
        return _execution_result(
            mode=mode,
            error=f"Coding task reached its limit of {budget.max_executions} code executions."
        )

    started_at = time.perf_counter()
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="my-agent-code-") as temporary_directory:
            work_dir = Path(temporary_directory)
            source_path = work_dir / "solution.py"
            cases_path = work_dir / "test_cases.json"
            runner_path = work_dir / "runner.py"
            result_path = work_dir / "runner_result.json"
            source_path.write_text(source, encoding="utf-8")
            cases_path.write_text(serialized_cases, encoding="utf-8")
            runner_path.write_text(RUNNER_SOURCE, encoding="utf-8")

            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(runner_path),
                    str(source_path),
                    str(cases_path),
                    str(result_path),
                    mode,
                ],
                cwd=temporary_directory,
                env=_safe_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                stdout, stderr = process.communicate()
                duration = time.perf_counter() - started_at
                return _execution_result(
                    mode=mode,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=process.returncode,
                    timeout=True,
                    duration_seconds=duration,
                    error=f"Execution exceeded the {timeout_seconds:g}-second timeout; the Python process was terminated.",
                    executed=True,
                )

            duration = time.perf_counter() - started_at

            if len(stdout.encode("utf-8")) > MAX_RESULT_BYTES:
                stdout = stdout.encode("utf-8")[:MAX_RESULT_BYTES].decode("utf-8", errors="ignore")
                output_error = "Captured stdout exceeded the output limit and was truncated."
            else:
                output_error = None

            if process.returncode != 0:
                error = stderr.strip() or "Python runner exited without a diagnostic message."
                return _execution_result(
                    mode=mode,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=process.returncode,
                    duration_seconds=duration,
                    error=error[-2000:],
                    executed=True,
                )

            if not result_path.is_file():
                return _execution_result(
                    mode=mode,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=process.returncode,
                    duration_seconds=duration,
                    error="Python runner did not produce a result file.",
                    executed=True,
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            results = payload.get("results", [])
            output_truncated = bool(payload.get("stdout_truncated"))
            if mode == "exact_code":
                all_passed = process.returncode == 0 and not output_truncated and output_error is None
                error = None if all_passed else output_error or "Exact-code execution output was truncated."
            else:
                all_passed = (
                    process.returncode == 0
                    and len(results) == len(test_cases)
                    and all(item.get("passed") for item in results)
                    and not output_truncated
                )
                error = None if all_passed else output_error or "One or more test cases failed."
            return _execution_result(
                mode=mode,
                success=all_passed,
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode,
                duration_seconds=duration,
                error=error,
                results=results,
                executed=True,
                stdout_truncated=output_truncated,
            )

    except Exception as error:
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
            process.communicate()
        return _execution_result(
            mode=mode,
            stdout="",
            stderr="",
            exit_code=process.returncode if process is not None else None,
            duration_seconds=time.perf_counter() - started_at,
            error=f"Test runner error: {type(error).__name__}: {error}",
            executed=process is not None,
        )


def run_solution_tests(
    source,
    test_cases,
    approved=False,
    execution_budget=None,
    timeout_seconds=MAX_EXECUTION_SECONDS,
):
    return _execute_code(
        source,
        mode="problem_solving",
        test_cases=test_cases,
        approved=approved,
        execution_budget=execution_budget,
        timeout_seconds=timeout_seconds,
    )


def run_exact_code(
    source,
    approved=False,
    execution_budget=None,
    timeout_seconds=MAX_EXECUTION_SECONDS,
):
    return _execute_code(
        source,
        mode="exact_code",
        test_cases=None,
        approved=approved,
        execution_budget=execution_budget,
        timeout_seconds=timeout_seconds,
    )
