import unittest

from coding_workflow import (
    EXACT_CODE_MODE,
    PROBLEM_SOLVING_MODE,
    detect_coding_mode,
    extract_exact_source,
    run_coding_task,
)
from safe_code_runner import (
    MAX_TEST_EXECUTIONS,
    ExecutionBudget,
    run_exact_code,
    run_solution_tests,
)


class SafeCodeRunnerTests(unittest.TestCase):
    def test_mode_detector_separates_exact_and_problem_requests(self):
        self.assertEqual(
            detect_coding_mode("Run the Python program `while True: pass` using the safe code runner."),
            EXACT_CODE_MODE,
        )
        self.assertEqual(
            detect_coding_mode("Given an integer n, return the sum from 1 to n."),
            PROBLEM_SOLVING_MODE,
        )

    def test_exact_source_extraction_preserves_multiline_code(self):
        source = "while True:\n    pass"
        request = (
            "Execute the following code:\n```python\n"
            + source
            + "\n```\nTerminate it when the timeout is reached."
        )
        self.assertEqual(extract_exact_source(request), source)

    def test_exact_workflow_does_not_call_qwen_or_make_tests(self):
        source = 'print("hello")'
        request = f"Run this exact code: `{source}`"

        def qwen_must_not_be_called(*args, **kwargs):
            self.fail("Exact-code mode must not call Qwen.")

        result = run_coding_task(
            request,
            ask_model=qwen_must_not_be_called,
            confirm_execution=lambda _: True,
        )

        self.assertEqual(result["mode"], EXACT_CODE_MODE)
        self.assertEqual(result["source"], source)
        self.assertEqual(result["execution_result"]["stdout"].strip(), "hello")
        self.assertEqual(result["execution_result"]["results"], [])

    def test_correct_solution_captures_output_and_passes(self):
        result = run_solution_tests(
            "def solve(input_data):\n    print('tested')\n    return sum(input_data)\n",
            [{"input": [2, 3], "expected": 5}],
            approved=True,
        )

        self.assertTrue(result["success"], result)
        self.assertEqual(result["stdout"], "tested\n")
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timeout"])
        self.assertGreaterEqual(result["duration_seconds"], 0)
        self.assertTrue(result["results"][0]["passed"])

    def test_incorrect_solution_returns_failed_test_details(self):
        result = run_solution_tests(
            "def solve(input_data):\n    return input_data[0]\n",
            [{"input": [2, 3], "expected": 5}],
            approved=True,
        )

        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("test cases failed", result["error"].lower())
        self.assertEqual(result["results"][0]["actual"], 2)
        self.assertEqual(result["results"][0]["expected"], 5)

    def test_infinite_loop_is_terminated_by_timeout(self):
        result = run_solution_tests(
            "def solve(input_data):\n    while True:\n        pass\n",
            [{"input": None, "expected": None}],
            approved=True,
            timeout_seconds=0.5,
        )

        self.assertTrue(result["executed"])
        self.assertFalse(result["success"])
        self.assertTrue(result["timeout"])
        self.assertIsInstance(result["exit_code"], int)
        self.assertIn("terminated", result["error"].lower())
        self.assertLess(result["duration_seconds"], 3)

    def test_task_execution_budget_is_enforced(self):
        budget = ExecutionBudget()
        source = "def solve(input_data):\n    return input_data\n"
        cases = [{"input": 3, "expected": 3}]

        results = [
            run_solution_tests(source, cases, approved=True, execution_budget=budget)
            for _ in range(MAX_TEST_EXECUTIONS)
        ]
        over_limit = run_solution_tests(source, cases, approved=True, execution_budget=budget)

        self.assertTrue(all(result["success"] for result in results), results)
        self.assertFalse(over_limit["executed"])
        self.assertIn("limit", over_limit["error"].lower())
        self.assertEqual(budget.executions, MAX_TEST_EXECUTIONS)

    def test_exact_code_succeeds_and_captures_stdout(self):
        source = 'print("hello")'
        result = run_exact_code(source, approved=True)

        self.assertEqual(source, 'print("hello")')
        self.assertEqual(result["mode"], "exact_code")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["stdout"].strip(), "hello")
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timeout"])

    def test_exact_code_failure_captures_traceback(self):
        result = run_exact_code('print(10 / 0)', approved=True)

        self.assertFalse(result["success"])
        self.assertIn("ZeroDivisionError", result["stderr"])
        self.assertNotEqual(result["exit_code"], 0)
        self.assertFalse(result["timeout"])

    def test_exact_infinite_loop_is_terminated(self):
        source = "while True:\n    pass"
        result = run_exact_code(source, approved=True, timeout_seconds=0.5)

        self.assertEqual(source, "while True:\n    pass")
        self.assertEqual(result["mode"], "exact_code")
        self.assertFalse(result["success"])
        self.assertTrue(result["timeout"])
        self.assertIsInstance(result["exit_code"], int)
        self.assertIn("terminated", result["error"].lower())

    def test_exact_code_still_rejects_shell_related_imports(self):
        result = run_exact_code(
            "import subprocess\nsubprocess.run(['cmd', '/c', 'echo unsafe'])",
            approved=True,
        )

        self.assertFalse(result["executed"])
        self.assertIn("not allowed", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
