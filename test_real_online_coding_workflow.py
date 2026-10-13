import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from real_online_coding_workflow import real_online_coding_workflow


class RealOnlineCodingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_url = Path(__file__).with_name("mock_run_coding_page.html").resolve().as_uri()
        cls.problem_statement = "Given an integer n, return the sum of all integers from 1 to n."
        cls.success_states = [
            "OPENING",
            "READING_PROBLEM",
            "PLANNING",
            "GENERATING_CODE",
            "FINDING_EDITOR",
            "ENTERING_CODE",
            "VERIFYING_CODE",
            "FINDING_RUN_CONTROL",
            "RUNNING",
            "READING_RESULT",
            "CORRECTING",
            "ENTERING_CODE",
            "VERIFYING_CODE",
            "FINDING_RUN_CONTROL",
            "RUNNING",
            "READING_RESULT",
            "VERIFIED",
            "READY_FOR_SUBMISSION",
        ]

    def _model(self, initial_code, corrected_codes=()):
        state = {"plan_calls": 0, "generation_calls": 0, "correction_prompts": []}

        def ask_model(prompt, max_tokens=150):
            if "Extract the coding problem from the visible webpage." in prompt:
                return json.dumps({
                    "problem_statement": self.problem_statement,
                    "examples": ["n = 5 -> 15"],
                    "constraints": ["n is an integer."],
                    "expected_input_output": "Implement solve(n) and return an integer.",
                    "language": "Python",
                    "requirements": ["Return the sum of all integers from 1 through n."],
                })
            if "Create a short implementation plan" in prompt:
                state["plan_calls"] += 1
                self.assertIn(self.problem_statement, prompt)
                return json.dumps({
                    "steps": ["Sum the inclusive integer range."],
                    "algorithm": "sum(range(1, n + 1))",
                    "complexity": "O(n) time.",
                })
            if "Your only task is code generation" in prompt:
                state["generation_calls"] += 1
                self.assertIn(self.problem_statement, prompt)
                return json.dumps({"code": initial_code})
            if "Diagnose this actual online website test failure" in prompt:
                state["correction_prompts"].append(prompt)
                index = len(state["correction_prompts"]) - 1
                corrected = corrected_codes[min(index, len(corrected_codes) - 1)] if corrected_codes else initial_code
                return json.dumps({
                    "diagnosis": "The candidate does not sum the inclusive range.",
                    "changes": "Return the sum from one through n.",
                    "code": corrected,
                })
            raise AssertionError("Unexpected model prompt:\n" + prompt)

        return ask_model, state

    def _run(self, initial_code, corrected_codes=(), query="", result_wait_seconds=0.1):
        parsed = urlsplit(self.mock_url)
        query_parts = ["integrated=1"]
        if query:
            query_parts.append(query)
        url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "&".join(query_parts), ""))
        observed = {}
        ask_model, model_state = self._model(initial_code, corrected_codes)

        def inspect_page(page, result):
            observed["run_count"] = page.locator("#run-count").inner_text()
            observed["submit_count"] = page.locator("#submit-count").inner_text()
            observed["editor_code"] = page.locator("#code-editor").input_value()

        with redirect_stdout(io.StringIO()):
            result = real_online_coding_workflow(
                url,
                ask_model=ask_model,
                headless=True,
                keep_open=True,
                inspection_callback=inspect_page,
                result_wait_seconds=result_wait_seconds,
            )
        return result, observed, model_state

    def test_complete_state_machine_reaches_ready_only_after_fresh_pass(self):
        wrong = "def solve(n):\n    return n"
        correct = "def solve(n):\n    return sum(range(1, n + 1))"
        result, observed, model_state = self._run(wrong, [correct])

        self.assertEqual(result["status"], "READY_FOR_SUBMISSION")
        self.assertEqual(result["state"], "READY_FOR_SUBMISSION")
        self.assertTrue(result["verified"])
        self.assertTrue(result["ready_for_submission"])
        self.assertEqual(result["result"]["status"], "passed")
        self.assertEqual(result["corrections"], 1)
        self.assertEqual(result["executions"], 2)
        self.assertEqual(result["state_history"], self.success_states)
        self.assertEqual(observed["run_count"], "2")
        self.assertEqual(observed["submit_count"], "0")
        self.assertEqual(observed["editor_code"], correct)
        self.assertFalse(result["submit_clicked"])
        self.assertEqual(model_state["plan_calls"], 1)
        self.assertEqual(model_state["generation_calls"], 1)
        correction_prompt = model_state["correction_prompts"][0]
        self.assertIn(self.problem_statement, correction_prompt)
        self.assertIn('"expected_result": "15"', correction_prompt)
        self.assertIn('"actual_result": "5"', correction_prompt)
        self.assertIn('"status": "wrong_answer"', correction_prompt)

    def test_maximum_corrections_end_in_failed_state(self):
        wrong = "def solve(n):\n    return n"
        corrections = [
            "def solve(n):\n    return n + 1",
            "def solve(n):\n    return n + 2",
            "def solve(n):\n    return n + 3",
        ]
        result, observed, _ = self._run(wrong, corrections)

        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(result["verified"])
        self.assertFalse(result["ready_for_submission"])
        self.assertIn("maximum correction attempts", result["summary"])
        self.assertEqual(result["corrections"], 3)
        self.assertEqual(result["executions"], 4)
        self.assertEqual(observed["run_count"], "4")
        self.assertEqual(observed["submit_count"], "0")

    def test_runtime_error_is_corrected_end_to_end(self):
        broken = "def solve(n):\n    raise RuntimeError('broken')"
        correct = "def solve(n):\n    return sum(range(1, n + 1))"
        result, _, model_state = self._run(broken, [correct])

        self.assertEqual(result["status"], "READY_FOR_SUBMISSION")
        self.assertIn('"status": "runtime_error"', model_state["correction_prompts"][0])

    def test_unknown_result_fails_without_correction(self):
        result, observed, model_state = self._run(
            "def solve(n):\n    return n",
            ["def solve(n):\n    return sum(range(1, n + 1))"],
            query="unknown=1",
            result_wait_seconds=0.1,
        )

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["summary"], "Online result could not be verified.")
        self.assertFalse(result["verified"])
        self.assertFalse(result["ready_for_submission"])
        self.assertEqual(result["corrections"], 0)
        self.assertEqual(observed["submit_count"], "0")
        self.assertEqual(model_state["correction_prompts"], [])

    def test_access_blocker_stops_before_model_and_run(self):
        model_calls = []

        def model_must_not_run(prompt, max_tokens=150):
            model_calls.append(prompt)
            raise AssertionError("Blocked content must not be sent to Qwen.")

        parsed = urlsplit(self.mock_url)
        url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "integrated=1&block=1", ""))
        observed = {}

        def inspect_page(page, result):
            observed["run_count"] = page.locator("#run-count").inner_text()
            observed["submit_count"] = page.locator("#submit-count").inner_text()

        with redirect_stdout(io.StringIO()):
            result = real_online_coding_workflow(
                url,
                ask_model=model_must_not_run,
                headless=True,
                keep_open=True,
                inspection_callback=inspect_page,
            )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("access restriction", result["summary"])
        self.assertEqual(result["state_history"], ["OPENING", "BLOCKED"])
        self.assertEqual(model_calls, [])
        self.assertEqual(observed["run_count"], "0")
        self.assertEqual(observed["submit_count"], "0")

    def test_missing_editor_stops_before_run(self):
        result, observed, _ = self._run(
            "def solve(n):\n    return n",
            query="no-editor=1",
        )

        self.assertEqual(result["status"], "FAILED")
        self.assertIn("visible", result["summary"])
        self.assertEqual(observed["run_count"], "0")
        self.assertEqual(observed["submit_count"], "0")

    def test_missing_run_control_stops_before_execution(self):
        result, observed, _ = self._run(
            "def solve(n):\n    return n",
            query="no-run=1",
        )

        self.assertEqual(result["status"], "FAILED")
        self.assertIn("Run/Test", result["summary"])
        self.assertEqual(observed["run_count"], "0")
        self.assertEqual(observed["submit_count"], "0")

    def test_duplicate_correction_is_not_reexecuted(self):
        wrong = "def solve(n):\n    return n"
        result, observed, _ = self._run(wrong, [wrong])

        self.assertEqual(result["status"], "FAILED")
        self.assertIn("already run", result["summary"])
        self.assertEqual(result["executions"], 1)
        self.assertEqual(observed["run_count"], "1")
        self.assertEqual(observed["submit_count"], "0")


if __name__ == "__main__":
    unittest.main()
