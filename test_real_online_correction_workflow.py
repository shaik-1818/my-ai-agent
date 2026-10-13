import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from real_online_correction_workflow import (
    MAX_ONLINE_CORRECTIONS,
    real_online_correction_workflow,
)


class RealOnlineCorrectionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_url = Path(__file__).with_name("mock_run_coding_page.html").resolve().as_uri()
        cls.problem_statement = "Given an integer n, return the sum of all integers from 1 to n."
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="online-correction-blocked-")
        cls.addClassCleanup(cls.temp_dir.cleanup)

    def _model(self, initial_code, corrected_codes=()):
        state = {"correction_prompts": [], "correction_calls": 0}

        def ask_model(prompt, max_tokens=150):
            if "Extract information about a coding problem" in prompt:
                return json.dumps({
                    "appears_coding_problem": True,
                    "problem_title": "Sum from 1 to n",
                    "problem_statement": self.problem_statement,
                    "examples": ["n = 5 -> 15"],
                    "constraints": ["n is an integer."],
                    "input_description": "An integer n.",
                    "output_description": "The sum from 1 through n.",
                    "programming_language": "Python",
                    "editor_information": "A code editor is present.",
                })
            if "Understand the extracted coding problem" in prompt:
                return json.dumps({
                    "problem_understanding": "Return the sum of integers from 1 through n.",
                    "algorithm": "Sum the inclusive integer range.",
                    "complexity": "O(n) time.",
                    "code": initial_code,
                })
            if "Diagnose this actual online website test failure" in prompt:
                state["correction_prompts"].append(prompt)
                state["correction_calls"] += 1
                index = state["correction_calls"] - 1
                corrected = corrected_codes[min(index, len(corrected_codes) - 1)] if corrected_codes else initial_code
                return json.dumps({
                    "diagnosis": "The candidate does not sum all integers from 1 through n.",
                    "changes": "Compute the inclusive range sum.",
                    "code": corrected,
                })
            raise AssertionError("Unexpected model prompt:\n" + prompt)

        return ask_model, state

    def _run(self, initial_code, corrected_codes=(), query="", result_wait_seconds=0.2):
        observed = {}
        ask_model, model_state = self._model(initial_code, corrected_codes)
        parsed = urlsplit(self.mock_url)
        url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "&".join(filter(None, ("correction=1", query))), ""))

        def inspect_page(page, result):
            observed["run_count"] = page.locator("#run-count").inner_text()
            observed["submit_count"] = page.locator("#submit-count").inner_text()
            observed["editor_code"] = page.locator("#code-editor").input_value()

        with redirect_stdout(io.StringIO()):
            result = real_online_correction_workflow(
                url,
                ask_model=ask_model,
                headless=True,
                keep_open=True,
                inspection_callback=inspect_page,
                result_wait_seconds=result_wait_seconds,
            )
        return result, observed, model_state

    def test_wrong_answer_is_corrected_and_only_online_pass_verifies(self):
        wrong = "def solve(n):\n    return n"
        corrected = "def solve(n):\n    return sum(range(1, n + 1))"
        result, observed, model_state = self._run(wrong, [corrected])

        self.assertEqual(result["status"], "VERIFIED")
        self.assertTrue(result["verified"])
        self.assertEqual(result["code"], corrected)
        self.assertEqual(result["result"]["status"], "passed")
        self.assertEqual(result["corrections"], 1)
        self.assertEqual(result["executions"], 2)
        self.assertEqual(observed["run_count"], "2")
        self.assertEqual(observed["submit_count"], "0")
        self.assertFalse(result["submit_clicked"])
        self.assertEqual(model_state["correction_calls"], 1)
        evidence_prompt = model_state["correction_prompts"][0]
        self.assertIn(self.problem_statement, evidence_prompt)
        self.assertIn('"current_code": "def solve(n):\\n    return n"', evidence_prompt)
        self.assertIn('"expected_result": "15"', evidence_prompt)
        self.assertIn('"actual_result": "5"', evidence_prompt)
        self.assertIn('"status": "wrong_answer"', evidence_prompt)

    def test_maximum_three_corrections_stops_not_verified(self):
        wrong = "def solve(n):\n    return n"
        corrections = [
            "def solve(n):\n    return n + 1",
            "def solve(n):\n    return n + 2",
            "def solve(n):\n    return n + 3",
        ]
        result, observed, state = self._run(wrong, corrections)

        self.assertEqual(MAX_ONLINE_CORRECTIONS, 3)
        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertFalse(result["verified"])
        self.assertEqual(result["summary"], "Could not verify solution after maximum correction attempts.")
        self.assertEqual(result["corrections"], 3)
        self.assertEqual(result["executions"], 4)
        self.assertEqual(state["correction_calls"], 3)
        self.assertEqual(observed["run_count"], "4")

    def test_runtime_error_is_sent_to_qwen_and_can_be_corrected(self):
        broken = "def solve(n):\n    raise RuntimeError('broken')"
        corrected = "def solve(n):\n    return sum(range(1, n + 1))"
        result, _, state = self._run(broken, [corrected])

        self.assertEqual(result["status"], "VERIFIED")
        self.assertIn('"status": "runtime_error"', state["correction_prompts"][0])
        self.assertIn("mock exception", state["correction_prompts"][0])

    def test_compilation_error_is_sent_to_qwen_and_can_be_corrected(self):
        broken = "def solve(n):\n    return COMPILE_ERROR"
        corrected = "def solve(n):\n    return sum(range(1, n + 1))"
        result, _, state = self._run(broken, [corrected])

        self.assertEqual(result["status"], "VERIFIED")
        self.assertIn('"status": "compilation_error"', state["correction_prompts"][0])
        self.assertIn("Compilation Error", state["correction_prompts"][0])

    def test_unknown_result_stops_without_correction_or_verification(self):
        wrong = "def solve(n):\n    return n"
        result, observed, state = self._run(wrong, query="unknown=1", result_wait_seconds=0.1)

        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertFalse(result["verified"])
        self.assertEqual(result["summary"], "Online result could not be verified.")
        self.assertEqual(result["corrections"], 0)
        self.assertEqual(state["correction_calls"], 0)
        self.assertEqual(observed["submit_count"], "0")

    def test_submit_button_is_never_clicked_during_correction(self):
        wrong = "def solve(n):\n    return n"
        corrected = "def solve(n):\n    return sum(range(1, n + 1))"
        result, observed, state = self._run(wrong, [corrected])

        self.assertTrue(result["run_clicked"])
        self.assertFalse(result["submit_clicked"])
        self.assertEqual(observed["submit_count"], "0")

    def test_repeated_identical_corrected_code_is_not_run_again(self):
        wrong = "def solve(n):\n    return n"
        result, observed, state = self._run(wrong, [wrong])

        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertFalse(result["verified"])
        self.assertIn("already run", result["summary"])
        self.assertEqual(result["corrections"], 1)
        self.assertEqual(result["executions"], 1)
        self.assertEqual(observed["run_count"], "1")
        self.assertEqual(state["correction_calls"], 1)

    def test_invalid_qwen_code_is_rejected_without_reexecution(self):
        wrong = "def solve(n):\n    return n"
        invalid = "import os\ndef solve(n):\n    return n"
        result, observed, state = self._run(wrong, [invalid])

        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertIn("failed static safety validation", result["summary"])
        self.assertEqual(result["executions"], 1)
        self.assertEqual(observed["run_count"], "1")
        self.assertEqual(state["correction_calls"], 1)

    def test_prose_only_qwen_response_is_rejected(self):
        wrong = "def solve(n):\n    return n"
        ask_model, _ = self._model(wrong, [wrong])

        def prose_model(prompt, max_tokens=150):
            if "Diagnose this actual online website test failure" in prompt:
                return "The bug is that the code returns n instead of summing the range."
            return ask_model(prompt, max_tokens=max_tokens)

        parsed = urlsplit(self.mock_url)
        url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "correction=1", ""))
        with redirect_stdout(io.StringIO()):
            result = real_online_correction_workflow(
                url,
                ask_model=prose_model,
                headless=True,
                keep_open=False,
                result_wait_seconds=0.2,
            )

        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertIn("exactly one valid JSON object", result["summary"])
        self.assertEqual(result["executions"], 1)

    def test_login_block_stops_before_model_or_run(self):
        blocked_page = Path(self.temp_dir.name) / "login-required.html"
        blocked_page.write_text(
            "<html><body><h1>Sign in to continue</h1>"
            "<input type='password' name='password'>"
            "<button id='run'>Run</button></body></html>",
            encoding="utf-8",
        )
        model_calls = []

        def model_must_not_run(prompt, max_tokens=150):
            model_calls.append(prompt)
            raise AssertionError("Blocked page must not be sent to Qwen.")

        with redirect_stdout(io.StringIO()):
            result = real_online_correction_workflow(
                blocked_page.resolve().as_uri(),
                ask_model=model_must_not_run,
                headless=True,
                keep_open=False,
            )

        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertIn("access is blocked", result["summary"])
        self.assertFalse(result["run_clicked"])
        self.assertEqual(model_calls, [])


if __name__ == "__main__":
    unittest.main()