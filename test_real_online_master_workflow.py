import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from real_online_coding_workflow import real_online_coding_workflow
from real_online_run_workflow import classify_online_result


class RealOnlineMasterWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_url = Path(__file__).with_name("mock_run_coding_page.html").resolve().as_uri()
        cls.problem_statement = "Given an integer n, return the sum of all integers from 1 to n."

    def _model(self, initial_code, corrected_codes=()):
        state = {"correction_prompts": [], "plan_calls": 0, "generation_calls": 0}

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
                return json.dumps({
                    "steps": ["Sum the inclusive integer range."],
                    "algorithm": "sum(range(1, n + 1))",
                    "complexity": "O(n) time.",
                })
            if "Your only task is code generation" in prompt:
                state["generation_calls"] += 1
                return json.dumps({"code": initial_code})
            if "Diagnose this actual online website test failure" in prompt:
                state["correction_prompts"].append(prompt)
                index = len(state["correction_prompts"]) - 1
                corrected = corrected_codes[min(index, len(corrected_codes) - 1)] if corrected_codes else initial_code
                return json.dumps({
                    "diagnosis": "The candidate did not produce the required inclusive sum.",
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

    def test_complete_successful_flow_is_ready_and_never_submits(self):
        code = "def solve(n):\n    return sum(range(1, n + 1))"
        result, observed, model_state = self._run(code)

        self.assertEqual(result["status"], "READY_FOR_SUBMISSION")
        self.assertEqual(result["state"], "READY_FOR_SUBMISSION")
        self.assertTrue(result["verified"])
        self.assertTrue(result["ready_for_submission"])
        self.assertEqual(result["state_history"][-2:], ["VERIFIED", "READY_FOR_SUBMISSION"])
        self.assertEqual(result["executions"], 1)
        self.assertEqual(observed["run_count"], "1")
        self.assertEqual(observed["submit_count"], "0")
        self.assertFalse(result["submit_clicked"])
        self.assertEqual(model_state["plan_calls"], 1)
        self.assertEqual(model_state["generation_calls"], 1)

    def test_wrong_answer_is_corrected_then_passes(self):
        result, observed, model_state = self._run(
            "def solve(n):\n    return n",
            ["def solve(n):\n    return sum(range(1, n + 1))"],
        )

        self.assertEqual(result["status"], "READY_FOR_SUBMISSION")
        self.assertEqual(result["corrections"], 1)
        self.assertEqual(result["executions"], 2)
        self.assertIn('"status": "wrong_answer"', model_state["correction_prompts"][0])
        self.assertEqual(observed["submit_count"], "0")

    def test_compile_error_is_corrected_then_passes(self):
        result, observed, model_state = self._run(
            "def solve(n):\n    return COMPILE_ERROR",
            ["def solve(n):\n    return sum(range(1, n + 1))"],
        )

        self.assertEqual(result["status"], "READY_FOR_SUBMISSION")
        self.assertIn('"status": "compilation_error"', model_state["correction_prompts"][0])
        self.assertEqual(observed["submit_count"], "0")

    def test_runtime_error_is_corrected_then_passes(self):
        result, observed, model_state = self._run(
            "def solve(n):\n    raise RuntimeError('broken')",
            ["def solve(n):\n    return sum(range(1, n + 1))"],
        )

        self.assertEqual(result["status"], "READY_FOR_SUBMISSION")
        self.assertIn('"status": "runtime_error"', model_state["correction_prompts"][0])
        self.assertEqual(observed["submit_count"], "0")

    def test_memory_limit_error_is_classified_and_corrected(self):
        result, observed, model_state = self._run(
            "def solve(n):\n    return MOCK_MEMORY_ERROR",
            ["def solve(n):\n    return sum(range(1, n + 1))"],
        )

        self.assertEqual(result["status"], "READY_FOR_SUBMISSION")
        self.assertIn('"status": "memory_limit_exceeded"', model_state["correction_prompts"][0])
        self.assertEqual(observed["submit_count"], "0")

    def test_memory_limit_parser_has_a_distinct_status(self):
        result = classify_online_result({"result_text": "Memory Limit Exceeded"})

        self.assertEqual(result["status"], "memory_limit_exceeded")

    def test_correction_limit_stops_after_three_corrections(self):
        result, observed, _ = self._run(
            "def solve(n):\n    return n",
            [
                "def solve(n):\n    return n + 1",
                "def solve(n):\n    return n + 2",
                "def solve(n):\n    return n + 3",
            ],
        )

        self.assertEqual(result["state"], "MAX_CORRECTIONS_REACHED")
        self.assertEqual(result["stop_reason"], "MAX_CORRECTIONS_REACHED")
        self.assertEqual(result["state_history"].count("MAX_CORRECTIONS_REACHED"), 1)
        self.assertEqual(result["corrections"], 3)
        self.assertEqual(result["executions"], 4)
        self.assertEqual(observed["run_count"], "4")
        self.assertEqual(observed["submit_count"], "0")

    def test_comment_only_correction_stops_as_no_progress(self):
        code = "def solve(n):\n    return n"
        result, observed, _ = self._run(code, [code + "  # no meaningful change"])

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["stop_reason"], "NO_PROGRESS")
        self.assertIn("meaningful change", result["summary"])
        self.assertEqual(result["executions"], 1)
        self.assertEqual(observed["run_count"], "1")
        self.assertEqual(observed["submit_count"], "0")

    def test_unknown_result_stops_without_correction(self):
        result, observed, model_state = self._run(
            "def solve(n):\n    return n",
            ["def solve(n):\n    return sum(range(1, n + 1))"],
            query="unknown=1",
        )

        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(result["verified"])
        self.assertEqual(result["corrections"], 0)
        self.assertEqual(model_state["correction_prompts"], [])
        self.assertEqual(observed["submit_count"], "0")

    def test_login_required_stops_before_model_or_run(self):
        result, observed, model_state = self._run(
            "def solve(n):\n    return n",
            query="login=1",
        )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["state"], "BLOCKED")
        self.assertFalse(result["run_clicked"])
        self.assertEqual(model_state["plan_calls"], 0)
        self.assertEqual(observed["run_count"], "0")
        self.assertEqual(observed["submit_count"], "0")

    def test_security_block_stops_before_model_or_run(self):
        result, observed, model_state = self._run(
            "def solve(n):\n    return n",
            query="block=1",
        )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["run_clicked"])
        self.assertEqual(model_state["plan_calls"], 0)
        self.assertEqual(observed["run_count"], "0")
        self.assertEqual(observed["submit_count"], "0")

    def test_problem_page_accepted_statistics_do_not_verify_run(self):
        result = classify_online_result({
            "result_text": "Test Result",
            "body_text": "Accepted\nAcceptance Rate\n58.0%",
        })

        self.assertEqual(result["status"], "unknown")


if __name__ == "__main__":
    unittest.main()