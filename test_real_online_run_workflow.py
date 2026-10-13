import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

from playwright.sync_api import sync_playwright

from real_online_run_workflow import (
    MAX_ONLINE_EXECUTIONS,
    OnlineExecutionBudget,
    classify_online_result,
    find_run_control,
    read_online_test_result,
    real_online_run_workflow,
    run_online_code,
)
from real_online_editor_workflow import enter_code, find_code_editor


class RealOnlineRunWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_url = Path(__file__).with_name("mock_run_coding_page.html").resolve().as_uri()

    @staticmethod
    def _model_for(code):
        def ask_model(prompt, max_tokens=150):
            if "Extract information about a coding problem" in prompt:
                return json.dumps({
                    "appears_coding_problem": True,
                    "problem_title": "Sum a List",
                    "problem_statement": "Given a list of integers, return their sum.",
                    "examples": ["[1, 2, 3] -> 6"],
                    "constraints": ["The list may be empty."],
                    "input_description": "A list of integers.",
                    "output_description": "An integer sum.",
                    "programming_language": "Python",
                    "editor_information": "A code editor is shown.",
                })
            if "Understand the extracted coding problem" in prompt:
                return json.dumps({
                    "problem_understanding": "Return the sum of all values.",
                    "algorithm": "Add the input values.",
                    "complexity": "O(n) time.",
                    "code": code,
                })
            raise AssertionError("Unexpected model prompt:\n" + prompt)

        return ask_model

    def _run_workflow(self, code, query="", result_wait_seconds=0.2, ask_model=None):
        observed = {}

        def inspect_page(page, result):
            observed["run_count"] = page.locator("#run-count").inner_text()
            observed["submit_count"] = page.locator("#submit-count").inner_text()
            observed["editor_code"] = page.locator("#code-editor").input_value()

        parsed = urlsplit(self.mock_url)
        url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))
        with redirect_stdout(io.StringIO()):
            result = real_online_run_workflow(
                url,
                ask_model=ask_model or self._model_for(code),
                headless=True,
                keep_open=True,
                inspection_callback=inspect_page,
                result_wait_seconds=result_wait_seconds,
            )
        return result, observed

    def test_correct_run_control_is_selected_and_submit_is_excluded(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.mock_url, wait_until="load")
            control = find_run_control(page)
            browser.close()

        self.assertTrue(control["found"], control)
        self.assertEqual(control["selector"], '[id="run-tests"]')
        self.assertEqual(control["visible_text"], "Run Tests")
        self.assertNotIn("submit", control["selector"].lower())

    def test_accessible_name_run_control_is_detected_without_id(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content("<button aria-label='Execute Code'>Execute</button>")
            control = find_run_control(page)
            browser.close()

        self.assertTrue(control["found"], control)
        self.assertIn("Execute", control["selector"])
        self.assertEqual(control["visible_text"], "Execute")

    def test_submit_type_input_is_never_accepted_as_a_run_control(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content("<form><input type='submit' value='Run Code'></form>")
            control = find_run_control(page)
            browser.close()

        self.assertFalse(control["found"])
        self.assertIn("No visible", control["reason"])

    def test_submit_type_input_cannot_be_clicked_even_with_forged_run_descriptor(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(
                "<form><textarea id='editor'></textarea>"
                "<button type='submit' id='run' aria-label='Run Code'>Run</button></form>"
            )
            page.locator("#editor").fill("answer = 42")
            budget = OnlineExecutionBudget()
            result = run_online_code(
                page,
                "#run",
                "#editor",
                "textarea",
                "answer = 42",
                {"found": True, "selector": "#run"},
                budget,
            )
            self.assertEqual(page.locator("#run").count(), 1)
            self.assertEqual(page.locator("#run-count").count(), 0)
            browser.close()

        self.assertFalse(result["executed"])
        self.assertEqual(budget.executions, 0)

    def test_correct_code_passes_and_submit_is_never_clicked(self):
        result, observed = self._run_workflow("def solve(input_data):\n    return sum(input_data)")

        self.assertTrue(result["executed"])
        self.assertEqual(result["status"], "passed")
        self.assertIn("All tests passed", result["summary"])
        self.assertEqual(observed["run_count"], "1")
        self.assertEqual(observed["submit_count"], "0")
        self.assertFalse(result["submit_clicked"])

    def test_wrong_answer_is_classified_as_failed(self):
        result, observed = self._run_workflow("def solve(input_data):\n    return 1")

        self.assertEqual(result["status"], "wrong_answer")
        self.assertIn("expected 6", result["output"])
        self.assertEqual(observed["submit_count"], "0")

    def test_runtime_error_is_detected_from_page_output(self):
        result, _ = self._run_workflow("def solve(input_data):\n    raise RuntimeError('bad')")

        self.assertEqual(result["status"], "runtime_error")
        self.assertIn("mock exception", result["error"])

    def test_timeout_is_detected_from_page_output(self):
        result, _ = self._run_workflow("def solve(input_data):\n    return 'MOCK_TIMEOUT'")

        self.assertEqual(result["status"], "timeout")
        self.assertIn("Time Limit Exceeded", result["summary"])

    def test_unknown_result_is_not_reported_as_success(self):
        result, observed = self._run_workflow(
            "def solve(input_data):\n    return None",
            query="unknown=1",
            result_wait_seconds=0.1,
        )

        self.assertTrue(result["executed"])
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["summary"], "Online result could not be verified.")
        self.assertEqual(observed["submit_count"], "0")

    def test_result_diagnostics_print_repr_and_poll_metadata(self):
        snapshot = {
            "title": "Run result",
            "result_text": "Failed:\tExpected 4, got 2",
            "body_text": "visible result\n" + ("x" * 5000) + "TAIL_NOT_PRINTED",
            "result_container_found": True,
            "result_container_selector": "[role='status']",
        }
        classification = {
            "status": "unknown",
            "summary": "Online result could not be verified.",
            "output": snapshot["result_text"],
            "error": "",
            "observation": snapshot,
            "polling_attempts": 17,
            "snapshot_text_changed": True,
        }
        output = io.StringIO()

        with patch(
            "real_online_run_workflow.read_online_test_result",
            return_value=classification,
        ), redirect_stdout(output):
            result = real_online_run_workflow(
                self.mock_url,
                ask_model=self._model_for("def solve(input_data):\n    return 1"),
                headless=True,
                keep_open=False,
            )

        diagnostic = output.getvalue()
        self.assertEqual(result["status"], "unknown")
        self.assertIn("result_text repr: " + repr(snapshot["result_text"]), diagnostic)
        self.assertIn("body_text repr: " + repr(snapshot["body_text"][:5000]), diagnostic)
        self.assertNotIn("TAIL_NOT_PRINTED", diagnostic)
        self.assertIn("current URL: " + self.mock_url, diagnostic)
        self.assertIn("result container found: True", diagnostic)
        self.assertIn("result container selector: [role='status']", diagnostic)
        self.assertIn("polling attempts: 17", diagnostic)
        self.assertIn("snapshot text changed during polling: True", diagnostic)

    def test_compilation_error_is_classified(self):
        result = classify_online_result({"result_text": "Compilation Error: invalid syntax"})

        self.assertEqual(result["status"], "compilation_error")
        self.assertIn("Compilation Error", result["error"])

    def test_login_required_takes_priority_over_problem_acceptance_statistics(self):
        result = classify_online_result({
            "result_text": "Test Result\nTest Result",
            "body_text": (
                "You need to log in / sign up to run or submit\n"
                "Accepted\n23,659,753/40.8M\nAcceptance Rate\n58.0%"
            ),
        })

        self.assertEqual(result["status"], "login_required")
        self.assertEqual(
            result["summary"],
            "LeetCode requires login before code can be run or submitted.",
        )
        self.assertFalse(result["verified"])

    def test_initial_login_prompt_returns_login_required_before_qwen_or_run(self):
        model_calls = []

        def model_must_not_run(prompt, max_tokens=150):
            model_calls.append(prompt)
            raise AssertionError("Login-required content must stop before Qwen.")

        result, observed = self._run_workflow(
            "def solve(input_data):\n    return sum(input_data)",
            query="login=1",
            ask_model=model_must_not_run,
        )

        self.assertEqual(result["status"], "LOGIN_REQUIRED")
        self.assertIn("requires login", result["summary"])
        self.assertFalse(result["executed"])
        self.assertEqual(model_calls, [])
        self.assertEqual(observed["run_count"], "0")
        self.assertEqual(observed["submit_count"], "0")

    def test_login_prompt_after_first_four_thousand_chars_returns_login_required(self):
        model_calls = []

        def model_must_not_run(prompt, max_tokens=150):
            model_calls.append(prompt)
            raise AssertionError("Login-required content must stop before Qwen.")

        result, observed = self._run_workflow(
            "def solve(input_data):\n    return sum(input_data)",
            query="login-late=1",
            ask_model=model_must_not_run,
        )

        self.assertEqual(result["status"], "LOGIN_REQUIRED")
        self.assertEqual(model_calls, [])
        self.assertEqual(observed["run_count"], "0")
        self.assertEqual(observed["submit_count"], "0")

    def test_security_block_remains_blocked_before_qwen_or_run(self):
        model_calls = []

        def model_must_not_run(prompt, max_tokens=150):
            model_calls.append(prompt)
            raise AssertionError("Security-blocked content must stop before Qwen.")

        result, observed = self._run_workflow(
            "def solve(input_data):\n    return sum(input_data)",
            query="security=1",
            ask_model=model_must_not_run,
        )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("access restriction", result["summary"])
        self.assertEqual(model_calls, [])
        self.assertEqual(observed["run_count"], "0")
        self.assertEqual(observed["submit_count"], "0")

    def test_accepted_in_result_panel_is_classified_as_passed(self):
        result = classify_online_result({
            "result_text": "Accepted",
            "body_text": "Acceptance Rate 58.0%",
        })

        self.assertEqual(result["status"], "passed")

    def test_unrelated_accepted_text_does_not_verify_test_result(self):
        result = classify_online_result({
            "result_text": "Test Result",
            "body_text": "Accepted\nAcceptance Rate\n58.0%",
        })

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["summary"], "Online result could not be verified.")

    def test_login_required_result_is_returned_without_fresh_panel_text(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(
                "<div id='test-result'>Test Result</div>"
                "<p>You need to log in / sign up to run or submit</p>"
                "<p>Accepted</p><p>Acceptance Rate 58.0%</p>"
            )
            result = read_online_test_result(
                page,
                previous_observation={"result_text": "Test Result"},
                timeout_seconds=1,
            )
            browser.close()

        self.assertEqual(result["status"], "login_required")
        self.assertFalse(result["verified"])
        self.assertEqual(result["polling_attempts"], 1)

    def test_execution_budget_prevents_repeated_run_clicks(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.mock_url, wait_until="load")
            editor = find_code_editor(page)
            control = find_run_control(page)
            code = "def solve(input_data):\n    return sum(input_data)"
            self.assertTrue(enter_code(page, editor["selector"], code, editor["editor_type"])["success"])
            budget = OnlineExecutionBudget(max_executions=5)
            first = run_online_code(
                page, control["selector"], editor["selector"], editor["editor_type"], code, control, budget
            )
            second = run_online_code(
                page, control["selector"], editor["selector"], editor["editor_type"], code, control, budget
            )
            run_count = page.locator("#run-count").inner_text()
            browser.close()

        self.assertEqual(MAX_ONLINE_EXECUTIONS, 1)
        self.assertTrue(first["executed"], first)
        self.assertFalse(second["executed"])
        self.assertIn("limit reached", second["reason"])
        self.assertEqual(run_count, "1")

    def test_zero_execution_budget_prevents_the_first_run(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.mock_url, wait_until="load")
            editor = find_code_editor(page)
            control = find_run_control(page)
            code = "def solve(input_data):\n    return sum(input_data)"
            enter_code(page, editor["selector"], code, editor["editor_type"])
            budget = OnlineExecutionBudget(max_executions=0)
            result = run_online_code(
                page, control["selector"], editor["selector"], editor["editor_type"], code, control, budget
            )
            run_count = page.locator("#run-count").inner_text()
            browser.close()

        self.assertFalse(result["executed"])
        self.assertEqual(budget.executions, 0)
        self.assertEqual(run_count, "0")

    def test_editor_mismatch_prevents_run(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.mock_url, wait_until="load")
            editor = find_code_editor(page)
            control = find_run_control(page)
            budget = OnlineExecutionBudget()
            result = run_online_code(
                page,
                control["selector"],
                editor["selector"],
                editor["editor_type"],
                "def solve(input_data):\n    return 42",
                control,
                budget,
            )
            run_count = page.locator("#run-count").inner_text()
            browser.close()

        self.assertFalse(result["executed"])
        self.assertIn("do not match", result["reason"])
        self.assertEqual(budget.executions, 0)
        self.assertEqual(run_count, "0")


if __name__ == "__main__":
    unittest.main()