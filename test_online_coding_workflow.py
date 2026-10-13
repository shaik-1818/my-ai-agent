import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from playwright.sync_api import sync_playwright

from browser_tools import browser_read
from coding_workflow import MAX_CORRECTIONS, run_coding_workflow, verify_solution
from online_coding_workflow import (
    access_block_reason,
    online_coding_workflow,
    safe_observation,
)
from safe_code_runner import run_solution_tests


class OnlineCodingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_url = Path(__file__).with_name("mock_coding_page.html").resolve().as_uri()

    def _fake_qwen(self):
        state = {"classifications": 0, "plan": None}

        def ask_model(prompt, max_tokens=150):
            if "Extract the coding problem" in prompt:
                return (
                    '{"problem_statement":"Given a list of integers named input_data, return their sum.",'
                    '"examples":["[1,2,3] -> 6"],"constraints":["List may be empty."],'
                    '"expected_input_output":"Return an integer.","language":"Python",'
                    '"requirements":["Return the sum of all list values."]}'
                )
            if "Create a short implementation plan" in prompt:
                return '{"steps":["Sum the input values."],"algorithm":"sum","complexity":"O(n)"}'
            if "Write the solution code now" in prompt:
                return '{"code":"def solve(input_data):\\n    return input_data[0]"}'
            if "Select the code editor" in prompt:
                return '{"selector":"#code-editor"}'
            if "Select the Run/Test/Execute control" in prompt:
                return '{"selector":"#run-tests"}'
            if "Classify the latest visible website test/run result" in prompt:
                state["classifications"] += 1
                if "Wrong Answer" in prompt:
                    return '{"status":"failed","summary":"Wrong Answer","error_output":"expected 6, received 1"}'
                return '{"status":"passed","summary":"Accepted. All tests passed.","error_output":""}'
            if "Correct this Python code now" in prompt:
                return '{"code":"def solve(input_data):\\n    return sum(input_data)"}'
            if "Verify the candidate using only the problem requirements" in prompt:
                return '{"verified":true,"summary":"The visible run passed.","remaining_concerns":[]}'
            if "Select the Submit control only" in prompt:
                return '{"selector":"#submit-solution"}'
            if "Read the post-submit browser observation" in prompt:
                return '{"status":"submitted","summary":"Submission received."}'
            raise AssertionError("Unexpected Qwen prompt:\n" + prompt)

        return ask_model, state

    def _run_with_confirmation(self, confirmed):
        ask_model, model_state = self._fake_qwen()
        clicks = []
        original_click = __import__("online_coding_workflow").browser_click

        def record_click(page, selector):
            clicks.append(selector)
            return original_click(page, selector)

        with redirect_stdout(io.StringIO()):
            with patch("online_coding_workflow.browser_click", side_effect=record_click):
                result = online_coding_workflow(
                    self.mock_url,
                    ask_model=ask_model,
                    confirm_submission=lambda message: confirmed,
                    headless=True,
                )

        return result, clicks, model_state

    def _run_local_correction(self, correction_sources, max_corrections=MAX_CORRECTIONS):
        model_state = {"corrections": 0, "failure_evidence": []}

        def ask_model(prompt, max_tokens=150):
            if "Read this programming problem" in prompt:
                return '{"requirements":["Return the sum of integers from 1 through n."],"test_cases":[]}'
            if "Create a short implementation plan" in prompt:
                return '{"steps":["Add integers from 1 through n."],"algorithm":"sum(range(1, n + 1))","complexity":"O(n)"}'
            if "Your only task is code generation" in prompt:
                return '{"code":"def solve(n):\\n    return n"}'
            if "Analyze this candidate against the stated requirements" in prompt:
                model_state["failure_evidence"].append(prompt)
                failed = '"actual": 5' in prompt and '"expected": 15' in prompt
                if failed:
                    return '{"correction_needed":true,"findings":["Returned 5; expected 15."],"next_action":"correct"}'
                return '{"correction_needed":false,"findings":[],"next_action":"verify"}'
            if "Correct the Python candidate using the concrete test failures" in prompt:
                index = model_state["corrections"]
                model_state["corrections"] += 1
                source = correction_sources[min(index, len(correction_sources) - 1)]
                return json.dumps({"code": source})
            raise AssertionError("Unexpected Qwen prompt:\n" + prompt)

        with redirect_stdout(io.StringIO()):
            result = run_coding_workflow(
                "Given an integer n, return the sum of all integers from 1 to n.",
                ask_model=ask_model,
                confirm_execution=lambda _: True,
                test_cases=[{"input": 5, "expected": 15}],
            )
        return result, model_state

    def test_wrong_integer_sum_candidate_fails_with_actual_and_expected(self):
        result = run_solution_tests(
            "def solve(n):\n    return n\n",
            [{"input": 5, "expected": 15}],
            approved=True,
        )

        self.assertFalse(result["all_passed"])
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["results"][0]["passed"])
        self.assertEqual(result["results"][0]["actual"], 5)
        self.assertEqual(result["results"][0]["expected"], 15)

    def test_local_qwen_correction_uses_failure_evidence_and_returns_corrected_code(self):
        corrected = "def solve(n):\n    return sum(range(1, n + 1))"
        result, state = self._run_local_correction([corrected])

        self.assertEqual(result["code"], corrected)
        self.assertEqual(result["corrections"], 1)
        self.assertTrue(state["failure_evidence"])
        self.assertIn('"actual": 5', state["failure_evidence"][0])
        self.assertIn('"expected": 15', state["failure_evidence"][0])
        self.assertIn('"stdout": ""', state["failure_evidence"][0])
        self.assertIn('"stderr": ""', state["failure_evidence"][0])
        self.assertIn('"exit_code": 0', state["failure_evidence"][0])

    def test_corrected_integer_sum_candidate_passes_in_safe_runner(self):
        result = run_solution_tests(
            "def solve(n):\n    return sum(range(1, n + 1))\n",
            [{"input": 5, "expected": 15}],
            approved=True,
        )

        self.assertTrue(result["executed"])
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["results"][0]["actual"], 15)
        self.assertEqual(result["results"][0]["expected"], 15)

    def test_verification_requires_actual_successful_execution(self):
        failed_execution = {
            "executed": True,
            "exit_code": 0,
            "all_passed": False,
            "results": [{"passed": False, "actual": 5, "expected": 15}],
        }

        verification = verify_solution("sum integers", [], "def solve(n): return n", failed_execution, None)

        self.assertFalse(verification["verified"])
        result, _ = self._run_local_correction(["def solve(n):\n    return sum(range(1, n + 1))"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["verification"]["verified"], result["test_result"]["all_passed"])

    def test_correction_stops_after_configured_maximum(self):
        wrong = "def solve(n):\n    return n"
        result, state = self._run_local_correction([wrong], max_corrections=MAX_CORRECTIONS)

        self.assertEqual(result["corrections"], MAX_CORRECTIONS)
        self.assertEqual(state["corrections"], MAX_CORRECTIONS)
        self.assertEqual(len(state["failure_evidence"]), MAX_CORRECTIONS + 1)
        self.assertFalse(result["test_result"]["all_passed"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["test_result"]["results"][0]["actual"], 5)
        self.assertIn("could not be verified", result["verification"]["summary"])

    def test_complete_workflow_corrects_and_stops_without_confirmation(self):
        result, clicks, state = self._run_with_confirmation(False)

        self.assertTrue(result["success"], result)
        self.assertEqual(result["status"], "verified")
        self.assertFalse(result["submitted"])
        self.assertEqual(result["corrections"], 1)
        self.assertEqual(result["code"], "def solve(input_data):\n    return sum(input_data)")
        self.assertEqual(clicks, ["#run-tests", "#run-tests"])
        self.assertEqual(state["classifications"], 2)

    def test_submit_occurs_only_after_confirmation(self):
        result, clicks, _ = self._run_with_confirmation(True)

        self.assertTrue(result["success"], result)
        self.assertEqual(result["status"], "submitted")
        self.assertTrue(result["submitted"])
        self.assertEqual(clicks, ["#run-tests", "#run-tests", "#submit-solution"])

    def test_observation_redacts_secret_values_before_model_use(self):
        safe = safe_observation({
            "title": "Problem",
            "body_text": "password=hunter2 access_token=abc.def.ghi",
            "inputs": [{"type": "password", "value": "hunter2"}],
            "buttons": [],
            "editable_regions": [{"name": "access_token", "value": "abc.def.ghi", "selector": "#access_token"}],
        })

        safe_text = str(safe)
        self.assertNotIn("hunter2", safe_text)
        self.assertNotIn("abc.def.ghi", safe_text)
        self.assertNotIn("#access_token", safe_text)

    def test_login_and_captcha_pages_are_detected_as_stop_conditions(self):
        captcha_reason = access_block_reason({"title": "Security", "body_text": "Please complete the CAPTCHA."})
        login_reason = access_block_reason({
            "title": "Sign in",
            "body_text": "Please sign in to continue.",
            "inputs": [{"type": "password", "name": "password"}],
        })

        self.assertIn("CAPTCHA", captcha_reason)
        self.assertIn("password", login_reason.lower())

    def test_local_browser_observation_finds_editor_buttons_and_language(self):
        with redirect_stdout(io.StringIO()):
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(self.mock_url, wait_until="domcontentloaded")
                observation = browser_read(page)
                browser.close()

        self.assertEqual(observation["selects"][0]["value"], "Python")
        self.assertTrue(any(item["selector"] == "#code-editor" for item in observation["editable_regions"]))
        self.assertTrue(any(item["selector"] == "#run-tests" for item in observation["buttons"]))
        self.assertTrue(any(item["selector"] == "#submit-solution" for item in observation["buttons"]))


if __name__ == "__main__":
    unittest.main()
