import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from real_online_read_workflow import (
    MAX_MODEL_INPUT_CHARS,
    _generate_solution,
    real_online_read_workflow,
)


class RealOnlineReadWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="real-online-read-tests-")
        cls.temp_path = Path(cls.temp_dir.name)
        cls.addClassCleanup(cls.temp_dir.cleanup)
        cls.mock_problem_url = Path(__file__).with_name("mock_coding_page.html").resolve().as_uri()

    def _write_page(self, name, content):
        path = self.temp_path / name
        path.write_text(content, encoding="utf-8")
        return path.as_uri()

    def _model(self, prompt_lengths=None):
        def ask_model(prompt, max_tokens=150):
            if prompt_lengths is not None:
                prompt_lengths.append(len(prompt))
            if "Extract information about a coding problem" in prompt:
                if "weather dashboard" in prompt.lower():
                    return json.dumps({
                        "appears_coding_problem": False,
                        "problem_title": "Weather Dashboard",
                        "problem_statement": "",
                        "examples": [],
                        "constraints": [],
                        "input_description": "",
                        "output_description": "",
                        "programming_language": "",
                        "editor_information": "",
                    })
                if "Two Sum" in prompt:
                    return json.dumps({
                        "appears_coding_problem": True,
                        "problem_title": "Two Sum",
                        "problem_statement": "Return indices of two values whose sum equals target.",
                        "examples": ["[2, 7], target 9 -> [0, 1]"],
                        "constraints": ["One answer exists."],
                        "input_description": "An integer array and a target integer.",
                        "output_description": "Two indices.",
                        "programming_language": "Python",
                        "editor_information": "A labeled code editor is present.",
                    })
                if "coming soon" in prompt.lower():
                    return json.dumps({
                        "appears_coding_problem": True,
                        "problem_title": "Problem of the Day",
                        "problem_statement": "",
                        "examples": [],
                        "constraints": [],
                        "input_description": "",
                        "output_description": "",
                        "programming_language": "",
                        "editor_information": "",
                    })
                return json.dumps({
                    "appears_coding_problem": True,
                    "problem_title": "Sum a List",
                    "problem_statement": "Given a list of integers, return their sum.",
                    "examples": ["[1, 2, 3] -> 6"],
                    "constraints": ["The list may be empty."],
                    "input_description": "A list of integers.",
                    "output_description": "The sum as an integer.",
                    "programming_language": "Python",
                    "editor_information": "A code editor is present.",
                })
            if "Understand the extracted coding problem" in prompt:
                return json.dumps({
                    "problem_understanding": "Return the indices of two values that add to the target.",
                    "algorithm": "Store previously seen values and look up each complement.",
                    "complexity": "O(n) time and O(n) space.",
                    "code": "def solve(nums, target):\n    seen = {}\n    for index, value in enumerate(nums):\n        complement = target - value\n        if complement in seen:\n            return [seen[complement], index]\n        seen[value] = index\n    return []",
                })
            raise AssertionError("Unexpected model prompt:\n" + prompt)

        return ask_model

    def _run(self, url, ask_model=None):
        with redirect_stdout(io.StringIO()):
            return real_online_read_workflow(
                url,
                ask_model or self._model(),
                headless=True,
                keep_open=False,
            )

    def _run_with_output(self, url, ask_model):
        output = io.StringIO()
        with redirect_stdout(output):
            result = real_online_read_workflow(
                url,
                ask_model,
                headless=True,
                keep_open=False,
            )
        return result, output.getvalue()

    def _solution_problem(self):
        return {
            "problem_title": "Two Sum",
            "problem_statement": "Return the indices of two numbers that add to a target value.",
            "examples": [],
            "constraints": [],
        }

    def _solution_response(self, **fields):
        return {
            "algorithm": "Use a hash map to find each complement.",
            "complexity": "O(n) time and O(n) space.",
            "code": "def solve(nums, target): return []",
            **fields,
        }

    def _solution_model(self, response):
        return lambda prompt, max_tokens=150: json.dumps(response)

    def test_solution_response_with_problem_understanding_is_normalized(self):
        solution = _generate_solution(
            self._solution_problem(),
            self._solution_model(self._solution_response(
                problem_understanding="Return indices for the matching pair."
            )),
        )

        self.assertEqual(solution["problem_understanding"], "Return indices for the matching pair.")
        self.assertEqual(solution["algorithm"], "Use a hash map to find each complement.")

    def test_solution_response_accepts_title_and_statement_without_understanding(self):
        solution = _generate_solution(
            self._solution_problem(),
            self._solution_model(self._solution_response(
                title="Two Sum",
                statement="Return the indices of two numbers that add to a target value.",
                examples=["Input: [2, 7], target 9; Output: [0, 1]"],
                constraints=["Exactly one answer exists."],
                input="An integer array and target.",
                output="The two indices.",
            )),
        )

        self.assertIn("Two Sum: Return the indices", solution["problem_understanding"])
        self.assertIn("Examples:", solution["problem_understanding"])
        self.assertIn("Constraints:", solution["problem_understanding"])
        self.assertIn("Input:", solution["problem_understanding"])
        self.assertIn("Output:", solution["problem_understanding"])
        self.assertEqual(set(solution), {"problem_understanding", "algorithm", "complexity", "code"})

    def test_solution_response_accepts_problem_statement_alias(self):
        solution = _generate_solution(
            self._solution_problem(),
            self._solution_model(self._solution_response(
                title="Two Sum",
                problem_statement="Return the indices of two numbers that add to a target value.",
            )),
        )

        self.assertIn("Two Sum: Return the indices", solution["problem_understanding"])

    def test_solution_response_rejects_missing_required_problem_content(self):
        with self.assertRaisesRegex(ValueError, "valid problem title and meaningful statement"):
            _generate_solution(
                self._solution_problem(),
                self._solution_model(self._solution_response(title="Two Sum")),
            )

    def test_extraction_response_aliases_normalize_to_problem_fields(self):
        url = self._write_page(
            "aliased-problem.html",
            "<html><head><title>Problem</title></head><body>Page text requiring model extraction.</body></html>",
        )

        def alias_model(prompt, max_tokens=150):
            if "Extract information about a coding problem" in prompt:
                return json.dumps({
                    "title": "Sum a List",
                    "statement": "Given a list of integers, calculate and return their total sum.",
                    "examples": ["[1, 2, 3] -> 6"],
                    "constraints": ["The list may be empty."],
                    "input": "A list of integers.",
                    "output": "The sum as an integer.",
                })
            return json.dumps(self._solution_response(
                problem_understanding="Return the total sum of the list."
            ))

        result = self._run(url, alias_model)

        self.assertTrue(result["success"], result)
        self.assertEqual(result["problem"]["problem_title"], "Sum a List")
        self.assertEqual(result["problem"]["problem_statement"], "Given a list of integers, calculate and return their total sum.")
        self.assertEqual(result["problem"]["input_description"], "A list of integers.")
        self.assertEqual(result["problem"]["output_description"], "The sum as an integer.")

    def test_public_read_only_succeeds_without_solution_fields(self):
        url = self._write_page(
            "public-problem-only.html",
            "<html><body><h1>Two Sum</h1><p>Return the indices of two numbers that add to a target value.</p></body></html>",
        )
        prompts = []
        output = io.StringIO()

        def extraction_only_model(prompt, max_tokens=150):
            prompts.append(prompt)
            return json.dumps({
                "title": "Two Sum",
                "statement": "Return the indices of two numbers that add to a target value.",
            })

        with patch("real_online_read_workflow._validate_public_url"), patch(
            "real_online_read_workflow.classify_public_navigation",
            return_value={"status": "CONTINUE", "redirected": False},
        ), redirect_stdout(output):
            result = real_online_read_workflow(
                url,
                extraction_only_model,
                headless=True,
                keep_open=False,
                public_only=True,
            )

        self.assertTrue(result["success"], result)
        self.assertEqual(result["problem"]["problem_title"], "Two Sum")
        self.assertEqual(result["problem"]["problem_statement"], "Return the indices of two numbers that add to a target value.")
        self.assertEqual(len(prompts), 1)
        self.assertIn("Extract information about a coding problem", prompts[0])
        self.assertNotIn("Understand the extracted coding problem", prompts[0])
        self.assertEqual(result["solution"], {
            "problem_understanding": "",
            "algorithm": "",
            "complexity": "",
            "code": "",
        })
        self.assertNotIn("GENERATED CODE", output.getvalue())

    def test_public_read_only_rejects_empty_problem_response(self):
        url = self._write_page(
            "public-empty-problem.html",
            "<html><body><h1>Unknown</h1><p>Visible page text without problem details.</p></body></html>",
        )
        calls = []

        def empty_model(prompt, max_tokens=150):
            calls.append(prompt)
            return json.dumps({})

        output = io.StringIO()
        with patch("real_online_read_workflow._validate_public_url"), patch(
            "real_online_read_workflow.classify_public_navigation",
            return_value={"status": "CONTINUE", "redirected": False},
        ), redirect_stdout(output):
            result = real_online_read_workflow(
                url,
                empty_model,
                headless=True,
                keep_open=False,
                public_only=True,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "insufficient_information")
        self.assertIn("problem statement", result["missing_information"])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("solution", result)

    def test_reads_existing_local_mock_without_editor_interaction(self):
        prompts = []
        model = self._model()

        def capture_prompt(prompt, max_tokens=150):
            prompts.append(prompt)
            return model(prompt, max_tokens=max_tokens)

        result = self._run(self.mock_problem_url, capture_prompt)

        self.assertTrue(result["success"], result)
        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["problem"]["problem_title"], "Sum a List")
        self.assertIn("return their sum", result["problem"]["problem_statement"])
        self.assertIn("def solve", result["solution"]["code"])
        self.assertIn('"value": "Python"', prompts[0])
        self.assertNotIn('"value": ""', prompts[0])

    def test_reads_generic_public_problem_page_and_bounds_model_input(self):
        large_visible_text = "Visible context. " * 3000
        url = self._write_page(
            "generic-problem.html",
            f"""<!doctype html><html><head><title>Two Sum</title></head><body>
            <main><h1>Two Sum</h1><p>Given an integer array nums and an integer target,
            return indices of the two numbers such that they add up to target.</p>
            <h2>Example</h2><pre>nums = [2, 7], target = 9, output = [0, 1]</pre>
            <h2>Constraints</h2><p>One valid answer exists.</p>
            <label for="language">Language</label><select id="language"><option>Python</option></select>
            <label for="editor">Code</label><textarea id="editor"></textarea></main>
            <p>{large_visible_text}</p></body></html>""",
        )
        prompt_lengths = []

        result = self._run(url, self._model(prompt_lengths))

        self.assertTrue(result["success"], result)
        self.assertEqual(result["problem"]["problem_title"], "Two Sum")
        self.assertEqual(result["problem"]["programming_language"], "Python")
        self.assertEqual(result["solution"]["complexity"], "O(n) time and O(n) space.")
        self.assertTrue(prompt_lengths)
        self.assertLessEqual(prompt_lengths[0], MAX_MODEL_INPUT_CHARS)
        self.assertLessEqual(max(prompt_lengths), MAX_MODEL_INPUT_CHARS)

    def test_deterministically_extracts_leetcode_description_and_examples(self):
        url = self._write_page(
            "leetcode-description.html",
            """<!doctype html><html><head><title>Two Sum - LeetCode</title></head><body>
            <h1>1. Two Sum</h1>
            <div data-track-load="description_content">
              Given an integer array nums and an integer target, return indices of the two numbers
              such that they add up to target.
              <h2>Example 1:</h2>
              <p>Input: nums = [2,7], target = 9</p>
              <p>Output: [0,1]</p>
              <h2>Constraints:</h2>
              <p>- 2 &lt;= nums.length &lt;= 10^4</p>
            </div>
            <button>Python3</button>
            <div role="textbox" contenteditable="true"></div>
            </body></html>""",
        )
        prompts = []
        model = self._model()

        def capture_prompt(prompt, max_tokens=150):
            prompts.append(prompt)
            return model(prompt, max_tokens=max_tokens)

        result = self._run(url, capture_prompt)

        self.assertTrue(result["success"], result)
        self.assertEqual(result["problem"]["title"], "Two Sum")
        self.assertIn("return indices of the two numbers", result["problem"]["statement"])
        self.assertEqual(len(result["problem"]["examples"]), 1)
        self.assertIn("nums = [2,7]", result["problem"]["input"])
        self.assertEqual(result["problem"]["language"], "Python3")
        self.assertTrue(result["problem"]["editor_detected"])
        self.assertNotIn("Extract information about a coding problem", prompts[0])

    def test_missing_optional_fields_do_not_prevent_problem_read(self):
        url = self._write_page(
            "leetcode-minimal.html",
            """<!doctype html><html><body><h1>Array Total</h1>
            <div id="description_content">
              Given a list of integers, compute and return the total sum of its values.
            </div></body></html>""",
        )

        result = self._run(url)

        self.assertTrue(result["success"], result)
        self.assertEqual(result["problem"]["problem_title"], "Array Total")
        self.assertEqual(result["problem"]["examples"], [])
        self.assertEqual(result["problem"]["constraints"], [])
        self.assertEqual(result["problem"]["programming_language"], "")
        self.assertFalse(result["problem"]["editor_detected"])

    def test_malformed_qwen_json_does_not_count_as_extraction(self):
        url = self._write_page(
            "malformed-model.html",
            "<html><head><title>Unknown</title></head><body>Visible page text.</body></html>",
        )
        called = []

        def malformed_model(prompt, max_tokens=150):
            called.append(prompt)
            return '{"appears_coding_problem": true'

        result, output = self._run_with_output(url, malformed_model)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "open_or_read_failed")
        self.assertIn("Qwen returned invalid JSON", result["error"])
        self.assertIn("Qwen returned invalid JSON", output)
        self.assertEqual(len(called), 1)

    def test_qwen_natural_language_does_not_count_as_extraction(self):
        url = self._write_page(
            "natural-language-model.html",
            "<html><head><title>Unknown</title></head><body>Visible page text.</body></html>",
        )

        result, output = self._run_with_output(
            url,
            lambda prompt, max_tokens=150: "This page appears to be a coding problem.",
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "open_or_read_failed")
        self.assertIn("Qwen returned invalid JSON", output)
        self.assertNotIn("solution", result)

    def test_extraction_exception_is_reported_instead_of_null_blocker(self):
        url = self._write_page(
            "extraction-error.html",
            "<html><head><title>Unknown</title></head><body>Visible page text.</body></html>",
        )

        def failing_model(prompt, max_tokens=150):
            raise RuntimeError("test extraction failure")

        result, output = self._run_with_output(url, failing_model)

        self.assertEqual(result["status"], "open_or_read_failed")
        self.assertIn("RuntimeError: test extraction failure", result["error"])
        self.assertIn("ERROR: RuntimeError: test extraction failure", output)
        self.assertNotIn("BLOCKER: None", output)

    def test_reports_missing_coding_information_without_generating_code(self):
        url = self._write_page(
            "insufficient.html",
            "<html><head><title>Problem of the Day</title></head><body>"
            "A coding problem is coming soon. Given details will appear here.</body></html>",
        )

        result = self._run(url)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "insufficient_information")
        self.assertIn("problem statement", result["missing_information"])
        self.assertNotIn("solution", result)

    def test_classifies_normal_non_coding_page_without_generating_code(self):
        url = self._write_page(
            "weather.html",
            "<html><head><title>Weather Dashboard</title></head><body>"
            "Current conditions: sunny, 21 degrees.</body></html>",
        )

        result = self._run(url)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "insufficient_information")
        self.assertIn("a page that appears to contain a coding problem", result["missing_information"])
        self.assertNotIn("solution", result)

    def test_stops_on_simulated_login_or_captcha_block(self):
        url = self._write_page(
            "blocked.html",
            "<html><head><title>Sign in</title></head><body>"
            "Sign in to continue. <input type='password' name='password'>"
            "</body></html>",
        )
        called = []

        def model_must_not_run(prompt, max_tokens=150):
            called.append(prompt)
            raise AssertionError("Blocked content must not be sent to Qwen.")

        result = self._run(url, model_must_not_run)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "blocked")
        self.assertIn("authentication is required", result["blocker"].lower())
        self.assertEqual(called, [])

    def test_stops_on_captcha_instead_of_treating_it_as_a_problem(self):
        url = self._write_page(
            "captcha.html",
            "<html><head><title>Security check</title></head><body>Complete the CAPTCHA to continue.</body></html>",
        )
        called = []

        def model_must_not_run(prompt, max_tokens=150):
            called.append(prompt)
            raise AssertionError("CAPTCHA content must not be sent to Qwen.")

        result = self._run(url, model_must_not_run)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "blocked")
        self.assertIn("captcha", result["blocker"].lower())
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()