import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from playwright.sync_api import sync_playwright

from real_online_editor_workflow import (
    MAX_CODE_LENGTH,
    enter_code,
    find_code_editor,
    real_online_code_entry_workflow,
)


class RealOnlineEditorWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="real-online-editor-tests-")
        cls.temp_path = Path(cls.temp_dir.name)
        cls.addClassCleanup(cls.temp_dir.cleanup)
        cls.mock_url = Path(__file__).with_name("mock_editor_page.html").resolve().as_uri()

    def _write_page(self, name, content):
        path = self.temp_path / name
        path.write_text(content, encoding="utf-8")
        return path.as_uri()

    @staticmethod
    def _model(prompt, max_tokens=150):
        if "Extract information about a coding problem" in prompt:
            return json.dumps({
                "appears_coding_problem": True,
                "problem_title": "Sum a List",
                "problem_statement": "Given a list of integers, return their sum.",
                "examples": ["[1, 2, 3] -> 6"],
                "constraints": [],
                "input_description": "A list of integers.",
                "output_description": "An integer sum.",
                "programming_language": "Python",
                "editor_information": "A code editor is present.",
            })
        if "Understand the extracted coding problem" in prompt:
            return json.dumps({
                "problem_understanding": "Return the sum of the input values.",
                "algorithm": "Use the built-in sum operation.",
                "complexity": "O(n) time.",
                "code": "def solve(input_data):\n    return sum(input_data)",
            })
        raise AssertionError("Unexpected model prompt:\n" + prompt)

    def _run(self, url, **kwargs):
        with redirect_stdout(io.StringIO()):
            return real_online_code_entry_workflow(
                url,
                ask_model=self._model,
                headless=True,
                keep_open=False,
                **kwargs,
            )

    def _assert_monaco_round_trip(self, code):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(
                """<div class='monaco-editor' data-uri='inmemory://test/two-sum'>
                <textarea id='code' aria-label='Code editor'></textarea></div>
                <script>
                const textarea = document.querySelector('#code');
                const model = {
                    uri: {toString: () => 'inmemory://test/two-sum'},
                    value: '',
                    setValue(value) {
                        this.value = value;
                        textarea.value = value;
                        window.modelSetCount += 1;
                    },
                    getValue() { return this.value; }
                };
                window.modelSetCount = 0;
                window.monaco = {editor: {getModels: () => [model]}};
                </script>"""
            )
            editor = find_code_editor(page)
            self.assertTrue(editor["found"], editor)
            self.assertEqual(editor["editor_type"], "monaco")
            with patch.object(page.keyboard, "insert_text", side_effect=AssertionError("Monaco must not type text")):
                result = enter_code(page, editor["selector"], code, editor["editor_type"])
            editor_value = page.locator(editor["selector"]).input_value()
            model_set_count = page.evaluate("window.modelSetCount")
            browser.close()

        self.assertTrue(result["success"], result)
        self.assertTrue(result["readback_matches"])
        self.assertEqual(result["readback"], code)
        self.assertEqual(editor_value, code)
        self.assertEqual(model_set_count, 1)
        self.assertEqual(result["insertion_method"], "monaco.model.setValue")

    def _run_with_model(self, url, ask_model):
        with redirect_stdout(io.StringIO()):
            return real_online_code_entry_workflow(
                url,
                ask_model=ask_model,
                headless=True,
                keep_open=False,
            )

    def test_workflow_finds_enters_and_reads_code_without_run_or_submit(self):
        button_counts = {}
        prompts = []

        def capture_prompt(prompt, max_tokens=150):
            prompts.append(prompt)
            return self._model(prompt, max_tokens=max_tokens)

        def inspect_page(page, result):
            button_counts["run"] = page.locator("#run-count").inner_text()
            button_counts["submit"] = page.locator("#submit-count").inner_text()
            button_counts["code"] = page.locator("#code-editor").input_value()

        with redirect_stdout(io.StringIO()):
            result = real_online_code_entry_workflow(
                self.mock_url,
                ask_model=capture_prompt,
                headless=True,
                keep_open=True,
                inspection_callback=inspect_page,
            )

        expected_code = "def solve(input_data):\n    return sum(input_data)"
        self.assertTrue(result["success"], result)
        self.assertEqual(result["status"], "entered")
        self.assertEqual(result["editor"]["selector"], '[id="code-editor"]')
        self.assertEqual(result["editor"]["editor_type"], "textarea")
        self.assertEqual(button_counts["code"], expected_code)
        self.assertTrue(result["entry"]["readback_matches"])
        self.assertFalse(result["run_clicked"])
        self.assertFalse(result["submit_clicked"])
        self.assertEqual(button_counts["run"], "0")
        self.assertEqual(button_counts["submit"], "0")
        self.assertNotIn("PRIVATE_EDITOR_DRAFT_123", "\n".join(prompts))

    def test_entry_verification_failure_prints_repr_and_first_difference(self):
        generated_code = "def solve(input_data):\n    return sum(input_data)"
        readback = "def solve(input_data):\n\treturn sum(input_data)"
        output = io.StringIO()

        with patch(
            "real_online_editor_workflow.enter_code",
            return_value={
                "success": False,
                "reason": "Editor readback did not match the generated code.",
                "readback_matches": False,
                "readback": readback,
            },
        ), redirect_stdout(output):
            result = real_online_code_entry_workflow(
                self.mock_url,
                ask_model=self._model,
                headless=True,
                keep_open=False,
            )

        diagnostic = output.getvalue()
        self.assertEqual(result["status"], "entry_failed")
        self.assertIn("Generated code repr: " + repr(generated_code), diagnostic)
        self.assertIn("Editor readback repr: " + repr(readback), diagnostic)
        self.assertIn("'editor_type': 'textarea'", diagnostic)
        self.assertIn(f"Generated code length: {len(generated_code)}", diagnostic)
        self.assertIn(f"Editor readback length: {len(readback)}", diagnostic)
        self.assertIn("First differing line: 2", diagnostic)
        self.assertIn("First differing character index: 0", diagnostic)
        self.assertIn("Generated character repr: ' '", diagnostic)
        self.assertIn("Readback character repr: '\\t'", diagnostic)
        self.assertNotIn("PRIVATE_EDITOR_DRAFT_123", diagnostic)

    def test_security_verification_page_is_blocked_before_model_or_editor_detection(self):
        url = self._write_page(
            "cloudflare-verification.html",
            """<html><head><title>Just a moment...</title></head><body>
            <h1>Performing security verification</h1>
            <p>This website uses a security service to protect against malicious bots.</p>
            <p>This page is displayed while the website verifies you are not a bot.</p>
            </body></html>""",
        )
        calls = []

        def model_must_not_run(prompt, max_tokens=150):
            calls.append(prompt)
            raise AssertionError("Verification pages must be blocked before Qwen is called.")

        result = self._run_with_model(url, model_must_not_run)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["stage"], "page_inspection")
        self.assertIn("verification", result["reason"].lower())
        self.assertEqual(result["page_type"], "security_verification")
        self.assertFalse(result["editor_found"])
        self.assertEqual(result["url"], url)
        self.assertEqual(calls, [])

    def test_missing_editor_returns_structured_entry_failed_result(self):
        url = self._write_page(
            "problem-without-editor.html",
            "<html><body><h1>Sum a List</h1><p>Problem description.</p></body></html>",
        )
        prompts = []

        def capture_prompt(prompt, max_tokens=150):
            prompts.append(prompt)
            return self._model(prompt, max_tokens=max_tokens)

        result = self._run_with_model(url, capture_prompt)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "entry_failed")
        self.assertEqual(result["stage"], "editor_detection")
        self.assertIn("No visible", result["reason"])
        self.assertEqual(result["url"], url)
        self.assertEqual(result["page_type"], "coding_problem")
        self.assertFalse(result["editor_found"])
        self.assertEqual(len(prompts), 2)

    def test_unlabelled_editor_value_is_not_sent_as_accessibility_text(self):
        from browser_tools import browser_read
        from real_online_editor_workflow import _problem_observation

        url = self._write_page(
            "unlabelled-private-editor.html",
            "<html><body><p>Given a value, return the result.</p>"
            "<textarea>PRIVATE_VALUE_THAT_MUST_NOT_LEAK</textarea></body></html>",
        )
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="load")
            with redirect_stdout(io.StringIO()):
                observation = browser_read(page)
            bounded = _problem_observation(observation, page)
            browser.close()

        self.assertNotIn("PRIVATE_VALUE_THAT_MUST_NOT_LEAK", str(bounded))

    def test_contenteditable_editor_is_filled_and_verified(self):
        url = self._write_page(
            "contenteditable.html",
            "<html><body><h1>Code editor</h1>"
            "<div id='editable-code' role='textbox' aria-label='Code editor' contenteditable='true'></div>"
            "</body></html>",
        )
        code = "def solve(n):\n    return n + 1"

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="load")
            editor = find_code_editor(page)
            result = enter_code(page, editor["selector"], code, editor["editor_type"])
            browser.close()

        self.assertTrue(editor["found"], editor)
        self.assertEqual(editor["editor_type"], "contenteditable")
        self.assertTrue(result["success"], result)
        self.assertTrue(result["readback_matches"])
        self.assertEqual(result["readback"], code)

    def test_monaco_multiline_python_indentation_is_preserved(self):
        code = (
            "class Solution:\n"
            "    def twoSum(self, nums, target):\n"
            "        num_map = {}\n"
            "        for i, num in enumerate(nums):\n"
            "            complement = target - num\n"
            "            if complement in num_map:\n"
            "                return [num_map[complement], i]\n"
            "            num_map[num] = i"
        )

        self._assert_monaco_round_trip(code)

    def test_monaco_nested_indentation_is_preserved(self):
        code = (
            "class Solution:\n"
            "    def solve(self, values):\n"
            "        for value in values:\n"
            "            if value > 0:\n"
            "                if value % 2 == 0:\n"
            "                    return value\n"
            "        return None"
        )

        self._assert_monaco_round_trip(code)

    def test_monaco_multiple_blank_lines_are_preserved(self):
        code = (
            "class Solution:\n"
            "\n"
            "    def solve(self, values):\n"
            "        total = 0\n"
            "\n"
            "        for value in values:\n"
            "            total += value\n"
            "\n"
            "        return total"
        )

        self._assert_monaco_round_trip(code)

    def test_associated_visible_label_provides_a_stable_selector(self):
        url = self._write_page(
            "label-editor.html",
            "<html><body><label>Solution code<textarea></textarea></label></body></html>",
        )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="load")
            editor = find_code_editor(page)
            result = enter_code(page, editor["selector"], "answer = 42", editor["editor_type"])
            browser.close()

        self.assertTrue(editor["found"], editor)
        self.assertIn("label:has-text", editor["selector"])
        self.assertTrue(result["success"], result)

    def test_codemirror_and_ace_surfaces_use_normal_fill_only(self):
        for class_name, expected_type in (
            ("CodeMirror", "codemirror"),
            ("ace_editor", "ace"),
        ):
            with self.subTest(editor_type=expected_type):
                url = self._write_page(
                    f"{expected_type}.html",
                    f"<html><body><div class='{class_name}'>"
                    "<div id='code' contenteditable='true' aria-label='Code editor'></div>"
                    "</div></body></html>",
                )
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="load")
                    editor = find_code_editor(page)
                    result = enter_code(page, editor["selector"], "def solve():\n    pass", editor["editor_type"])
                    browser.close()

                self.assertTrue(editor["found"], editor)
                self.assertEqual(editor["editor_type"], expected_type)
                self.assertTrue(result["success"], result)
                self.assertEqual(result["readback"], "def solve():\n    pass")

    def test_editor_not_found_on_page_without_editable_controls(self):
        url = self._write_page("no-editor.html", "<html><body><p>Problem only.</p></body></html>")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="load")
            result = find_code_editor(page)
            browser.close()

        self.assertFalse(result["found"])
        self.assertIn("No visible", result["reason"])

    def test_hidden_editor_is_not_selected(self):
        url = self._write_page(
            "hidden-editor.html",
            "<html><body><textarea id='hidden-code' style='display:none' aria-label='Code editor'></textarea></body></html>",
        )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="load")
            result = find_code_editor(page)
            browser.close()

        self.assertFalse(result["found"])
        self.assertIn("visible", result["reason"])

    def test_rich_editor_with_only_hidden_input_is_reported_unsupported(self):
        url = self._write_page(
            "unsupported-monaco.html",
            "<html><body><div class='monaco-editor' style='width:500px;height:300px'>"
            "<textarea class='inputarea' style='display:none'></textarea>"
            "</div></body></html>",
        )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="load")
            result = find_code_editor(page)
            browser.close()

        self.assertTrue(result["found"], result)
        self.assertFalse(result["supported"])
        self.assertIn("no visible native editable surface", result["reason"])

    def test_code_insertion_readback_mismatch_is_reported(self):
        url = self._write_page(
            "readback.html",
            "<html><body><textarea id='code' aria-label='Code editor'></textarea></body></html>",
        )
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="load")
            with patch("real_online_editor_workflow._read_editor_value", return_value="not the code"):
                result = enter_code(page, "#code", "print('hello')", "textarea")
            browser.close()

        self.assertFalse(result["success"])
        self.assertFalse(result["readback_matches"])

    def test_rich_editor_interaction_failure_is_reported_unsupported(self):
        url = self._write_page(
            "rich-interaction-failure.html",
            "<html><body><div class='CodeMirror' style='width:400px;height:200px'>"
            "<div id='code' contenteditable='true' aria-label='Code editor'></div>"
            "</div></body></html>",
        )
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="load")
            editor = find_code_editor(page)
            with patch(
                "real_online_editor_workflow._read_editor_value",
                side_effect=RuntimeError("editor state is not readable"),
            ):
                result = enter_code(page, editor["selector"], "print('safe')", editor["editor_type"])
            browser.close()

        self.assertFalse(result["success"])
        self.assertTrue(result["unsupported"])
        self.assertIn("normal interaction", result["reason"])

    def test_maximum_code_length_is_enforced_before_interaction(self):
        result = enter_code(None, "#code", "x" * (MAX_CODE_LENGTH + 1), "textarea")

        self.assertFalse(result["success"])
        self.assertIn("character limit", result["reason"])

    def test_ordinary_text_input_requires_editor_semantics_and_accepts_single_line(self):
        url = self._write_page(
            "text-input.html",
            "<html><body><label for='solution'>Solution code</label>"
            "<input id='solution' aria-label='Code editor'></body></html>",
        )
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="load")
            editor = find_code_editor(page)
            result = enter_code(page, editor["selector"], "return value", editor["editor_type"])
            browser.close()

        self.assertTrue(editor["found"], editor)
        self.assertEqual(editor["editor_type"], "text_input")
        self.assertTrue(result["success"], result)


if __name__ == "__main__":
    unittest.main()