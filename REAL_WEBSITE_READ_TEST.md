# Manual Real-Website Read-Only Test

This is a manual integration procedure. Do not treat local mock tests as evidence that a public website was opened.

1. Start Ollama and make the configured Qwen3 4B model available.
2. In a terminal at the project directory, activate the project environment if needed, then run `python agent.py`.
3. Choose `real_website_readonly`.
4. Enter a public coding-problem URL using `https://` or `http://`, for example `https://public-host/path/to/problem`.
5. Confirm that Chromium opens the entered URL and inspect the terminal report. On success, verify the title, statement, examples, constraints, input/output descriptions, generated understanding, algorithm, complexity, and candidate code.
6. Confirm the report ends with `STATUS: READY_FOR_EDITOR` and the browser remains open for inspection.
7. Press Enter in the terminal when finished inspecting the page.

The workflow is read-only: it does not enter code, click Run/Test, or click Submit. Same-host path redirects and redirects to a subdomain of the requested host are allowed when the final page has no access blocker. Cross-origin redirects are reported as unexpected navigation. Login, CAPTCHA, anti-bot/access blocking, and missing content are reported as stop conditions; the workflow does not attempt to bypass them. Do not provide account credentials or use an authenticated account for this test.