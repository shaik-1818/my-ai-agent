import json
import re
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

from browser_tools import browser_click, browser_open, browser_read, browser_type
from coding_workflow import (
    create_solution_plan,
)


MAX_CORRECTION_ATTEMPTS = 3
ONLINE_CODING_MODE = "ONLINE_CODING"
SECRET_VALUE_PATTERNS = (
    re.compile(
        r"(?i)\b(password|passwd|secret|api[_ -]?key|access[_ -]?token|"
        r"authorization|cookie|csrf[_ -]?token)\b\s*[:=]\s*['\"]?[^\s,;\"'<>]+"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
)
SENSITIVE_FIELD_PATTERN = re.compile(
    r"(?i)(password|passwd|secret|token|api[_ -]?key|authorization|cookie|csrf|session)"
)
BLOCKED_PAGE_PATTERNS = (
    re.compile(r"(?i)\bcaptcha\b"),
    re.compile(r"(?i)verify (that )?you are (a )?human"),
    re.compile(r"(?i)unusual traffic"),
    re.compile(r"(?i)automated requests? (are )?not allowed"),
    re.compile(r"(?i)access denied"),
    re.compile(r"(?i)security check"),
    re.compile(r"(?i)anti[- ]bot"),
)
LOGIN_REQUIRED_PATTERNS = (
    re.compile(r"(?i)sign in to continue"),
    re.compile(r"(?i)log in to continue"),
    re.compile(r"(?i)authentication required"),
    re.compile(r"(?i)login required"),
    re.compile(r"(?i)you need to\s+(?:log\s*in|sign\s*in|login)\s*/\s*(?:sign\s*up|register)"),
    re.compile(r"(?i)\b(?:you need to\s+)?(?:log\s*in|sign\s*in|login|sign-in)\b.{0,120}\b(?:sign\s*up|run|submit)\b"),
    re.compile(r"(?i)\b(?:run|submit)\b.{0,120}\b(?:log\s*in|sign\s*in|login|sign-in)\b"),
)


def _ask_json(ask_model, prompt, max_tokens=1200):
    response = ask_model(prompt, max_tokens=max_tokens)
    start = response.find("{")
    end = response.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"Qwen did not return a JSON object: {response}")
    try:
        result = json.loads(response[start:end])
    except json.JSONDecodeError as error:
        raise ValueError(f"Qwen returned invalid JSON: {error}\n{response}") from error
    if not isinstance(result, dict):
        raise ValueError("Qwen response must be a JSON object.")
    return result


def _redact_text(value):
    text = str(value or "")
    for pattern in SECRET_VALUE_PATTERNS:
        text = pattern.sub(r"\1=[REDACTED]" if pattern.groups else "[REDACTED]", text)
    return text


def _safe_element(element, include_value=False):
    identity = " ".join(
        str(element.get(key, ""))
        for key in ("id", "name", "type", "placeholder", "text", "selector", "value")
    )
    if SENSITIVE_FIELD_PATTERN.search(identity):
        return None

    safe = {}
    keys = ["tag", "role", "type", "placeholder", "text", "selector", "contenteditable"]
    if include_value:
        keys.append("value")
    for key in keys:
        if element.get(key):
            safe[key] = _redact_text(element[key])
    return safe


def safe_observation(observation):
    return {
        "title": _redact_text(observation.get("page_title") or observation.get("title", "")),
        "body_text": _redact_text(observation.get("body_text", ""))[:10000],
        "inputs": [
            safe
            for safe in (_safe_element(item, include_value=True) for item in observation.get("inputs", []))
            if safe is not None
        ],
        "buttons": [
            safe
            for safe in (_safe_element(item) for item in observation.get("buttons", []))
            if safe is not None
        ],
        "editable_regions": [
            safe
            for safe in (_safe_element(item, include_value=True) for item in observation.get("editable_regions", []))
            if safe is not None
        ],
        "selects": [
            safe
            for safe in (_safe_element(item, include_value=True) for item in observation.get("selects", []))
            if safe is not None
        ],
        "elements": [
            {
                "id": el.get("id"),
                "selector": el.get("selector"),
                "tag": el.get("tag"),
                "text": _redact_text(el.get("text", "")),
            }
            for el in observation.get("elements", [])
            if el.get("id") and not SENSITIVE_FIELD_PATTERN.search(el.get("id", ""))
        ],
    }


def access_block_reason(observation):
    visible_text = " ".join(
        [
            str(observation.get("page_title") or observation.get("title") or ""),
            str(observation.get("body_text") or ""),
        ]
    )
    if any(pattern.search(visible_text) for pattern in BLOCKED_PAGE_PATTERNS):
        return "The page shows a CAPTCHA, anti-bot challenge, or access restriction. Automation stopped without attempting to bypass it."

    for field in observation.get("inputs", []):
        field_type = str(field.get("type", "")).lower()
        if field_type == "password":
            return "The page contains a password field. Authentication is required; the workflow will not enter or send credentials."

    if any(pattern.search(visible_text) for pattern in LOGIN_REQUIRED_PATTERNS):
        return "The page requires login. The workflow will not bypass authentication or handle credentials."

    return None


def _read_page(page):
    observation = browser_read(page)
    if not observation.get("success"):
        raise RuntimeError(f"Could not read the webpage: {observation.get('error', 'unknown error')}")
    return observation


def read_coding_problem(observation, ask_model):
    print("\n[ONLINE_CODING] Identify problem and constraints")
    problem_view = safe_observation(observation)
    prompt = f"""
Extract the coding problem from the visible webpage. Do not solve it yet.
Return only JSON with these keys:
- problem_statement: string
- examples: list of visible examples
- constraints: list of visible constraints
- expected_input_output: string describing the site's required interface
- language: current programming language if visible, otherwise "Python"
- requirements: list of concise behavioral requirements

Do not infer hidden content. Use only this browser observation.

OBSERVATION:
{json.dumps(problem_view, ensure_ascii=True)}
"""
    details = _ask_json(ask_model, prompt, max_tokens=1800)
    details["problem_statement"] = str(details.get("problem_statement") or problem_view["body_text"])
    for key in ("examples", "constraints", "requirements"):
        if not isinstance(details.get(key), list):
            value = details.get(key)
            details[key] = [str(value)] if value else []
    if not details["requirements"]:
        details["requirements"] = [details["problem_statement"]]
    details["expected_input_output"] = str(details.get("expected_input_output") or "Follow the function signature and input/output format shown on the page.")
    details["language"] = str(details.get("language") or "Python")
    return details


def _problem_context(problem):
    return json.dumps(problem, ensure_ascii=True, indent=2)


def _online_requirements(details):
    return details.get("requirements", []) + [
        f"Examples: {details.get('examples', [])}",
        f"Constraints: {details.get('constraints', [])}",
        f"Expected input/output: {details.get('expected_input_output', '')}",
    ]


def _code_from_model_response(ask_model, prompt):
    for attempt in range(2):
        response = _ask_json(ask_model, prompt, max_tokens=1200)
        code = response.get("code")
        if isinstance(code, str) and code.strip():
            code = code.strip()
            fenced = re.fullmatch(r"```(?:python|py|[a-zA-Z0-9+#.-]+)?\s*\n([\s\S]*?)\n?```", code)
            return fenced.group(1) if fenced else code

        prompt += f"""
The previous response was not source code: {json.dumps(response, ensure_ascii=True)}
This is still the code-generation stage. Do not repeat the problem extraction,
requirements, or plan JSON. Return exactly one JSON object with a non-empty
string field named code containing only the complete {ONLINE_CODING_MODE} solution.
"""
    raise ValueError("Qwen did not return source code in the 'code' field after one format retry.")


def generate_solution(problem, details, plan, ask_model):
    print("\n[ONLINE_CODING] Generate platform solution")
    algorithm = plan.get("algorithm", "Follow the solution plan from the previous stage.")
    prompt = f"""
Write the solution code now in {details['language']}. Do not summarize or
extract the problem. Return exactly JSON:
{{"code":"<complete source code>"}}.
Follow the function signature and input/output contract explicitly shown in
the problem. If no signature is shown, use a simple function for the stated
input and return value. Do not invent an uncertain class or function name.
Do not submit or execute the code.

Problem: {problem}
Required interface: {details.get('expected_input_output', '')}
Algorithm: {algorithm}
"""
    return _code_from_model_response(ask_model, prompt)


def find_code_editor(observation, ask_model):
    editors = safe_observation(observation)["editable_regions"]
    if not editors:
        raise RuntimeError("No generic editable code region was found in the browser observation.")

    prompt = f"""
Select the code editor from the observed editable regions. Do not select an
ordinary search, email, password, or login field. Return only JSON with key
selector, copied exactly from one listed region, or null if none is a code editor.

EDITABLE REGIONS:
{json.dumps(editors, ensure_ascii=True)}
"""
    choice = _ask_json(ask_model, prompt, max_tokens=500)
    selector = choice.get("selector")
    valid_selectors = {item.get("selector") for item in editors}
    if not selector or selector not in valid_selectors:
        raise RuntimeError("Qwen did not identify a code editor from the observed editable regions.")
    return selector


def enter_code(page, editor_selector, code):
    result = browser_type(page, editor_selector, code)
    if not result.get("success"):
        raise RuntimeError(f"Could not enter code in the observed editor: {result.get('error', 'unknown error')}")
    return result


def _find_button(observation, purpose, ask_model):
    buttons = safe_observation(observation)["buttons"]
    if purpose == "run":
        candidates = [
            item for item in buttons
            if re.search(r"\b(run|test|execute|check|evaluate)\b", item.get("text", ""), re.IGNORECASE)
            and not re.search(r"\bsubmit\b", item.get("text", ""), re.IGNORECASE)
        ]
        instruction = "Select the Run/Test/Execute control; never select Submit."
    else:
        candidates = [
            item for item in buttons
            if re.search(r"\bsubmit\b", item.get("text", ""), re.IGNORECASE)
        ]
        instruction = "Select the Submit control only."

    if not candidates:
        raise RuntimeError(f"No observed {purpose} button was found.")

    prompt = f"""
{instruction}
Return only JSON with key selector, copied exactly from one listed button,
or null if none is suitable.

OBSERVED BUTTONS:
{json.dumps(candidates, ensure_ascii=True)}
"""
    choice = _ask_json(ask_model, prompt, max_tokens=500)
    selector = choice.get("selector")
    if not selector or selector not in {item.get("selector") for item in candidates}:
        raise RuntimeError(f"Qwen did not select an observed {purpose} button.")
    return selector


def run_code(page, observation, ask_model):
    print("\n[ONLINE_CODING] Locate and click Run/Test")
    current_observation = _read_page(page)
    if access_block_reason(current_observation):
        raise RuntimeError(access_block_reason(current_observation))
    run_selector = _find_button(current_observation, "run", ask_model)
    result = browser_click(page, run_selector)
    if not result.get("success"):
        raise RuntimeError(f"Could not run the code: {result.get('error', 'unknown error')}")
    return result


def read_test_result(page, ask_model):
    print("\n[ONLINE_CODING] Read and classify webpage test result")
    observation = _read_page(page)
    blocked = access_block_reason(observation)
    if blocked:
        return {
            "status": "blocked",
            "summary": blocked,
            "error_output": "",
            "observation": safe_observation(observation),
        }

    prompt = f"""
Classify the latest visible website test/run result. Do not claim success unless
there is visible evidence. Return only JSON with keys:
- status: exactly one of "passed", "failed", "unknown"
- summary: concise visible result summary
- error_output: exact relevant failure/runtime/compiler text if visible, otherwise empty

PAGE OBSERVATION:
{json.dumps(safe_observation(observation), ensure_ascii=True)}
"""
    result = _ask_json(ask_model, prompt, max_tokens=900)
    status = result.get("status")
    if status not in {"passed", "failed", "unknown"}:
        status = "unknown"
    return {
        "status": status,
        "summary": _redact_text(result.get("summary", "No clear test result was identified.")),
        "error_output": _redact_text(result.get("error_output", "")),
        "observation": safe_observation(observation),
    }


def correct_solution(problem, details, plan, code, test_result, ask_model):
    print("\n[ONLINE_CODING] Ask Qwen to correct the failing solution")
    prompt = f"""
Correct this {details['language']} code now using the actual site failure.
Do not restate the problem or plan. Return exactly JSON:
{{"code":"<complete corrected source>"}}.

Problem: {problem}
Required behavior: {json.dumps(details.get('requirements', []), ensure_ascii=True)}
Current code: {code}
Observed failure: {test_result.get('summary', '')}
Error/output: {test_result.get('error_output', '')}
"""
    return _code_from_model_response(ask_model, prompt)


def verify_solution(problem, details, code, test_result, ask_model):
    print("\n[ONLINE_CODING] Verify successful run against the problem")
    prompt = f"""
Verify the candidate using only the problem requirements and visible website
run result. A solution is verified only when the site result is explicitly
passed and the code matches the stated interface. Do not submit anything.

PROBLEM DETAILS:
{_problem_context(details)}

CANDIDATE CODE:
{code}

VISIBLE TEST RESULT:
{json.dumps(test_result, ensure_ascii=True)}

Return only JSON with keys: verified (boolean), summary (string),
remaining_concerns (list).
"""
    verification = _ask_json(ask_model, prompt, max_tokens=1000)
    verification["verified"] = bool(verification.get("verified")) and test_result.get("status") == "passed"
    if not isinstance(verification.get("remaining_concerns"), list):
        verification["remaining_concerns"] = []
    return verification


def request_submission_confirmation(confirm_callback):
    prompt = (
        "The solution has been tested and verified.\n"
        "Do you want me to submit it? Confirm explicitly with SUBMIT."
    )
    return bool(confirm_callback(prompt))


def submit_solution(page, observation, ask_model, confirmed=False):
    if not confirmed:
        return {"success": False, "submitted": False, "error": "Submission was not explicitly confirmed."}

    print("\n[ONLINE_CODING] Submit after explicit user confirmation")
    current_observation = _read_page(page)
    blocked = access_block_reason(current_observation)
    if blocked:
        return {"success": False, "submitted": False, "error": blocked}

    submit_selector = _find_button(current_observation, "submit", ask_model)
    click_result = browser_click(page, submit_selector)
    if not click_result.get("success"):
        return {
            "success": False,
            "submitted": False,
            "error": click_result.get("error", "Submit click failed."),
        }

    final_observation = _read_page(page)
    blocked = access_block_reason(final_observation)
    if blocked:
        return {"success": False, "submitted": True, "error": blocked}

    return {
        "success": True,
        "submitted": True,
        "observation": safe_observation(final_observation),
    }


def _validate_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https", "file"}:
        raise ValueError("Use an http, https, or local file URL.")
    if parsed.scheme in {"http", "https"} and not parsed.netloc:
        raise ValueError("The coding website URL is invalid.")


def online_coding_workflow(
    url,
    ask_model,
    confirm_submission,
    headless=False,
    max_corrections=MAX_CORRECTION_ATTEMPTS,
):
    print(f"\nMODE: {ONLINE_CODING_MODE}")
    _validate_url(url)
    max_corrections = max(0, min(int(max_corrections), MAX_CORRECTION_ATTEMPTS))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            opened = browser_open(page, url)
            if not opened.get("success"):
                return {"success": False, "status": "open_failed", "error": opened.get("error")}

            observation = _read_page(page)
            blocked = access_block_reason(observation)
            if blocked:
                print(blocked)
                return {"success": False, "status": "blocked", "error": blocked}

            details = read_coding_problem(observation, ask_model)
            problem_context = details["problem_statement"]
            requirements = _online_requirements(details)
            plan = create_solution_plan(
                problem_context,
                requirements,
                ask_model,
                language=details["language"],
                online=True,
            )
            code = generate_solution(problem_context, details, plan, ask_model)

            editor_selector = find_code_editor(observation, ask_model)
            enter_code(page, editor_selector, code)

            result = None
            verification = {"verified": False, "summary": "No successful run was observed.", "remaining_concerns": []}
            corrections = 0
            for attempt in range(max_corrections + 1):
                run_code(page, observation, ask_model)
                result = read_test_result(page, ask_model)

                if result["status"] == "blocked":
                    print(result["summary"])
                    return {
                        "success": False,
                        "status": "blocked",
                        "problem": details,
                        "language": details["language"],
                        "code": code,
                        "test_result": result,
                        "corrections": corrections,
                    }

                if result["status"] == "passed":
                    verification = verify_solution(problem_context, details, code, result, ask_model)
                    if verification["verified"]:
                        break
                    result["status"] = "unknown"
                    result["summary"] = "The site reported a pass, but independent verification was inconclusive."
                    break

                if result["status"] != "failed" or corrections >= max_corrections:
                    break

                code = correct_solution(problem_context, details, plan, code, result, ask_model)
                corrections += 1
                enter_code(page, editor_selector, code)

            if result is None or result["status"] != "passed" or not verification["verified"]:
                print("\nSOLUTION NOT VERIFIED")
                if result:
                    print("Test result:", result.get("summary"), result.get("error_output", ""))
                return {
                    "success": False,
                    "status": result["status"] if result else "unknown",
                    "problem": details,
                    "language": details["language"],
                    "code": code,
                    "test_result": result,
                    "verification": verification,
                    "corrections": corrections,
                    "submitted": False,
                }

            print("\nSOLUTION VERIFIED")
            print("\nProblem:\n" + details["problem_statement"])
            print("\nLanguage:\n" + details["language"])
            print("\nCode:\n" + code)
            print("\nTest result:\n" + result["summary"])

            confirmed = request_submission_confirmation(confirm_submission)
            if not confirmed:
                print("\nSubmission skipped; the solution remains in the editor.")
                return {
                    "success": True,
                    "status": "verified",
                    "problem": details,
                    "language": details["language"],
                    "code": code,
                    "test_result": result,
                    "verification": verification,
                    "corrections": corrections,
                    "submitted": False,
                }

            submission = submit_solution(page, result["observation"], ask_model, confirmed=True)
            if submission.get("submitted") and submission.get("success"):
                submission_result = _ask_json(ask_model, f"""
Read the post-submit browser observation and report the visible submission status.
Do not claim acceptance unless shown in the page.
Return JSON with keys status and summary.

POST-SUBMIT OBSERVATION:
{json.dumps(submission['observation'], ensure_ascii=True)}
""", max_tokens=700)
                print("\nSubmission result:", submission_result.get("summary", submission_result.get("status")))
            else:
                submission_result = submission
                print("\nSubmission was not completed:", submission.get("error"))

            return {
                "success": True,
                "status": "submitted" if submission.get("submitted") else "verified",
                "problem": details,
                "language": details["language"],
                "code": code,
                "test_result": result,
                "verification": verification,
                "corrections": corrections,
                "submitted": bool(submission.get("submitted")),
                "submission_result": submission_result,
            }
        finally:
            browser.close()
