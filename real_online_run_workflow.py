import re
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from browser_tools import browser_open, browser_read
from online_coding_workflow import access_block_reason, safe_observation
from real_online_editor_workflow import (
    _problem_observation,
    _read_editor_value,
    _wait_for_page,
    enter_code,
    find_code_editor,
)
from real_online_read_workflow import (
    _extract_problem,
    _generate_solution,
    _validate_url,
)


REAL_ONLINE_RUN_MODE = "REAL_ONLINE_RUN"
MAX_ONLINE_EXECUTIONS = 1
MAX_ONLINE_TASK_EXECUTIONS = 4
MAX_RESULT_WAIT_SECONDS = 10
MAX_RESULT_TEXT_CHARS = 2400
RUN_LABEL_PATTERN = re.compile(r"(?i)\b(run|test|execute|play)\b")
SUBMIT_LABEL_PATTERN = re.compile(r"(?i)\b(submit|send|publish)\b")
RESULT_SELECTOR = (
    "[role='status'], [role='alert'], [aria-live], output, "
    "[data-testid*='result' i], [data-test*='result' i], [data-cy*='result' i], "
    "[id*='result' i], [class*='result' i], [id*='output' i], [class*='output' i], "
    "[id*='console' i], [class*='console' i]"
)


class OnlineExecutionBudget:
    def __init__(
        self,
        max_executions=MAX_ONLINE_EXECUTIONS,
        max_limit=MAX_ONLINE_EXECUTIONS,
    ):
        hard_limit = max(0, min(int(max_limit), MAX_ONLINE_TASK_EXECUTIONS))
        self.max_executions = max(0, min(int(max_executions), hard_limit))
        self.executions = 0

    def consume(self):
        if self.executions >= self.max_executions:
            return False
        self.executions += 1
        return True


def _css_string(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _unique_control_selector(page, selectors):
    for selector in selectors:
        try:
            if page.locator(selector).count() == 1:
                return selector
        except Exception:
            continue
    return None


def _control_details(locator):
    return locator.evaluate(
        """element => {
            const tag = element.tagName.toLowerCase();
            const text = (element.innerText || element.value || "").trim();
            const data = {};
            for (const attribute of Array.from(element.attributes || [])) {
                if (attribute.name.startsWith("data-") && /test|run|action|control/i.test(attribute.name)) {
                    data[attribute.name] = attribute.value;
                }
            }
            return {
                tag,
                text,
                aria_label: element.getAttribute("aria-label") || "",
                title: element.getAttribute("title") || "",
                role: element.getAttribute("role") || "",
                id: element.id || "",
                type: element.getAttribute("type") || "",
                submits_form: (element.tagName.toLowerCase() === "input" && element.type === "submit") ||
                    (element.tagName.toLowerCase() === "button" && element.type === "submit" && Boolean(element.form)),
                data
            };
        }"""
    )


def _control_selector(page, locator, details):
    selectors = []
    if details["id"]:
        selectors.append(f"[id={_css_string(details['id'])}]")
    for name, value in details["data"].items():
        selectors.append(f"[{name}={_css_string(value)}]")
    if details["aria_label"]:
        selectors.append(f"[aria-label={_css_string(details['aria_label'])}]")
    if details["title"]:
        selectors.append(f"[title={_css_string(details['title'])}]")
    selector = _unique_control_selector(page, selectors)
    if selector:
        return selector

    accessible_name = details["aria_label"] or details["title"] or details["text"]
    if accessible_name:
        role = details["role"] or ("button" if details["tag"] == "button" else "button")
        try:
            role_locator = page.get_by_role(role, name=accessible_name, exact=True)
            if role_locator.count() == 1:
                return f"role={role}[name={_css_string(accessible_name)}]"
        except Exception:
            pass

    tag = details["tag"]
    base = "button, [role='button'], input[type='button'], input[type='submit']"
    if tag == "button":
        selectors = ["button"]
    elif details["role"] == "button":
        selectors = ["[role='button']"]
    elif tag == "input" and details["type"] in {"button", "submit"}:
        selectors = [f"input[type='{details['type']}']"]
    else:
        selectors = [base]
    if _unique_control_selector(page, selectors):
        return selectors[0]
    try:
        return f"{base} >> nth={locator.evaluate('(element) => Array.from(document.querySelectorAll(\"button, [role=\\\"button\\\"], input[type=\\\"button\\\"], input[type=\\\"submit\\\"]\")).indexOf(element)')}"
    except Exception:
        return None


def find_run_control(page):
    controls = page.locator("button, [role='button'], input[type='button'], input[type='submit']")
    candidates = []
    for index in range(controls.count()):
        locator = controls.nth(index)
        try:
            if not locator.is_visible() or not locator.is_enabled():
                continue
            details = _control_details(locator)
        except Exception:
            continue

        if details["submits_form"]:
            continue
        label = " ".join((details["text"], details["aria_label"], details["title"], *details["data"].values())).strip()
        if SUBMIT_LABEL_PATTERN.search(label) or not RUN_LABEL_PATTERN.search(label):
            continue
        selector = _control_selector(page, locator, details)
        if not selector:
            continue
        candidates.append({
            "found": True,
            "selector": selector,
            "visible_text": details["text"] or details["aria_label"] or details["title"],
            "accessible_name": details["aria_label"] or details["title"] or details["text"],
            "confidence": 0.98 if details["id"] or details["data"] else 0.85,
        })

    if not candidates:
        return {"found": False, "reason": "No visible Run/Test/Execute/Play control was found."}
    if len(candidates) > 1:
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        if candidates[0]["confidence"] == candidates[1]["confidence"]:
            return {"found": False, "reason": "Multiple equally likely Run/Test controls were found; refusing to guess."}
    return candidates[0]


def _normalize_code(value):
    return str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")


def _result_snapshot(page):
    messages = []
    candidates = page.locator(RESULT_SELECTOR)
    seen = set()
    for index in range(min(candidates.count(), 20)):
        locator = candidates.nth(index)
        try:
            if not locator.is_visible():
                continue
            text = locator.inner_text(timeout=300).strip()
            if text and text not in seen:
                messages.append(text[:MAX_RESULT_TEXT_CHARS])
                seen.add(text)
        except Exception:
            continue
    try:
        body_text = page.locator("body").inner_text(timeout=1000).strip()[:4000]
    except Exception:
        body_text = ""
    try:
        title = page.title()[:200]
    except Exception:
        title = ""
    return {
        "title": title,
        "result_text": "\n".join(messages)[:MAX_RESULT_TEXT_CHARS],
        "body_text": body_text,
        "result_container_found": bool(messages),
        "result_container_selector": RESULT_SELECTOR,
    }


def _access_block_status(blocker):
    if blocker.startswith(("The page requires login.", "The page contains a password field.")):
        return "LOGIN_REQUIRED"
    return "blocked"


def classify_online_result(observation):
    result_text = str(observation.get("result_text", "")).strip()
    body_text = str(observation.get("body_text", "")).strip()
    text = result_text or body_text
    login_required = re.search(
        r"(?i)\b(?:log\s*in|sign\s*in|login|sign-in)\b.{0,100}\b(?:to|before)\s+(?:run|submit)\b|"
        r"\b(?:run|submit)\b.{0,100}\b(?:log\s*in|sign\s*in|login|sign-in)\b",
        body_text,
    )
    if login_required:
        return {
            "status": "login_required",
            "summary": "LeetCode requires login before code can be run or submitted.",
            "output": result_text[:MAX_RESULT_TEXT_CHARS],
            "error": "",
            "verified": False,
        }

    patterns = (
        ("memory_limit_exceeded", re.compile(r"(?i)\b(memory limit exceeded|out of memory|memory limit)\b")),
        ("timeout", re.compile(r"(?i)\b(time limit exceeded|timed?\s*out|timeout|time limit)\b")),
        ("compilation_error", re.compile(r"(?i)\b(compilation error|compile error|compilation failed|compiler error)\b")),
        ("runtime_error", re.compile(r"(?i)\b(runtime\s*error|runtime exception|exception occurred|traceback|execution error)\b")),
        ("wrong_answer", re.compile(r"(?i)\b(wrong answer|incorrect answer|expected\b.{0,80}\b(received|got|actual)\b)")),
        ("failed", re.compile(r"(?i)\b(failed|failure|test failures?|not accepted|rejected)\b")),
    )
    for status, pattern in patterns:
        match = pattern.search(text)
        if match:
            summary = text[max(0, match.start() - 80):min(len(text), match.end() + 160)].strip()
            error = summary if status in {"timeout", "compilation_error", "runtime_error", "wrong_answer", "failed"} else ""
            return {
                "status": status,
                "summary": summary or status.replace("_", " ").title(),
                "output": result_text[:MAX_RESULT_TEXT_CHARS],
                "error": error,
            }
    passed_pattern = re.compile(
        r"(?i)\b(accepted|all tests passed|all test cases passed|tests? passed|passed successfully)\b"
    )
    match = passed_pattern.search(result_text)
    if match:
        summary = result_text[max(0, match.start() - 80):min(len(result_text), match.end() + 160)].strip()
        return {
            "status": "passed",
            "summary": summary or "passed",
            "output": result_text[:MAX_RESULT_TEXT_CHARS],
            "error": "",
        }
    return {
        "status": "unknown",
        "summary": "Online result could not be verified.",
        "output": result_text[:MAX_RESULT_TEXT_CHARS],
        "error": "",
    }


def read_online_test_result(page, previous_observation=None, timeout_seconds=MAX_RESULT_WAIT_SECONDS):
    baseline = (previous_observation or {}).get("result_text", "")
    deadline = time.monotonic() + max(0, timeout_seconds)
    latest = _result_snapshot(page)
    initial_snapshot_text = (latest["result_text"], latest["body_text"])
    polling_attempts = 1
    snapshot_text_changed = False
    while True:
        classification = classify_online_result(latest)
        has_fresh_result = bool(latest["result_text"]) and latest["result_text"] != baseline
        if classification["status"] == "login_required" or (
            classification["status"] != "unknown" and (has_fresh_result or not previous_observation)
        ):
            classification["observation"] = latest
            classification["polling_attempts"] = polling_attempts
            classification["snapshot_text_changed"] = snapshot_text_changed
            return classification
        if time.monotonic() >= deadline:
            classification = classify_online_result(latest)
            classification["status"] = "unknown"
            classification["summary"] = "Online result could not be verified."
            classification["observation"] = latest
            classification["polling_attempts"] = polling_attempts
            classification["snapshot_text_changed"] = snapshot_text_changed
            return classification
        page.wait_for_timeout(200)
        latest = _result_snapshot(page)
        polling_attempts += 1
        snapshot_text_changed = snapshot_text_changed or (
            (latest["result_text"], latest["body_text"]) != initial_snapshot_text
        )


def run_online_code(
    page,
    selector,
    editor_selector,
    editor_type,
    intended_code,
    run_control,
    execution_budget,
):
    if not isinstance(intended_code, str) or not intended_code:
        return {"executed": False, "reason": "Intended code is empty or invalid."}
    try:
        editor = page.locator(editor_selector)
        if editor.count() != 1 or not editor.is_visible():
            return {"executed": False, "reason": "The intended editor is no longer uniquely visible."}
        editor_value = _read_editor_value(editor, editor_type)
        if _normalize_code(editor_value) != _normalize_code(intended_code):
            return {"executed": False, "reason": "Editor contents do not match the intended code."}

        if not run_control or not run_control.get("found") or selector != run_control.get("selector"):
            return {"executed": False, "reason": "Run control was not verified from current page controls."}
        candidates = page.locator("button, [role='button'], input[type='button'], input[type='submit']")
        matching = []
        for index in range(candidates.count()):
            candidate = candidates.nth(index)
            try:
                if not candidate.is_visible() or not candidate.is_enabled():
                    continue
                details = _control_details(candidate)
            except Exception:
                continue
            if details["submits_form"]:
                continue
            label = " ".join((details["text"], details["aria_label"], details["title"], *details["data"].values())).strip()
            if SUBMIT_LABEL_PATTERN.search(label):
                continue
            if RUN_LABEL_PATTERN.search(label):
                matching.append((candidate, details, label))
        selected = None
        for candidate, details, label in matching:
            if _control_selector(page, candidate, details) == selector:
                selected = (candidate, details, label)
                break
        if selected is None:
            return {"executed": False, "reason": "Selected control is no longer an eligible Run/Test control."}
        if not execution_budget.consume():
            return {"executed": False, "reason": "Online execution limit reached; repeated Run clicks are blocked."}

        previous_observation = _result_snapshot(page)
        visible_text = selected[1]["text"] or selected[1]["aria_label"] or selected[1]["title"]
        print(f"\n[REAL_ONLINE_RUN] Clicking Run control selector={selector!r}, text={visible_text!r}")
        selected[0].click(timeout=5000)
        return {
            "executed": True,
            "selector": selector,
            "visible_text": visible_text,
            "previous_observation": previous_observation,
        }
    except Exception as error:
        return {
            "executed": False,
            "reason": f"Could not safely run the online code: {type(error).__name__}: {error}",
        }


def real_online_run_workflow(
    url,
    ask_model,
    headless=False,
    keep_open=True,
    inspection_callback=None,
    max_executions=MAX_ONLINE_EXECUTIONS,
    result_wait_seconds=MAX_RESULT_WAIT_SECONDS,
):
    _validate_url(url)
    result = None
    run_was_clicked = False
    budget = OnlineExecutionBudget(max_executions)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            opened = browser_open(page, url)
            _wait_for_page(page)
            observation = browser_read(page)
            try:
                full_body_text = page.locator("body").inner_text(timeout=1000).strip()
            except Exception:
                full_body_text = observation.get("body_text", "")
            access_observation = {**observation, "body_text": full_body_text}
            blocker = access_block_reason(access_observation)
            if blocker:
                result = {
                    "executed": False,
                    "status": _access_block_status(blocker),
                    "summary": blocker,
                    "error": blocker,
                }
            elif not observation.get("success"):
                result = {"executed": False, "status": "read_failed", "error": observation.get("error", "Could not read the page.")}
            else:
                problem = _extract_problem(_problem_observation(observation, page), ask_model)
                if not problem["appears_coding_problem"] or not problem["problem_statement"]:
                    result = {
                        "executed": False,
                        "status": "insufficient_information",
                        "problem": problem,
                        "missing_information": problem["missing_information"],
                    }
                else:
                    solution = _generate_solution(problem, ask_model)
                    editor = find_code_editor(page)
                    if not editor["found"]:
                        result = {"executed": False, "status": "editor_not_found", "editor": editor}
                    elif not editor.get("supported", True):
                        result = {"executed": False, "status": "editor_unsupported", "editor": editor}
                    else:
                        entry = enter_code(page, editor["selector"], solution["code"], editor["editor_type"])
                        if not entry["success"]:
                            result = {"executed": False, "status": "entry_failed", "entry": entry}
                        else:
                            control = find_run_control(page)
                            if not control["found"]:
                                result = {"executed": False, "status": "run_control_not_found", "run_control": control}
                            else:
                                action = run_online_code(
                                    page,
                                    control["selector"],
                                    editor["selector"],
                                    editor["editor_type"],
                                    solution["code"],
                                    control,
                                    budget,
                                )
                                if not action["executed"]:
                                    result = {
                                        "executed": False,
                                        "status": "run_not_executed",
                                        "error": action["reason"],
                                    }
                                else:
                                    run_was_clicked = True
                                    try:
                                        classification = read_online_test_result(
                                            page,
                                            previous_observation=action["previous_observation"],
                                            timeout_seconds=result_wait_seconds,
                                        )
                                    except Exception as error:
                                        classification = {
                                            "status": "unknown",
                                            "summary": "Online result could not be verified.",
                                            "output": "",
                                            "error": f"Result reading failed: {type(error).__name__}: {error}",
                                        }
                                    snapshot = classification.get("observation", {})
                                    safe_result_text = safe_observation({
                                        "body_text": snapshot.get("result_text", ""),
                                    })["body_text"]
                                    safe_body_text = safe_observation({
                                        "body_text": snapshot.get("body_text", ""),
                                    })["body_text"][:5000]
                                    print("\n[REAL_ONLINE_RUN] Result diagnostics")
                                    print("result_text repr:", repr(safe_result_text))
                                    print("body_text repr:", repr(safe_body_text))
                                    print("current URL:", page.url)
                                    print("result container found:", snapshot.get("result_container_found", False))
                                    print("result container selector:", snapshot.get("result_container_selector", RESULT_SELECTOR))
                                    print("polling attempts:", classification.get("polling_attempts", "unavailable"))
                                    print("snapshot text changed during polling:", classification.get("snapshot_text_changed", "unavailable"))
                                    result = {
                                        "executed": True,
                                        **classification,
                                        "selector": action["selector"],
                                        "run_clicked": True,
                                        "submit_clicked": False,
                                        "execution_count": budget.executions,
                                        "problem": problem,
                                        "solution": solution,
                                        "editor": editor,
                                        "entry": entry,
                                    }
        except Exception as error:
            result = {
                "executed": False,
                "status": "workflow_error",
                "summary": f"Online result could not be verified: {type(error).__name__}: {error}",
                "run_clicked": run_was_clicked,
                "submit_clicked": False,
            }

        result.setdefault("run_clicked", run_was_clicked)
        result.setdefault("submit_clicked", False)
        print(f"\nMODE: {REAL_ONLINE_RUN_MODE}")
        print(f"Result: {result.get('status', 'unknown')} - {result.get('summary', result.get('error', ''))}")
        print("Submit was not clicked. Workflow stopped after reading the Run result.")
        if keep_open:
            if inspection_callback:
                inspection_callback(page, result)
            else:
                input("\nBrowser is open for inspection. Press Enter to close it: ")
        browser.close()
    return result