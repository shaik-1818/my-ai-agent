import ast
import hashlib
import json
import re

from playwright.sync_api import sync_playwright

from browser_tools import browser_open, browser_read
from coding_workflow import PROBLEM_SOLVING_MODE
from online_coding_workflow import _redact_text, access_block_reason, safe_observation
from real_online_editor_workflow import (
    _problem_observation,
    _read_editor_value,
    _wait_for_page,
    enter_code,
    find_code_editor,
    MAX_CODE_LENGTH,
)
from real_online_read_workflow import (
    _extract_problem,
    _generate_solution,
    _validate_url,
)
from real_online_run_workflow import (
    MAX_ONLINE_TASK_EXECUTIONS,
    OnlineExecutionBudget,
    find_run_control,
    read_online_test_result,
    run_online_code,
)
from safe_code_runner import validate_solution


REAL_ONLINE_CORRECTION_MODE = "REAL_ONLINE_CORRECTION"
MAX_ONLINE_CORRECTIONS = 3
MAX_ONLINE_CORRECTION_EXECUTIONS = min(
    MAX_ONLINE_CORRECTIONS + 1,
    MAX_ONLINE_TASK_EXECUTIONS,
)
MAX_CORRECTION_RESPONSE_CHARS = 40000
FAILURE_STATUSES = {
    "wrong_answer",
    "failed",
    "runtime_error",
    "compilation_error",
    "timeout",
    "memory_limit_exceeded",
}
EXPECTED_PATTERN = re.compile(
    r"(?i)\bexpected\b\s*[:=]?\s*(.+?)(?=\s+\b(?:actual|received|got)\b|[\r\n]|$)"
)
ACTUAL_PATTERN = re.compile(r"(?i)\b(?:actual|received|got)\b\s*[:=]?\s*([^\s,;]+)")


def _code_fingerprint(code):
    normalized = _normalize_code(code)
    try:
        normalized = ast.dump(ast.parse(normalized), include_attributes=False)
    except (SyntaxError, ValueError, TypeError):
        normalized = normalized.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _failure_fingerprint(result):
    evidence = " ".join((
        str(result.get("status", "unknown")),
        str(result.get("summary", "")),
        str(result.get("error", "")),
        str(result.get("output", "")),
    ))
    normalized = " ".join(evidence.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_code(code):
    return str(code).replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")


def _extract_expected_actual(observation):
    text = "\n".join((observation.get("result_text", ""), observation.get("body_text", "")))
    expected_match = EXPECTED_PATTERN.search(text)
    actual_match = ACTUAL_PATTERN.search(text)
    return {
        "expected_result": _redact_text(expected_match.group(1)[:500]) if expected_match else "",
        "actual_result": _redact_text(actual_match.group(1)[:500]) if actual_match else "",
    }


def _failure_evidence(problem, code, result):
    observation = result.get("observation") or {}
    expected_actual = _extract_expected_actual(observation)
    safe_page = safe_observation({
        "page_title": observation.get("title", ""),
        "body_text": observation.get("body_text", ""),
    })
    return {
        "problem": problem,
        "current_code": code,
        "status": result.get("status", "unknown"),
        "summary": _redact_text(result.get("summary", "")),
        "expected_result": expected_actual["expected_result"],
        "actual_result": expected_actual["actual_result"],
        "error_output": _redact_text(result.get("error", ""))[:1600],
        "output": _redact_text(result.get("output", ""))[:2400],
        "page_observation": safe_page,
    }


def _correction_prompt(original_problem, evidence):
    return f"""
Diagnose this actual online website test failure and return corrected code.
The original problem statement below is authoritative and must not be changed.
Use the page's actual result evidence; do not claim tests passed or invent a
result. Do not omit the required solution behavior to silence the failure.

Return exactly one JSON object, with no markdown or prose outside JSON, using
these non-empty string keys:
- diagnosis
- changes
- code

The code value must be the complete corrected source code.

ORIGINAL PROBLEM (immutable):
{original_problem}

ACTUAL ONLINE FAILURE EVIDENCE:
{json.dumps(evidence, ensure_ascii=True)}
"""


def _request_corrected_code(original_problem, evidence, ask_model):
    prompt = _correction_prompt(original_problem, evidence)
    if len(prompt) > 30000:
        raise ValueError("Correction evidence exceeds the Qwen prompt limit.")
    response = ask_model(prompt, max_tokens=2600)
    if not isinstance(response, str) or len(response) > MAX_CORRECTION_RESPONSE_CHARS:
        raise ValueError("Qwen correction response is empty or exceeds the response limit.")
    try:
        result = json.loads(response.strip())
    except json.JSONDecodeError as error:
        raise ValueError("Qwen correction must be exactly one valid JSON object.") from error
    if not isinstance(result, dict):
        raise ValueError("Qwen correction must be a JSON object.")
    for key in ("diagnosis", "changes", "code"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"Qwen correction must contain a non-empty '{key}' string.")
    code = result["code"].strip()
    if len(code) > MAX_CODE_LENGTH:
        raise ValueError(f"Corrected code exceeds the {MAX_CODE_LENGTH}-character limit.")
    result["code"] = code
    validation = validate_solution(code, mode=PROBLEM_SOLVING_MODE)
    if not validation["success"]:
        raise ValueError(f"Corrected code failed static safety validation: {validation['error']}")
    return result


def _page_blocker(page):
    if page.is_closed():
        return "The website page was closed or became inaccessible."
    try:
        observation = browser_read(page)
    except Exception as error:
        return f"The website page became inaccessible: {type(error).__name__}: {error}"
    if not observation.get("success"):
        return observation.get("error", "The website page became inaccessible.")
    return access_block_reason(observation)


def _blocker_state(blocker):
    return "LOGIN_REQUIRED" if re.search(r"(?i)login|log in|sign in|authentication", blocker or "") else "BLOCKED"


def _not_verified(summary, **details):
    state = details.pop("state", "FAILED")
    state_callback = details.pop("state_callback", None)
    if state_callback:
        state_callback(state)
    return {
        "status": "NOT_VERIFIED",
        "state": state,
        "verified": False,
        "summary": summary,
        "run_clicked": details.pop("run_clicked", False),
        "submit_clicked": False,
        **details,
    }


def online_correction_loop(
    problem,
    current_code,
    editor,
    page,
    run_control,
    ask_model,
    max_corrections=MAX_ONLINE_CORRECTIONS,
    execution_budget=None,
    result_wait_seconds=10,
    state_callback=None,
):
    max_corrections = max(0, min(int(max_corrections), MAX_ONLINE_CORRECTIONS))
    budget = execution_budget or OnlineExecutionBudget(
        max_executions=MAX_ONLINE_CORRECTION_EXECUTIONS,
        max_limit=MAX_ONLINE_TASK_EXECUTIONS,
    )
    code = current_code
    seen_code = {_code_fingerprint(code)}
    corrections = 0
    history = []
    seen_failures = {}
    run_clicked = False
    transition = state_callback or (lambda _: None)

    for _ in range(max_corrections + 1):
        blocker = _page_blocker(page)
        if blocker:
            return _not_verified(f"Online execution stopped because access is blocked: {blocker}", state=_blocker_state(blocker), state_callback=state_callback, run_clicked=run_clicked, corrections=corrections, history=history)

        try:
            editor_locator = page.locator(editor["selector"])
            if editor_locator.count() != 1 or not editor_locator.is_visible():
                return _not_verified("The code editor is no longer uniquely visible.", run_clicked=run_clicked, corrections=corrections, history=history)
            editor_value = _read_editor_value(editor_locator, editor["editor_type"])
        except Exception as error:
            return _not_verified(f"Could not read the editor before online execution: {type(error).__name__}: {error}", run_clicked=run_clicked, corrections=corrections, history=history)
        if _normalize_code(editor_value) != _normalize_code(code):
            return _not_verified("Editor contents do not match the intended code; online execution stopped.", run_clicked=run_clicked, corrections=corrections, history=history)

        current_run_control = find_run_control(page)
        if not current_run_control.get("found"):
            return _not_verified(current_run_control.get("reason", "Run control is unavailable."), run_clicked=run_clicked, corrections=corrections, history=history)
        if run_control and current_run_control["selector"] != run_control.get("selector"):
            return _not_verified("The Run control changed; refusing to use a stale selector.", run_clicked=run_clicked, corrections=corrections, history=history)
        run_control = current_run_control

        transition("RUNNING")
        action = run_online_code(
            page,
            run_control["selector"],
            editor["selector"],
            editor["editor_type"],
            code,
            run_control,
            budget,
        )
        if not action.get("executed"):
            return _not_verified(action.get("reason", "Online execution did not occur."), run_clicked=run_clicked, corrections=corrections, history=history)
        run_clicked = True

        transition("READING_RESULT")
        try:
            result = read_online_test_result(
                page,
                previous_observation=action["previous_observation"],
                timeout_seconds=result_wait_seconds,
            )
        except Exception as error:
            return _not_verified(
                "Online result could not be verified.",
                run_clicked=run_clicked,
                corrections=corrections,
                history=history,
                result_error=f"{type(error).__name__}: {error}",
            )

        blocker = _page_blocker(page)
        if blocker:
            return _not_verified(f"Online result stopped because access is blocked: {blocker}", state=_blocker_state(blocker), state_callback=state_callback, run_clicked=run_clicked, corrections=corrections, history=history, latest_result=result)

        if result.get("status") == "login_required":
            return _not_verified(
                result.get("summary", "LeetCode requires login before code can be run or submitted."),
                state="LOGIN_REQUIRED",
                state_callback=state_callback,
                run_clicked=run_clicked,
                corrections=corrections,
                executions=budget.executions,
                latest_result=result,
                history=history,
            )

        history.append({
            "code": code,
            "status": result.get("status", "unknown"),
            "summary": result.get("summary", ""),
            "output": result.get("output", ""),
        })
        if result.get("status") == "passed":
            transition("VERIFIED")
            return {
                "status": "VERIFIED",
                "state": "VERIFIED",
                "verified": True,
                "summary": result.get("summary", "Website reported all tests passed."),
                "result": result,
                "code": code,
                "corrections": corrections,
                "executions": budget.executions,
                "history": history,
                "run_clicked": run_clicked,
                "submit_clicked": False,
            }
        if result.get("status") == "unknown":
            transition("FAILED")
            return _not_verified(
                "Online result could not be verified.",
                run_clicked=run_clicked,
                corrections=corrections,
                executions=budget.executions,
                result=result,
                history=history,
            )
        if result.get("status") not in FAILURE_STATUSES:
            return _not_verified("Online result could not be verified.", run_clicked=run_clicked, corrections=corrections, executions=budget.executions, result=result, history=history)

        if corrections >= max_corrections:
            return _not_verified(
                "Could not verify solution after maximum correction attempts.",
                state="MAX_CORRECTIONS_REACHED",
                state_callback=state_callback,
                run_clicked=run_clicked,
                corrections=corrections,
                executions=budget.executions,
                result=result,
                history=history,
                stop_reason="MAX_CORRECTIONS_REACHED",
            )

        failure_fingerprint = _failure_fingerprint(result)
        code_fingerprint = _code_fingerprint(code)
        previous_code_fingerprint = seen_failures.get(failure_fingerprint)
        if previous_code_fingerprint == code_fingerprint:
            transition("FAILED")
            return _not_verified(
                "The same failure repeated without a meaningful code change; stopping correction.",
                run_clicked=run_clicked,
                corrections=corrections,
                executions=budget.executions,
                result=result,
                history=history,
                stop_reason="NO_PROGRESS",
            )
        seen_failures[failure_fingerprint] = code_fingerprint

        evidence = _failure_evidence(problem, code, result)
        transition("CORRECTING")
        try:
            correction = _request_corrected_code(problem, evidence, ask_model)
        except Exception as error:
            return _not_verified(
                f"Corrected code was rejected: {type(error).__name__}: {error}",
                run_clicked=run_clicked,
                corrections=corrections,
                executions=budget.executions,
                result=result,
                failure_evidence=evidence,
                history=history,
            )

        corrections += 1
        corrected_code = correction["code"]
        fingerprint = _code_fingerprint(corrected_code)
        if fingerprint in seen_code:
            transition("FAILED")
            return _not_verified(
                "Qwen returned code with no meaningful change that was already run; stopping to prevent repeated execution.",
                run_clicked=run_clicked,
                corrections=corrections,
                executions=budget.executions,
                result=result,
                diagnosis=correction["diagnosis"],
                changes=correction["changes"],
                history=history,
                stop_reason="NO_PROGRESS",
            )
        seen_code.add(fingerprint)

        transition("ENTERING_CODE")
        entered = enter_code(page, editor["selector"], corrected_code, editor["editor_type"])
        if not entered.get("success"):
            return _not_verified(
                f"Corrected code could not be safely entered and verified: {entered.get('reason', 'unknown error')}",
                run_clicked=run_clicked,
                corrections=corrections,
                executions=budget.executions,
                result=result,
                history=history,
            )
        transition("VERIFYING_CODE")
        try:
            readback = _read_editor_value(page.locator(editor["selector"]), editor["editor_type"])
        except Exception as error:
            return _not_verified(
                f"Corrected editor contents could not be read back: {type(error).__name__}: {error}",
                run_clicked=run_clicked,
                corrections=corrections,
                executions=budget.executions,
                history=history,
            )
        if _normalize_code(readback) != _normalize_code(corrected_code):
            return _not_verified(
                "Corrected editor contents failed readback verification.",
                run_clicked=run_clicked,
                corrections=corrections,
                executions=budget.executions,
                history=history,
            )
        code = corrected_code
        transition("FINDING_RUN_CONTROL")

    transition("FAILED")
    return _not_verified(
        "Could not verify solution after maximum correction attempts.",
        run_clicked=run_clicked,
        corrections=corrections,
        executions=budget.executions,
        history=history,
    )


def real_online_correction_workflow(
    url,
    ask_model,
    headless=False,
    keep_open=True,
    inspection_callback=None,
    result_wait_seconds=10,
):
    _validate_url(url)
    result = None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            browser_open(page, url)
            _wait_for_page(page)
            observation = browser_read(page)
            blocker = access_block_reason(observation)
            if blocker:
                result = _not_verified(f"Online execution stopped because access is blocked: {blocker}")
            elif not observation.get("success"):
                result = _not_verified("The problem page could not be read.")
            else:
                problem = _extract_problem(_problem_observation(observation, page), ask_model)
                if not problem["appears_coding_problem"] or not problem["problem_statement"]:
                    result = _not_verified(
                        "The coding problem could not be extracted sufficiently.",
                        problem=problem,
                        missing_information=problem["missing_information"],
                    )
                else:
                    solution = _generate_solution(problem, ask_model)
                    initial_validation = validate_solution(solution["code"], mode=PROBLEM_SOLVING_MODE)
                    if not initial_validation["success"]:
                        result = _not_verified(
                            f"Generated code failed static safety validation: {initial_validation['error']}",
                            problem=problem,
                        )
                    else:
                        editor = find_code_editor(page)
                        if not editor.get("found"):
                            result = _not_verified(editor.get("reason", "Code editor was not found."), problem=problem)
                        elif not editor.get("supported", True):
                            result = _not_verified(editor.get("reason", "Code editor is unsupported."), problem=problem)
                        else:
                            entered = enter_code(page, editor["selector"], solution["code"], editor["editor_type"])
                            if not entered.get("success"):
                                result = _not_verified(entered.get("reason", "Initial code could not be entered."), problem=problem)
                            else:
                                control = find_run_control(page)
                                if not control.get("found"):
                                    result = _not_verified(control.get("reason", "Run control was not found."), problem=problem)
                                else:
                                    budget = OnlineExecutionBudget(
                                        max_executions=MAX_ONLINE_CORRECTION_EXECUTIONS,
                                        max_limit=MAX_ONLINE_TASK_EXECUTIONS,
                                    )
                                    result = online_correction_loop(
                                        problem=problem,
                                        current_code=solution["code"],
                                        editor=editor,
                                        page=page,
                                        run_control=control,
                                        ask_model=ask_model,
                                        execution_budget=budget,
                                        result_wait_seconds=result_wait_seconds,
                                    )
                                    result["problem"] = problem
                                    result["initial_code"] = solution["code"]
        except Exception as error:
            result = _not_verified(
                f"Online result could not be verified: {type(error).__name__}: {error}",
                run_clicked=False,
            )

        print(f"\nMODE: {REAL_ONLINE_CORRECTION_MODE}")
        print(f"Final status: {result.get('status', 'NOT_VERIFIED')}")
        print(result.get("summary", "Online result could not be verified."))
        print("Submit was not clicked.")
        if keep_open:
            if inspection_callback:
                inspection_callback(page, result)
            else:
                input("\nBrowser is open for inspection. Press Enter to close it: ")
        browser.close()
    return result