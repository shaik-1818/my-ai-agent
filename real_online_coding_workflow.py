from playwright.sync_api import sync_playwright

from browser_tools import browser_open, browser_read
from coding_workflow import create_solution_plan, generate_solution
from online_coding_workflow import (
    ONLINE_CODING_MODE,
    _online_requirements,
    access_block_reason,
    read_coding_problem,
)
from real_online_correction_workflow import (
    MAX_ONLINE_CORRECTIONS,
    online_correction_loop,
)
from real_online_editor_workflow import (
    _problem_observation,
    _read_editor_value,
    _wait_for_page,
    enter_code,
    find_code_editor,
)
from real_online_read_workflow import _validate_url
from real_online_run_workflow import (
    MAX_ONLINE_TASK_EXECUTIONS,
    OnlineExecutionBudget,
    find_run_control,
)
from safe_code_runner import validate_solution


MAX_ONLINE_CORRECTION_EXECUTIONS = min(
    MAX_ONLINE_CORRECTIONS + 1,
    MAX_ONLINE_TASK_EXECUTIONS,
)


def real_online_coding_workflow(
    url,
    ask_model,
    headless=False,
    keep_open=True,
    inspection_callback=None,
    max_corrections=MAX_ONLINE_CORRECTIONS,
    result_wait_seconds=10,
    event_callback=None,
):
    _validate_url(url)
    states = []

    def transition(state):
        states.append(state)
        print(f"[STATE] {state}")
        event_name = {
            "OPENING": "browser_opening",
            "READING_PROBLEM": "reading_problem",
            "PLANNING": "planning",
            "GENERATING_CODE": "generating_code",
            "FINDING_EDITOR": "finding_editor",
            "ENTERING_CODE": "entering_code",
            "VERIFYING_CODE": "verifying_code",
            "FINDING_RUN_CONTROL": "finding_run_control",
            "RUNNING": "running",
            "READING_RESULT": "reading_result",
            "CORRECTING": "correcting",
            "VERIFIED": "verified",
            "READY_FOR_SUBMISSION": "ready_for_submission",
            "BLOCKED": "blocked",
            "LOGIN_REQUIRED": "blocked",
            "FAILED": "failed",
            "MAX_CORRECTIONS_REACHED": "failed",
        }.get(state)
        if event_callback and event_name:
            event_callback(event_name)

    def stop(state, summary, **details):
        transition(state)
        return {
            "status": state,
            "state": state,
            "verified": False,
            "ready_for_submission": False,
            "summary": summary,
            **details,
        }

    result = None
    run_clicked = False
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            transition("OPENING")
            opened = browser_open(page, url)
            if not opened.get("success"):
                result = stop("FAILED", opened.get("error", "The problem page could not be opened."))
            else:
                if event_callback:
                    event_callback("browser_opened")
                _wait_for_page(page)
                observation = browser_read(page)
                if not observation.get("success"):
                    result = stop("FAILED", observation.get("error", "The problem page could not be read."))
                else:
                    blocker = access_block_reason(observation)
                    if blocker:
                        result = stop("BLOCKED", blocker)
                    else:
                        transition("READING_PROBLEM")
                        details = read_coding_problem(_problem_observation(observation, page), ask_model)
                        if event_callback:
                            event_callback("problem_read")
                        problem = details["problem_statement"]
                        if not problem.strip():
                            result = stop(
                                "FAILED",
                                "The coding problem statement could not be extracted.",
                                problem=details,
                            )
                        else:
                            requirements = _online_requirements(details)
                            transition("PLANNING")
                            plan = create_solution_plan(
                                problem,
                                requirements,
                                ask_model,
                                language=details["language"],
                                online=True,
                            )
                            transition("GENERATING_CODE")
                            code = generate_solution(
                                problem,
                                requirements,
                                plan,
                                ask_model,
                                language=details["language"],
                                online=True,
                            )
                            validation = validate_solution(code)
                            if not validation["success"]:
                                result = stop(
                                    "FAILED",
                                    f"Generated code failed static safety validation: {validation['error']}",
                                    problem=details,
                                    plan=plan,
                                )
                            else:
                                transition("FINDING_EDITOR")
                                editor = find_code_editor(page)
                                if not editor.get("found"):
                                    result = stop(
                                        "FAILED",
                                        editor.get("reason", "Code editor was not found."),
                                        problem=details,
                                        plan=plan,
                                        editor=editor,
                                    )
                                elif not editor.get("supported", True):
                                    result = stop(
                                        "FAILED",
                                        editor.get("reason", "Code editor is unsupported."),
                                        problem=details,
                                        plan=plan,
                                        editor=editor,
                                    )
                                else:
                                    transition("ENTERING_CODE")
                                    entered = enter_code(page, editor["selector"], code, editor["editor_type"])
                                    if not entered.get("success"):
                                        result = stop(
                                            "FAILED",
                                            entered.get("reason", "Could not enter generated code."),
                                            problem=details,
                                            plan=plan,
                                            editor=editor,
                                            entry=entered,
                                        )
                                    else:
                                        transition("VERIFYING_CODE")
                                        try:
                                            readback = _read_editor_value(
                                                page.locator(editor["selector"]),
                                                editor["editor_type"],
                                            )
                                        except Exception as error:
                                            readback = None
                                            readback_error = f"{type(error).__name__}: {error}"
                                        else:
                                            readback_error = None
                                        normalize = lambda value: str(value).replace("\r\n", "\n").replace("\u00a0", " ")
                                        if readback is None or normalize(readback) != normalize(code):
                                            result = stop(
                                                "FAILED",
                                                readback_error or "Editor contents did not match generated code.",
                                                problem=details,
                                                plan=plan,
                                                editor=editor,
                                            )
                                        else:
                                            transition("FINDING_RUN_CONTROL")
                                            run_control = find_run_control(page)
                                            if not run_control.get("found"):
                                                result = stop(
                                                    "FAILED",
                                                    run_control.get("reason", "Run/Test control was not found."),
                                                    problem=details,
                                                    plan=plan,
                                                    editor=editor,
                                                    run_control=run_control,
                                                )
                                            else:
                                                budget = OnlineExecutionBudget(
                                                    max_executions=MAX_ONLINE_CORRECTION_EXECUTIONS,
                                                    max_limit=MAX_ONLINE_TASK_EXECUTIONS,
                                                )
                                                loop_result = online_correction_loop(
                                                    problem=problem,
                                                    current_code=code,
                                                    editor=editor,
                                                    page=page,
                                                    run_control=run_control,
                                                    ask_model=ask_model,
                                                    max_corrections=max_corrections,
                                                    execution_budget=budget,
                                                    result_wait_seconds=result_wait_seconds,
                                                    state_callback=transition,
                                                )
                                                run_clicked = loop_result.get("run_clicked", False)
                                                if (
                                                    loop_result.get("verified")
                                                    and loop_result.get("result", {}).get("status") == "passed"
                                                ):
                                                    transition("READY_FOR_SUBMISSION")
                                                    result = {
                                                        **loop_result,
                                                        "status": "READY_FOR_SUBMISSION",
                                                        "state": "READY_FOR_SUBMISSION",
                                                        "ready_for_submission": True,
                                                        "problem": details,
                                                        "plan": plan,
                                                        "initial_code": code,
                                                    }
                                                else:
                                                    final_state = loop_result.get("state", "FAILED")
                                                    if final_state not in {"FAILED", "BLOCKED", "LOGIN_REQUIRED", "MAX_CORRECTIONS_REACHED"}:
                                                        final_state = "FAILED"
                                                    if not states or states[-1] != final_state:
                                                        transition(final_state)
                                                    result = {
                                                        **loop_result,
                                                        "status": "FAILED" if final_state == "MAX_CORRECTIONS_REACHED" else final_state,
                                                        "state": final_state,
                                                        "ready_for_submission": False,
                                                        "problem": details,
                                                        "plan": plan,
                                                        "initial_code": code,
                                                    }
        except Exception as error:
            if not states or states[-1] != "FAILED":
                transition("FAILED")
            result = {
                "status": "FAILED",
                "state": "FAILED",
                "verified": False,
                "ready_for_submission": False,
                "summary": f"Online coding workflow stopped: {type(error).__name__}: {error}",
                "run_clicked": run_clicked,
                "submit_clicked": False,
            }

        result.setdefault("run_clicked", run_clicked)
        result.setdefault("submit_clicked", False)
        result["state_history"] = states
        print(f"\nMODE: {ONLINE_CODING_MODE}")
        print(f"Final state: {result['state']}")
        print(result.get("summary", ""))
        print("Submission was not attempted.")
        if keep_open:
            if inspection_callback:
                inspection_callback(page, result)
            else:
                input("\nBrowser is open for inspection. Press Enter to close it: ")
        browser.close()
    return result
