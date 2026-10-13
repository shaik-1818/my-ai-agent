import json
import re

from safe_code_runner import (
    MAX_CASES,
    MAX_TEST_EXECUTIONS,
    ExecutionBudget,
    run_exact_code,
    run_solution_tests,
    validate_solution,
)


MAX_CORRECTIONS = 3
EXACT_CODE_MODE = "exact_code"
PROBLEM_SOLVING_MODE = "problem_solving"

EXACT_CODE_MARKERS = (
    "execute this exact code",
    "run this exact code",
    "execute the following code",
    "execute following code",
    "run the following code",
    "run this python program",
    "run the python program",
    "execute this python program",
    "execute the python program",
    "execute this python code",
    "execute the python code",
    "run this program exactly",
    "execute this program",
    "do not modify the code",
    "do not modify code",
    "exact source code",
    "exact python code",
)


def detect_coding_mode(user_request):
    request = (user_request or "").lower()
    if any(marker in request for marker in EXACT_CODE_MARKERS):
        return EXACT_CODE_MODE
    return PROBLEM_SOLVING_MODE


def extract_exact_source(user_request):
    request = user_request or ""
    marker_positions = [
        request.lower().find(marker)
        for marker in EXACT_CODE_MARKERS
        if request.lower().find(marker) >= 0
    ]
    tail = request[min(marker_positions):] if marker_positions else request

    fenced_blocks = re.findall(
        r"```(?:python|py)?[ \t]*\r?\n?(.*?)```",
        tail,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_blocks:
        return fenced_blocks[0].strip("\r\n")

    inline_blocks = re.findall(r"`([^`\r\n]+)`", tail)
    if inline_blocks:
        return inline_blocks[-1].strip()

    colon_match = re.search(r":\s*([\s\S]+)", tail)
    if colon_match:
        source = colon_match.group(1).strip("\r\n ")
        source = re.split(
            r"(?im)^\s*(?:terminate it|terminate the process|stop it when|"
            r"kill it when|the process should stop|timeout should)",
            source,
            maxsplit=1,
        )[0].rstrip()
        return source or None

    return None


def _request_json(ask_model, prompt, max_tokens=1800):
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


def read_problem_and_identify_requirements(problem, ask_model):
    print("\n[1/8] Read problem and identify requirements")
    prompt = f"""
Read this programming problem. Do not write a solution yet.
The required public interface is a Python function named solve(input_data).
input_data and the returned value must be JSON-compatible. input_data can be
any JSON value, so do not assume a platform-specific class or judge.

Return only JSON with these keys:
- requirements: list of concrete behavioral requirements
- assumptions: list of assumptions
- edge_cases: list of important edge cases
- test_cases: list of objects with input and expected keys, using JSON values

PROBLEM:
{problem}
"""
    result = _request_json(ask_model, prompt)
    requirements = result.get("requirements")
    if isinstance(requirements, str):
        requirements = [requirements]
    if not isinstance(requirements, list) or not requirements:
        requirements = []
        for key in ("required_public_interface", "input_data", "returned_value"):
            value = result.get(key)
            if value:
                requirements.append(f"{key}: {value}")
        edge_cases = result.get("edge_cases", [])
        if isinstance(edge_cases, list):
            requirements.extend(f"Edge case: {item}" for item in edge_cases)
    if not requirements:
        raise ValueError("Qwen did not identify any concrete requirements.")
    result["requirements"] = requirements
    if not isinstance(result.get("test_cases"), list):
        result["test_cases"] = []
    return result


def create_solution_plan(problem, requirements, ask_model, language="Python", online=False):
    print("\n[2/8] Create solution plan")
    prompt = f"""
Create a short implementation plan for this coding problem in {language}.
Do not generate code yet. Follow the required function signature and input /
output contract from the problem statement. Consider listed edge cases.
Online platform: {online}

PROBLEM:
{problem}

REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=True)}

Return only JSON with keys: steps (list), algorithm, complexity.
"""
    result = _request_json(ask_model, prompt, max_tokens=1200)
    if not isinstance(result.get("steps"), list) or not result["steps"]:
        raise ValueError("Qwen did not return a usable solution plan.")
    return result


def generate_solution(problem, requirements, plan, ask_model, language="Python", online=False):
    print("\n[3/8] Generate code")
    prompt = f"""
Your only task is code generation. Do not return a plan, analysis, or steps.
Return exactly one JSON object with exactly one key named code. Its value
must be the complete Python source as a string.

Required output shape:
{{"code":"def solve(input_data):\\n    return sum(input_data)"}}

Target language: {language}
Online platform code: {online}
Follow the exact signature, platform language, and input/output format in the
problem statement. Do not add markdown fences or explanations to the code.
When online platform code is false, retain the local solve(input_data) and
safe-runner restrictions used by the coding workflow.

PROBLEM:
{problem}

REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=True)}

SOLUTION PLAN:
{json.dumps(plan, ensure_ascii=True)}
"""
    result = _request_json(ask_model, prompt, max_tokens=3500)
    code = result.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Qwen did not return source code in the 'code' field.")
    return code


def run_tests(code, test_cases, approved, execution_budget, rerun=False):
    stage = "[7/8] Run tests again after correction" if rerun else "[4/8] Run initial tests"
    print(f"\n{stage} in the guarded subprocess")
    return run_solution_tests(
        code,
        test_cases,
        approved=approved,
        execution_budget=execution_budget,
    )


def analyze_test_results(problem, requirements, code, test_result, ask_model):
    print("\n[5/8] Read and analyze test results")
    prompt = f"""
Analyze this candidate against the stated requirements and test results.
The tests and errors are evidence; do not claim tests passed if they did not.
Do not write corrected code in this step.

PROBLEM:
{problem}

REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=True)}

CANDIDATE CODE:
{code}

TEST RESULT:
{json.dumps(test_result, ensure_ascii=True)}

Return only JSON with keys:
- correction_needed: boolean
- findings: list of concise findings
- next_action: either "correct" or "verify"
"""
    result = _request_json(ask_model, prompt, max_tokens=1200)
    if not isinstance(result.get("findings"), list):
        result["findings"] = []
    result["correction_needed"] = bool(result.get("correction_needed"))
    return result


def correct_solution(
    problem,
    requirements,
    plan,
    code,
    test_result,
    analysis,
    ask_model,
    language="Python",
    online=False,
):
    print("\n[6/8] Correct code based on the test evidence")
    prompt = f"""
Correct the {language} candidate using the concrete test failures and findings.
Return only JSON with one key, code, containing the complete corrected source.
Follow the required signature and platform input/output contract. Online
platform code is {online}.
For local code, keep the solve(input_data) interface and execution restrictions.
Do not hide a failing case by hardcoding its expected output.

PROBLEM:
{problem}

REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=True)}

PLAN:
{json.dumps(plan, ensure_ascii=True)}

CURRENT CODE:
{code}

TEST RESULT:
{json.dumps(test_result, ensure_ascii=True)}

ANALYSIS:
{json.dumps(analysis, ensure_ascii=True)}
"""
    result = _request_json(ask_model, prompt, max_tokens=3500)
    corrected_code = result.get("code")
    if not isinstance(corrected_code, str) or not corrected_code.strip():
        raise ValueError("Qwen did not return corrected source code.")
    return corrected_code


def verify_solution(problem, requirements, code, test_result, ask_model):
    print("\n[8/8] Verify requirements and final test results")
    results = test_result.get("results", [])
    executed_cases = (
        test_result.get("executed")
        and test_result.get("exit_code") == 0
        and test_result.get("all_passed")
        and bool(results)
        and all(
            item.get("passed")
            and "actual" in item
            and "expected" in item
            and item["actual"] == item["expected"]
            for item in results
        )
    )
    if executed_cases:
        return {
            "verified": True,
            "summary": f"All {len(results)} test case(s) passed in the safe code runner.",
            "remaining_concerns": [],
        }
    return {
        "verified": False,
        "summary": "The safe code runner did not confirm that every actual result matched its expected result.",
        "remaining_concerns": ["The solution was not verified by successful execution."],
    }


def _validate_test_cases(test_cases):
    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("No test cases are available. Provide at least one input/expected pair.")
    if len(test_cases) > MAX_CASES:
        raise ValueError(f"Provide no more than {MAX_CASES} test cases.")
    for index, case in enumerate(test_cases, start=1):
        if not isinstance(case, dict) or "input" not in case or "expected" not in case:
            raise ValueError(f"Test case {index} must contain 'input' and 'expected'.")
    return test_cases


def run_exact_code_workflow(user_request, confirm_execution):
    print("\nMODE: EXACT_CODE")
    source = extract_exact_source(user_request)
    if not source:
        raise ValueError(
            "Exact-code mode was selected, but no source was found. "
            "Put the exact Python source in backticks or a fenced code block."
        )

    execution_budget = ExecutionBudget(1)
    policy = validate_solution(source, mode=EXACT_CODE_MODE)
    if policy["success"]:
        approved = bool(confirm_execution(
            "This will execute only the exact source below, without Qwen edits "
            "or generated tests. Execution uses the current Python interpreter, "
            "a restricted import/builtin policy, a temporary workspace, a "
            "5-second timeout, and no shell command. This is not an OS-level "
            f"sandbox.\n\nEXACT SOURCE:\n{source}"
        ))
        result = run_exact_code(
            source,
            approved=approved,
            execution_budget=execution_budget,
        )
    else:
        result = run_exact_code(
            source,
            approved=False,
            execution_budget=execution_budget,
        )

    return {
        "mode": EXACT_CODE_MODE,
        "source": source,
        "execution_result": result,
        "approval_required": True,
    }


def run_coding_workflow(problem, ask_model, confirm_execution, test_cases=None):
    print("\nMODE: PROBLEM_SOLVING")
    requirements = read_problem_and_identify_requirements(problem, ask_model)
    cases = test_cases if test_cases is not None else requirements.get("test_cases")
    cases = _validate_test_cases(cases)

    plan = create_solution_plan(problem, requirements, ask_model)
    code = generate_solution(problem, requirements, plan, ask_model)

    approved = False
    execution_declined = False
    execution_budget = ExecutionBudget(MAX_TEST_EXECUTIONS)
    policy = validate_solution(code)
    if policy["success"]:
        approved = bool(confirm_execution(
            "Review the generated code and test cases below. The code passed "
            "static restrictions and will run in a "
            "temporary directory using a restricted Python subprocess, safe "
            "built-ins/imports, the current Python interpreter, no shell "
            "command, a 5-second timeout, and at most "
            f"{MAX_TEST_EXECUTIONS} test executions. This is not an OS-level sandbox. "
            "Approve running these tests?\n\nCODE:\n"
            f"{code}\n\nTEST CASES:\n"
            f"{json.dumps(cases, indent=2, ensure_ascii=True)}"
        ))
        execution_declined = not approved
        result = run_tests(code, cases, approved=approved, execution_budget=execution_budget)
    else:
        result = run_tests(
            code,
            cases,
            approved=False,
            execution_budget=execution_budget,
        )

    analysis = analyze_test_results(problem, requirements["requirements"], code, result, ask_model)
    corrections = 0

    while not execution_declined and corrections < MAX_CORRECTIONS and (
        not result.get("all_passed") or analysis.get("correction_needed")
    ):
        code = correct_solution(
            problem,
            requirements["requirements"],
            plan,
            code,
            result,
            analysis,
            ask_model,
        )
        corrections += 1
        policy = validate_solution(code)
        if not policy["success"]:
            result = run_tests(
                code,
                cases,
                approved=False,
                execution_budget=execution_budget,
                rerun=True,
            )
            print("\n[7/8] Corrected source rejected by static restrictions; no execution attempted.")
        else:
            if not approved:
                approved = bool(confirm_execution(
                    "The corrected code passed static restrictions. Review it and "
                    "the test cases, then approve execution?\n\nCODE:\n"
                    f"{code}\n\nTEST CASES:\n"
                    f"{json.dumps(cases, indent=2, ensure_ascii=True)}"
                ))
                execution_declined = not approved
            result = run_tests(
                code,
                cases,
                approved=approved,
                execution_budget=execution_budget,
                rerun=True,
            )
        analysis = analyze_test_results(problem, requirements["requirements"], code, result, ask_model)

    if not approved:
        verification = {
            "verified": False,
            "summary": (
                "Execution was declined, so the solution was not run."
                if execution_declined
                else "No candidate passed static validation, so the solution was not run."
            ),
            "remaining_concerns": ["Tests were not executed."],
        }
    else:
        verification = verify_solution(problem, requirements["requirements"], code, result, ask_model)
        if not result.get("all_passed") and corrections >= MAX_CORRECTIONS:
            verification["summary"] = (
                f"Maximum of {MAX_CORRECTIONS} correction attempts reached; "
                "the solution could not be verified."
            )
            verification["remaining_concerns"] = [
                "The solution did not pass the safe runner after all correction attempts."
            ]

    return {
        "mode": PROBLEM_SOLVING_MODE,
        "requirements": requirements,
        "plan": plan,
        "code": code,
        "test_cases": cases,
        "test_result": result,
        "analysis": analysis,
        "corrections": corrections,
        "verification": verification,
        "verified": bool(result.get("all_passed")) and verification["verified"],
    }


def run_coding_task(user_request, ask_model, confirm_execution, test_cases=None):
    mode = detect_coding_mode(user_request)
    if mode == EXACT_CODE_MODE:
        return run_exact_code_workflow(user_request, confirm_execution)
    return run_coding_workflow(
        user_request,
        ask_model,
        confirm_execution,
        test_cases=test_cases,
    )
