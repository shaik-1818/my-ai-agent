import json
import ipaddress
import re
from urllib.parse import urlsplit

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from browser_tools import browser_open, browser_read
from online_coding_workflow import access_block_reason, safe_observation


REAL_ONLINE_READ_MODE = "REAL_ONLINE_READ"
MAX_VISIBLE_TEXT_CHARS = 6000
MAX_MODEL_INPUT_CHARS = 9000
MAX_OBSERVATION_CHARS = MAX_MODEL_INPUT_CHARS - 1200
MAX_CONTROL_ITEMS = 3
MAX_FIELD_CHARS = 100
MAX_RENDERED_PROBLEM_CHARS = 12000
INFORMATION_FIELDS = (
    ("problem_title", "problem title"),
    ("problem_statement", "problem statement"),
    ("examples", "examples"),
    ("constraints", "constraints"),
    ("input_description", "input description"),
    ("output_description", "output description"),
    ("programming_language", "programming language"),
    ("editor_information", "editor information"),
)
SECURITY_CHALLENGE_PATTERNS = (
    ("captcha", re.compile(r"(?i)\bcaptcha\b")),
    ("captcha", re.compile(r"(?i)verify (that )?you are (a )?human")),
    ("captcha", re.compile(r"(?i)\brobot verification\b|\bverify you are not a robot\b")),
    ("security_challenge", re.compile(r"(?i)\bunusual traffic\b")),
    ("security_challenge", re.compile(r"(?i)automated requests? (are )?not allowed")),
    ("security_challenge", re.compile(r"(?i)\bsecurity (check|challenge)\b")),
    ("security_challenge", re.compile(r"(?i)anti[- ]bot")),
    ("access_denied", re.compile(r"(?i)\b(access denied|forbidden)\b")),
)
AUTH_REQUIRED_PATTERNS = (
    re.compile(r"(?i)\b(sign|log) in to continue\b"),
    re.compile(r"(?i)\b(authentication required|login required)\b"),
    re.compile(r"(?i)\bplease (sign|log) in\b"),
    re.compile(r"(?i)\b(sign|log) in to (view|access|continue)\b"),
)


def _validate_url(url):
    parsed = urlsplit(url or "")
    if parsed.scheme not in {"http", "https", "file"}:
        raise ValueError("Use an http, https, or local file URL.")
    if parsed.scheme in {"http", "https"} and not parsed.netloc:
        raise ValueError("The website URL is invalid.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")


def _validate_public_url(url):
    _validate_url(url)
    parsed = urlsplit(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Read-only real website mode requires a public http:// or https:// URL.")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise ValueError("Read-only real website mode does not open localhost or local-network URLs.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("Read-only real website mode requires a publicly routable host.")
    return hostname


def classify_public_navigation(requested_url, final_url, observation):
    requested = urlsplit(requested_url or "")
    final = urlsplit(final_url or "")
    requested_host = (requested.hostname or "").lower().rstrip(".")
    final_host = (final.hostname or "").lower().rstrip(".")
    redirected = requested_url != final_url
    result = {
        "requested_url": requested_url,
        "final_url": final_url,
        "redirected": redirected,
        "blocker": None,
    }

    visible_text = " ".join((
        str(observation.get("page_title") or observation.get("title") or ""),
        str(observation.get("body_text") or ""),
    ))
    for blocker, pattern in SECURITY_CHALLENGE_PATTERNS:
        if pattern.search(visible_text):
            return {
                **result,
                "status": "BLOCKED_SECURITY_CHALLENGE",
                "blocker": blocker,
            }

    access_reason = access_block_reason(observation)
    has_password_field = any(
        str(field.get("type", "")).lower() == "password"
        for field in observation.get("inputs", [])
    )
    auth_path = re.search(
        r"(?i)(?:^|/)(?:login|log-in|signin|sign-in|authenticate|authentication)(?:/|$)",
        final.path,
    )
    if (
        has_password_field
        or auth_path
        or any(pattern.search(visible_text) for pattern in AUTH_REQUIRED_PATTERNS)
    ):
        return {**result, "status": "AUTH_REQUIRED", "blocker": "login_required"}
    if access_reason:
        return {
            **result,
            "status": "BLOCKED_SECURITY_CHALLENGE",
            "blocker": "security_challenge",
        }

    requested_is_web = requested.scheme.lower() in {"http", "https"}
    final_is_web = final.scheme.lower() in {"http", "https"}
    same_host = final_host == requested_host
    accepted_subdomain = bool(
        requested_host
        and final_host.endswith("." + requested_host)
    )
    if not requested_is_web or not final_is_web or not (same_host or accepted_subdomain):
        return {
            **result,
            "status": "UNEXPECTED_NAVIGATION",
            "blocker": "cross_origin_navigation",
        }
    return {**result, "status": "CONTINUE"}


def _ask_json(ask_model, prompt, max_tokens):
    if len(prompt) > MAX_MODEL_INPUT_CHARS:
        raise ValueError(
            f"Qwen input exceeds the {MAX_MODEL_INPUT_CHARS}-character limit."
        )
    response = ask_model(prompt, max_tokens=max_tokens)
    try:
        result = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(f"Qwen returned invalid JSON: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("Qwen response must be a JSON object.")
    return result


def _rendered_problem_content(page, leetcode_page=False):
    return page.evaluate("""isLeetCodePage => {
        const visible = element => {
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden' &&
                rect.width > 0 && rect.height > 0;
        };
        const text = element => (element?.innerText || '').trim();
        const selectors = [
            '[data-track-load="description_content"]',
            '#description_content',
            '[class*="description"]',
            'article',
            'main'
        ];
        let description = '';
        let descriptionDetected = false;
        for (let index = 0; index < selectors.length; index += 1) {
            const selector = selectors[index];
            const element = Array.from(document.querySelectorAll(selector))
                .find(candidate => visible(candidate) && text(candidate).length > 30);
            if (element) {
                description = text(element);
                descriptionDetected = index < selectors.length - 1 || isLeetCodePage;
                break;
            }
        }
        const title = Array.from(document.querySelectorAll('h1'))
            .find(visible);
        const bodyText = text(document.body);
        if (!description && isLeetCodePage) {
            description = bodyText;
            descriptionDetected = true;
        }
        const controls = Array.from(document.querySelectorAll('button, [role="button"], select'))
            .filter(visible)
            .map(element => element.tagName === 'SELECT'
                ? (element.selectedOptions[0]?.textContent || element.value || '')
                : text(element))
            .filter(Boolean);
        const editor = Array.from(document.querySelectorAll(
            'textarea, [contenteditable="true"], [role="textbox"]'
        )).some(visible);
        return {
            title: (title ? text(title) : document.title).replace(/^\\s*\\d+\\.\\s*/, '').replace(/\\s*[-|]\\s*LeetCode.*$/i, ''),
            description,
            description_detected: descriptionDetected,
            body_text: bodyText.slice(0, 12000),
            controls,
            editor_detected: editor
        };
    }""", leetcode_page)


def _meaningful_statement(statement):
    return len(statement.strip()) >= 30 and len(statement.split()) >= 5


def _extract_leetcode_problem(rendered):
    description = str(rendered.get("description") or "").strip()
    if not description or not rendered.get("description_detected"):
        return None

    title = str(rendered.get("title") or "").strip()
    content = description
    if title:
        content = re.sub(
            rf"(?im)^\s*(?:\d+\.\s*)?{re.escape(title)}\s*$\n?",
            "",
            content,
            count=1,
        ).strip()

    section_markers = list(re.finditer(
        r"(?im)^\s*(Example(?:\s+\d+)?|Constraints)\s*:", content
    ))
    first_section = section_markers[0].start() if section_markers else len(content)
    statement = content[:first_section].strip()
    if not _meaningful_statement(statement):
        return None

    examples = []
    constraints = []
    for index, marker in enumerate(section_markers):
        end = section_markers[index + 1].start() if index + 1 < len(section_markers) else len(content)
        section = content[marker.start():end].strip()
        if marker.group(1).lower().startswith("example"):
            examples.append(section[:240])
        else:
            constraints.extend(
                line.strip().lstrip("-•* ")[:240]
                for line in re.sub(r"(?i)^\s*Constraints\s*:\s*", "", section).splitlines()
                if line.strip()
            )

    example_text = "\n".join(examples)
    input_matches = re.findall(r"(?im)^\s*Input\s*:\s*(.+)$", example_text)
    output_matches = re.findall(r"(?im)^\s*Output\s*:\s*(.+)$", example_text)
    controls = rendered.get("controls", [])
    language = next((
        str(control).strip()
        for control in controls
        if re.search(r"(?i)\b(Python3?|Java|C\+\+|JavaScript|TypeScript|C#|Go|Rust|Kotlin|Swift)\b", str(control))
    ), "")
    editor_detected = rendered.get("editor_detected") is True
    problem = {
        "appears_coding_problem": True,
        "problem_title": title,
        "problem_statement": statement[:2000],
        "examples": examples[:5],
        "constraints": constraints[:5],
        "input_description": "\n".join(input_matches)[:250],
        "output_description": "\n".join(output_matches)[:250],
        "programming_language": language[:200],
        "editor_information": "Code editor is visible." if editor_detected else "",
        "editor_detected": editor_detected,
        "confidence": 0.95,
    }
    return _add_problem_aliases(problem)


def _add_problem_aliases(problem):
    problem.update({
        "title": problem.get("problem_title", ""),
        "statement": problem.get("problem_statement", ""),
        "input": problem.get("input_description", ""),
        "output": problem.get("output_description", ""),
        "language": problem.get("programming_language", ""),
    })
    problem.setdefault("editor_detected", bool(problem.get("editor_information")))
    problem.setdefault("confidence", 0.0)
    return problem


def _bounded_elements(elements, include_value=False):
    fields = ("tag", "role", "type", "placeholder", "text")
    if include_value:
        fields += ("value",)
    bounded = []
    for element in elements[:MAX_CONTROL_ITEMS]:
        bounded.append({
            key: str(element.get(key, ""))[:MAX_FIELD_CHARS]
            for key in fields
            if element.get(key)
        })
    return bounded


def bounded_observation(observation):
    safe = safe_observation(observation)
    bounded = {
        "title": safe["title"][:MAX_FIELD_CHARS],
        "body_text": safe["body_text"][:MAX_VISIBLE_TEXT_CHARS],
        "editable_regions": _bounded_elements(safe["editable_regions"]),
        "selects": _bounded_elements(safe["selects"], include_value=True),
    }
    serialized = json.dumps(bounded, ensure_ascii=True)
    while len(serialized) > MAX_OBSERVATION_CHARS:
        if bounded["body_text"]:
            overflow = len(serialized) - MAX_OBSERVATION_CHARS
            bounded["body_text"] = bounded["body_text"][:max(0, len(bounded["body_text"]) - overflow)]
        elif bounded["editable_regions"]:
            bounded["editable_regions"].pop()
        elif bounded["selects"]:
            bounded["selects"].pop()
        else:
            bounded["title"] = bounded["title"][:max(0, len(bounded["title"]) - 100)]
        serialized = json.dumps(bounded, ensure_ascii=True)
    return bounded


def _extract_problem(observation, ask_model, rendered=None):
    deterministic = _extract_leetcode_problem(rendered or {})
    if deterministic:
        deterministic["missing_information"] = [
            label for key, label in INFORMATION_FIELDS if not deterministic[key]
        ]
        return deterministic

    rendered_context = {
        "title": str((rendered or {}).get("title") or "")[:200],
        "description": str((rendered or {}).get("description") or "")[:900],
        "controls": (rendered or {}).get("controls", [])[:8],
        "editor_detected": (rendered or {}).get("editor_detected") is True,
    }
    prompt = f"""
Extract information about a coding problem from this visible page observation.
Do not solve it. Use only text and controls actually shown. Never fill gaps by
guessing. Use empty strings or empty lists for information that is not visible.
Return one strict JSON object with these keys:
- appears_coding_problem: boolean
- problem_title: string
- problem_statement: string
- examples: list of strings
- constraints: list of strings
- input_description: string
- output_description: string
- programming_language: string
- editor_information: string

PAGE OBSERVATION:
{json.dumps(observation, ensure_ascii=True)}

RENDERED DOM DETAILS:
{json.dumps(rendered_context, ensure_ascii=True)}
"""
    while len(prompt) > MAX_MODEL_INPUT_CHARS and observation.get("body_text"):
        overflow = len(prompt) - MAX_MODEL_INPUT_CHARS
        observation["body_text"] = observation["body_text"][:-max(1, overflow)]
        prompt = prompt.rsplit("PAGE OBSERVATION:\n", 1)[0] + "PAGE OBSERVATION:\n" + json.dumps(observation, ensure_ascii=True) + "\n\nRENDERED DOM DETAILS:\n" + json.dumps(rendered_context, ensure_ascii=True) + "\n"
    extracted = _ask_json(ask_model, prompt, max_tokens=1400)
    title_value = extracted.get("problem_title") or extracted.get("title")
    statement_value = (
        extracted.get("problem_statement")
        or extracted.get("statement")
        or extracted.get("problem_understanding")
    )
    normalized = {
        "appears_coding_problem": extracted.get("appears_coding_problem") is True
        or bool(str(title_value or "").strip() and _meaningful_statement(str(statement_value or "")))
    }
    field_aliases = {
        "problem_title": ("problem_title", "title"),
        "problem_statement": ("problem_statement", "statement", "problem_understanding"),
        "input_description": ("input_description", "input"),
        "output_description": ("output_description", "output"),
        "programming_language": ("programming_language", "language"),
        "editor_information": ("editor_information",),
    }
    for key, _ in INFORMATION_FIELDS:
        value = next((extracted.get(alias) for alias in field_aliases.get(key, (key,)) if extracted.get(alias)), None)
        if key in {"examples", "constraints"}:
            if isinstance(value, str):
                value = [value]
            normalized[key] = [str(item)[:240] for item in value[:5] if item] if isinstance(value, list) else []
        else:
            field_limit = 2000 if key == "problem_statement" else 200 if key == "problem_title" else 250
            normalized[key] = str(value or "")[:field_limit]
    normalized["missing_information"] = [
        label
        for key, label in INFORMATION_FIELDS
        if not normalized[key]
    ]
    normalized["confidence"] = 0.0
    return _add_problem_aliases(normalized)


def _generate_solution(problem, ask_model):
    prompt = f"""
Understand the extracted coding problem, create a solution plan, and write code.
Do not claim that the code was executed or tested. Do not invent constraints or
an interface that is absent from the extracted information. Mention any
necessary assumptions in problem_understanding.

Return exactly one JSON object with these string keys:
- problem_understanding
- algorithm
- complexity
- code

EXTRACTED PROBLEM INFORMATION:
{json.dumps(problem, ensure_ascii=True)}
"""
    result = _ask_json(ask_model, prompt, max_tokens=2600)
    understanding = result.get("problem_understanding")
    if not isinstance(understanding, str) or not understanding.strip():
        title = result.get("title") or result.get("problem_title")
        statement = result.get("statement") or result.get("problem_statement")
        if (
            not isinstance(title, str)
            or not title.strip()
            or not isinstance(statement, str)
            or not _meaningful_statement(statement)
        ):
            raise ValueError(
                "Qwen response must include problem_understanding or a valid problem title and meaningful statement."
            )
        understanding_parts = [f"{title.strip()}: {statement.strip()}"]
        for label, aliases in (
            ("Examples", ("examples",)),
            ("Constraints", ("constraints",)),
            ("Input", ("input", "input_description")),
            ("Output", ("output", "output_description")),
        ):
            value = next((result.get(alias) for alias in aliases if result.get(alias)), None)
            if isinstance(value, list):
                value = "; ".join(str(item).strip() for item in value if str(item).strip())
            if isinstance(value, str) and value.strip():
                understanding_parts.append(f"{label}: {value.strip()}")
        understanding = "\n".join(understanding_parts)

    solution = {}
    for key in ("problem_understanding", "algorithm", "complexity", "code"):
        value = understanding if key == "problem_understanding" else result.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Qwen did not return a non-empty '{key}' field.")
        solution[key] = value.strip()[:12000]
    return solution


def _display_result(result, public_mode=False):
    if public_mode:
        stopped = (
            result.get("status") in {"blocked", "unexpected_navigation", "missing_content"}
            or not result.get("success")
        )
        print("\nREAL WEBSITE READ STOPPED" if stopped else "\nREAL WEBSITE READ COMPLETE")
        print("URL:", result.get("url", ""))
        if result.get("status") in {
            "blocked",
            "unexpected_navigation",
            "UNEXPECTED_NAVIGATION",
            "AUTH_REQUIRED",
            "BLOCKED_SECURITY_CHALLENGE",
        }:
            print("BLOCKER:", result.get("blocker", "Unexpected navigation detected."))
            print("STATUS:", result["status"])
            return
        if result.get("status") in {"missing_content", "MISSING_CONTENT"}:
            print("Could not extract: page content, problem title, statement, examples, constraints, input, output.")
            print("STATUS: INSUFFICIENT_INFORMATION")
            return
        if result.get("status") == "insufficient_information":
            print("PROBLEM:", result.get("problem", {}).get("problem_statement", ""))
            print("Could not extract:", ", ".join(result.get("missing_information", [])))
            print("STATUS: INSUFFICIENT_INFORMATION")
            return
        if not result.get("success"):
            error = result.get("error")
            if error:
                print("ERROR:", error)
            else:
                print("BLOCKER:", result.get("blocker") or "The page could not be read.")
            print("STATUS: STOPPED")
            return
        problem = result.get("problem", {})
        solution = result.get("solution", {})
        print("\nPROBLEM:")
        print("Title:", problem.get("problem_title", ""))
        print("Statement:", problem.get("problem_statement", ""))
        print("Examples:", json.dumps(problem.get("examples", []), ensure_ascii=True))
        print("Constraints:", json.dumps(problem.get("constraints", []), ensure_ascii=True))
        print("Input:", problem.get("input_description", ""))
        print("Output:", problem.get("output_description", ""))
        print("\nSTATUS:\nREADY_FOR_EDITOR")
        print("Read-only mode stopped. No editor, Run, or Submit controls were used.")
        return

    print(f"\nMODE: {REAL_ONLINE_READ_MODE}")
    if not result.get("success"):
        error = result.get("error")
        if error:
            print("ERROR:", error)
        else:
            print("READ STOPPED:", result.get("blocker") or "The page could not be read.")
        return
    if result["status"] == "blocked":
        print("READ STOPPED:", result["blocker"])
        return
    if result["status"] == "insufficient_information":
        print("INSUFFICIENT CODING-PROBLEM INFORMATION")
        print("Could not extract:", ", ".join(result["missing_information"]))
        return
    print("\nExtracted problem information:")
    print(json.dumps(result["problem"], indent=2, ensure_ascii=True))
    print("\nGenerated solution (not entered or executed):")
    print(json.dumps(result["solution"], indent=2, ensure_ascii=True))
    print("\nREAD-ONLY workflow stopped. No editor, Run, or Submit controls were used.")


def real_online_read_workflow(
    url,
    ask_model,
    headless=False,
    keep_open=True,
    inspection_callback=None,
    public_only=False,
):
    if public_only:
        _validate_public_url(url)
    if not public_only:
        _validate_url(url)
    result = None

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            opened = browser_open(page, url)
            try:
                page.wait_for_load_state("load", timeout=5000)
            except PlaywrightTimeoutError:
                pass
            try:
                page.wait_for_load_state("networkidle", timeout=1500)
            except PlaywrightTimeoutError:
                pass
            page.locator("body").wait_for(state="visible", timeout=5000)
            final_url = page.url
            final_host = (urlsplit(final_url).hostname or "").lower().rstrip(".")
            leetcode_page = final_host == "leetcode.com" or final_host.endswith(".leetcode.com")
            if leetcode_page:
                try:
                    page.wait_for_function("""() => {
                        const visible = element => {
                            const rect = element.getBoundingClientRect();
                            const style = window.getComputedStyle(element);
                            return rect.width > 0 && rect.height > 0 &&
                                style.display !== 'none' && style.visibility !== 'hidden';
                        };
                        const description = document.querySelector(
                            '[data-track-load="description_content"], #description_content, [class*="description"]'
                        );
                        const bodyText = (document.body.innerText || '').trim();
                        return (description && visible(description) && (description.innerText || '').trim().length > 30) ||
                            (document.querySelector('h1') && bodyText.length > 150 &&
                                /\\bExample(?:\\s+\\d+)?\\s*:|\\bConstraints\\s*:?/i.test(bodyText));
                    }""", timeout=10000)
                except PlaywrightTimeoutError:
                    pass
            observation = browser_read(page)
            if not observation.get("success"):
                result = {
                    "success": False,
                    "status": "read_failed",
                    "error": observation.get("error", "Could not read the page."),
                }
            else:
                navigation = (
                    classify_public_navigation(url, final_url, observation)
                    if public_only
                    else {"status": "CONTINUE"}
                )
                if navigation["status"] != "CONTINUE":
                    result = {
                        "success": False,
                        **navigation,
                        "url": final_url,
                    }
                else:
                    blocker = access_block_reason(observation)
                    if blocker and not public_only:
                        result = {
                            "success": False,
                            "status": "blocked",
                            "blocker": blocker,
                            "url": final_url,
                        }
                    elif public_only and not str(observation.get("body_text", "")).strip():
                        result = {
                            "success": False,
                            "status": "MISSING_CONTENT",
                            "url": final_url,
                            "requested_url": url,
                            "final_url": final_url,
                            "redirected": navigation.get("redirected", False),
                            "missing_information": [
                                "page content",
                                "problem title",
                                "problem statement",
                                "examples",
                                "constraints",
                                "input description",
                                "output description",
                            ],
                        }
                    else:
                        bounded = bounded_observation(observation)
                        rendered = _rendered_problem_content(page, leetcode_page=leetcode_page)
                        problem = _extract_problem(bounded, ask_model, rendered)
                        has_title = bool(problem["problem_title"].strip())
                        has_statement = _meaningful_statement(problem["problem_statement"])
                        has_supporting_content = bool(
                            problem["examples"] or problem["constraints"] or problem["problem_statement"].strip()
                        )
                        if not problem["appears_coding_problem"] or not has_title or not has_statement or not has_supporting_content:
                            missing_information = list(problem["missing_information"])
                            if not problem["appears_coding_problem"]:
                                missing_information.insert(
                                    0,
                                    "a page that appears to contain a coding problem",
                                )
                            if not missing_information:
                                missing_information.append("a usable coding problem statement")
                            if not has_title and "problem title" not in missing_information:
                                missing_information.insert(0, "problem title")
                            if not has_statement and "problem statement" not in missing_information:
                                missing_information.append("problem statement")
                            result = {
                                "success": False,
                                "status": "insufficient_information",
                                "url": final_url,
                                "requested_url": url,
                                "final_url": final_url,
                                "redirected": navigation.get("redirected", False),
                                "problem": problem,
                                "missing_information": missing_information,
                            }
                        else:
                            solution = (
                                {
                                    "problem_understanding": "",
                                    "algorithm": "",
                                    "complexity": "",
                                    "code": "",
                                }
                                if public_only
                                else _generate_solution(problem, ask_model)
                            )
                            result = {
                                "success": True,
                                "status": "READY_FOR_EDITOR" if public_only else "generated",
                                "url": final_url,
                                "requested_url": url if public_only else None,
                                "final_url": final_url if public_only else None,
                                "redirected": navigation.get("redirected", False) if public_only else None,
                                "blocker": None if public_only else None,
                                "problem": problem,
                                "solution": solution,
                            }
        except Exception as error:
            result = {
                "success": False,
                "status": "open_or_read_failed",
                "error": f"{type(error).__name__}: {error}",
            }

        if public_only:
            try:
                final_url = page.url or url
            except Exception:
                final_url = url
            result.setdefault("requested_url", url)
            result.setdefault("final_url", final_url)
            result.setdefault("url", final_url)
            result.setdefault("redirected", final_url != url)
            result.setdefault("blocker", None)
        _display_result(result, public_mode=public_only)
        if keep_open:
            wait_for_inspection = inspection_callback or input
            wait_for_inspection("\nBrowser is open for inspection. Press Enter to close it: ")
        browser.close()

    return result