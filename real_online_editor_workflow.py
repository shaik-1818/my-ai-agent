import json
import re
from urllib.parse import urlsplit

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from browser_tools import browser_open, browser_read
from online_coding_workflow import access_block_reason, safe_observation
from real_online_read_workflow import (
    _extract_problem,
    _generate_solution,
    _validate_url,
    bounded_observation,
)


REAL_ONLINE_EDITOR_MODE = "REAL_ONLINE_EDITOR"
MAX_CODE_LENGTH = 20000
EDITOR_TARGET_SELECTOR = (
    "textarea:not([disabled]):not([readonly]), "
    "input:not([type]):not([disabled]):not([readonly]), "
    "input[type='text']:not([disabled]):not([readonly]), "
    "[contenteditable='true'], [role='textbox']"
)
EDITOR_HINT_PATTERN = re.compile(
    r"(?i)\b(code|editor|solution|source|program|monaco|codemirror|ace)\b"
)
SENSITIVE_HINT_PATTERN = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_ -]?key|authorization|cookie|csrf|session)\b"
)
SEARCH_HINT_PATTERN = re.compile(r"(?i)\b(search|email|username|login|comment|chat)\b")
STABLE_DATA_ATTRIBUTES = (
    "data-testid",
    "data-test",
    "data-cy",
    "data-qa",
    "data-editor",
)
SECURITY_PAGE_PATTERNS = (
    re.compile(r"(?i)\bcaptcha\b"),
    re.compile(r"(?i)verify (that )?you are (a )?human|verify you are not a (?:robot|bot)"),
    re.compile(r"(?i)performing security verification|security verification required"),
    re.compile(r"(?i)security service.{0,120}malicious bots"),
    re.compile(r"(?i)unusual traffic|automated requests? (are )?not allowed"),
    re.compile(r"(?i)security (?:check|challenge)|anti[- ]bot"),
    re.compile(r"(?i)access denied|\bforbidden\b"),
)


def _css_string(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\a ") + '"'


def _unique_selector(page, selectors):
    for selector in selectors:
        try:
            if page.locator(selector).count() == 1:
                return selector
        except Exception:
            continue
    return None


def _selector_for_candidate(page, locator, metadata):
    selectors = []
    if metadata["id"]:
        selectors.append(f"[id={_css_string(metadata['id'])}]")

    for attribute in STABLE_DATA_ATTRIBUTES:
        value = metadata["data_attributes"].get(attribute)
        if value:
            selectors.append(f"[{attribute}={_css_string(value)}]")

    if metadata["aria_label"]:
        selectors.append(
            f"[aria-label={_css_string(metadata['aria_label'])}]"
        )
    if metadata["label_text"]:
        selectors.append(
            f"label:has-text({_css_string(metadata['label_text'])}) {metadata['tag']}"
        )
    if metadata["aria_labelledby"]:
        selectors.append(
            f"[aria-labelledby={_css_string(metadata['aria_labelledby'])}]"
        )
    if metadata["role"]:
        selectors.append(f"[role={_css_string(metadata['role'])}]")
    if metadata["placeholder"]:
        selectors.append(
            f"[placeholder={_css_string(metadata['placeholder'])}]"
        )
    if metadata["name"]:
        selectors.append(f"[name={_css_string(metadata['name'])}]")

    selector = _unique_selector(page, selectors)
    if selector:
        return selector

    tag = metadata["tag"]
    generic_selectors = {
        "textarea": "textarea:not([disabled]):not([readonly])",
        "input": (
            "input:not([type]):not([disabled]):not([readonly]), "
            "input[type='text']:not([disabled]):not([readonly])"
        ),
    }
    if metadata["contenteditable"]:
        generic_selectors[tag] = f"{tag}[contenteditable='true']"
    elif metadata["role"] == "textbox":
        generic_selectors[tag] = f"{tag}[role='textbox']"

    return _unique_selector(page, [generic_selectors.get(tag, tag)])


def _editor_metadata(locator):
    return locator.evaluate(
        """element => {
            const ancestors = [];
            let current = element;
            for (let depth = 0; current && depth < 4; depth += 1, current = current.parentElement) {
                ancestors.push(current);
            }
            const labels = element.labels ? Array.from(element.labels).map(label => label.innerText || label.textContent || "") : [];
            const ariaLabelledBy = element.getAttribute("aria-labelledby") || "";
            const labelledBy = ariaLabelledBy
                .split(/\\s+/).filter(Boolean)
                .map(id => document.getElementById(id))
                .filter(Boolean)
                .map(node => node.innerText || node.textContent || "");
            const dataAttributes = {};
            for (const attribute of Array.from(element.attributes || [])) {
                if (attribute.name.startsWith("data-")) dataAttributes[attribute.name] = attribute.value;
            }
            return {
                tag: element.tagName.toLowerCase(),
                id: element.id || "",
                name: element.getAttribute("name") || "",
                type: element.getAttribute("type") || "",
                role: element.getAttribute("role") || "",
                aria_label: element.getAttribute("aria-label") || "",
                aria_labelledby: ariaLabelledBy,
                placeholder: element.getAttribute("placeholder") || "",
                contenteditable: element.getAttribute("contenteditable") === "true",
                readonly: Boolean(element.readOnly),
                disabled: Boolean(element.disabled),
                classes: ancestors.map(node => typeof node.className === "string" ? node.className : "").join(" ").slice(0, 800),
                labels: labels.concat(labelledBy).join(" ").slice(0, 500),
                label_text: labels.join(" ").trim().slice(0, 200),
                nearby_text: (element.parentElement ? element.parentElement.innerText : "").slice(0, 500),
                data_attributes: dataAttributes
            };
        }"""
    )


def _editor_type(metadata):
    classes = metadata["classes"].lower()
    if "monaco-editor" in classes:
        return "monaco"
    if "codemirror" in classes or "cm-editor" in classes or "cm-content" in classes:
        return "codemirror"
    if "ace_editor" in classes or "ace_text-input" in classes:
        return "ace"
    if metadata["tag"] == "textarea":
        return "textarea"
    if metadata["tag"] == "input":
        return "text_input"
    if metadata["contenteditable"]:
        return "contenteditable"
    if metadata["role"] == "textbox":
        return "role_textbox"
    return "unknown"


def _inspect_page_access(observation):
    visible_text = " ".join((
        str(observation.get("page_title") or observation.get("title") or ""),
        str(observation.get("body_text") or ""),
    ))
    for pattern in SECURITY_PAGE_PATTERNS:
        match = pattern.search(visible_text)
        if match:
            return {
                "blocked": True,
                "page_type": "security_verification",
                "reason": f"Security or verification challenge detected: {match.group(0)}",
            }

    reason = access_block_reason(observation)
    if reason:
        login_required = re.search(r"(?i)login|log in|sign in|authentication|password", reason)
        return {
            "blocked": True,
            "page_type": "login_required" if login_required else "access_restricted",
            "reason": reason,
        }
    return {"blocked": False, "page_type": "normal_page", "reason": None}


def find_code_editor(page):
    try:
        candidates = page.locator(EDITOR_TARGET_SELECTOR)
        candidate_count = candidates.count()
    except Exception as error:
        return {"found": False, "reason": f"Could not inspect editor candidates: {error}"}

    scored = []
    saw_hidden_candidate = False
    saw_ambiguous_candidate = False
    for index in range(candidate_count):
        locator = candidates.nth(index)
        try:
            if not locator.is_visible():
                saw_hidden_candidate = True
                continue
            metadata = _editor_metadata(locator)
        except Exception:
            continue

        semantic_text = " ".join((
            metadata["id"],
            metadata["name"],
            metadata["type"],
            metadata["role"],
            metadata["aria_label"],
            metadata["placeholder"],
            metadata["classes"],
            metadata["labels"],
            metadata["nearby_text"],
        ))
        if SENSITIVE_HINT_PATTERN.search(semantic_text):
            continue
        has_editor_hint = bool(EDITOR_HINT_PATTERN.search(semantic_text))
        if metadata["tag"] == "input" and (
            not has_editor_hint or SEARCH_HINT_PATTERN.search(semantic_text)
        ):
            continue
        if SEARCH_HINT_PATTERN.search(semantic_text) and not has_editor_hint:
            continue

        editor_type = _editor_type(metadata)
        confidence = 0.5
        if has_editor_hint:
            confidence += 0.32
        if metadata["aria_label"] or metadata["labels"] or metadata["role"]:
            confidence += 0.1
        if editor_type in {"textarea", "contenteditable", "monaco", "codemirror", "ace"}:
            confidence += 0.08
        confidence = min(confidence, 0.99)

        selector = _selector_for_candidate(page, locator, metadata)
        if not selector:
            saw_ambiguous_candidate = True
            continue
        scored.append({
            "found": True,
            "selector": selector,
            "editor_type": editor_type,
            "confidence": round(confidence, 2),
            "supported": True,
        })

    if scored:
        scored.sort(key=lambda item: item["confidence"], reverse=True)
        if len(scored) > 1 and scored[0]["confidence"] == scored[1]["confidence"]:
            return {
                "found": False,
                "reason": "Multiple equally likely visible code editors were found; refusing to guess.",
            }
        return scored[0]
    for selector, editor_type in (
        (".monaco-editor", "monaco"),
        (".CodeMirror", "codemirror"),
        (".cm-editor", "codemirror"),
        (".ace_editor", "ace"),
    ):
        roots = page.locator(selector)
        visible_roots = [
            roots.nth(index)
            for index in range(roots.count())
            if roots.nth(index).is_visible()
        ]
        if len(visible_roots) == 1:
            return {
                "found": True,
                "selector": selector,
                "editor_type": editor_type,
                "confidence": 0.72,
                "supported": False,
                "reason": (
                    "The rich-editor container is visible, but no visible native editable "
                    "surface is available for safe normal Playwright interaction."
                ),
            }
        if len(visible_roots) > 1:
            return {
                "found": False,
                "reason": "Multiple rich-editor containers are visible; refusing to guess.",
            }
    if saw_hidden_candidate:
        reason = "Editor-like elements were found, but none were visible for normal interaction."
    elif saw_ambiguous_candidate:
        reason = "An editor-like element was found, but no stable unique selector could be identified."
    else:
        reason = "No visible, semantically identifiable code editor was found."
    return {"found": False, "reason": reason}


def _monaco_model_state(locator, value=None):
    return locator.evaluate(
        """(element, requestedValue) => {
            const models = window.monaco?.editor?.getModels?.() || [];
            if (!models.length) return {available: false, reason: 'Monaco model API is not exposed.'};

            const root = element.closest('.monaco-editor');
            const rootUri = root?.getAttribute('data-uri') ||
                root?.getAttribute('data-model-uri') || root?.dataset?.uri || '';
            let matchingModels = rootUri
                ? models.filter(model => model.uri?.toString() === rootUri)
                : [];
            if (!matchingModels.length && models.length === 1) matchingModels = models;
            if (matchingModels.length !== 1) {
                return {available: false, reason: 'Could not uniquely associate a Monaco model with the detected editor.'};
            }

            const model = matchingModels[0];
            if (requestedValue !== null) model.setValue(requestedValue);
            return {
                available: true,
                uri: model.uri?.toString() || '',
                value: model.getValue()
            };
        }""",
        value,
    )


def _paste_monaco_code(page, locator, code):
    parsed_url = urlsplit(page.url)
    origin = f"{parsed_url.scheme}://{parsed_url.netloc}" if parsed_url.netloc else None
    permissions = ["clipboard-write"]
    if origin:
        page.context.grant_permissions(permissions, origin=origin)
    else:
        page.context.grant_permissions(permissions)
    page.evaluate("text => navigator.clipboard.writeText(text)", code)
    locator.press("Control+V", timeout=5000)


def _read_editor_value(locator, editor_type):
    if editor_type == "monaco":
        model_state = _monaco_model_state(locator)
        if model_state.get("available"):
            return model_state["value"]
    if editor_type in {"textarea", "text_input", "monaco", "codemirror", "ace"}:
        try:
            return locator.input_value(timeout=2000)
        except Exception:
            pass
    return locator.inner_text(timeout=2000)


def enter_code(page, selector, code, editor_type):
    if not isinstance(code, str) or not code:
        return {"success": False, "reason": "Generated code is empty or invalid."}
    if len(code) > MAX_CODE_LENGTH:
        return {
            "success": False,
            "reason": f"Generated code exceeds the {MAX_CODE_LENGTH}-character limit.",
        }
    if editor_type not in {"textarea", "text_input", "contenteditable", "role_textbox", "monaco", "codemirror", "ace"}:
        return {"success": False, "reason": f"Unsupported editor type: {editor_type}."}

    try:
        locator = page.locator(selector)
        if locator.count() != 1:
            return {"success": False, "reason": "The editor selector did not resolve to exactly one element."}
        if not locator.is_visible():
            return {"success": False, "reason": "The editor is not visible."}
        if not locator.is_enabled():
            return {"success": False, "reason": "The editor is disabled."}
        metadata = _editor_metadata(locator)
        if metadata["readonly"] or metadata["disabled"]:
            return {"success": False, "reason": "The editor is read-only or disabled."}

        locator.focus(timeout=3000)
        insertion_method = "locator.fill"
        if editor_type == "monaco":
            model_state = _monaco_model_state(locator, code)
            if model_state.get("available"):
                page.wait_for_function(
                    """({uri, value}) => {
                        const models = window.monaco?.editor?.getModels?.() || [];
                        const matching = uri
                            ? models.filter(model => model.uri?.toString() === uri)
                            : models;
                        return matching.length === 1 && matching[0].getValue() === value;
                    }""",
                    arg={"uri": model_state["uri"], "value": code},
                    timeout=5000,
                )
                insertion_method = "monaco.model.setValue"
            else:
                locator.fill("", timeout=5000)
                locator.focus(timeout=3000)
                _paste_monaco_code(page, locator, code)
                insertion_method = "clipboard paste"
        else:
            locator.fill(code, timeout=5000)
        entered_code = _read_editor_value(locator, editor_type)
        normalize = lambda value: str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
        normalized_readback = entered_code if editor_type == "monaco" else normalize(entered_code)
        comparison_code = code if editor_type == "monaco" else normalize(code)
        comparison_readback = normalized_readback if editor_type == "monaco" else normalize(normalized_readback)
        if comparison_readback != comparison_code:
            return {
                "success": False,
                "reason": "Editor readback did not match the generated code.",
                "readback_matches": False,
                "readback": normalized_readback,
                "insertion_method": insertion_method,
            }
        return {
            "success": True,
            "readback_matches": True,
            "readback": normalized_readback,
            "characters_entered": len(code),
            "editor_type": editor_type,
            "insertion_method": insertion_method,
        }
    except Exception as error:
        rich_editor = editor_type in {"monaco", "codemirror", "ace"}
        prefix = "Rich editor does not support safe normal interaction" if rich_editor else "Could not safely enter code"
        return {
            "success": False,
            "reason": f"{prefix}: {type(error).__name__}: {error}",
            "unsupported": rich_editor,
        }


def _problem_observation(observation, page):
    bounded = bounded_observation(observation)
    bounded["editable_regions"] = [
        {key: value for key, value in editor.items() if key != "value"}
        for editor in bounded["editable_regions"]
    ]
    for selector in (
        "textarea",
        "input:not([type='hidden'])",
        "[contenteditable='true']",
        "[role='textbox']",
    ):
        elements = page.locator(selector)
        for index in range(min(elements.count(), 20)):
            element = elements.nth(index)
            try:
                value = element.input_value(timeout=300)
            except Exception:
                try:
                    value = element.inner_text(timeout=300)
                except Exception:
                    continue
            if value:
                bounded["body_text"] = bounded["body_text"].replace(
                    value,
                    "[editable content omitted]",
                )
                for editor in bounded["editable_regions"]:
                    if editor.get("text") == value:
                        editor["text"] = "[editable content omitted]"
    return bounded


def _wait_for_page(page):
    try:
        page.wait_for_load_state("load", timeout=5000)
    except PlaywrightTimeoutError:
        pass
    try:
        page.wait_for_load_state("networkidle", timeout=1500)
    except PlaywrightTimeoutError:
        pass
    page.locator("body").wait_for(state="visible", timeout=5000)


def real_online_code_entry_workflow(
    url,
    ask_model,
    headless=False,
    keep_open=True,
    inspection_callback=None,
):
    _validate_url(url)
    result = None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            opened = browser_open(page, url)
            _wait_for_page(page)
            observation = browser_read(page)
            if not observation.get("success"):
                result = {
                    "success": False,
                    "status": "read_failed",
                    "error": observation.get("error", "Could not read the page."),
                }
            else:
                page_access = _inspect_page_access(observation)
                if page_access["blocked"]:
                    result = {
                        "success": False,
                        "status": "BLOCKED",
                        "stage": "page_inspection",
                        "blocker": page_access["reason"],
                        "reason": page_access["reason"],
                        "url": page.url,
                        "page_type": page_access["page_type"],
                        "editor_found": False,
                    }
                else:
                    problem = _extract_problem(_problem_observation(observation, page), ask_model)
                    if not problem["appears_coding_problem"] or not problem["problem_statement"]:
                        result = {
                            "success": False,
                            "status": "insufficient_information",
                            "problem": problem,
                            "missing_information": problem["missing_information"],
                            "url": page.url,
                            "page_type": "normal_page",
                            "editor_found": False,
                        }
                    else:
                        solution = _generate_solution(problem, ask_model)
                        editor = find_code_editor(page)
                        if not editor["found"]:
                            result = {
                                "success": False,
                                "status": "entry_failed",
                                "stage": "editor_detection",
                                "reason": editor["reason"],
                                "url": page.url,
                                "page_type": "coding_problem",
                                "editor_found": False,
                                "problem": problem,
                                "solution": solution,
                                "editor": editor,
                            }
                        elif not editor.get("supported", True):
                            result = {
                                "success": False,
                                "status": "editor_unsupported",
                                "problem": problem,
                                "solution": solution,
                                "editor": editor,
                                "run_clicked": False,
                                "submit_clicked": False,
                            }
                        else:
                            entered = enter_code(
                                page,
                                editor["selector"],
                                solution["code"],
                                editor["editor_type"],
                            )
                            if entered.get("unsupported"):
                                editor["supported"] = False
                                editor["reason"] = entered["reason"]
                            result = {
                                "success": entered["success"],
                                "status": (
                                    "entered"
                                    if entered["success"]
                                    else "editor_unsupported"
                                    if entered.get("unsupported")
                                    else "entry_failed"
                                ),
                                "url": opened["url"],
                                "problem": problem,
                                "solution": solution,
                                "editor": editor,
                                "entry": entered,
                                "run_clicked": False,
                                "submit_clicked": False,
                                **({
                                    "stage": "code_entry",
                                    "reason": entered.get("reason", "Code entry failed."),
                                    "url": page.url,
                                    "page_type": "coding_problem",
                                    "editor_found": True,
                                } if not entered["success"] else {}),
                            }
        except Exception as error:
            result = {
                "success": False,
                "status": "workflow_error",
                "error": f"{type(error).__name__}: {error}",
                "run_clicked": False,
                "submit_clicked": False,
            }

        print(f"\nMODE: {REAL_ONLINE_EDITOR_MODE}")
        if result["success"]:
            print("Code entered and read back successfully.")
            print("Run and Submit were not clicked. Workflow stopped.")
        elif result.get("status") == "BLOCKED":
            print("BLOCKED:", result.get("reason"))
            print("Stage:", result.get("stage"))
            print("URL:", result.get("url"))
            print("Page type:", result.get("page_type"))
            print("Editor found:", result.get("editor_found"))
        else:
            print("Code entry stopped:", result.get("status"))
            print("Stage:", result.get("stage", "workflow"))
            print("Reason:", result.get("reason") or result.get("error") or result.get("blocker") or result.get("editor", {}).get("reason") or result.get("status"))
            print("URL:", result.get("url") or page.url)
            print("Page type:", result.get("page_type", "unknown"))
            print("Editor found:", result.get("editor_found", result.get("editor", {}).get("found", False)))
            if result.get("status") == "entry_failed" and result.get("stage") == "code_entry":
                generated_code = result.get("solution", {}).get("code")
                readback = result.get("entry", {}).get("readback")
                editor = result.get("editor", {})
                editor_metadata = {
                    key: editor.get(key)
                    for key in ("found", "editor_type", "confidence", "supported", "reason")
                    if key in editor
                }
                print("Generated code repr:", repr(generated_code))
                print("Editor readback repr:", repr(readback))
                print("Editor metadata:", repr(editor_metadata))
                print("Generated code length:", len(generated_code) if isinstance(generated_code, str) else None)
                print("Editor readback length:", len(readback) if isinstance(readback, str) else None)

                if isinstance(generated_code, str) and isinstance(readback, str):
                    generated_lines = generated_code.splitlines(keepends=True)
                    readback_lines = readback.splitlines(keepends=True)
                    difference = next((
                        index
                        for index in range(max(len(generated_lines), len(readback_lines)))
                        if (generated_lines[index] if index < len(generated_lines) else None)
                        != (readback_lines[index] if index < len(readback_lines) else None)
                    ), None)
                    if difference is None:
                        print("First differing line/character: none (exact match)")
                    else:
                        generated_line = generated_lines[difference] if difference < len(generated_lines) else ""
                        readback_line = readback_lines[difference] if difference < len(readback_lines) else ""
                        char_index = next((
                            index
                            for index in range(max(len(generated_line), len(readback_line)))
                            if (generated_line[index] if index < len(generated_line) else None)
                            != (readback_line[index] if index < len(readback_line) else None)
                        ), None)
                        print("First differing line:", difference + 1)
                        print("Generated line repr:", repr(generated_line))
                        print("Readback line repr:", repr(readback_line))
                        print("First differing character index:", char_index)
                        generated_char = generated_line[char_index] if char_index is not None and char_index < len(generated_line) else None
                        readback_char = readback_line[char_index] if char_index is not None and char_index < len(readback_line) else None
                        print("Generated character repr:", repr(generated_char))
                        print("Readback character repr:", repr(readback_char))
            print("Run and Submit were not clicked.")

        if keep_open:
            if inspection_callback:
                inspection_callback(page, result)
            else:
                input("\nBrowser is open for inspection. Press Enter to close it: ")
        browser.close()

    return result