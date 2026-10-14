import json
import requests

from playwright.sync_api import sync_playwright

from browser_tools import (
    browser_open,
    browser_read,
    browser_click,
    browser_type,
    browser_scroll,
    browser_press,
    browser_screenshot
)


# ============================================================
# CONFIGURATION
# ============================================================

def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            safe_args = [
                arg.encode("ascii", errors="replace").decode("ascii") if isinstance(arg, str) else arg
                for arg in args
            ]
            print(*safe_args, **kwargs)
        except Exception:
            pass


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:4b"

MAX_STEPS = 15
MAX_SAME_ACTION_RETRIES = 2

BROWSER_ACTIONS = {
    "browser_open",
    "browser_read",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_press",
    "browser_screenshot",
}


def action_retry_key(action):
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def classify_browser_error(error_text):
    message = str(error_text).lower()

    if "invalid selector" in message or "not a valid selector" in message or "unexpected token" in message:
        return "invalid selector"
    if "not visible" in message or "hidden" in message or "intercepts pointer events" in message:
        return "element not visible or interactable"
    if "not found" in message or "no element" in message or "resolved to 0" in message:
        return "element not found"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "navigation" in message or "net::err" in message or "failed to load" in message:
        return "page navigation failure"
    if "detached" in message or "stale" in message or "execution context was destroyed" in message:
        return "stale or dynamic page state"

    return "browser action failure"


def format_action_failure(action, error_text, failure_counts):
    key = action_retry_key(action)
    failure_counts[key] = failure_counts.get(key, 0) + 1
    retries_used = failure_counts[key] - 1
    retries_left = max(0, MAX_SAME_ACTION_RETRIES - retries_used)
    classification = classify_browser_error(error_text)

    return key, (
        f"{action.get('action')} failed ({classification}).\n"
        f"Original error: {error_text}\n"
        f"Same-action retries used: {retries_used}/"
        f"{MAX_SAME_ACTION_RETRIES}; retries remaining: {retries_left}.\n"
        "Choose a recovery action. Do not repeat the same action unchanged "
        "unless retrying is appropriate."
    )


# ============================================================
# ASK QWEN
# ============================================================

def ask_ai(prompt, max_tokens=400):

    for attempt in range(1, 4):

        try:

            response = requests.post(
                OLLAMA_URL,

                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "think": False,

                    "options": {
                        "temperature": 0,
                        "num_predict": max_tokens
                    }
                },

                timeout=300
            )

            response.raise_for_status()

            data = response.json()

            result = data.get(
                "response",
                ""
            ).strip()

            _safe_print(
                f"\n📡 Ollama response length: {len(result)}"
            )

            if result:
                return result

            _safe_print(
                f"\n⚠️ Empty AI response "
                f"(attempt {attempt}/3)"
            )

        except Exception as e:

            _safe_print(
                f"\n⚠️ Ollama error "
                f"(attempt {attempt}/3):"
            )

            _safe_print(e)

    raise RuntimeError(
        "Ollama failed to return a response."
    )


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("AI returned an empty response.")

    text = text.strip()

    # First, try parsing the complete response.
    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        action = None

    if action is None:
        # Handle a response containing a JSON code fence.
        cleaned = text

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        try:
            action = json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: decode one JSON object from the response.
            decoder = json.JSONDecoder()
            action = None

            for index, character in enumerate(cleaned):
                if character != "{":
                    continue

                try:
                    candidate, _ = decoder.raw_decode(cleaned[index:])
                except json.JSONDecodeError:
                    continue

                if isinstance(candidate, dict):
                    action = candidate
                    break

    if not isinstance(action, dict):
        raise ValueError(
            f"AI response does not contain a valid JSON object: {text[:500]}"
        )

    if not isinstance(action.get("action"), str):
        raise ValueError(
            "AI JSON must contain an 'action' string."
        )

    return action


# ============================================================
# BUILD AVAILABLE ACTIONS
# ============================================================

def get_available_actions(
    page_opened,
    page_read,
    last_action=None
):

    # --------------------------------------------------------
    # Nothing open
    # --------------------------------------------------------

    if not page_opened:

        return """
AVAILABLE ACTION:

browser_open

JSON:
{
    "action": "browser_open",
    "url": "https://example.com"
}

finish

JSON:
{
    "action": "finish",
    "reason": "The page could not be opened."
}

IMPORTANT:
The webpage is NOT open.
Use browser_open to retry with a corrected URL, or finish if recovery is not possible.
"""


    # --------------------------------------------------------
    # Immediately after browser_open, force browser_read
    # --------------------------------------------------------

    if last_action == "browser_open":

        return """
AVAILABLE ACTIONS:

browser_read

JSON:
{
    "action": "browser_read"
}

IMPORTANT:
The page was just opened.
The next action must be browser_read.
"""


    # --------------------------------------------------------
    # Immediately after browser_type, do not allow it again
    # --------------------------------------------------------

    if last_action == "browser_type":

        return """
AVAILABLE ACTIONS:

browser_click

JSON:
{
    "action": "browser_click",
    "selector": "#submit"
}

browser_read

JSON:
{
    "action": "browser_read"
}

finish

JSON:
{
    "action": "finish",
    "reason": "Task completed."
}

IMPORTANT:
Do NOT use browser_type again immediately.
Use browser_click, browser_read, or finish.
"""


    # --------------------------------------------------------
    # Immediately after browser_click, prefer read/finish
    # --------------------------------------------------------

    if last_action == "browser_click":

        return """
AVAILABLE ACTIONS:

browser_read

JSON:
{
    "action": "browser_read"
}

browser_scroll

JSON:
{
    "action": "browser_scroll",
    "direction": "down",
    "amount": 500
}

browser_press

JSON:
{
    "action": "browser_press",
    "key": "Enter",
    "selector": "#name"
}

browser_screenshot

JSON:
{
    "action": "browser_screenshot",
    "name": "after_click_check"
}

finish

JSON:
{
    "action": "finish",
    "reason": "Task completed."
}

IMPORTANT:
Do not click again unnecessarily.
Use browser_read, browser_scroll, browser_press, browser_screenshot, or finish.
"""


    # --------------------------------------------------------
    # Immediately after browser_scroll, avoid redundant scrolls
    # --------------------------------------------------------

    if last_action == "browser_scroll":

        return """
AVAILABLE ACTIONS:

browser_read

JSON:
{
    "action": "browser_read"
}

browser_type

JSON:
{
    "action": "browser_type",
    "selector": "#name",
    "text": "Moula"
}

browser_click

JSON:
{
    "action": "browser_click",
    "selector": "#submit"
}

browser_press

JSON:
{
    "action": "browser_press",
    "key": "Enter",
    "selector": "#name"
}

browser_screenshot

JSON:
{
    "action": "browser_screenshot",
    "name": "after_scroll_check"
}

finish

JSON:
{
    "action": "finish",
    "reason": "Task completed."
}

IMPORTANT:
Do not scroll again immediately.
Use browser_read, browser_type, browser_click, browser_press, browser_screenshot, or finish.
"""


    # --------------------------------------------------------
    # Page open but not read
    # --------------------------------------------------------

    if not page_read:

        return """
AVAILABLE ACTIONS:

browser_read

JSON:
{
    "action": "browser_read"
}

browser_scroll

JSON:
{
    "action": "browser_scroll",
    "direction": "down",
    "amount": 500
}

browser_press

JSON:
{
    "action": "browser_press",
    "key": "Enter",
    "selector": "#name"
}

browser_type

JSON:
{
    "action": "browser_type",
    "selector": "#name",
    "text": "Moula"
}

browser_click

JSON:
{
    "action": "browser_click",
    "selector": "#submit"
}

browser_screenshot

JSON:
{
    "action": "browser_screenshot",
    "name": "visual_check"
}

finish

JSON:
{
    "action": "finish",
    "reason": "Task completed."
}

IMPORTANT:
The webpage is already OPEN.

DO NOT use browser_open again.

Use browser_read to inspect the page.
If required information is not visible,
use browser_scroll to move the page.
If a focused field needs keyboard input,
use browser_press with a safe key.
If visual confirmation is useful,
use browser_screenshot to capture the current page state.
"""


    # --------------------------------------------------------
    # Page open and read
    # --------------------------------------------------------

    return """
AVAILABLE ACTIONS:

browser_read

JSON:
{
    "action": "browser_read"
}

browser_scroll

JSON:
{
    "action": "browser_scroll",
    "direction": "down",
    "amount": 500
}

browser_press

JSON:
{
    "action": "browser_press",
    "key": "Enter",
    "selector": "#name"
}

browser_type

JSON:
{
    "action": "browser_type",
    "selector": "#name",
    "text": "Moula"
}

browser_click

JSON:
{
    "action": "browser_click",
    "selector": "#submit"
}

browser_screenshot

JSON:
{
    "action": "browser_screenshot",
    "name": "visual_check"
}

finish

JSON:
{
    "action": "finish",
    "reason": "Task completed."
}

IMPORTANT:

The webpage is already OPEN.

DO NOT use browser_open.

Use the interactive elements discovered
by browser_read.

If the required information is not visible,
use browser_scroll before acting.

If a focused field needs a safe keyboard action,
use browser_press with only one of:
Enter, Escape, Tab, ArrowUp, ArrowDown.

Use browser_screenshot only when a visual check is needed.
Do not take screenshots repeatedly.

Prefer element IDs.

Example:

If:

button
id: submit
text: Submit

use:

{
    "action": "browser_click",
    "selector": "#submit"
}
"""


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    user_task,
    observation,
    page_opened,
    page_read,
    last_action=None
):

    available_actions = get_available_actions(
        page_opened,
        page_read,
        last_action
    )

    return f"""
You are the decision-making brain of a browser
automation AI agent.

Your job is to complete the user's task using
the browser tools.

============================================================
USER TASK
============================================================

{user_task}


============================================================
CURRENT OBSERVATION
============================================================

{observation}


============================================================
CURRENT STATE
============================================================

Page opened:
{page_opened}

Page read:
{page_read}

Last action:
{last_action}


============================================================
{available_actions}
============================================================


============================================================
GENERAL RULES
============================================================

RULE 1:
Choose exactly ONE action.

RULE 2:
Return ONLY valid JSON.

RULE 3:
Do not explain your reasoning.

RULE 4:
Do not return markdown.

RULE 5:
Do not return code fences.

RULE 6:
Never use browser_open if Page opened is True.

RULE 7:
If Page opened is True and Page read is False,
browser_read is normally the correct next action.

RULE 8:
If the user asks to enter text,
use browser_type.

RULE 9:
If the user asks to click something,
use browser_click.

RULE 10:
Use information from interactive_elements
to select the correct selector.

RULE 11:
Prefer IDs when available.

RULE 12:
If the information you need is not visible on the current page,
use browser_scroll before continuing.

RULE 13:
Use browser_press only for safe keys:
Enter, Escape, Tab, ArrowUp, ArrowDown.
Do not use arbitrary or dangerous keys.

RULE 13B:
Use browser_screenshot only when visual inspection is genuinely useful.
Do not take screenshots repeatedly or unnecessarily.

RULE 14:
After typing or clicking, read the page again
when verification is needed.

RULE 15:
Do not repeat an action unnecessarily.
Never click a button or link that has already been clicked.
If an action has already been performed and the expected result is visible in the observation, use finish.

RULE 16:
Do not scroll repeatedly in a row.

RULE 17:
Use finish only when the user's task is completed
or no further browser action is appropriate.

When using finish, the "reason" field MUST contain
the actual final answer to the user's task.

FINAL ANSWER RULES:

1. If the user asks questions about a webpage, answer
   those questions using the actual page content.

2. If the user asks for the main heading, identify it
   from the visible page content. Do not use navigation
   menus as the heading.

3. If the user asks about the purpose of a website,
   explain its purpose using evidence from the page.

4. If the user requests a summary, summarize the
   meaningful page content. Ignore navigation menus,
   cookie banners, unrelated links, and code examples
   unless they are relevant to the request.

5. Follow the requested answer format and number
   of sentences or bullet points.

6. Do not invent facts or claim information was found
   if it is not present in the observation.

7. If the available content is insufficient, explain
   what information is missing or why the page could
   not be accessed.

8. If the user asks multiple questions, answer each
   question separately.

9. Before using finish, check whether every part of
   the user's request has been addressed.

10. Never use generic finish reasons such as "done",
    "completed", "task completed", or "success".

Never use generic finish reasons such as:
"done", "completed", "task completed", or "success".

If the webpage contains information needed to answer
the user, read the page before finishing.

Do not claim success if the task has not been completed.

RULE 18:
If CURRENT OBSERVATION says a browser action failed,
read the full original error and choose a recovery action.
Retry the same action only when reasonable, and change its
target or wait for the page when the error indicates that is needed.
Do not repeat an unchanged action after its retry limit is reached.


============================================================
EXAMPLE
============================================================

User task:

Open the test page, enter my name Moula,
click Submit, and read the result.

Correct sequence:

browser_open
↓
browser_read
↓
browser_type
↓
browser_click
↓
browser_read
↓
finish


============================================================
FINAL RESPONSE FORMAT
============================================================

Return exactly ONE JSON object.

Example:

{{
    "action": "browser_read"
}}

OR:

{{
    "action": "browser_type",
    "selector": "#name",
    "text": "Moula"
}}

OR:

{{
    "action": "browser_click",
    "selector": "#submit"
}}

OR:

{{
    "action": "finish",
    "reason": "Task completed."
}}
"""


# ============================================================
# MAIN AGENT
# ============================================================

def main():

    print("""
╔══════════════════════════════════════════════╗
║          MY AI COMPUTER AGENT                ║
║                                              ║
║          Ollama + Qwen3 4B + Playwright      ║
╚══════════════════════════════════════════════╝
""")

    mode = input(
        "\nChoose workflow (browser/coding/online_coding/real_online_read/real_website_readonly/real_online_editor/real_online_run/real_online_correction) [browser]: "
    ).strip().lower() or "browser"

    if mode not in {"browser", "coding", "online_coding", "real_online_read", "real_website_readonly", "real_online_editor", "real_online_run", "real_online_correction"}:

        print("Choose 'browser', 'coding', 'online_coding', 'real_online_read', 'real_website_readonly', 'real_online_editor', 'real_online_run', or 'real_online_correction'.")

        return

    if mode == "real_online_correction":

        from real_online_correction_workflow import real_online_correction_workflow

        url = input("\nCoding problem URL for online correction (no submission): ").strip()

        try:

            real_online_correction_workflow(
                url=url,
                ask_model=ask_ai,
                headless=False,
                keep_open=True,
            )

        except Exception as e:

            print("\n❌ Online correction workflow stopped:")

            print(e)

        return

    if mode == "real_online_run":

        from real_online_run_workflow import real_online_run_workflow

        url = input("\nCoding problem URL to run/test once: ").strip()

        try:

            real_online_run_workflow(
                url=url,
                ask_model=ask_ai,
                headless=False,
                keep_open=True,
            )

        except Exception as e:

            print("\n❌ Online Run/Test workflow stopped:")

            print(e)

        return

    if mode == "real_online_editor":

        from real_online_editor_workflow import real_online_code_entry_workflow

        url = input("\nCoding problem URL to enter generated code into: ").strip()

        try:

            real_online_code_entry_workflow(
                url=url,
                ask_model=ask_ai,
                headless=False,
                keep_open=True,
            )

        except Exception as e:

            print("\n❌ Read-and-enter workflow stopped:")

            print(e)

        return

    if mode in {"real_online_read", "real_website_readonly"}:

        from real_online_read_workflow import real_online_read_workflow

        url = input("\nCoding problem URL to read (no editor interaction): ").strip()

        try:

            real_online_read_workflow(
                url=url,
                ask_model=ask_ai,
                headless=False,
                keep_open=True,
                public_only=(mode == "real_website_readonly"),
            )

        except Exception as e:

            print("\n❌ Read-only coding workflow stopped:")

            print(e)

        return

    if mode == "online_coding":

        from real_online_coding_workflow import real_online_coding_workflow

        url = input("\nCoding website URL (Run/Test only; no submission): ").strip()

        try:

            real_online_coding_workflow(
                url=url,
                ask_model=ask_ai,
                headless=False,
                keep_open=True,
            )

        except Exception as e:

            print("\n❌ Online coding workflow stopped:")

            print(e)

        return

    user_task = input(
        "\n🤖 What should I do? "
    )

    if mode == "coding":

        from coding_workflow import (
            PROBLEM_SOLVING_MODE,
            detect_coding_mode,
            run_coding_task,
        )

        try:

            coding_mode = detect_coding_mode(user_task)

            if coding_mode == PROBLEM_SOLVING_MODE:

                raw_test_cases = input(
                    "\nOptional test cases as JSON [{\"input\": ..., \"expected\": ...}] (Enter to let Qwen propose): "
                ).strip()

                test_cases = json.loads(raw_test_cases) if raw_test_cases else None

            else:

                test_cases = None

            result = run_coding_task(
                user_request=user_task,
                ask_model=ask_ai,
                confirm_execution=lambda message: input(
                    f"\nSAFETY CHECK: {message}\nType RUN to approve: "
                ).strip() == "RUN",
                test_cases=test_cases,
            )

        except Exception as e:

            print("\n❌ Coding workflow failed:")

            print(e)

            return

        if result["mode"] == "exact_code":

            print("\nExact source executed (unchanged):")

            print(result["source"])

            print("\nExecution result:")

            print(json.dumps(result["execution_result"], indent=2, ensure_ascii=True))

            print("\nExact-code workflow stopped; no solution generation or correction was run.")

            return

        print("\nGenerated solution:")

        print(result["code"])

        print("\nTest result:")

        print(json.dumps(result["test_result"], indent=2, ensure_ascii=True))

        print("\nVerification:")

        print(json.dumps(result["verification"], indent=2, ensure_ascii=True))

        print("\nFinal status:", "VERIFIED" if result["verified"] else "NOT VERIFIED")

        return

    with sync_playwright() as p:

        print(
            "\n🌐 Starting Chromium..."
        )

        browser = p.chromium.launch(
            headless=False
        )

        page = browser.new_page()

        # ====================================================
        # STATE
        # ====================================================

        page_opened = False

        page_read = False

        last_action = None

        failure_counts = {}

        observation = (
            "No webpage is open yet."
        )

        # ====================================================
        # AGENT LOOP
        # ====================================================

        for step in range(
            1,
            MAX_STEPS + 1
        ):

            print(
                "\n"
                + "=" * 50
            )

            print(
                f"STEP {step}"
            )

            print(
                "=" * 50
            )

            # =================================================
            # CREATE PROMPT
            # =================================================

            prompt = build_prompt(
                user_task=user_task,
                observation=observation,
                page_opened=page_opened,
                page_read=page_read,
                last_action=last_action
            )

            # =================================================
            # ASK QWEN
            # =================================================

            print(
                "\n🧠 Qwen3 4B is thinking..."
            )

            try:

                ai_response = ask_ai(
                    prompt
                )

                print(
                    "\n🤖 AI response:"
                )

                print(
                    ai_response
                )

                action = extract_json(
                    ai_response
                )

            except Exception as e:

                print(
                    "\n❌ AI error:"
                )

                print(e)

                break

            # =================================================
            # SHOW ACTION
            # =================================================

            print(
                "\n⚡ Selected action:"
            )

            print(
                action
            )

            action_name = action.get(
                "action"
            )

            # =================================================
            # SAFETY GUARD
            # =================================================
            #
            # Qwen should never reopen an already-open page.
            # But if it does, correct it automatically.
            # =================================================

            if (
                action_name == "browser_open"
                and page_opened
            ):

                print(
                    "\n⚠️ AI attempted to reopen "
                    "an already-open webpage."
                )

                print(
                    "🔧 Automatically changing "
                    "action to browser_read."
                )

                action = {
                    "action": "browser_read"
                }

                action_name = "browser_read"

            elif (
                page_opened and
                last_action == "browser_open" and
                action_name != "browser_read"
            ):

                print(
                    "\n⚠️ AI attempted an invalid action "
                    "right after browser_open."
                )

                print(
                    "🔧 Automatically changing "
                    "action to browser_read."
                )

                action = {
                    "action": "browser_read"
                }

                action_name = "browser_read"

            elif (
                last_action == "browser_type" and
                action_name == "browser_type"
            ):

                print(
                    "\n⚠️ AI attempted browser_type again "
                    "immediately after typing."
                )

                print(
                    "🔧 Automatically changing "
                    "action to browser_read."
                )

                action = {
                    "action": "browser_read"
                }

                action_name = "browser_read"

            elif (
                last_action == "browser_click" and
                action_name not in {"browser_read", "browser_scroll", "browser_screenshot", "finish"}
            ):

                print(
                    "\n⚠️ AI attempted an unnecessary action "
                    "right after browser_click."
                )

                print(
                    "🔧 Automatically changing "
                    "action to browser_read."
                )

                action = {
                    "action": "browser_read"
                }

                action_name = "browser_read"

            elif (
                last_action == "browser_scroll" and
                action_name == "browser_scroll"
            ):

                print(
                    "\n⚠️ AI attempted repeated scrolling "
                    "without checking the page."
                )

                print(
                    "🔧 Automatically changing "
                    "action to browser_read."
                )

                action = {
                    "action": "browser_read"
                }


                action_name = "browser_read"

            elif (
                last_action == "browser_press" and
                action_name == "browser_press"
            ):

                print(
                    "\n⚠️ AI attempted repeated keyboard presses "
                    "without checking the page."
                )

                print(
                    "🔧 Automatically changing "
                    "action to browser_read."
                )

                action = {
                    "action": "browser_read"
                }

                action_name = "browser_read"

            elif (
                last_action == "browser_screenshot" and
                action_name == "browser_screenshot"
            ):

                print(
                    "\n⚠️ AI attempted repeated screenshots "
                    "without inspecting the result."
                )

                print(
                    "🔧 Automatically changing "
                    "action to browser_read."
                )

                action = {
                    "action": "browser_read"
                }

                action_name = "browser_read"

            if action_name in BROWSER_ACTIONS:

                retry_key = action_retry_key(action)

                if failure_counts.get(retry_key, 0) > MAX_SAME_ACTION_RETRIES:

                    observation = (
                        f"Retry limit reached for {action_name}; this exact action "
                        f"has failed {failure_counts[retry_key]} times.\n\n"
                        f"Previous error and recovery context:\n{observation}\n\n"
                        "Choose a different recovery action or finish. "
                        "Do not repeat this exact action."
                    )

                    last_action = "action_failed"

                    print(
                        "\n⛔ Same-action retry limit reached; "
                        "asking Qwen for a different recovery action."
                    )

                    continue

            # =================================================
            # BROWSER OPEN
            # =================================================

            if action_name == "browser_open":

                url = action.get(
                    "url"
                )

                if not url:

                    print(
                        "\n❌ browser_open requires a URL."
                    )

                    observation = (
                        "browser_open failed because "
                        "no URL was provided."
                    )

                    _, observation = format_action_failure(
                        action,
                        
                        
                        observation,
                        failure_counts
                    )

                    last_action = "action_failed"

                    continue

                print(
                    "\n🌐 Opening:"
                )

                print(
                    url
                )

                try:

                    result = browser_open(
                        page,
                        url
                    )

                    print(
                        "\n✅ Browser opened successfully."
                    )

                    print(
                        "Title:",
                        result.get(
                            "title"
                        )
                    )

                    print(
                        "URL:",
                        result.get(
                            "url"
                        )
                    )

                    # -----------------------------------------
                    # UPDATE STATE
                    # -----------------------------------------

                    page_opened = True

                    page_read = False

                    last_action = "browser_open"

                    observation = (
                        "Website opened successfully.\n\n"

                        f"Title: {result.get('title')}\n"

                        f"URL: {result.get('url')}\n\n"

                        "IMPORTANT:\n"
                        "The webpage is now OPEN.\n"
                        "Do NOT use browser_open again.\n"
                        "Use browser_read to inspect it."
                    )

                except Exception as e:

                    print(
                        "\n❌ Browser open failed:"
                    )

                    print(e)

                    observation = (
                        "browser_open failed:\n"
                        f"{e}"
                    )

            # =================================================
            # BROWSER READ
            # =================================================

            elif action_name == "browser_read":

                if not page_opened:

                    print(
                        "\n❌ Cannot read."
                    )

                    observation = (
                        "browser_read failed: no webpage is open.\n"
                        "Use browser_open first."
                    )

                    _, observation = format_action_failure(
                        action,
                        observation,
                        failure_counts
                    )

                    last_action = "action_failed"

                    continue

                print(
                    "\n📖 Reading webpage..."
                )

                try:

                    result = browser_read(
                        page
                    )

                    content = result.get(
                        "content",
                        ""
                    )

                    interactive_elements = result.get(
                        "interactive_elements",
                        []
                    )

                    print(
                        "\n✅ Page read successfully."
                    )

                    print(
                        "\n📄 Page summary:"
                    )

                    print(
                        json.dumps(
                            {
                                "url": result.get("page_url") or result.get("url"),
                                "title": result.get("page_title") or result.get("title"),
                                "body_text": result.get("body_text", "")[:1500],
                                "links": result.get("links", [])[:10],
                                "buttons": result.get("buttons", [])[:10],
                                "inputs": result.get("inputs", [])[:10],
                                "textareas": result.get("textareas", [])[:10],
                                "selects": result.get("selects", [])[:10],
                                "checkboxes": result.get("checkboxes", [])[:10],
                                "radio_buttons": result.get("radio_buttons", [])[:10],
                            },
                            indent=2
                        )
                    )

                    # -----------------------------------------
                    # UPDATE STATE
                    # -----------------------------------------

                    page_read = True

                    last_action = "browser_read"

                    observation = (
                        "The webpage has been successfully read.\n\n"

                        f"Title: {result.get('page_title') or result.get('title')}\n"

                        f"URL: {result.get('page_url') or result.get('url')}\n\n"

                        "BODY TEXT:\n"

                        f"{result.get('body_text', '')[:1500]}\n\n"

                        "INTERACTIVE ELEMENTS:\n"

                        f"{json.dumps((result.get('interactive_elements') or [])[:20], indent=2)}"
                    )

                except Exception as e:

                    print(
                        "\n❌ Browser read failed:"
                    )

                    print(e)

                    observation = (
                        "browser_read failed:\n"
                        f"{e}"
                    )

            # =================================================
            # BROWSER TYPE
            # =================================================

            elif action_name == "browser_type":

                selector = action.get(
                    "selector"
                )

                text = action.get(
                    "text"
                )

                if not selector:

                    print(
                        "\n❌ No selector provided."
                    )

                    observation = (
                        "browser_type failed.\n"
                        "No selector was provided."
                    )

                    _, observation = format_action_failure(
                        action,
                        observation,
                        failure_counts
                    )

                    last_action = "action_failed"

                    continue

                if text is None:

                    print(
                        "\n❌ No text provided."
                    )

                    observation = (
                        "browser_type failed.\n"
                        "No text was provided."
                    )

                    _, observation = format_action_failure(
                        action,
                        observation,
                        failure_counts
                    )

                    last_action = "action_failed"

                    continue

                print(
                    "\n⌨️ Typing..."
                )

                print(
                    "Selector:",
                    selector
                )

                print(
                    "Text:",
                    text
                )

                try:

                    result = browser_type(
                        page,
                        selector,
                        text
                    )

                    if result.get(
                        "success"
                    ):

                        print(
                            "\n✅ Text entered successfully."
                        )

                        # -------------------------------------
                        # Page state
                        # -------------------------------------

                        page_read = False

                        last_action = "browser_type"

                        observation = (
                            "Text was entered successfully.\n\n"

                            f"Selector: {selector}\n"

                            f"Text: {text}\n\n"

                            "The page should be inspected again "
                            "before deciding the next action."
                        )

                    else:

                        print(
                            "\n❌ Typing failed."
                        )

                        observation = (
                            "browser_type failed:\n"
                            f"{result.get('error')}"
                        )

                except Exception as e:

                    print(
                        "\n❌ Typing error:"
                    )

                    print(e)

                    observation = (
                        "browser_type failed:\n"
                        f"{e}"
                    )

            # =================================================
            # BROWSER CLICK
            # =================================================

            elif action_name == "browser_click":

                selector = action.get(
                    "selector"
                )

                if not selector:

                    print(
                        "\n❌ No selector provided."
                    )

                    observation = (
                        "browser_click failed.\n"
                        "No selector was provided."
                    )

                    _, observation = format_action_failure(
                        action,
                        observation,
                        failure_counts
                    )

                    last_action = "action_failed"

                    continue

                print(
                    "\n🖱️ Clicking..."
                )

                print(
                    "Selector:",
                    selector
                )

                try:

                    result = browser_click(
                        page,
                        selector
                    )

                    if result.get(
                        "success"
                    ):

                        print(
                            "\n✅ Click successful."
                        )

                        # -------------------------------------
                        # Page may have changed
                        # -------------------------------------

                        page_read = False

                        last_action = "browser_click"

                        observation = (
                            "Click was successful.\n\n"

                            f"URL: {result.get('url')}\n\n"

                            "The page may have changed.\n"
                            "Read the webpage again to verify "
                            "the result."
                        )

                    else:

                        print(
                            "\n❌ Click failed."
                        )

                        observation = (
                            "browser_click failed:\n"
                            f"{result.get('error')}"
                        )

                except Exception as e:

                    print(
                        "\n❌ Click error:"
                    )

                    print(e)

                    observation = (
                        "browser_click failed:\n"
                        f"{e}"
                    )

            # =================================================
            # BROWSER SCROLL
            # =================================================

            elif action_name == "browser_scroll":

                direction = action.get(
                    "direction",
                    "down"
                )

                amount = action.get(
                    "amount",
                    500
                )

                print(
                    "\n📜 Scrolling..."
                )

                print(
                    "Direction:",
                    direction
                )

                print(
                    "Amount:",
                    amount
                )

                try:

                    result = browser_scroll(
                        page,
                        direction,
                        amount
                    )

                    if result.get(
                        "success"
                    ):

                        print(
                            "\n✅ Scroll successful."
                        )

                        print(
                            result
                        )

                        page_read = False

                        last_action = "browser_scroll"

                        observation = (
                            "The page was scrolled successfully.\n\n"

                            f"URL: {result.get('url')}\n"

                            f"Direction: {result.get('direction')}\n"

                            f"Amount: {result.get('amount')}\n"

                            "The page may now show new content.\n"
                            "Read the page again to inspect what is visible."
                        )

                    else:

                        print(
                            "\n❌ Scroll failed."
                        )

                        observation = (
                            "browser_scroll failed:\n"
                            f"{result.get('error')}"
                        )

                except Exception as e:

                    print(
                        "\n❌ Scroll error:"
                    )

                    print(e)

                    observation = (
                        "browser_scroll failed:\n"
                        f"{e}"
                    )

            # =================================================
            # BROWSER PRESS
            # =================================================

            elif action_name == "browser_press":

                key = action.get(
                    "key"
                )

                selector = action.get(
                    "selector"
                )

                if not key:

                    print(
                        "\n❌ No key provided for browser_press."
                    )

                    observation = (
                        "browser_press failed.\n"
                        "No key was provided."
                    )

                    _, observation = format_action_failure(
                        action,
                        observation,
                        failure_counts
                    )

                    last_action = "action_failed"

                    continue

                print(
                    "\n⌨️ Pressing key..."
                )

                print(
                    "Key:",
                    key
                )

                print(
                    "Selector:",
                    selector
                )

                try:

                    result = browser_press(
                        page,
                        key,
                        selector
                    )

                    if result.get(
                        "success"
                    ):

                        print(
                            "\n✅ Key press successful."
                        )

                        print(
                            result
                        )

                        page_read = False

                        last_action = "browser_press"

                        observation = (
                            "A safe key was pressed successfully.\n\n"

                            f"URL: {result.get('url')}\n"

                            f"Key: {result.get('key')}\n"

                            "The browser may now be in a new state.\n"
                            "Read the page again if you need to verify the result."
                        )

                    else:

                        print(
                            "\n❌ Key press failed."
                        )

                        observation = (
                            "browser_press failed:\n"
                            f"{result.get('error')}"
                        )

                except Exception as e:

                    print(
                        "\n❌ Key press error:"
                    )

                    print(e)

                    observation = (
                        "browser_press failed:\n"
                        f"{e}"
                    )

            # =================================================
            # BROWSER SCREENSHOT
            # =================================================

            elif action_name == "browser_screenshot":

                screenshot_name = action.get(
                    "name",
                    "browser_screenshot"
                )

                print(
                    "\n📸 Capturing screenshot..."
                )

                print(
                    "Name:",
                    screenshot_name
                )

                try:

                    result = browser_screenshot(
                        page,
                        name=screenshot_name,
                        full_page=False
                    )

                    if result.get("success"):

                        print(
                            "\n✅ Screenshot captured successfully."
                        )

                        print(
                            result
                        )

                        page_read = False

                        last_action = "browser_screenshot"

                        observation = (
                            "A screenshot was captured successfully.\n\n"

                            f"URL: {result.get('url')}\n"

                            f"Path: {result.get('relative_path')}\n\n"

                            "Use this for visual confirmation if needed.\n"
                            "Do not repeat screenshots unnecessarily."
                        )

                    else:

                        print(
                            "\n❌ Screenshot failed."
                        )

                        observation = (
                            "browser_screenshot failed:\n"
                            f"{result.get('error')}"
                        )

                except Exception as e:

                    print(
                        "\n❌ Screenshot error:"
                    )

                    print(e)

                    observation = (
                        "browser_screenshot failed:\n"
                        f"{e}"
                    )

            # =================================================
            # FINISH
            # =================================================

            elif action_name == "finish":

                print(
                    "\n"
                    + "=" * 50
                )

                print(
                    "✅ AGENT FINISHED"
                )

                print(
                    "=" * 50
                )

                print(
                    "\nReason:"
                )

                print(
                    action.get(
                        "reason",
                        "Task completed."
                    )
                )

                last_action = "finish"

                break

            # =================================================
            # UNKNOWN ACTION
            # =================================================

            else:

                print(
                    "\n❌ Unknown action:"
                )

                print(
                    action_name
                )

                observation = (
                    f"Unknown action: {action_name}\n\n"

                    "Valid actions are:\n"

                    "- browser_open\n"
                    "- browser_read\n"
                    "- browser_scroll\n"
                    "- browser_press\n"
                    "- browser_screenshot\n"
                    "- browser_type\n"
                    "- browser_click\n"
                    "- finish"
                )

            if action_name in BROWSER_ACTIONS:

                if f"{action_name} failed" in observation.lower():

                    _, observation = format_action_failure(
                        action,
                        observation,
                        failure_counts
                    )

                    last_action = "action_failed"

                else:

                    failure_counts.pop(
                        action_retry_key(action),
                        None
                    )

        else:

            print(
                "\n⚠️ Maximum number of steps reached."
            )

        # ====================================================
        # CLOSE BROWSER
        # ====================================================

        print(
            "\n🌐 Closing browser..."
        )

        browser.close()

        print(
            "👋 Agent stopped."
        )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()