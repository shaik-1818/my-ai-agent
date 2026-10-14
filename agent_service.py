import re
import threading
from typing import Any, Callable
from uuid import uuid4

from agent_api_models import TaskEventType, TaskRequest, TaskResult, TaskStatus


SUPPORTED_TASK_TYPES = frozenset({"computer", "coding"})
READY_FOR_SUBMISSION_SUMMARY = "Solution prepared — awaiting your review."
EventCallback = Callable[[TaskEventType, TaskStatus, str, dict[str, Any] | None], None]
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(password|passwd|secret|token|authorization|cookie|api[_ -]?key|csrf|session)"
)
SECRET_TEXT_PATTERNS = (
    re.compile(
        r"(?i)\b(password|passwd|secret|api[_ -]?key|access[_ -]?token|"
        r"authorization|cookie|csrf[_ -]?token)\b\s*[:=]\s*['\"]?[^\s,;\"'<>]+"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
)
SENSITIVE_CONTROL_PATTERN = re.compile(
    r"(?i)\b(submit|login|log\s*in|sign\s*in|authenticate|password|passwd|"
    r"secret|token|authorization|cookie|csrf|session)\b"
)
MAX_COMPUTER_STEPS = 15
MAX_THINKING_REPEATS = 4
MAX_VERIFY_ATTEMPTS = 2
MAX_PARSE_RETRIES = 3      # total attempts per step (1 original + 2 retries)
MAX_FINISH_REJECTIONS = 4  # stop after this many rejected finish actions
_PLAN_STEP_MIN = 2
_PLAN_STEP_MAX = 6
APPROVAL_TIMEOUT_SECONDS = 600  # 10 minutes; never auto-approves on timeout


class ApprovalGate:
    """
    Thread-safe one-shot gate that pauses a worker thread until the user
    approves or rejects a proposed browser action.

    Lifecycle
    ---------
    1. Worker calls ``request(action_snapshot)`` — blocks until decided or timed out.
    2. API endpoint calls ``decide(action_id, approved)`` — unblocks the worker.
    3. Worker reads ``self.approved`` to know whether to proceed.

    Safety guarantees
    -----------------
    - Each gate is consumed exactly once (``_decided`` flag + lock).
    - Timeout never auto-approves; it sets ``approved = False``.
    - ``cancel()`` unblocks the worker with ``approved = False`` for clean shutdown.
    """

    def __init__(self, timeout: float = APPROVAL_TIMEOUT_SECONDS):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._timeout = timeout
        self.action_id: str = ""
        self.action_snapshot: dict[str, Any] = {}
        self.approved: bool = False
        self._decided: bool = False
        self._timed_out: bool = False

    def prepare(self, action_snapshot: dict[str, Any]) -> str:
        """Assign a UUID and store the snapshot. Returns the action_id.
        Must be called before ``wait()``.
        """
        self.action_id = str(uuid4())
        self.action_snapshot = action_snapshot
        self._event.clear()
        return self.action_id

    def wait(self) -> bool:
        """Block until a decision is made. Returns True iff approved.
        Must be called after ``prepare()``.
        """
        fired = self._event.wait(timeout=self._timeout)
        if not fired:
            # Timeout — never approve automatically.
            with self._lock:
                self._timed_out = True
                self._decided = True
                self.approved = False
        return self.approved

    def request(self, action_snapshot: dict[str, Any]) -> bool:
        """Convenience: prepare + wait in one call. Returns True iff approved."""
        self.prepare(action_snapshot)
        return self.wait()

    def decide(self, action_id: str, approved: bool) -> None:
        """
        Called by the API endpoint. Raises ValueError on stale/duplicate decisions.
        """
        with self._lock:
            if self._decided:
                raise ValueError("Approval gate has already been decided.")
            if action_id != self.action_id:
                raise ValueError("Action ID does not match the pending action.")
            self._decided = True
            self.approved = approved
        self._event.set()

    def cancel(self) -> None:
        """Unblock the worker without approving (used on shutdown/cancellation)."""
        with self._lock:
            if not self._decided:
                self._decided = True
                self.approved = False
        self._event.set()

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    @property
    def is_pending(self) -> bool:
        """True while the worker is blocked waiting for a decision."""
        with self._lock:
            return not self._decided


class UnsupportedTaskTypeError(ValueError):
    pass


class TaskBlockedError(RuntimeError):
    pass


def sanitize_event_text(value: Any) -> str:
    text = str(value or "")
    for pattern in SECRET_TEXT_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1)}=[REDACTED]" if match.lastindex else "[REDACTED]",
            text,
        )
    return text[:1000]


def sanitize_event_data(value: Any):
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SENSITIVE_KEY_PATTERN.search(str(key)) else sanitize_event_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_event_data(item) for item in value]
    if isinstance(value, str):
        return sanitize_event_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_event_text(value)


def _emit(
    event_callback: EventCallback,
    event_type: TaskEventType,
    status: TaskStatus,
    message: str,
    data: dict[str, Any] | None = None,
):
    event_callback(
        event_type,
        status,
        sanitize_event_text(message),
        sanitize_event_data(data) if data is not None else None,
    )


def _default_ask_model():
    from agent import ask_ai

    return ask_ai


def _is_ready_for_submission(raw_result: dict[str, Any], raw_status: str) -> bool:
    """True when the coding workflow stopped before Submit and is ready for review."""
    if raw_result.get("submit_clicked") or raw_result.get("submitted"):
        return False
    return raw_status == "READY_FOR_SUBMISSION" or bool(raw_result.get("ready_for_submission"))


def _run_local_coding_workflow(task_request: TaskRequest, ask_model):
    from coding_workflow import run_coding_task

    return run_coding_task(
        task_request.task,
        ask_model=ask_model,
        confirm_execution=lambda _: not task_request.require_confirmation,
    )


def _run_online_coding_workflow(task_request, ask_model, event_callback, workflow=None):
    if workflow is None:
        from real_online_coding_workflow import real_online_coding_workflow

        workflow = real_online_coding_workflow
    return workflow(
        url=task_request.url,
        ask_model=ask_model,
        headless=True,
        keep_open=False,
        event_callback=event_callback,
    )


def _read_page_safely(page):
    from browser_tools import browser_read
    from online_coding_workflow import access_block_reason

    observation = browser_read(page)
    try:
        full_body_text = page.locator("body").inner_text(timeout=1000).strip()
    except Exception:
        full_body_text = observation.get("body_text", "")
    observation = {**observation, "body_text": full_body_text}
    blocker = access_block_reason(observation)
    if blocker:
        raise TaskBlockedError(blocker)
    return observation


def _observed_target(observation, selector, categories):
    if not selector:
        return None
    sel_clean = str(selector).strip()

    # 1. Exact match on selector
    for category in categories:
        for item in observation.get(category, []):
            if item.get("selector") == sel_clean:
                return item

    # 2. Match by id (#id or id)
    for category in categories:
        for item in observation.get(category, []):
            item_id = item.get("id", "").strip()
            if item_id and (sel_clean == f"#{item_id}" or sel_clean == item_id or sel_clean.endswith(f"#{item_id}")):
                return item

    # 3. Match by selector suffix or prefix (e.g. button#showName or #showName)
    for category in categories:
        for item in observation.get(category, []):
            item_sel = item.get("selector", "").strip()
            if item_sel and (sel_clean.endswith(item_sel) or item_sel.endswith(sel_clean)):
                return item

    # 4. Match by text (case-insensitive)
    for category in categories:
        for item in observation.get(category, []):
            item_text = (item.get("text") or "").strip().lower()
            if item_text and (sel_clean.lower() == item_text or sel_clean.lower() in item_text or item_text in sel_clean.lower()):
                return item

    # 5. If only a single candidate exists across the requested categories, fall back to it
    all_items = [
        item
        for category in categories
        for item in observation.get(category, [])
    ]
    if len(all_items) == 1:
        return all_items[0]

    return None


def _target_is_sensitive(target):
    identity = " ".join(
        str(target.get(key, ""))
        for key in ("id", "name", "type", "placeholder", "text", "href")
    )
    return bool(SENSITIVE_CONTROL_PATTERN.search(identity)) or str(target.get("type", "")).lower() == "submit"


def _is_approval_required(action_name: str, target: dict[str, Any] | None) -> bool:
    """
    Return True when the proposed action touches a sensitive control and
    require_confirmation is in effect.  The caller must have already
    confirmed the target exists in the observed page.
    """
    if target is None:
        return False
    return _target_is_sensitive(target)

def _simple_read_task_completed(user_task: str, observation: dict[str, Any]) -> bool:
    """
    Deterministically complete simple read-only information tasks after
    the page has been successfully read.

    This avoids asking the small local model to repeatedly choose
    browser_read for tasks whose answer is already available.
    """
    task = str(user_task or "").strip().lower()
    import re

    # Navigation alone cannot count as successful completion when interaction was requested.
    interaction_patterns = [
        r"\bclick\b", r"\btype\b", r"\benter\b", r"\bfill\b", r"\bpress\b",
        r"\bsubmit\b", r"\bselect\b", r"\bcheck\b", r"\buncheck\b",
        r"\bbutton\b", r"\binput\b", r"\bform\b", r"#[a-zA-Z0-9_-]+",
        r"\bverify\b", r"\bassert\b", r"\bresult\b", r"\btest\b",
        r"\blogin\b", r"\bsignin\b", r"\bsearch\b",
    ]
    if any(re.search(pat, task) for pat in interaction_patterns):
        return False

    # Do not finish early when the task requires multiple websites.
    website_count = len(
        re.findall(r"https?://", str(user_task or ""), re.IGNORECASE)
    )

    if website_count > 1:
        return False
    

    title_requested = any(
        phrase in task
        for phrase in (
            "page title",
            "website title",
            "site title",
            "title of the page",
            "title of this page",
            "what is the title",
            "tell me the title",
        )
    )

    url_requested = any(
        phrase in task
        for phrase in (
            "page url",
            "website url",
            "site url",
            "url of the page",
            "what is the url",
        )
    )

    summarize_requested = any(
        phrase in task
        for phrase in (
            "summarize",
            "summary",
            "bullet point",
            "bullet points",
            "what is the website about",
            "what is this website about",
            "what is the page about",
            "describe the website",
            "describe the page",
            "what does the website say",
            "what does the page say",
            "read the description",
            "read the description page",
            "read the page",
            "read this page",
            "read the webpage",
            "read the web page",
            "read the given url",
            "read the url",
            "when given in url",
            "when given a url",
            "get the page description",
            "extract the page description",
        )
    )

    if title_requested:
        title = str(
            observation.get("title")
            or observation.get("page_title")
            or ""
        ).strip()

        return bool(title)

    if url_requested:
        url = str(
            observation.get("url")
            or observation.get("page_url")
            or ""
        ).strip()

        return bool(url)

    if summarize_requested:
        body = str(observation.get("body_text", "")).strip()
        return bool(body)

    return False
def _build_bullet_summary(body_text: str, n: int = 3) -> str:
    """Generate a concise, source-grounded webpage summary using Ollama."""

    import json
    import re

    if not body_text or not body_text.strip():
        return "No readable webpage content was found."

    # Validate the requested bullet count.
    if n < 1:
        return "The number of bullet points must be at least 1."

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", body_text).strip()

    # Short text is still useful content. Do not reject it just because
    # it contains fewer than 100 characters.
    if not text:
        return "No readable webpage content was found."

    # Handle short text without calling the AI model.
    if len(text) < 100:
        # Split the text into sentences.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        # If there is only one sentence, return one bullet.
        if len(sentences) <= 1:
            sentence = text.strip()

            if sentence and sentence[-1] not in ".!?":
                sentence += "."

            return f"• {sentence}"

        # Produce up to n bullets, preserving all source sentences.
        bullet_count = min(n, len(sentences))
        groups = []
        start = 0

        for i in range(bullet_count):
            remaining_sentences = len(sentences) - start
            remaining_groups = bullet_count - i

            # Distribute sentences across the remaining groups.
            group_size = (
                remaining_sentences + remaining_groups - 1
            ) // remaining_groups

            group = sentences[start:start + group_size]
            groups.append(" ".join(group))
            start += group_size

        return "\n".join(f"• {group}" for group in groups)

        # Limit input size to avoid overwhelming the local model.
    text = text[:7000]

    from agent import ask_ai

    prompt = f"""
You are an accurate webpage content analyst.

TASK:
Summarize the supplied webpage in exactly {n} bullet points.

RULES:
1. Use only information supported by the source text.
2. Prioritize the main article, documentation, tutorial, or explanation.
3. Ignore navigation menus, headers, footers, and promotions when possible.
4. Each bullet must contain a distinct, useful fact.
5. Use clear, simple English.
6. Each bullet must be a complete sentence.
7. Do not invent facts or repeat ideas.
8. Return ONLY valid JSON in this format:
   {{"bullets": ["First complete sentence.", "Second complete sentence."]}}

SOURCE TEXT:
{text}
"""

    try:
        raw = ask_ai(prompt, max_tokens=2000)

        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start < 0 or end <= start:
            raise ValueError("The model did not return valid JSON.")

        result = json.loads(raw[start:end])
        bullets = result.get("bullets")

        if not isinstance(bullets, list):
            raise ValueError("The bullets field is not a list.")

        cleaned_bullets = []

        for item in bullets:
            if not isinstance(item, str):
                continue

            bullet = re.sub(
                r"^\s*(?:[-*•]|\d+[.)])\s*",
                "",
                item,
            ).strip()

            if bullet:
                cleaned_bullets.append(bullet)

        if len(cleaned_bullets) != n:
            raise ValueError(
                f"Expected {n} bullet points, "
                f"received {len(cleaned_bullets)}."
            )

        # Reject very short bullets.
        if any(len(bullet) < 20 for bullet in cleaned_bullets):
            raise ValueError(
                "A bullet point is too short to be useful."
            )

        # Reject duplicate bullets.
        normalized = [
            re.sub(r"\W+", " ", bullet.lower()).strip()
            for bullet in cleaned_bullets
        ]

        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "The model returned duplicate bullet points."
            )

        # Reject bullets that clearly end mid-explanation.
        incomplete_endings = (
            " the", " a", " an", " and", " or", " to", " of",
            " for", " with", " by", " if", " when", " because",
            " which", " that", " such as", " including",
        )

        for bullet in cleaned_bullets:
            if bullet.lower().endswith(incomplete_endings):
                raise ValueError(
                    f"Incomplete bullet point detected: {bullet}"
                )

        # Ensure sentence-ending punctuation.
        cleaned_bullets = [
            bullet if bullet.endswith((".", "!", "?")) else bullet + "."
            for bullet in cleaned_bullets
        ]

        return "\n".join(
            f"• {bullet}" for bullet in cleaned_bullets
        )

    except Exception as exc:
        import traceback

        print("\n========== SUMMARY GENERATION ERROR ==========")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        traceback.print_exc()
        print("==============================================\n")

        return f"Summary failed: {type(exc).__name__}: {exc}"

def _plan_browser_task(task: str, url: str | None, ask_model) -> list[str]:
    """
    Ask the model for a concise ordered plan (2-6 steps) before the action loop.
    Returns a validated list of step strings. On any failure returns a safe fallback.
    Never executes browser actions.
    """
    import json as _json

    url_hint = f" starting at {url}" if url else ""
    prompt = (
        f"You are a browser-task planner.\n"
        f"Task: {task}{url_hint}\n"
        f"Return ONLY a JSON object with key \"steps\": an ordered list of "
        f"{_PLAN_STEP_MIN}-{_PLAN_STEP_MAX} concise plain-English steps to complete the task.\n"
        f"Do NOT include credentials, passwords, or tokens.\n"
        f"Do NOT include markdown or explanation.\n"
        f"Example: {{\"steps\": [\"Open the URL\", \"Read the page\", \"Finish with the answer\"]}}"
    )
    try:
        raw = ask_model(prompt)
        text = str(raw or "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON in planning response.")
        parsed = _json.loads(text[start:end])
        steps = parsed.get("steps")
        if not isinstance(steps, list):
            raise ValueError("'steps' is not a list.")
        steps = [SENSITIVE_KEY_PATTERN.sub("[REDACTED]", str(s).strip()) for s in steps if str(s).strip()]
        if not (_PLAN_STEP_MIN <= len(steps) <= _PLAN_STEP_MAX):
            raise ValueError(f"Plan has {len(steps)} steps; expected {_PLAN_STEP_MIN}-{_PLAN_STEP_MAX}.")
        return steps
    except Exception:
        fallback = ["Open the URL", "Read the page content", "Finish with the answer"]
        if not url:
            fallback[0] = "Identify the correct URL and open it"
        return fallback


def _verify_browser_task(
    task: str,
    plan: list[str],
    observation: str,
    ask_model,
) -> tuple[bool, str]:
    """
    Ask the model to verify whether the task was actually completed.
    Returns (passed, feedback). On model failure returns (False, reason).
    Never fabricates success.
    """
    import json as _json

    _GENERIC = {
        "task completed", "task completed.", "done", "done.",
        "yes", "yes.", "completed", "completed.", "",
    }
    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
    obs_snippet = observation[:1200]
    prompt = (
        f"You are a task-completion verifier for a browser agent.\n"
        f"Original task: {task}\n"
        f"Planned steps:\n{plan_text}\n"
        f"Current observation (page evidence):\n{obs_snippet}\n"
        f"Based ONLY on the observation above, did the agent fully complete the task?\n"
        f"Return ONLY a JSON object: {{\"passed\": true/false, \"reason\": \"one sentence\"}}.\n"
        f"Do NOT return true unless the observation contains clear evidence of completion.\n"
        f"Do NOT return markdown or explanation outside the JSON."
    )
    try:
        raw = ask_model(prompt)
        text = str(raw or "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON in verification response.")
        parsed = _json.loads(text[start:end])
        passed = bool(parsed.get("passed", False))
        reason = str(parsed.get("reason", "")).strip()
        if not reason or reason.lower() in _GENERIC:
            return False, "Verification returned no meaningful evidence."
        return passed, sanitize_event_text(reason)
    except Exception:
        return False, "Verification could not be completed due to a model error."


def _inspect_and_verify_result(
    task: str,
    observation: dict[str, Any],
    page=None,
) -> tuple[bool, bool, str, str, str]:
    """
    Inspects DOM for verification targets specified in the task (e.g. #result).
    Returns (is_verification_task, matches, observed_text, expected_text, target_selector).
    """
    if not task:
        return False, False, "", "", "#result"

    task_str = str(task).strip()
    import re

    # Check if task requests verification
    if not re.search(r"\b(verify|check|assert|inspect|equals?)\b", task_str, re.IGNORECASE):
        return False, False, "", "", "#result"

    # 1. Identify target selector and ID
    target_selector = None
    target_id = None

    selectors = re.findall(r"#([a-zA-Z0-9_-]+)", task_str)
    for sel in selectors:
        sel_lower = sel.lower()
        if any(keyword in sel_lower for keyword in ("result", "output", "message", "greeting", "text")):
            target_selector = f"#{sel}"
            target_id = sel
            break

    if not target_selector and selectors:
        input_selectors = {
            item.get("selector", "").lstrip("#")
            for item in observation.get("inputs", [])
        }
        button_selectors = {
            item.get("selector", "").lstrip("#")
            for item in observation.get("buttons", [])
        }
        for sel in selectors:
            if sel not in input_selectors and sel not in button_selectors:
                target_selector = f"#{sel}"
                target_id = sel
                break

    if not target_selector and re.search(r"\bresult\b", task_str, re.IGNORECASE):
        target_selector = "#result"
        target_id = "result"

    if not target_selector:
        for el in observation.get("elements", []):
            el_id = el.get("id", "")
            if el_id and el_id.lower() == "result":
                target_selector = f"#{el_id}"
                target_id = el_id
                break

    if not target_selector:
        target_selector = "#result"
        target_id = "result"

    # 2. Extract expected text
    expected_text = ""
    m_q = re.search(
        r'(?:verify|check|assert|equals?|result)[^"\'\n\r]*(?:the\s+exact\s+text\s*)?["\']([^"\']+)["\']',
        task_str,
        re.IGNORECASE,
    )
    if m_q:
        expected_text = m_q.group(1).strip()
    else:
        m_u = re.search(
            r'(?:verify|check|assert|equals?)\s+(?:the\s+exact\s+(?:text|value)\s+)?(?:that\s+)?(?:#?\w+\s+(?:is|equals?|to be)\s+)?([^\n\r]+)',
            task_str,
            re.IGNORECASE,
        )
        if m_u:
            expected_candidate = m_u.group(1).strip()
            if expected_candidate.endswith(".") and not expected_candidate.endswith(".."):
                expected_candidate = expected_candidate[:-1].strip()
            expected_candidate = expected_candidate.strip("\"'")
            expected_text = expected_candidate

    if not expected_text:
        return False, False, "", "", target_selector

    # 3. Read observed text from DOM / observation
    observed_text = ""
    if page and target_selector:
        try:
            loc = page.locator(target_selector)
            if loc.count() > 0:
                observed_text = loc.first.inner_text(timeout=1000).strip()
        except Exception:
            observed_text = ""

    if not observed_text and target_id:
        for el in observation.get("elements", []):
            if el.get("id") == target_id or el.get("selector") == target_selector:
                observed_text = str(el.get("text", "")).strip()
                break

    if not observed_text and page:
        try:
            body = page.locator("body").inner_text(timeout=1000).strip()
            if expected_text in body:
                observed_text = expected_text
        except Exception:
            pass

    matches = (observed_text == expected_text)
    return True, matches, observed_text, expected_text, target_selector



def _run_computer_workflow(
    task_request: TaskRequest,
    emit: EventCallback,
    ask_model,
    approval_gate: ApprovalGate | None = None,
):
    from agent import build_prompt, extract_json
    from browser_tools import (
        browser_click,
        browser_open,
        browser_press,
        browser_screenshot,
        browser_scroll,
        browser_type,
    )
    from online_coding_workflow import safe_observation
    from playwright.sync_api import sync_playwright
    from real_online_read_workflow import _validate_url

    _emit(emit, TaskEventType.BROWSER_OPENING, TaskStatus.RUNNING, "Starting an isolated browser session.")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page_opened = False
        page_read = False
        last_action = None
        observation_text = "No webpage is open yet."
        try:
            if task_request.url:
                _validate_url(task_request.url)
                opened = browser_open(page, task_request.url)
                page_opened = bool(opened.get("success"))
                if not page_opened:
                    error_msg = opened.get("error", f"Failed to open {task_request.url}")
                    _emit(
                        emit,
                        TaskEventType.TASK_FAILED,
                        TaskStatus.FAILED,
                        f"Failed to open URL: {error_msg}",
                        {"url": task_request.url, "error": error_msg},
                    )
                    return {
                        "success": False,
                        "verified": False,
                        "state": "FAILED",
                        "summary": f"Failed to open URL: {error_msg}",
                        "url": task_request.url,
                        "error": error_msg,
                        "steps": 0,
                    }
                last_action = "browser_open"
                _emit(
                    emit,
                    TaskEventType.BROWSER_OPENED,
                    TaskStatus.RUNNING,
                    "Browser opened the requested page.",
                    {"url": page.url},
                )
                observation_text = f"Website opened. Title: {page.title()} URL: {page.url}. Read the page next."

            # --- Pre-task plan (generated once before the loop) ----------
            plan = _plan_browser_task(task_request.task, task_request.url, ask_model)
            plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
            _emit(
                emit,
                TaskEventType.THINKING,
                TaskStatus.RUNNING,
                "Task plan generated.",
                {"plan": plan},
            )
            # Inject plan into the initial observation so the loop sees it.
            observation_text = observation_text + f"\nPLAN:\n{plan_text}"
            # -------------------------------------------------------------

            consecutive_thinking = 0
            last_thinking_action = None
            verify_attempts = 0
            finish_rejected_count = 0
            clicked_selectors: set[str] = set()
            for step in range(1, MAX_COMPUTER_STEPS + 1):
                _emit(
                    emit,
                    TaskEventType.THINKING,
                    TaskStatus.RUNNING,
                    "Selecting the next safe browser action.",
                    {"step": step},
                )
                prompt = build_prompt(
                    task_request.task,
                    observation_text,
                    page_opened,
                    page_read,
                    last_action=last_action,
                )

                # --- Bounded parse-retry loop ---
                action = None
                last_parse_error = ""
                for parse_attempt in range(1, MAX_PARSE_RETRIES + 1):
                    try:
                        raw = ask_model(prompt)
                        action = extract_json(raw)
                        break
                    except (ValueError, Exception) as _parse_exc:
                        last_parse_error = sanitize_event_text(str(_parse_exc))
                        if parse_attempt < MAX_PARSE_RETRIES:
                            _emit(
                                emit,
                                TaskEventType.THINKING,
                                TaskStatus.RUNNING,
                                f"Model response could not be parsed (attempt {parse_attempt}/{MAX_PARSE_RETRIES}); retrying.",
                                {"parse_error": last_parse_error},
                            )
                            # Corrective prompt: explain the error and require valid JSON.
                            prompt = (
                                f"Your previous response could not be parsed: {last_parse_error}\n"
                                "Return ONLY a single valid JSON object with an \"action\" key.\n"
                                "Do not include markdown, code fences, or explanation.\n"
                                "Example: {\"action\": \"browser_read\"}\n"
                                f"Original task context:\n{observation_text}"
                            )
                if action is None:
                    _emit(
                        emit,
                        TaskEventType.THINKING,
                        TaskStatus.RUNNING,
                        "Model failed to return a valid action after all retries; skipping step.",
                        {"parse_error": last_parse_error},
                    )
                    observation_text = (
                        f"Your last response could not be parsed after {MAX_PARSE_RETRIES} attempts: "
                        f"{last_parse_error}\n"
                        "Return ONLY a single valid JSON object with an \"action\" key."
                    )
                    last_action = "parse_failed"
                    continue
                # --------------------------------

                action_name = action.get("action")

                # Guard: bail out if the model repeats a non-productive action.
                if action_name == last_thinking_action and action_name not in (
                    "browser_open", "browser_read", "browser_click",
                    "browser_type", "browser_scroll", "browser_press",
                    "browser_screenshot", "finish",
                ):
                    consecutive_thinking += 1
                else:
                    consecutive_thinking = 0
                last_thinking_action = action_name

                if consecutive_thinking >= MAX_THINKING_REPEATS:
                    return {
                        "success": False,
                        "verified": False,
                        "state": "FAILED",
                        "summary": (
                            f"Task stopped: the model repeated '{action_name}' "
                            f"{consecutive_thinking} times without progress."
                        ),
                        "url": page.url if page_opened else None,
                        "steps": step,
                    }

                if page_opened and last_action == "browser_open" and action_name != "browser_read":
                    action_name = "browser_read"
                elif last_action == "browser_type" and action_name == "browser_type":
                    action_name = "browser_read"

                if action_name == "finish":
                    reason = sanitize_event_text(action.get("reason", "")).strip()
                    print("\n========== AGENT FINISH DEBUG ==========")
                    print("Task:", task_request.task)
                    print("Step:", step)
                    print("Page opened:", page_opened)
                    print("Page read:", page_read)
                    print("Current URL:", page.url if page_opened else "No page")
                    print("Last action:", last_action)
                    print("Finish reason:", repr(reason))
                    print("Observation preview:", observation_text[:1500])
                    print("========================================\n")
                    _GENERIC = {"task completed", "task completed.", "done", "done.", ""}
                    if reason.lower() in _GENERIC:
                        finish_rejected_count += 1
                        if finish_rejected_count >= MAX_FINISH_REJECTIONS:
                            return {
                                "success": False,
                                "verified": False,
                                "state": "FAILED",
                                "summary": (
                                    f"Task stopped: model returned a generic finish reason "
                                    f"{finish_rejected_count} times without a meaningful answer."
                                ),
                                "url": page.url if page_opened else None,
                                "steps": step,
                            }
                        observation_text = (
                            f"{observation_text}\n\n"
                            "PREVIOUS FINISH REJECTED:\n"
                            "Your previous finish action did not contain a meaningful answer.\n"
                            "Use the task instructions and the webpage content in the observation above.\n"
                            "If the task asks a question, provide the actual answer.\n"
                            "If the task asks for a coding solution, provide the solution "
                            "based on the problem statement and available information.\n"
                            'Return a JSON object with action="finish" and a meaningful reason '
                            "only when you have enough information to answer.\n"
                            "Otherwise, choose an appropriate browser action to gather more information."
                        )
                        last_action = "finish_rejected"
                        continue
                    # --- Post-completion verification ---
                    dom_is_verif, dom_matches, dom_obs_text, dom_exp_text, dom_target_sel = _inspect_and_verify_result(
                        task_request.task, observation if 'observation' in locals() else {}, page if page_opened else None
                    )
                    if dom_is_verif and not dom_matches:
                        dom_obs_msg = f"Observed {dom_target_sel} text is '{dom_obs_text}' (expected '{dom_exp_text}')."
                        finish_rejected_count += 1
                        if finish_rejected_count >= MAX_FINISH_REJECTIONS:
                            return {
                                "success": False,
                                "verified": False,
                                "state": "FAILED",
                                "summary": f"Task stopped: DOM verification failed ({dom_obs_msg}).",
                                "url": page.url if page_opened else None,
                                "steps": step,
                            }
                        _emit(
                            emit,
                            TaskEventType.THINKING,
                            TaskStatus.RUNNING,
                            f"Verification failed: {dom_obs_msg}; continuing task.",
                            {"selector": dom_target_sel, "observed": dom_obs_text, "expected": dom_exp_text},
                        )
                        observation_text = (
                            f"{observation_text}\n\n"
                            f"VERIFICATION FAILED: {dom_obs_msg}\n"
                            "Review the observation and take further actions "
                            "to fully complete the task before finishing."
                        )
                        last_action = "finish_rejected"
                        continue
                    elif dom_is_verif and dom_matches:
                        _emit(
                            emit,
                            TaskEventType.THINKING,
                            TaskStatus.RUNNING,
                            f"Verification passed: {dom_target_sel} matches '{dom_obs_text}'",
                            {"verified": True, "selector": dom_target_sel, "value": dom_obs_text},
                        )
                        return {
                            "success": True,
                            "verified": True,
                            "state": "VERIFIED",
                            "summary": reason or f"Observed {dom_target_sel} with text: '{dom_obs_text}'.",
                            "url": page.url if page_opened else None,
                            "steps": step,
                            "result": {
                                "selector": dom_target_sel,
                                "text": dom_obs_text,
                            },
                        }

                    if verify_attempts < MAX_VERIFY_ATTEMPTS:
                        verify_attempts += 1
                        passed, feedback = _verify_browser_task(
                            task_request.task, plan, observation_text, ask_model
                        )
                        if not passed:
                            finish_rejected_count += 1
                            if finish_rejected_count >= MAX_FINISH_REJECTIONS:
                                return {
                                    "success": False,
                                    "verified": False,
                                    "state": "FAILED",
                                    "summary": (
                                        f"Task stopped: finish was rejected {finish_rejected_count} times "
                                        f"(verification never passed: {feedback})."
                                    ),
                                    "url": page.url if page_opened else None,
                                    "steps": step,
                                }
                            _emit(
                                emit,
                                TaskEventType.THINKING,
                                TaskStatus.RUNNING,
                                f"Verification failed: {feedback}; continuing task.",
                                {"feedback": feedback},
                            )
                            observation_text = (
                                f"{observation_text}\n\n"
                                f"VERIFICATION FAILED: {feedback}\n"
                                "Review the observation and take further actions "
                                "to fully complete the task before finishing."
                            )
                            last_action = "finish_rejected"
                            continue
                        else:
                            _emit(
                                emit,
                                TaskEventType.THINKING,
                                TaskStatus.RUNNING,
                                f"Verification passed: {feedback}",
                                {"verified": True, "feedback": feedback},
                            )

                    if verify_attempts > 0 and not passed:
                        return {
                            "success": False,
                            "verified": False,
                            "state": "FAILED",
                            "summary": f"Task verification failed: {feedback}",
                            "url": page.url if page_opened else None,
                            "steps": step,
                        }

                    verified_flag = verify_attempts > 0 and passed
                    return {
                        "success": True,
                        "verified": verified_flag,
                        "state": "VERIFIED" if verified_flag else "COMPLETED",
                        "summary": reason,
                        "url": page.url if page_opened else None,
                        "steps": step,
                    }
                elif action_name == "browser_open":
                    if page_opened:
                        raise ValueError("The agent cannot reopen an already-open page.")
                    url = action.get("url")
                    if not url:
                        raise ValueError("The browser_open action did not include a URL.")
                    _validate_url(url)
                    opened = browser_open(page, url)
                    page_opened = bool(opened.get("success"))
                    if not page_opened:
                        error_msg = opened.get("error", f"Failed to open {url}")
                        observation_text = f"browser_open failed: {error_msg}\nPLAN:\n{plan_text}"
                        last_action = "browser_open_failed"
                        continue
                    last_action = "browser_open"
                    page_read = False
                    _emit(
                        emit,
                        TaskEventType.BROWSER_OPENED,
                        TaskStatus.RUNNING,
                        "Browser opened a page requested by the task.",
                        {"url": page.url},
                    )
                    observation_text = f"Website opened. Title: {page.title()} URL: {page.url}. Read the page next.\nPLAN:\n{plan_text}"
                elif action_name == "browser_read":
                    if not page_opened:
                        raise ValueError("Cannot read before opening a webpage.")

                    # Do not read the exact same page repeatedly.
                    if page_read and last_action == "browser_read":
                        observation_text = (
                            "The page has already been read successfully.\n"
                            "Do NOT use browser_read again unless the page has changed.\n"
                            "If the user's task can be answered from the current observation, "
                            "finish the task."
                        )
                        last_action = "repeated_browser_read_blocked"
                        continue

                    _emit(
                        emit,
                        TaskEventType.BROWSER_READING,
                        TaskStatus.RUNNING,
                        "Reading visible page content.",
                    )

                    observation = _read_page_safely(page)

                    page_read = True
                    last_action = "browser_read"

                    page_read_data = {
                        "title": observation.get("title", ""),
                        "url": page.url,
                    }
                    if observation.get("elements"):
                        page_read_data["elements"] = observation["elements"]
                    _emit(
                        emit,
                        TaskEventType.PAGE_READ,
                        TaskStatus.RUNNING,
                        "Page content was read safely.",
                        page_read_data,
                    )

                    observation_text = f"{safe_observation(observation)}\n\nPLAN:\n{plan_text}"

                    dom_is_verif, dom_matches, dom_obs_text, dom_exp_text, dom_target_sel = _inspect_and_verify_result(
                        task_request.task, observation, page
                    )
                    if dom_is_verif and dom_matches and clicked_selectors:
                        feedback = f"Observed {dom_target_sel} matches expected text: '{dom_obs_text}'."
                        _emit(
                            emit,
                            TaskEventType.THINKING,
                            TaskStatus.RUNNING,
                            f"Verification passed: {feedback}",
                            {"verified": True, "selector": dom_target_sel, "value": dom_obs_text},
                        )
                        return {
                            "success": True,
                            "verified": True,
                            "state": "VERIFIED",
                            "summary": f"Observed {dom_target_sel} with text: '{dom_obs_text}'. Task verified successfully.",
                            "url": page.url,
                            "steps": step,
                            "result": {
                                "selector": dom_target_sel,
                                "text": dom_obs_text,
                            },
                        }

                    # Deterministically finish simple information requests.
                    if _simple_read_task_completed(task_request.task, observation):
                        title = str(
                            observation.get("title")
                            or observation.get("page_title")
                            or ""
                        ).strip()

                        url = str(
                            observation.get("url")
                            or observation.get("page_url")
                            or page.url
                            or ""
                        ).strip()

                        task_lower = task_request.task.lower()
                        if any(p in task_lower for p in (
                            "page title", "website title", "site title",
                            "title of", "what is the title", "tell me the title",
                        )):
                            summary = f"Page title: {title}"
                        elif any(p in task_lower for p in (
                            "page url", "website url", "site url",
                            "url of", "what is the url",
                        )):
                            summary = f"Page URL: {url}"
                        else:
                            body = str(observation.get("body_text", "")).strip()
                            import re

                            match = re.search(
                                r"\b(?:exactly\s+)?(\d+)\s+bullet\s+points?\b",
                                task_request.task,
                                re.IGNORECASE,
                            )

                            bullet_count = int(match.group(1)) if match else 3
                            
                            print(f"\nDEBUG: Requested bullet count = {bullet_count}")
                            print(f"DEBUG: Extracted body length = {len(body)}")
                            print(f"DEBUG: Extracted body preview = {body[:500]}")
                            
                            summary = _build_bullet_summary(body, n=bullet_count)

                        return {
                            "success": True,
                            "verified": True,
                            "state": "VERIFIED",
                            "summary": sanitize_event_text(summary),
                            "url": page.url,
                            "steps": step,
                            "result": {
                                "title": title,
                                "url": url,
                            },
                        }
                elif action_name == "browser_click":
                    observation = _read_page_safely(page)
                    selector = action.get("selector", "")
                    target = _observed_target(observation, selector, ("buttons", "links"))
                    if target is None:
                        raise ValueError("Click rejected: target is not an observed control.")
                    effective_selector = target.get("selector") or selector
                    target_id = target.get("id", "")
                    already_clicked = (
                        effective_selector in clicked_selectors
                        or selector in clicked_selectors
                        or (target_id and (target_id in clicked_selectors or f"#{target_id}" in clicked_selectors))
                    )
                    if already_clicked:
                        # Do not repeat a successful click unnecessarily.
                        observation = _read_page_safely(page)
                        page_read = True
                        dom_is_verif, dom_matches, dom_obs_text, dom_exp_text, dom_target_sel = _inspect_and_verify_result(
                            task_request.task, observation, page
                        )
                        if dom_is_verif and dom_matches:
                            feedback = f"Observed {effective_selector} was already clicked and {dom_target_sel} with '{dom_obs_text}' verified in DOM."
                            _emit(
                                emit,
                                TaskEventType.THINKING,
                                TaskStatus.RUNNING,
                                f"Verification passed: {feedback}",
                                {"verified": True, "selector": dom_target_sel, "value": dom_obs_text},
                            )
                            return {
                                "success": True,
                                "verified": True,
                                "state": "VERIFIED",
                                "summary": f"Observed {dom_target_sel} with text: '{dom_obs_text}'. Task verified successfully.",
                                "url": page.url,
                                "steps": step,
                                "result": {"selector": dom_target_sel, "text": dom_obs_text},
                            }

                        _emit(
                            emit,
                            TaskEventType.THINKING,
                            TaskStatus.RUNNING,
                            f"Repeated click blocked: '{effective_selector}' has already been clicked.",
                            {"action": "browser_click", "selector": effective_selector},
                        )
                        obs_note = f"NOTE: The control '{effective_selector}' has already been clicked successfully. Do NOT click it again."
                        if dom_is_verif and not dom_matches:
                            obs_note += f" Actual observation: {dom_target_sel} is '{dom_obs_text}', expected '{dom_exp_text}'."
                        observation_text = f"{safe_observation(observation)}\n\n{obs_note}\nPLAN:\n{plan_text}"
                        last_action = "repeated_click_blocked"
                        continue

                    if _is_approval_required(action_name, target):
                        if approval_gate is None or not task_request.require_confirmation:
                            raise ValueError("Click rejected: sensitive target requires user approval.")
                        action_snapshot = sanitize_event_data({
                            "action": "browser_click",
                            "selector": effective_selector,
                            "target_text": target.get("text", ""),
                            "target_type": target.get("type", ""),
                        })
                        # Assign UUID first so the event carries the real action_id.
                        action_id = approval_gate.prepare(action_snapshot)
                        _emit(
                            emit,
                            TaskEventType.CONFIRMATION_REQUIRED,
                            TaskStatus.WAITING_CONFIRMATION,
                            "Approval required before clicking a sensitive control.",
                            {
                                "action_id": action_id,
                                "action": action_snapshot,
                            },
                        )
                        # --- PAUSE: block worker thread until user decides ---
                        approved = approval_gate.wait()
                        if approval_gate.timed_out:
                            return {
                                "success": False, "verified": False, "state": "FAILED",
                                "summary": "Approval timed out; sensitive action was not executed.",
                                "url": page.url if page_opened else None, "steps": step,
                            }
                        if not approved:
                            _emit(emit, TaskEventType.APPROVAL_REJECTED, TaskStatus.RUNNING,
                                  "User rejected the sensitive click action.")
                            return {
                                "success": False, "verified": False, "state": "FAILED",
                                "summary": "Action rejected by user; sensitive click was not executed.",
                                "url": page.url if page_opened else None, "steps": step,
                            }
                        _emit(emit, TaskEventType.APPROVAL_GRANTED, TaskStatus.RUNNING,
                              "User approved the sensitive click action.")
                    _emit(
                        emit,
                        TaskEventType.EXECUTING,
                        TaskStatus.RUNNING,
                        f"Clicking observed control {effective_selector}.",
                        {"action": "browser_click", "selector": effective_selector},
                    )
                    outcome = browser_click(page, effective_selector)
                    if not outcome.get("success"):
                        raise RuntimeError(outcome.get("error", "Browser click failed."))

                    clicked_selectors.add(effective_selector)
                    clicked_selectors.add(selector)
                    if target_id:
                        clicked_selectors.add(target_id)
                        clicked_selectors.add(f"#{target_id}")

                    # After clicking, read the current DOM again immediately
                    _emit(
                        emit,
                        TaskEventType.BROWSER_READING,
                        TaskStatus.RUNNING,
                        "Reading visible page content after click.",
                    )
                    observation = _read_page_safely(page)
                    page_read = True
                    last_action = "browser_click"

                    page_read_data = {
                        "title": observation.get("title", ""),
                        "url": page.url,
                    }
                    if observation.get("elements"):
                        page_read_data["elements"] = observation["elements"]
                    _emit(
                        emit,
                        TaskEventType.PAGE_READ,
                        TaskStatus.RUNNING,
                        "Page content was read safely.",
                        page_read_data,
                    )

                    observation_text = f"{safe_observation(observation)}\n\nPLAN:\n{plan_text}"

                    # Inspect DOM and check verification
                    dom_is_verif, dom_matches, dom_obs_text, dom_exp_text, dom_target_sel = _inspect_and_verify_result(
                        task_request.task, observation, page
                    )
                    if dom_is_verif:
                        if dom_matches:
                            feedback = f"Observed {dom_target_sel} matches expected text: '{dom_obs_text}'."
                            _emit(
                                emit,
                                TaskEventType.THINKING,
                                TaskStatus.RUNNING,
                                f"Verification passed: {feedback}",
                                {"verified": True, "selector": dom_target_sel, "value": dom_obs_text},
                            )
                            return {
                                "success": True,
                                "verified": True,
                                "state": "VERIFIED",
                                "summary": f"Observed {dom_target_sel} with text: '{dom_obs_text}'. Task verified successfully.",
                                "url": page.url,
                                "steps": step,
                                "result": {
                                    "selector": dom_target_sel,
                                    "text": dom_obs_text,
                                },
                            }
                        else:
                            obs_msg = f"Observed {dom_target_sel} text is '{dom_obs_text}' (expected '{dom_exp_text}')."
                            _emit(
                                emit,
                                TaskEventType.THINKING,
                                TaskStatus.RUNNING,
                                f"Verification pending: {obs_msg}",
                                {"selector": dom_target_sel, "observed": dom_obs_text, "expected": dom_exp_text},
                            )
                            observation_text = f"{safe_observation(observation)}\n\nACTUAL OBSERVATION:\n{obs_msg}\nPLAN:\n{plan_text}"
                elif action_name == "browser_type":
                    observation = _read_page_safely(page)
                    selector = action.get("selector", "")
                    target = _observed_target(observation, selector, ("inputs", "editable_regions"))
                    if target is None:
                        raise ValueError("Typing rejected: target is not an observed field.")
                    text = str(action.get("text", ""))
                    if not text:
                        raise ValueError("The browser_type action did not include text.")
                    effective_selector = target.get("selector") or selector
                    if target.get("value") == text:
                        observation_text = (
                            f"{safe_observation(observation)}\n\n"
                            f"NOTE: The field '{effective_selector}' already contains the required text '{text}'.\n"
                            "Do NOT type into it again.\n"
                            "Choose an observed button to click (for example from the buttons list above) or finish the task.\n\n"
                            f"PLAN:\n{plan_text}"
                        )
                        last_action = "browser_type_already_filled"
                        continue
                    if _is_approval_required(action_name, target):
                        if approval_gate is None or not task_request.require_confirmation:
                            raise ValueError("Typing rejected: sensitive target requires user approval.")
                        action_snapshot = sanitize_event_data({
                            "action": "browser_type",
                            "selector": effective_selector,
                            "target_type": target.get("type", ""),
                            "target_placeholder": target.get("placeholder", ""),
                            # Never include the text value for sensitive fields.
                        })
                        # Assign UUID first so the event carries the real action_id.
                        action_id = approval_gate.prepare(action_snapshot)
                        _emit(
                            emit,
                            TaskEventType.CONFIRMATION_REQUIRED,
                            TaskStatus.WAITING_CONFIRMATION,
                            "Approval required before typing into a sensitive field.",
                            {
                                "action_id": action_id,
                                "action": action_snapshot,
                            },
                        )
                        approved = approval_gate.wait()
                        if approval_gate.timed_out:
                            return {
                                "success": False, "verified": False, "state": "FAILED",
                                "summary": "Approval timed out; sensitive action was not executed.",
                                "url": page.url if page_opened else None, "steps": step,
                            }
                        if not approved:
                            _emit(emit, TaskEventType.APPROVAL_REJECTED, TaskStatus.RUNNING,
                                  "User rejected the sensitive type action.")
                            return {
                                "success": False, "verified": False, "state": "FAILED",
                                "summary": "Action rejected by user; sensitive typing was not executed.",
                                "url": page.url if page_opened else None, "steps": step,
                            }
                        _emit(emit, TaskEventType.APPROVAL_GRANTED, TaskStatus.RUNNING,
                              "User approved the sensitive type action.")
                    _emit(
                        emit,
                        TaskEventType.EXECUTING,
                        TaskStatus.RUNNING,
                        f"Entering text into {effective_selector}.",
                        {"action": "browser_type", "selector": effective_selector},
                    )
                    outcome = browser_type(page, effective_selector, text)
                    if not outcome.get("success"):
                        raise RuntimeError(outcome.get("error", "Browser typing failed."))
                    page_read = False
                    last_action = "browser_type"
                    observation_text = f"Text entered into {effective_selector}. Read the page to verify; do not submit forms.\nPLAN:\n{plan_text}"
                elif action_name == "browser_scroll":
                    if not page_opened:
                        raise ValueError("Cannot scroll before opening a webpage.")
                    _read_page_safely(page)
                    outcome = browser_scroll(page, action.get("direction", "down"), action.get("amount", 500))
                    if not outcome.get("success"):
                        raise RuntimeError(outcome.get("error", "Browser scroll failed."))
                    page_read = False
                    last_action = "browser_scroll"
                    observation_text = outcome.get("message", "Scrolled. Read the page.")
                elif action_name == "browser_press":
                    if action.get("key") not in {"Escape", "Tab", "ArrowUp", "ArrowDown"}:
                        raise ValueError("Only non-submitting browser keys are allowed.")
                    observation = _read_page_safely(page) if page_opened else {}
                    selector = action.get("selector")
                    if selector:
                        target = _observed_target(observation, selector, ("inputs", "editable_regions"))
                        if target is None or _target_is_sensitive(target):
                            raise ValueError("Keyboard target is not an observed non-sensitive field.")
                    outcome = browser_press(page, action["key"], selector)
                    if not outcome.get("success"):
                        raise RuntimeError(outcome.get("error", "Browser key action failed."))
                    page_read = False
                    last_action = "browser_press"
                    observation_text = outcome.get("message", "Key pressed. Read the page.")
                elif action_name == "browser_screenshot":
                    if not page_opened:
                        raise ValueError("Cannot capture a screenshot before opening a webpage.")
                    _read_page_safely(page)
                    outcome = browser_screenshot(page, action.get("name", "agent_task"), full_page=False)
                    if not outcome.get("success"):
                        raise RuntimeError(outcome.get("error", "Screenshot failed."))
                    page_read = False
                    last_action = "browser_screenshot"
                    observation_text = sanitize_event_text(outcome.get("message", "Screenshot captured."))
                else:
                    _emit(
                        emit,
                        TaskEventType.THINKING,
                        TaskStatus.RUNNING,
                        f"Unsupported or unsafe action: '{sanitize_event_text(action_name)}'.",
                        {"action": action_name},
                    )
                    observation_text = f"Unsupported or unsafe action: {sanitize_event_text(action_name)}. Choose a safe available action or finish."
                    last_action = "action_failed"
                    continue

            return {
    "success": False,
    "verified": False,
    "state": "FAILED",
    "summary": (
        f"Computer task stopped because it reached the "
        f"{MAX_COMPUTER_STEPS}-step safety limit before completion."
    ),
    "url": page.url if page_opened else None,
    "steps": MAX_COMPUTER_STEPS,
}
        finally:
            browser.close()


def run_task(
    task_id: str,
    task_request: TaskRequest,
    event_callback: EventCallback,
    *,
    ask_model=None,
    computer_workflow=None,
    coding_workflow=None,
    online_coding_workflow=None,
    approval_gate: ApprovalGate | None = None,
) -> TaskResult:
    task_type = task_request.task_type
    if task_type not in SUPPORTED_TASK_TYPES:
        raise UnsupportedTaskTypeError(f"Unsupported task type: {sanitize_event_text(task_type)}")

    ask_model = ask_model or _default_ask_model()
    _emit(event_callback, TaskEventType.TASK_STARTED, TaskStatus.RUNNING, "Task started.")
    current_stage = ["STARTING"]
    terminal_events = set()

    def report_online_progress(event_name, data=None):
        event_name = str(event_name).lower()
        event_type = {
            "blocked": TaskEventType.TASK_BLOCKED,
            "failed": TaskEventType.TASK_FAILED,
        }.get(event_name)
        if event_type is None:
            event_type = TaskEventType(event_name.upper())
        status = {
            "browser_opening": TaskStatus.OPENING,
            "browser_opened": TaskStatus.OPENING,
            "reading_problem": TaskStatus.READING,
            "problem_read": TaskStatus.READING,
            "planning": TaskStatus.PLANNING,
            "generating_code": TaskStatus.GENERATING,
            "finding_editor": TaskStatus.EDITING,
            "entering_code": TaskStatus.EDITING,
            "verifying_code": TaskStatus.VERIFYING,
            "finding_run_control": TaskStatus.VERIFYING,
            "running": TaskStatus.RUNNING,
            "reading_result": TaskStatus.RUNNING,
            "correcting": TaskStatus.CORRECTING,
            "verified": TaskStatus.VERIFYING,
            "ready_for_submission": TaskStatus.RUNNING,
            "blocked": TaskStatus.BLOCKED,
            "failed": TaskStatus.FAILED,
        }[event_name]
        current_stage[0] = event_name.upper()
        if event_name in {"blocked", "failed"}:
            terminal_events.add(event_name)
        event_data = {"stage": event_name}
        if data:
            event_data.update(data)
        _emit(event_callback, event_type, status, event_name.replace("_", " ").capitalize(), event_data)

    try:
        if task_type == "computer":
            if computer_workflow is None:
                # Real workflow: pass approval_gate as keyword arg.
                raw_result = _run_computer_workflow(
                    task_request,
                    lambda event_type, status, message, data=None: _emit(
                        event_callback, event_type, status, message, data
                    ),
                    ask_model,
                    approval_gate=approval_gate,
                )
            else:
                # Injected workflow (tests / custom callers): call with 3 positional
                # args for backward compatibility; pass approval_gate only if the
                # callable explicitly accepts it (keyword-only or **kwargs).
                import inspect as _inspect
                sig = _inspect.signature(computer_workflow)
                params = sig.parameters
                accepts_gate = (
                    "approval_gate" in params
                    or any(
                        p.kind == _inspect.Parameter.VAR_KEYWORD
                        for p in params.values()
                    )
                )
                emit_lambda = lambda event_type, status, message, data=None: _emit(
                    event_callback, event_type, status, message, data
                )
                if accepts_gate:
                    raw_result = computer_workflow(
                        task_request, emit_lambda, ask_model, approval_gate=approval_gate
                    )
                else:
                    raw_result = computer_workflow(task_request, emit_lambda, ask_model)
        else:
            if task_request.url:
                raw_result = _run_online_coding_workflow(
                    task_request,
                    ask_model,
                    report_online_progress,
                    workflow=online_coding_workflow,
                )
            else:
                workflow = coding_workflow or _run_local_coding_workflow
                _emit(event_callback, TaskEventType.PLANNING, TaskStatus.RUNNING, "Preparing the local coding workflow.")
                _emit(event_callback, TaskEventType.THINKING, TaskStatus.RUNNING, "Generating and validating a local solution.")
                raw_result = workflow(task_request, ask_model)

        if not isinstance(raw_result, dict):
            raise TypeError("Workflow returned a non-dictionary result.")
        raw_status = str(raw_result.get("state", raw_result.get("status", ""))).upper()
        is_blocked = bool(raw_result.get("blocked")) or raw_status in {
            "BLOCKED", "LOGIN_REQUIRED", "BLOCKED_SECURITY_CHALLENGE"
        }
        if is_blocked:
            summary = sanitize_event_text(raw_result.get("summary", "Task stopped because page access is blocked."))
            if "blocked" not in terminal_events:
                _emit(event_callback, TaskEventType.TASK_BLOCKED, TaskStatus.BLOCKED, summary)
            current_stage[0] = "BLOCKED"
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.BLOCKED,
                success=False,
                summary=summary,
                result=raw_result,
                error=sanitize_event_text(raw_result.get("error", summary)),
                current_stage=current_stage[0],
                verified=False,
                blocked=True,
                workflow_result=raw_result,
            )

        failed = (
            raw_status in {"FAILED", "MAX_CORRECTIONS_REACHED"}
            or raw_result.get("success") is False
        )
        success = bool(raw_result.get("success", raw_result.get("verified", not failed)))
        verified = bool(raw_result.get("verified", success)) and not failed
        verification = raw_result.get("verification")
        summary = raw_result.get("summary")
        if not summary and isinstance(verification, dict):
            summary = verification.get("summary")
        if not summary:
            summary = "Task completed." if success else "Task finished without verified success."
        ready_for_review = (not failed) and _is_ready_for_submission(raw_result, raw_status)
        if ready_for_review:
            summary = READY_FOR_SUBMISSION_SUMMARY
            current_stage[0] = "READY_FOR_SUBMISSION"
        summary = sanitize_event_text(summary)
        if failed:
            if "failed" not in terminal_events:
                _emit(event_callback, TaskEventType.TASK_FAILED, TaskStatus.FAILED, summary)
            current_stage[0] = "FAILED"
        else:
            _emit(event_callback, TaskEventType.TASK_COMPLETED, TaskStatus.COMPLETED, summary)
            if current_stage[0] == "STARTING":
                current_stage[0] = "COMPLETED"
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED if failed else TaskStatus.COMPLETED,
            success=success,
            summary=summary,
            result=raw_result,
            error=sanitize_event_text(raw_result.get("error", summary)) if failed else None,
            current_stage=current_stage[0],
            verified=verified,
            blocked=False,
            workflow_result=raw_result,
        )
    except TaskBlockedError as error:
        summary = sanitize_event_text(error)
        _emit(event_callback, TaskEventType.TASK_BLOCKED, TaskStatus.BLOCKED, summary)
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.BLOCKED,
            success=False,
            summary=summary,
            result={"blocked": True},
            error=summary,
            current_stage="BLOCKED",
            blocked=True,
            workflow_result={"blocked": True},
        )
    except Exception as error:
        error_message = sanitize_event_text(f"{type(error).__name__}: {error}")
        _emit(event_callback, TaskEventType.TASK_FAILED, TaskStatus.FAILED, "Task failed.", {"error": error_message})
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            success=False,
            summary="Task failed.",
            error=error_message,
            current_stage="FAILED",
            workflow_result=None,
        )