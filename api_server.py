from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from threading import RLock
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from agent_api_models import (
    ApprovalRequest,
    HealthResponse,
    TaskEvent,
    TaskEventType,
    TaskRequest,
    TaskResponse,
    TaskResult,
    TaskStatus,
)
from agent_service import (
    SUPPORTED_TASK_TYPES,
    ApprovalGate,
    run_task,
    sanitize_event_data,
    sanitize_event_text,
)


MODEL_NAME = "qwen3:4b"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_HEALTH_TIMEOUT_SECONDS = 1

app = FastAPI(title="AI Computer Agent API")

tasks: dict[str, TaskResult] = {}
events: dict[str, list[TaskEvent]] = {}
task_requests: dict[str, TaskRequest] = {}
approval_gates: dict[str, ApprovalGate] = {}
_task_store_lock = RLock()
task_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-task")


def check_ollama_availability():
    try:
        with urlopen(OLLAMA_TAGS_URL, timeout=OLLAMA_HEALTH_TIMEOUT_SECONDS) as response:
            return "available" if response.status == 200 else "unavailable"
    except (OSError, TimeoutError, URLError):
        return "unavailable"


def check_playwright_availability():
    try:
        return "available" if find_spec("playwright") is not None else "unavailable"
    except (ImportError, ValueError):
        return "unavailable"


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    return HealthResponse(
        status="healthy",
        backend="healthy",
        ollama=check_ollama_availability(),
        model=MODEL_NAME,
        playwright=check_playwright_availability(),
    )


def emit_event(task_id: str, event_type: TaskEventType, message: str, data=None):
    with _task_store_lock:
        task = tasks.get(task_id)
        if task is None:
            return None
        event = TaskEvent(
            task_id=task_id,
            event_type=event_type,
            status=task.status,
            message=sanitize_event_text(message),
            data=sanitize_event_data(data) if data is not None else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        events.setdefault(task_id, []).append(event)
        return event


def _set_task_status(task_id: str, status: TaskStatus, current_stage=None, pending_action=None):
    with _task_store_lock:
        task = tasks.get(task_id)
        if task is not None:
            updates: dict = {"status": status}
            if current_stage:
                updates["current_stage"] = str(current_stage).upper()
            if pending_action is not None:
                updates["pending_action"] = pending_action
            elif status != TaskStatus.WAITING_CONFIRMATION:
                # Clear pending_action once we leave the waiting state.
                updates["pending_action"] = None
            tasks[task_id] = task.model_copy(update=updates)


def _execute_task(task_id: str, task_request: TaskRequest):
    gate = ApprovalGate()
    with _task_store_lock:
        approval_gates[task_id] = gate

    def event_callback(event_type, status, message, data=None):
        current_stage = data.get("stage") if isinstance(data, dict) else None
        # When the workflow emits CONFIRMATION_REQUIRED, surface the action snapshot.
        pending = None
        if event_type == TaskEventType.CONFIRMATION_REQUIRED and isinstance(data, dict):
            pending = {
                "action_id": gate.action_id,
                "action": data.get("action", {}),
            }
        _set_task_status(task_id, status, current_stage, pending_action=pending)
        emit_event(task_id, event_type, message, data)

    try:
        result = run_task(task_id, task_request, event_callback, approval_gate=gate)
    except Exception as error:
        error_message = sanitize_event_text(f"{type(error).__name__}: {error}")
        event_callback(TaskEventType.TASK_FAILED, TaskStatus.FAILED, "Task failed.", {"error": error_message})
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            success=False,
            summary="Task failed.",
            error=error_message,
        )
    finally:
        gate.cancel()  # Unblock any waiting thread on error/completion.
        with _task_store_lock:
            approval_gates.pop(task_id, None)

    with _task_store_lock:
        if task_id in tasks:
            tasks[task_id] = result
    return result


@app.post("/api/tasks", response_model=TaskResponse)
def create_task(request: TaskRequest):
    if request.task_type not in SUPPORTED_TASK_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported task_type")

    task_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    task_result = TaskResult(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        success=False,
        summary="Task queued.",
    )
    task_event = TaskEvent(
        task_id=task_id,
        event_type=TaskEventType.TASK_CREATED,
        status=TaskStatus.QUEUED,
        message="Task queued.",
        timestamp=timestamp,
        data={"task_type": request.task_type},
    )

    with _task_store_lock:
        tasks[task_id] = task_result
        events[task_id] = [task_event]
        task_requests[task_id] = request

    task_executor.submit(_execute_task, task_id, request)
    return TaskResponse(task_id=task_id, status=TaskStatus.QUEUED)


@app.get("/api/tasks/{task_id}", response_model=TaskResult)
def get_task(task_id: str):
    with _task_store_lock:
        task_result = tasks.get(task_id)
    if task_result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_result


@app.get("/api/tasks/{task_id}/events", response_model=list[TaskEvent])
def get_task_events(task_id: str):
    with _task_store_lock:
        task_events = events.get(task_id)
    if task_events is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_events


@app.post("/api/tasks/{task_id}/approve", response_model=TaskResult)
def approve_action(task_id: str, body: ApprovalRequest):
    """
    Approve or reject the pending action for a task that is
    WAITING_CONFIRMATION.  The ``action_id`` in the request body must
    match the exact pending action stored on the task.
    """
    with _task_store_lock:
        task = tasks.get(task_id)
        gate = approval_gates.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.WAITING_CONFIRMATION:
        raise HTTPException(
            status_code=409,
            detail=f"Task is not waiting for approval (status: {task.status}).",
        )

    if gate is None or not gate.is_pending:
        raise HTTPException(status_code=409, detail="No pending approval gate for this task.")

    if task.pending_action is None or task.pending_action.get("action_id") != body.action_id:
        raise HTTPException(
            status_code=409,
            detail="action_id does not match the current pending action.",
        )

    try:
        gate.decide(body.action_id, body.approved)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Emit the decision event immediately (before the worker resumes).
    decision_event_type = (
        TaskEventType.APPROVAL_GRANTED if body.approved else TaskEventType.APPROVAL_REJECTED
    )
    decision_msg = (
        "User approved the pending action." if body.approved
        else "User rejected the pending action."
    )
    emit_event(task_id, decision_event_type, decision_msg, {"action_id": body.action_id})

    # Return the current task snapshot; the worker will update it shortly.
    with _task_store_lock:
        return tasks[task_id]
