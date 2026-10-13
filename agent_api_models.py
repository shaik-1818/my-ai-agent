from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    OPENING = "OPENING"
    READING = "READING"
    PLANNING = "PLANNING"
    GENERATING = "GENERATING"
    EDITING = "EDITING"
    VERIFYING = "VERIFYING"
    RUNNING = "RUNNING"
    CORRECTING = "CORRECTING"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class TaskEventType(str, Enum):
    TASK_CREATED = "TASK_CREATED"
    TASK_STARTED = "TASK_STARTED"
    BROWSER_OPENING = "BROWSER_OPENING"
    BROWSER_OPENED = "BROWSER_OPENED"
    BROWSER_READING = "BROWSER_READING"
    PAGE_READ = "PAGE_READ"
    READING_PROBLEM = "READING_PROBLEM"
    PROBLEM_READ = "PROBLEM_READ"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    PLANNING = "PLANNING"
    GENERATING_CODE = "GENERATING_CODE"
    FINDING_EDITOR = "FINDING_EDITOR"
    ENTERING_CODE = "ENTERING_CODE"
    VERIFYING_CODE = "VERIFYING_CODE"
    FINDING_RUN_CONTROL = "FINDING_RUN_CONTROL"
    RUNNING = "RUNNING"
    READING_RESULT = "READING_RESULT"
    CORRECTING = "CORRECTING"
    VERIFIED = "VERIFIED"
    READY_FOR_SUBMISSION = "READY_FOR_SUBMISSION"
    CODE_GENERATED = "CODE_GENERATED"
    EDITOR_FOUND = "EDITOR_FOUND"
    CODE_ENTERED = "CODE_ENTERED"
    RUN_STARTED = "RUN_STARTED"
    RUN_RESULT = "RUN_RESULT"
    CORRECTION_STARTED = "CORRECTION_STARTED"
    CORRECTION_COMPLETED = "CORRECTION_COMPLETED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    APPROVAL_GRANTED = "APPROVAL_GRANTED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_BLOCKED = "TASK_BLOCKED"
    TASK_CANCELLED = "TASK_CANCELLED"


class AgentAPIModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)


class TaskRequest(AgentAPIModel):
    task: str
    task_type: str = "computer"
    url: str | None = None
    require_confirmation: bool = True


class TaskResponse(AgentAPIModel):
    task_id: str
    status: TaskStatus


class TaskEvent(AgentAPIModel):
    task_id: str
    event_type: TaskEventType
    status: TaskStatus
    message: str
    data: dict[str, Any] | None = None
    timestamp: str


class TaskResult(AgentAPIModel):
    task_id: str
    status: TaskStatus
    success: bool
    summary: str
    result: dict[str, Any] | None = None
    error: str | None = None
    current_stage: str = "QUEUED"
    verified: bool = False
    blocked: bool = False
    workflow_result: dict[str, Any] | None = None
    pending_action: dict[str, Any] | None = None


class ApprovalRequest(AgentAPIModel):
    approved: bool
    action_id: str


class HealthResponse(AgentAPIModel):
    status: str
    ollama: str
    model: str
    playwright: str
    backend: str = "healthy"