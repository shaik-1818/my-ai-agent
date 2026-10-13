export type TaskStatus =
  | 'QUEUED'
  | 'STARTING'
  | 'OPENING'
  | 'READING'
  | 'PLANNING'
  | 'GENERATING'
  | 'EDITING'
  | 'VERIFYING'
  | 'RUNNING'
  | 'CORRECTING'
  | 'WAITING_CONFIRMATION'
  | 'COMPLETED'
  | 'FAILED'
  | 'BLOCKED'
  | 'CANCELLED'

export type TaskEventType = string

export type TaskType = 'computer' | 'coding'

export const READY_FOR_SUBMISSION_MESSAGE =
  'Solution prepared — awaiting your review.'

export interface TaskRequest {
  task: string
  task_type: TaskType | string
  url: string | null
  require_confirmation: boolean
}

export interface TaskResponse {
  task_id: string
  status: TaskStatus
}

export interface TaskEvent {
  task_id: string
  event_type: TaskEventType
  status: TaskStatus
  message: string
  data: Record<string, unknown> | null
  timestamp: string
}

export interface PendingActionSnapshot {
  action: 'browser_click' | 'browser_type' | string
  selector: string
  target_text?: string
  target_type?: string
  target_placeholder?: string
}

export interface PendingAction {
  action_id: string
  action: PendingActionSnapshot
}

export interface TaskResult {
  task_id: string
  status: TaskStatus
  success: boolean
  summary: string
  result: Record<string, unknown> | null
  error: string | null
  current_stage: string
  verified: boolean
  blocked: boolean
  workflow_result: Record<string, unknown> | null
  pending_action: PendingAction | null
}

function workflowFlag(result: TaskResult, key: string): unknown {
  const workflow = result.workflow_result
  if (!workflow) return undefined
  return workflow[key]
}

export function isReadyForSubmission(result: TaskResult | null | undefined): boolean {
  if (!result) return false
  if (workflowFlag(result, 'submit_clicked') === true) return false
  if (workflowFlag(result, 'submitted') === true) return false
  return (
    result.current_stage === 'READY_FOR_SUBMISSION' ||
    workflowFlag(result, 'state') === 'READY_FOR_SUBMISSION' ||
    workflowFlag(result, 'status') === 'READY_FOR_SUBMISSION' ||
    workflowFlag(result, 'ready_for_submission') === true
  )
}