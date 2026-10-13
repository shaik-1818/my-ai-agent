export type TaskStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'blocked'

export type TaskEventType = string

export interface TaskRequest {
  task: string
  task_type: string
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
}