import type {
  TaskEvent,
  TaskRequest,
  TaskResponse,
  TaskResult,
} from './types';

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let message = `API request failed: ${response.status}`;

    try {
      const body = await response.json();
      if (body?.detail) {
        message = `${message} - ${body.detail}`;
      }
    } catch {
      // Keep the default error message.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export function createTask(
  requestData: TaskRequest,
): Promise<TaskResponse> {
  return request<TaskResponse>('/api/tasks', {
    method: 'POST',
    body: JSON.stringify(requestData),
  });
}

export function getTask(
  taskId: string,
): Promise<TaskResult> {
  return request<TaskResult>(
    `/api/tasks/${encodeURIComponent(taskId)}`,
  );
}

export function getTaskEvents(
  taskId: string,
): Promise<TaskEvent[]> {
  return request<TaskEvent[]>(
    `/api/tasks/${encodeURIComponent(taskId)}/events`,
  );
}

/**
 * POST /api/tasks/{taskId}/approve
 *
 * Sends an approval or rejection decision for the pending action.
 * The action_id must exactly match the pending_action.action_id on the task.
 * Returns the updated TaskResult snapshot.
 * Throws on any HTTP error (including 409 for duplicate/wrong action_id).
 */
export function approveAction(
  taskId: string,
  approved: boolean,
  actionId: string,
): Promise<TaskResult> {
  return request<TaskResult>(
    `/api/tasks/${encodeURIComponent(taskId)}/approve`,
    {
      method: 'POST',
      body: JSON.stringify({ approved, action_id: actionId }),
    },
  );
}