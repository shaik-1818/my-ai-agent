import unittest

from pydantic import ValidationError

from agent_api_models import (
    HealthResponse,
    TaskEvent,
    TaskEventType,
    TaskRequest,
    TaskResponse,
    TaskResult,
    TaskStatus,
)


class AgentAPIModelTests(unittest.TestCase):
    def test_task_request_defaults(self):
        request = TaskRequest(task="Read this page")

        self.assertEqual(request.task, "Read this page")
        self.assertEqual(request.task_type, "computer")
        self.assertIsNone(request.url)
        self.assertTrue(request.require_confirmation)

    def test_task_response(self):
        response = TaskResponse(task_id="task-1", status=TaskStatus.QUEUED)

        self.assertEqual(response.task_id, "task-1")
        self.assertEqual(response.status, "QUEUED")

    def test_task_event(self):
        event = TaskEvent(
            task_id="task-1",
            event_type=TaskEventType.TASK_STARTED,
            status=TaskStatus.STARTING,
            message="Task started",
            timestamp="2026-10-04T12:00:00Z",
        )

        self.assertEqual(event.event_type, "TASK_STARTED")
        self.assertEqual(event.status, "STARTING")
        self.assertIsNone(event.data)

    def test_task_result(self):
        result = TaskResult(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            success=True,
            summary="Task completed",
            result={"answer": 42},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.result, {"answer": 42})
        self.assertIsNone(result.error)

    def test_health_response(self):
        health = HealthResponse(
            status="ok",
            ollama="available",
            model="qwen3:4b",
            playwright="available",
        )

        self.assertEqual(health.model, "qwen3:4b")
        self.assertEqual(health.status, "ok")

    def test_all_canonical_status_values_are_valid(self):
        for status in TaskStatus:
            with self.subTest(status=status.value):
                response = TaskResponse(task_id="task-1", status=status.value)
                self.assertEqual(response.status, status.value)

    def test_invalid_status_and_event_type_are_rejected(self):
        with self.assertRaises(ValidationError):
            TaskResponse(task_id="task-1", status="NOT_A_STATUS")

        with self.assertRaises(ValidationError):
            TaskEvent(
                task_id="task-1",
                event_type="NOT_AN_EVENT",
                status="QUEUED",
                message="Invalid event",
                timestamp="2026-10-04T12:00:00Z",
            )

    def test_required_fields_are_validated(self):
        invalid_models = (
            (TaskRequest, {}),
            (TaskResponse, {"task_id": "task-1"}),
            (TaskEvent, {"task_id": "task-1"}),
            (TaskResult, {"task_id": "task-1"}),
            (HealthResponse, {"status": "ok"}),
        )
        for model, values in invalid_models:
            with self.subTest(model=model.__name__):
                with self.assertRaises(ValidationError):
                    model(**values)


if __name__ == "__main__":
    unittest.main()