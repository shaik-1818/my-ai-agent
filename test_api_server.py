import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api_server import _task_store_lock, app, events, task_executor, task_requests, tasks
from agent_api_models import TaskEventType, TaskResult, TaskStatus


class APIServerTests(unittest.TestCase):
    def setUp(self):
        self.ollama_patch = patch("api_server.check_ollama_availability", return_value="available")
        self.playwright_patch = patch("api_server.check_playwright_availability", return_value="available")
        self.ollama_patch.start()
        self.playwright_patch.start()
        self.addCleanup(self.ollama_patch.stop)
        self.addCleanup(self.playwright_patch.stop)
        self.executor_patch = patch.object(task_executor, "submit")
        self.executor_mock = self.executor_patch.start()
        self.addCleanup(self.executor_patch.stop)
        self.client = TestClient(app)
        with _task_store_lock:
            tasks.clear()
            events.clear()
            task_requests.clear()

    def test_health_returns_200(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)

    def test_health_contains_backend_status_and_model(self):
        response = self.client.get("/api/health")
        body = response.json()

        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["backend"], "healthy")
        self.assertEqual(body["model"], "qwen3:4b")
        self.assertEqual(body["ollama"], "available")
        self.assertEqual(body["playwright"], "available")

    def test_post_task_returns_200_and_task_id(self):
        response = self.client.post("/api/tasks", json={"task": "Read a page"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["task_id"])

    def test_post_task_starts_in_queued_state(self):
        response = self.client.post("/api/tasks", json={"task": "Read a page"})

        self.assertEqual(response.json()["status"], "QUEUED")
        self.assertTrue(task_requests[response.json()["task_id"]].require_confirmation)
        self.executor_mock.assert_called_once()

    def test_background_worker_stores_result_and_lifecycle_events(self):
        def fake_run_task(task_id, task_request, event_callback, **kwargs):
            observed_stages = []
            event_callback(TaskEventType.TASK_STARTED, TaskStatus.RUNNING, "Task started.")
            event_callback(
                TaskEventType.READING_PROBLEM,
                TaskStatus.READING,
                "Reading problem.",
                {"stage": "reading_problem"},
            )
            observed_stages.append(tasks[task_id].current_stage)
            event_callback(TaskEventType.TASK_COMPLETED, TaskStatus.COMPLETED, "Task completed.")
            self.assertEqual(observed_stages, ["READING_PROBLEM"])
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                success=True,
                summary="Task completed.",
                result={"task": task_request.task},
                current_stage="COMPLETED",
                verified=True,
                workflow_result={"task": task_request.task},
            )

        with patch("api_server.run_task", side_effect=fake_run_task):
            created = self.client.post("/api/tasks", json={"task": "Test task"}).json()
            submitted_callable, *submitted_args = self.executor_mock.call_args.args
            submitted_callable(*submitted_args)

        task_id = created["task_id"]
        result_response = self.client.get(f"/api/tasks/{task_id}")
        events_response = self.client.get(f"/api/tasks/{task_id}/events")
        lifecycle = [event["status"] for event in events_response.json()]

        self.assertEqual(result_response.json()["status"], "COMPLETED")
        self.assertEqual(result_response.json()["result"], {"task": "Test task"})
        self.assertEqual(result_response.json()["workflow_result"], {"task": "Test task"})
        self.assertEqual(result_response.json()["current_stage"], "COMPLETED")
        self.assertTrue(result_response.json()["verified"])
        self.assertFalse(result_response.json()["blocked"])
        self.assertEqual(lifecycle, ["QUEUED", "RUNNING", "READING", "COMPLETED"])
        self.assertEqual(
            [event["event_type"] for event in events_response.json()],
            ["TASK_CREATED", "TASK_STARTED", "READING_PROBLEM", "TASK_COMPLETED"],
        )

    def test_task_request_returns_while_executor_owns_execution(self):
        response = self.client.post("/api/tasks", json={"task": "Long-running task"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "QUEUED")
        self.executor_mock.assert_called_once()
        self.assertEqual(tasks[response.json()["task_id"]].status, "QUEUED")

    def test_post_coding_task_with_url_is_queued(self):
        response = self.client.post(
            "/api/tasks",
            json={
                "task": "Solve the problem",
                "task_type": "coding",
                "url": "https://example.test/problem",
                "require_confirmation": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "QUEUED")
        stored = task_requests[response.json()["task_id"]]
        self.assertEqual(stored.task_type, "coding")
        self.assertEqual(stored.url, "https://example.test/problem")
        self.assertTrue(stored.require_confirmation)
        self.executor_mock.assert_called_once()

    def test_post_computer_task_remains_the_default_type(self):
        response = self.client.post("/api/tasks", json={"task": "Read a page"})

        stored = task_requests[response.json()["task_id"]]
        self.assertEqual(stored.task_type, "computer")
        created_event = events[response.json()["task_id"]][0]
        self.assertEqual(created_event.data["task_type"], "computer")

    def test_unsupported_task_type_is_rejected_without_scheduling(self):
        response = self.client.post(
            "/api/tasks",
            json={"task": "Do something", "task_type": "real_online_coding"},
        )

        self.assertEqual(response.status_code, 422)
        self.executor_mock.assert_not_called()

    def test_get_task_returns_created_task(self):
        created = self.client.post("/api/tasks", json={"task": "Read a page"}).json()
        response = self.client.get(f"/api/tasks/{created['task_id']}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task_id"], created["task_id"])
        self.assertEqual(response.json()["status"], "QUEUED")
        self.assertFalse(response.json()["success"])

    def test_get_task_events_returns_list(self):
        created = self.client.post("/api/tasks", json={"task": "Read a page"}).json()
        response = self.client.get(f"/api/tasks/{created['task_id']}/events")

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        self.assertEqual(response.json()[0]["event_type"], "TASK_CREATED")

    def test_unknown_task_returns_404(self):
        response = self.client.get("/api/tasks/unknown-task")

        self.assertEqual(response.status_code, 404)

    def test_unknown_task_events_return_404(self):
        response = self.client.get("/api/tasks/unknown-task/events")

        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------
    # Phase 2: /approve endpoint tests
    # ------------------------------------------------------------------

    def _make_waiting_task(self, action_id="test-action-id"):
        """Helper: insert a WAITING_CONFIRMATION task and a live ApprovalGate."""
        from api_server import approval_gates
        from agent_service import ApprovalGate
        from agent_api_models import TaskResult, TaskStatus

        task_id = "task-waiting"
        gate = ApprovalGate(timeout=5)
        gate.prepare({"action": "browser_click", "selector": "#login"})
        # Override the UUID so tests can use a known value.
        gate.action_id = action_id

        task = TaskResult(
            task_id=task_id,
            status=TaskStatus.WAITING_CONFIRMATION,
            success=False,
            summary="Waiting for approval.",
            pending_action={"action_id": action_id, "action": {"action": "browser_click"}},
        )
        with _task_store_lock:
            tasks[task_id] = task
            events[task_id] = []
            approval_gates[task_id] = gate
        return task_id, gate

    def test_approve_endpoint_returns_404_for_unknown_task(self):
        response = self.client.post(
            "/api/tasks/no-such-task/approve",
            json={"approved": True, "action_id": "any"},
        )
        self.assertEqual(response.status_code, 404)

    def test_approve_endpoint_returns_409_when_task_not_waiting(self):
        created = self.client.post("/api/tasks", json={"task": "Read a page"}).json()
        response = self.client.post(
            f"/api/tasks/{created['task_id']}/approve",
            json={"approved": True, "action_id": "any"},
        )
        self.assertEqual(response.status_code, 409)

    def test_approve_endpoint_returns_409_for_wrong_action_id(self):
        task_id, gate = self._make_waiting_task(action_id="correct-id")
        response = self.client.post(
            f"/api/tasks/{task_id}/approve",
            json={"approved": True, "action_id": "wrong-id"},
        )
        self.assertEqual(response.status_code, 409)
        gate.cancel()  # clean up

    def test_approve_endpoint_accepts_correct_action_id(self):
        """A correct action_id calls gate.decide() without error."""
        import threading, time
        from api_server import approval_gates
        from agent_service import ApprovalGate

        task_id, gate = self._make_waiting_task(action_id="real-id")
        # gate.prepare() was already called; gate is blocking on wait().
        # We need to call wait() in a thread so decide() can unblock it.
        wait_results = []
        def do_wait():
            wait_results.append(gate.wait())
        t = threading.Thread(target=do_wait)
        t.start()
        time.sleep(0.05)

        response = self.client.post(
            f"/api/tasks/{task_id}/approve",
            json={"approved": True, "action_id": "real-id"},
        )
        t.join(timeout=2)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(wait_results, [True])

    def test_approve_endpoint_duplicate_decision_returns_409(self):
        """A second approve call after the gate is decided returns 409."""
        import threading, time
        from api_server import approval_gates

        task_id, gate = self._make_waiting_task(action_id="dup-id")
        wait_results = []
        def do_wait():
            wait_results.append(gate.wait())
        t = threading.Thread(target=do_wait)
        t.start()
        time.sleep(0.05)

        # First call succeeds.
        self.client.post(
            f"/api/tasks/{task_id}/approve",
            json={"approved": True, "action_id": "dup-id"},
        )
        t.join(timeout=2)

        # Second call must be rejected.
        response = self.client.post(
            f"/api/tasks/{task_id}/approve",
            json={"approved": True, "action_id": "dup-id"},
        )
        self.assertEqual(response.status_code, 409)

    def test_approve_endpoint_rejection_returns_200(self):
        """Rejection (approved=False) is a valid decision and returns 200."""
        import threading, time

        task_id, gate = self._make_waiting_task(action_id="reject-id")
        wait_results = []
        def do_wait():
            wait_results.append(gate.wait())
        t = threading.Thread(target=do_wait)
        t.start()
        time.sleep(0.05)

        response = self.client.post(
            f"/api/tasks/{task_id}/approve",
            json={"approved": False, "action_id": "reject-id"},
        )
        t.join(timeout=2)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(wait_results, [False])

    def test_approve_endpoint_emits_granted_event(self):
        """Approval emits an APPROVAL_GRANTED event."""
        import threading, time
        from agent_api_models import TaskEventType

        task_id, gate = self._make_waiting_task(action_id="grant-id")
        def do_wait(): gate.wait()
        t = threading.Thread(target=do_wait)
        t.start()
        time.sleep(0.05)

        self.client.post(
            f"/api/tasks/{task_id}/approve",
            json={"approved": True, "action_id": "grant-id"},
        )
        t.join(timeout=2)

        with _task_store_lock:
            task_events = events.get(task_id, [])
        event_types = [e.event_type for e in task_events]
        self.assertIn(TaskEventType.APPROVAL_GRANTED, event_types)

    def test_approve_endpoint_emits_rejected_event(self):
        """Rejection emits an APPROVAL_REJECTED event."""
        import threading, time
        from agent_api_models import TaskEventType

        task_id, gate = self._make_waiting_task(action_id="reject-ev-id")
        def do_wait(): gate.wait()
        t = threading.Thread(target=do_wait)
        t.start()
        time.sleep(0.05)

        self.client.post(
            f"/api/tasks/{task_id}/approve",
            json={"approved": False, "action_id": "reject-ev-id"},
        )
        t.join(timeout=2)

        with _task_store_lock:
            task_events = events.get(task_id, [])
        event_types = [e.event_type for e in task_events]
        self.assertIn(TaskEventType.APPROVAL_REJECTED, event_types)

    def test_approve_endpoint_returns_409_when_no_gate(self):
        """If the gate was already removed (task finished), return 409."""
        from agent_api_models import TaskResult, TaskStatus

        task_id = "task-no-gate"
        task = TaskResult(
            task_id=task_id,
            status=TaskStatus.WAITING_CONFIRMATION,
            success=False,
            summary="Waiting.",
            pending_action={"action_id": "some-id", "action": {}},
        )
        with _task_store_lock:
            tasks[task_id] = task
            events[task_id] = []
            # Deliberately do NOT add a gate.

        response = self.client.post(
            f"/api/tasks/{task_id}/approve",
            json={"approved": True, "action_id": "some-id"},
        )
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()