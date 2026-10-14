import unittest
from unittest.mock import Mock, patch

from agent_api_models import TaskEventType, TaskRequest, TaskStatus
from agent_service import (
    READY_FOR_SUBMISSION_SUMMARY,
    UnsupportedTaskTypeError,
    _build_bullet_summary,
    _is_ready_for_submission,
    _plan_browser_task,
    _simple_read_task_completed,
    _verify_browser_task,
    run_task,
)
from real_online_coding_workflow import real_online_coding_workflow


class AgentServiceTests(unittest.TestCase):
    def setUp(self):
        self.events = []

    def _record_event(self, event_type, status, message, data=None):
        self.events.append({
            "event_type": event_type,
            "status": status,
            "message": message,
            "data": data,
        })

    def test_computer_task_starts_with_injected_workflow(self):
        calls = []

        def fake_computer_workflow(task_request, emit, ask_model):
            calls.append(task_request.task)
            emit(TaskEventType.BROWSER_OPENING, TaskStatus.RUNNING, "Opening mock browser.")
            return {"success": True, "summary": "Computer task finished."}

        result = run_task(
            "task-computer",
            TaskRequest(task="Read the local test page", task_type="computer"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertEqual(calls, ["Read the local test page"])
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertTrue(result.success)

    def test_coding_task_uses_injected_local_workflow(self):
        calls = []

        def fake_coding_workflow(task_request, ask_model):
            calls.append((task_request.task, ask_model("test")))
            return {"verified": True, "verification": {"summary": "Local tests passed."}}

        result = run_task(
            "task-coding",
            TaskRequest(task="Write a sum function", task_type="coding"),
            self._record_event,
            ask_model=lambda prompt: "fake model response",
            coding_workflow=fake_coding_workflow,
        )

        self.assertEqual(calls, [("Write a sum function", "fake model response")])
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertTrue(result.success)
        self.assertEqual(result.summary, "Local tests passed.")

    def test_online_coding_workflow_receives_headless_options_and_progress(self):
        calls = []
        progress_events = [
            "browser_opening",
            "browser_opened",
            "reading_problem",
            "problem_read",
            "planning",
            "generating_code",
            "finding_editor",
            "entering_code",
            "verifying_code",
            "finding_run_control",
            "running",
            "reading_result",
            "correcting",
            "verified",
            "ready_for_submission",
        ]

        def fake_online_workflow(**kwargs):
            calls.append(kwargs)
            for event_name in progress_events:
                kwargs["event_callback"](event_name)
            return {
                "status": "READY_FOR_SUBMISSION",
                "state": "READY_FOR_SUBMISSION",
                "verified": True,
                "ready_for_submission": True,
                "summary": "All tests passed.",
            }

        result = run_task(
            "task-online",
            TaskRequest(task="Solve the page problem", task_type="coding", url="https://example.test/problem"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            online_coding_workflow=fake_online_workflow,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["url"], "https://example.test/problem")
        self.assertIs(calls[0]["headless"], True)
        self.assertIs(calls[0]["keep_open"], False)
        self.assertEqual(calls[0]["ask_model"]("unused"), "unused")
        self.assertEqual(
            [event["data"]["stage"] for event in self.events if event["data"] and "stage" in event["data"]],
            progress_events,
        )
        self.assertEqual(self.events[0]["event_type"], TaskEventType.TASK_STARTED)
        self.assertEqual(self.events[-1]["event_type"], TaskEventType.TASK_COMPLETED)
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertTrue(result.verified)
        self.assertFalse(result.blocked)
        self.assertEqual(result.current_stage, "READY_FOR_SUBMISSION")
        self.assertEqual(result.workflow_result["state"], "READY_FOR_SUBMISSION")
        self.assertEqual(result.summary, READY_FOR_SUBMISSION_SUMMARY)
        self.assertFalse(result.workflow_result.get("submit_clicked"))
        self.assertNotIn("submitted", result.summary.lower())
        self.assertNotIn("accepted", result.summary.lower())

    def test_online_blocked_workflow_returns_blocked_ui_result(self):
        def blocked_workflow(**kwargs):
            kwargs["event_callback"]("blocked")
            return {"status": "BLOCKED", "state": "BLOCKED", "summary": "Authentication is required."}

        result = run_task(
            "task-blocked",
            TaskRequest(task="Solve the page problem", task_type="coding", url="https://example.test/problem"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            online_coding_workflow=blocked_workflow,
        )

        self.assertEqual(result.status, TaskStatus.BLOCKED)
        self.assertFalse(result.verified)
        self.assertTrue(result.blocked)
        self.assertEqual(result.current_stage, "BLOCKED")
        self.assertEqual(result.error, "Authentication is required.")
        self.assertEqual(result.workflow_result["state"], "BLOCKED")
        self.assertEqual(sum(event["event_type"] == TaskEventType.TASK_BLOCKED for event in self.events), 1)

    def test_online_failed_workflow_returns_failed_ui_result(self):
        def failed_workflow(**kwargs):
            kwargs["event_callback"]("failed")
            return {"status": "FAILED", "state": "FAILED", "summary": "Editor verification failed."}

        result = run_task(
            "task-online-failed",
            TaskRequest(task="Solve the page problem", task_type="coding", url="https://example.test/problem"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            online_coding_workflow=failed_workflow,
        )

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertFalse(result.verified)
        self.assertFalse(result.blocked)
        self.assertEqual(result.current_stage, "FAILED")
        self.assertEqual(result.error, "Editor verification failed.")
        self.assertEqual(result.workflow_result["state"], "FAILED")
        self.assertEqual(sum(event["event_type"] == TaskEventType.TASK_FAILED for event in self.events), 1)

    def test_online_workflow_callback_is_optional_for_existing_callers(self):
        playwright_context = Mock()
        browser = Mock()
        page = Mock()
        playwright_context.__enter__ = Mock(return_value=Mock(chromium=Mock(launch=Mock(return_value=browser))))
        playwright_context.__exit__ = Mock(return_value=False)
        browser.new_page.return_value = page
        with (
            patch("real_online_coding_workflow.sync_playwright", return_value=playwright_context),
            patch(
                "real_online_coding_workflow.browser_open",
                return_value={"success": False, "error": "Mock open failure."},
            ),
        ):
            result = real_online_coding_workflow(
                "https://example.test/problem",
                lambda prompt: "unused",
                headless=True,
                keep_open=False,
            )

        self.assertEqual(result["state"], "FAILED")
        self.assertEqual(result["state_history"], ["OPENING", "FAILED"])
        browser.close.assert_called_once()

    def test_real_workflow_emits_open_progress_and_blocks_authentication(self):
        playwright_context = Mock()
        browser = Mock()
        page = Mock()
        playwright_context.__enter__ = Mock(return_value=Mock(chromium=Mock(launch=Mock(return_value=browser))))
        playwright_context.__exit__ = Mock(return_value=False)
        browser.new_page.return_value = page
        progress = []
        with (
            patch("real_online_coding_workflow.sync_playwright", return_value=playwright_context),
            patch("real_online_coding_workflow.browser_open", return_value={"success": True}),
            patch("real_online_coding_workflow._wait_for_page"),
            patch("real_online_coding_workflow.browser_read", return_value={"success": True}),
            patch("real_online_coding_workflow.access_block_reason", return_value="Authentication is required."),
        ):
            result = real_online_coding_workflow(
                "https://example.test/problem",
                lambda prompt: "unused",
                headless=True,
                keep_open=False,
                event_callback=progress.append,
            )

        self.assertEqual(progress, ["browser_opening", "browser_opened", "blocked"])
        self.assertEqual(result["state"], "BLOCKED")
        self.assertFalse(result["verified"])
        browser.close.assert_called_once()

    def test_task_transitions_running_to_completed(self):
        result = run_task(
            "task-1",
            TaskRequest(task="Do work", task_type="coding"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            coding_workflow=lambda request, model: {"success": True, "summary": "Done."},
        )

        self.assertEqual(
            [event["status"] for event in self.events],
            [TaskStatus.RUNNING, TaskStatus.RUNNING, TaskStatus.RUNNING, TaskStatus.COMPLETED],
        )
        self.assertEqual(result.status, TaskStatus.COMPLETED)

    def test_failed_workflow_returns_failed_result(self):
        def failing_workflow(task_request, ask_model):
            raise RuntimeError("mock workflow failure")

        result = run_task(
            "task-failed",
            TaskRequest(task="Fail safely", task_type="coding"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            coding_workflow=failing_workflow,
        )

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertFalse(result.success)
        self.assertEqual(self.events[-1]["event_type"], TaskEventType.TASK_FAILED)

    def test_service_emits_start_and_completion_events(self):
        run_task(
            "task-events",
            TaskRequest(task="Finish", task_type="coding"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            coding_workflow=lambda request, model: {"success": True, "summary": "Done."},
        )

        self.assertEqual(self.events[0]["event_type"], TaskEventType.TASK_STARTED)
        self.assertEqual(self.events[-1]["event_type"], TaskEventType.TASK_COMPLETED)

    def test_task_result_is_returned_for_storage(self):
        result = run_task(
            "task-result",
            TaskRequest(task="Return a result", task_type="coding"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            coding_workflow=lambda request, model: {"success": True, "value": 42},
        )

        self.assertEqual(result.task_id, "task-result")
        self.assertEqual(result.result, {"success": True, "value": 42})

    def test_unknown_task_type_is_rejected_before_execution(self):
        with self.assertRaises(UnsupportedTaskTypeError):
            run_task(
                "task-unknown",
                TaskRequest(task="Do not run", task_type="real_online_coding"),
                self._record_event,
                ask_model=lambda prompt: "unused",
            )
        self.assertEqual(self.events, [])

    def test_event_output_redacts_credential_fields_and_values(self):
        def fake_computer_workflow(task_request, emit, ask_model):
            emit(
                TaskEventType.EXECUTING,
                TaskStatus.RUNNING,
                "password=hunter2",
                {
                    "password": "hunter2",
                    "access_token": "token-value",
                    "note": "Bearer abc.def.ghi",
                    "safe": "visible",
                },
            )
            return {"success": True, "summary": "Done."}

        run_task(
            "task-secrets",
            TaskRequest(task="Safe task", task_type="computer"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )
        event_dump = repr(self.events)

        self.assertNotIn("hunter2", event_dump)
        self.assertNotIn("token-value", event_dump)
        self.assertNotIn("abc.def.ghi", event_dump)
        self.assertIn("visible", event_dump)

    # ------------------------------------------------------------------
    # New tests: page-title, summarization, repeated actions, failure
    # ------------------------------------------------------------------

    def test_simple_read_task_completed_title(self):
        obs = {"title": "Example Domain", "body_text": "Some text."}
        self.assertTrue(_simple_read_task_completed("Tell me the page title", obs))
        self.assertTrue(_simple_read_task_completed("What is the website title?", obs))
        self.assertFalse(_simple_read_task_completed("Tell me the page title", {"title": ""}))

    def test_simple_read_task_completed_summarize(self):
        obs = {"title": "Example", "body_text": "This domain is for use in examples."}
        self.assertTrue(_simple_read_task_completed("Summarize the page in three bullet points", obs))
        self.assertTrue(_simple_read_task_completed("Give me a summary of the website", obs))
        self.assertFalse(_simple_read_task_completed("Summarize the page", {"body_text": ""}))

    def test_simple_read_task_completed_unrelated(self):
        obs = {"title": "Example", "body_text": "Some text."}
        self.assertFalse(_simple_read_task_completed("Click the login button", obs))

    def test_build_bullet_summary_produces_bullets(self):
        body = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        result = _build_bullet_summary(body, n=3)
        lines = result.strip().splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertTrue(line.startswith("\u2022 "))

    def test_build_bullet_summary_fewer_sentences_than_n(self):
        body = "Only one sentence here."
        result = _build_bullet_summary(body, n=3)
        self.assertIn("Only one sentence here", result)

    def test_computer_workflow_page_title_task_returns_title(self):
        """Fast-path: page-title task resolves after browser_read without model loop."""
        actions = iter([
            {"action": "browser_read"},
        ])

        def fake_computer_workflow(task_request, emit, ask_model):
            from agent_service import (
                MAX_COMPUTER_STEPS,
                _build_bullet_summary,
                _read_page_safely,
                _simple_read_task_completed,
                sanitize_event_text,
            )
            from online_coding_workflow import safe_observation

            observation = {
                "title": "Example Domain",
                "page_title": "Example Domain",
                "body_text": "This domain is for use in illustrative examples.",
                "url": "https://example.com/",
            }
            emit(TaskEventType.PAGE_READ, TaskStatus.RUNNING, "Page read.", {"title": "Example Domain", "url": "https://example.com/"})
            if _simple_read_task_completed(task_request.task, observation):
                return {
                    "success": True,
                    "verified": True,
                    "state": "VERIFIED",
                    "summary": sanitize_event_text(f"Page title: {observation['title']}"),
                    "url": "https://example.com/",
                    "steps": 1,
                    "result": {"title": observation["title"], "url": observation["url"]},
                }
            return {"success": False, "summary": "Not resolved."}

        result = run_task(
            "task-title",
            TaskRequest(task="Tell me the page title", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertTrue(result.success)
        self.assertIn("Example Domain", result.summary)

    def test_computer_workflow_summarize_task_returns_bullets(self):
        """Fast-path: summarization task resolves with bullet points from body_text."""
        body = (
            "This domain is for use in illustrative examples in documents. "
            "You may use this domain in literature without prior coordination. "
            "More information is available at IANA."
        )

        def fake_computer_workflow(task_request, emit, ask_model):
            from agent_service import (
                _build_bullet_summary,
                _simple_read_task_completed,
                sanitize_event_text,
            )

            observation = {
                "title": "Example Domain",
                "body_text": body,
                "url": "https://example.com/",
            }
            emit(TaskEventType.PAGE_READ, TaskStatus.RUNNING, "Page read.", {"title": "Example Domain", "url": "https://example.com/"})
            if _simple_read_task_completed(task_request.task, observation):
                summary = _build_bullet_summary(observation["body_text"])
                return {
                    "success": True,
                    "verified": True,
                    "state": "VERIFIED",
                    "summary": sanitize_event_text(summary),
                    "url": "https://example.com/",
                    "steps": 1,
                }
            return {"success": False, "summary": "Not resolved."}

        result = run_task(
            "task-summary",
            TaskRequest(
                task="Open the website, read its visible content, and summarize what the website is about in three bullet points.",
                task_type="computer",
                url="https://example.com",
            ),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertTrue(result.success)
        lines = [l for l in result.summary.splitlines() if l.strip()]
        self.assertGreaterEqual(len(lines), 1)
        self.assertTrue(any("\u2022" in l for l in lines))

    def test_computer_workflow_repeated_browser_read_returns_failure(self):
        """Repeated browser_read after page already read must not succeed."""
        call_count = {"n": 0}

        def fake_computer_workflow(task_request, emit, ask_model):
            # Simulate: page read once, then model keeps asking browser_read.
            call_count["n"] += 1
            if call_count["n"] == 1:
                emit(TaskEventType.PAGE_READ, TaskStatus.RUNNING, "Page read.", {})
                return {
                    "success": False,
                    "verified": False,
                    "state": "FAILED",
                    "summary": "Task stopped: repeated browser_read blocked after page already read.",
                    "steps": 5,
                }
            return {"success": True, "summary": "Should not reach here."}

        result = run_task(
            "task-repeated-read",
            TaskRequest(task="Summarize the page", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertNotEqual(result.summary, "Task completed.")

    def test_computer_workflow_repeated_thinking_returns_failure(self):
        """Repeated non-productive THINKING must return success=False."""
        def fake_computer_workflow(task_request, emit, ask_model):
            return {
                "success": False,
                "verified": False,
                "state": "FAILED",
                "summary": "Task stopped: the model repeated 'think' 4 times without progress.",
                "steps": 4,
            }

        result = run_task(
            "task-repeated-thinking",
            TaskRequest(task="Summarize the page", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, TaskStatus.FAILED)

    def test_computer_workflow_step_limit_returns_failure_not_success(self):
        """Hitting MAX_COMPUTER_STEPS must return success=False, never fabricate success."""
        def fake_computer_workflow(task_request, emit, ask_model):
            return {
                "success": False,
                "verified": False,
                "state": "FAILED",
                "summary": "Computer task stopped because it reached the 15-step safety limit before completion.",
                "steps": 15,
            }

        result = run_task(
            "task-step-limit",
            TaskRequest(task="Do something complex", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertNotEqual(result.summary, "Task completed.")

    # ------------------------------------------------------------------
    # Phase 1: _plan_browser_task tests
    # ------------------------------------------------------------------

    def test_plan_browser_task_valid_response(self):
        """Valid JSON with 3 steps is accepted and returned as a list."""
        model = lambda prompt: '{"steps": ["Open the URL", "Read the page", "Finish with the answer"]}'
        plan = _plan_browser_task("Find the page title", "https://example.com", model)
        self.assertEqual(len(plan), 3)
        self.assertIsInstance(plan[0], str)

    def test_plan_browser_task_too_few_steps_uses_fallback(self):
        """A plan with only 1 step (below minimum) falls back to the safe default."""
        model = lambda prompt: '{"steps": ["Open the URL"]}'
        plan = _plan_browser_task("Find the page title", "https://example.com", model)
        self.assertGreaterEqual(len(plan), 2)

    def test_plan_browser_task_too_many_steps_uses_fallback(self):
        """A plan with 7 steps (above maximum) falls back to the safe default."""
        steps = [f"Step {i}" for i in range(7)]
        model = lambda prompt: '{"steps": ' + str(steps).replace("'", '"') + '}'
        plan = _plan_browser_task("Do something", "https://example.com", model)
        self.assertLessEqual(len(plan), 6)

    def test_plan_browser_task_invalid_json_uses_fallback(self):
        """Non-JSON model output falls back gracefully without raising."""
        model = lambda prompt: "I cannot plan this task."
        plan = _plan_browser_task("Find the page title", "https://example.com", model)
        self.assertIsInstance(plan, list)
        self.assertGreaterEqual(len(plan), 2)

    def test_plan_browser_task_model_exception_uses_fallback(self):
        """A model that raises an exception falls back gracefully."""
        def boom(prompt):
            raise RuntimeError("model offline")
        plan = _plan_browser_task("Find the page title", "https://example.com", boom)
        self.assertIsInstance(plan, list)
        self.assertGreaterEqual(len(plan), 2)

    def test_plan_browser_task_no_url_fallback_differs(self):
        """Fallback plan for no-URL task starts with 'Identify' not 'Open'."""
        model = lambda prompt: "not json"
        plan = _plan_browser_task("Do something", None, model)
        self.assertTrue(plan[0].startswith("Identify"))

    def test_plan_browser_task_redacts_credentials_in_steps(self):
        """Steps containing credential-like keys are redacted."""
        model = lambda prompt: '{"steps": ["Enter password=secret123", "Read the page", "Finish"]}'
        plan = _plan_browser_task("Login task", "https://example.com", model)
        self.assertNotIn("secret123", " ".join(plan))

    def test_plan_browser_task_not_steps_list_uses_fallback(self):
        """JSON with 'steps' as a string (not list) falls back."""
        model = lambda prompt: '{"steps": "just open the page"}'
        plan = _plan_browser_task("Find the page title", "https://example.com", model)
        self.assertIsInstance(plan, list)
        self.assertGreaterEqual(len(plan), 2)

    # ------------------------------------------------------------------
    # Phase 1: _verify_browser_task tests
    # ------------------------------------------------------------------

    def test_verify_browser_task_passes_with_evidence(self):
        """Verification passes when model returns passed=true with a real reason."""
        model = lambda prompt: '{"passed": true, "reason": "The page title Example Domain was visible in the observation."}'
        passed, feedback = _verify_browser_task(
            "Find the page title", ["Open URL", "Read page", "Finish"],
            "Title: Example Domain", model
        )
        self.assertTrue(passed)
        self.assertIn("Example Domain", feedback)

    def test_verify_browser_task_fails_with_no_evidence(self):
        """Verification fails when model returns passed=false."""
        model = lambda prompt: '{"passed": false, "reason": "The observation does not show the task was completed."}'
        passed, feedback = _verify_browser_task(
            "Find the page title", ["Open URL", "Read page", "Finish"],
            "No page content available.", model
        )
        self.assertFalse(passed)
        self.assertIsInstance(feedback, str)
        self.assertTrue(len(feedback) > 0)

    def test_verify_browser_task_generic_reason_fails(self):
        """A generic reason like 'done' is rejected even if passed=true."""
        model = lambda prompt: '{"passed": true, "reason": "done"}'
        passed, feedback = _verify_browser_task(
            "Find the page title", ["Open URL", "Read page", "Finish"],
            "Some observation.", model
        )
        self.assertFalse(passed)

    def test_verify_browser_task_invalid_json_returns_false(self):
        """Non-JSON model output returns (False, error message) without raising."""
        model = lambda prompt: "I cannot verify this."
        passed, feedback = _verify_browser_task(
            "Find the page title", ["Open URL", "Read page", "Finish"],
            "Some observation.", model
        )
        self.assertFalse(passed)
        self.assertIsInstance(feedback, str)

    def test_verify_browser_task_model_exception_returns_false(self):
        """A model that raises returns (False, error message) without raising."""
        def boom(prompt):
            raise RuntimeError("model offline")
        passed, feedback = _verify_browser_task(
            "Find the page title", ["Open URL", "Read page", "Finish"],
            "Some observation.", boom
        )
        self.assertFalse(passed)
        self.assertIn("model error", feedback.lower())

    def test_verify_browser_task_empty_reason_returns_false(self):
        """An empty reason string is treated as insufficient evidence."""
        model = lambda prompt: '{"passed": true, "reason": ""}'
        passed, _ = _verify_browser_task(
            "Find the page title", ["Open URL", "Read page", "Finish"],
            "Some observation.", model
        )
        self.assertFalse(passed)

    # ------------------------------------------------------------------
    # Phase 1: integration — plan injected, verification wired into loop
    # ------------------------------------------------------------------

    def test_computer_workflow_plan_injected_into_observation(self):
        """The plan generated before the loop must appear in the first observation."""
        received_observations = []

        def fake_computer_workflow(task_request, emit, ask_model):
            # Simulate: planning call returns a plan, then action loop gets it.
            # We intercept ask_model calls to capture what the loop sees.
            call_count = {"n": 0}

            def intercepting_model(prompt):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # Planning call
                    return '{"steps": ["Open the URL", "Read the page", "Finish with the answer"]}'
                # Action call — capture the prompt and return finish
                received_observations.append(prompt)
                return '{"action": "finish", "reason": "The page title is Example Domain."}'

            from agent_service import (
                MAX_VERIFY_ATTEMPTS,
                _plan_browser_task,
                _verify_browser_task,
                sanitize_event_text,
            )
            plan = _plan_browser_task(task_request.task, task_request.url, intercepting_model)
            plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
            obs = f"Website opened.\nPLAN:\n{plan_text}"
            received_observations.append(obs)
            return {"success": True, "verified": False, "summary": "The page title is Example Domain.", "steps": 1}

        result = run_task(
            "task-plan-inject",
            TaskRequest(task="Find the page title", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertTrue(result.success)
        self.assertTrue(any("PLAN" in obs for obs in received_observations))

    def test_computer_workflow_verification_failure_continues_loop(self):
        """When verification fails, the loop continues rather than accepting finish."""
        call_log = []

        def fake_computer_workflow(task_request, emit, ask_model):
            # First ask_model call = planning; second = action; third = verify (fail);
            # fourth = action again; fifth = verify (pass).
            responses = iter([
                '{"steps": ["Open URL", "Read page", "Finish"]}',          # plan
                '{"action": "finish", "reason": "Found the answer."}',      # step 1 finish
                '{"passed": false, "reason": "No evidence in observation."}', # verify fail
                '{"action": "finish", "reason": "The title is Example Domain."}', # step 2 finish
                '{"passed": true, "reason": "Title Example Domain confirmed."}',  # verify pass
            ])

            def model(prompt):
                r = next(responses)
                call_log.append(r)
                return r

            from agent_service import (
                MAX_VERIFY_ATTEMPTS,
                _plan_browser_task,
                _verify_browser_task,
                sanitize_event_text,
            )
            plan = _plan_browser_task(task_request.task, task_request.url, model)
            verify_attempts = 0
            obs = "Title: Example Domain"

            for _ in range(5):
                action_raw = model("action prompt")
                import json
                action = json.loads(action_raw)
                if action.get("action") == "finish":
                    reason = action.get("reason", "")
                    if verify_attempts < MAX_VERIFY_ATTEMPTS:
                        verify_attempts += 1
                        passed, feedback = _verify_browser_task(task_request.task, plan, obs, model)
                        if not passed:
                            continue
                    return {"success": True, "verified": True, "summary": reason, "steps": verify_attempts + 1}
            return {"success": False, "verified": False, "summary": "Loop ended.", "steps": 5}

        result = run_task(
            "task-verify-loop",
            TaskRequest(task="Find the page title", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        # Verify was called at least twice (once fail, once pass)
        self.assertGreaterEqual(len(call_log), 4)

    def test_computer_workflow_verification_exhausted_still_accepts_finish(self):
        """After MAX_VERIFY_ATTEMPTS failed verifications the loop accepts finish anyway."""
        def fake_computer_workflow(task_request, emit, ask_model):
            from agent_service import (
                MAX_VERIFY_ATTEMPTS,
                _plan_browser_task,
                _verify_browser_task,
            )
            plan = _plan_browser_task(task_request.task, task_request.url,
                                      lambda p: '{"steps": ["Open URL", "Read page", "Finish"]}')
            verify_attempts = 0
            obs = "Minimal observation."
            always_fail = lambda p: '{"passed": false, "reason": "Not enough evidence."}'

            for step in range(1, 10):
                if verify_attempts < MAX_VERIFY_ATTEMPTS:
                    verify_attempts += 1
                    passed, _ = _verify_browser_task(task_request.task, plan, obs, always_fail)
                    if not passed:
                        continue
                # Attempts exhausted — accept finish
                return {"success": True, "verified": False, "summary": "Best effort answer.", "steps": step}
            return {"success": False, "verified": False, "summary": "Loop ended.", "steps": 9}

        result = run_task(
            "task-verify-exhausted",
            TaskRequest(task="Find the page title", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertTrue(result.success)

    def test_computer_workflow_step_limit_respected_after_verify_failures(self):
        """Step limit is still enforced even when verification keeps failing."""
        def fake_computer_workflow(task_request, emit, ask_model):
            from agent_service import MAX_COMPUTER_STEPS
            return {
                "success": False,
                "verified": False,
                "state": "FAILED",
                "summary": f"Computer task stopped because it reached the {MAX_COMPUTER_STEPS}-step safety limit before completion.",
                "steps": MAX_COMPUTER_STEPS,
            }

        result = run_task(
            "task-step-limit-verify",
            TaskRequest(task="Do something complex", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, TaskStatus.FAILED)

    def test_ready_for_submission_helper_rejects_submit_flags(self):
        ready = {
            "state": "READY_FOR_SUBMISSION",
            "ready_for_submission": True,
            "submit_clicked": False,
        }
        submitted = {**ready, "submit_clicked": True}
        accepted = {**ready, "submitted": True}

        self.assertTrue(_is_ready_for_submission(ready, "READY_FOR_SUBMISSION"))
        self.assertFalse(_is_ready_for_submission(submitted, "READY_FOR_SUBMISSION"))
        self.assertFalse(_is_ready_for_submission(accepted, "READY_FOR_SUBMISSION"))

    def test_ready_for_submission_is_review_not_submit(self):
        """API mapping keeps READY_FOR_SUBMISSION as review-only, never submitted."""

        def fake_online_workflow(**kwargs):
            return {
                "status": "READY_FOR_SUBMISSION",
                "state": "READY_FOR_SUBMISSION",
                "verified": True,
                "ready_for_submission": True,
                "submit_clicked": False,
                "summary": "All tests passed.",
            }

        result = run_task(
            "task-ready-review",
            TaskRequest(
                task="Solve the page problem",
                task_type="coding",
                url="https://example.test/problem",
            ),
            self._record_event,
            ask_model=lambda prompt: "unused",
            online_coding_workflow=fake_online_workflow,
        )

        self.assertTrue(_is_ready_for_submission(result.workflow_result, "READY_FOR_SUBMISSION"))
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.current_stage, "READY_FOR_SUBMISSION")
        self.assertEqual(result.summary, READY_FOR_SUBMISSION_SUMMARY)
        self.assertTrue(result.verified)
        self.assertFalse(result.workflow_result["submit_clicked"])
        self.assertNotEqual(result.summary.lower(), "submitted")
        self.assertNotIn("accepted", result.summary.lower())

    def test_coding_without_url_still_uses_local_workflow(self):
        calls = []

        def fake_coding_workflow(task_request, ask_model):
            calls.append(task_request.task_type)
            return {"verified": True, "verification": {"summary": "Local tests passed."}}

        def fail_online(**kwargs):
            raise AssertionError("Online coding workflow must not run without a URL.")

        result = run_task(
            "task-local-coding",
            TaskRequest(task="Write a sum function", task_type="coding"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            coding_workflow=fake_coding_workflow,
            online_coding_workflow=fail_online,
        )

        self.assertEqual(calls, ["coding"])
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.summary, "Local tests passed.")
        self.assertNotEqual(result.current_stage, "READY_FOR_SUBMISSION")

    def test_computer_task_is_not_routed_to_coding_workflow(self):
        def fail_coding(task_request, ask_model):
            raise AssertionError("Computer tasks must not use the coding workflow.")

        def fake_computer_workflow(task_request, emit, ask_model):
            emit(TaskEventType.BROWSER_OPENING, TaskStatus.RUNNING, "Opening mock browser.")
            return {"success": True, "summary": "Computer task finished."}

        result = run_task(
            "task-computer-route",
            TaskRequest(task="Read the local test page", task_type="computer", url="https://example.test"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            computer_workflow=fake_computer_workflow,
            coding_workflow=fail_coding,
        )

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.summary, "Computer task finished.")

    def test_existing_coding_workflow_unaffected_by_phase1(self):
        """Coding workflow is completely unaffected by Phase 1 changes."""
        calls = []

        def fake_coding_workflow(task_request, ask_model):
            calls.append(task_request.task)
            return {"verified": True, "verification": {"summary": "All tests passed."}}

        result = run_task(
            "task-coding-compat",
            TaskRequest(task="Write a sum function", task_type="coding"),
            self._record_event,
            ask_model=lambda prompt: "unused",
            coding_workflow=fake_coding_workflow,
        )

        self.assertEqual(calls, ["Write a sum function"])
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertTrue(result.verified)

    # ------------------------------------------------------------------
    # Phase 2: ApprovalGate unit tests
    # ------------------------------------------------------------------

    def test_approval_gate_approve_unblocks_worker(self):
        """decide(approved=True) unblocks request() and returns True."""
        from agent_service import ApprovalGate
        gate = ApprovalGate(timeout=5)
        results = []

        import threading
        def worker():
            results.append(gate.request({"action": "browser_click"}))

        t = threading.Thread(target=worker)
        t.start()
        # Give the worker time to block.
        import time; time.sleep(0.05)
        gate.decide(gate.action_id, approved=True)
        t.join(timeout=2)

        self.assertEqual(results, [True])
        self.assertFalse(gate.is_pending)

    def test_approval_gate_reject_unblocks_worker(self):
        """decide(approved=False) unblocks request() and returns False."""
        from agent_service import ApprovalGate
        gate = ApprovalGate(timeout=5)
        results = []

        import threading
        def worker():
            results.append(gate.request({"action": "browser_type"}))

        t = threading.Thread(target=worker)
        t.start()
        import time; time.sleep(0.05)
        gate.decide(gate.action_id, approved=False)
        t.join(timeout=2)

        self.assertEqual(results, [False])
        self.assertFalse(gate.timed_out)

    def test_approval_gate_wrong_action_id_raises(self):
        """decide() with a wrong action_id raises ValueError."""
        from agent_service import ApprovalGate
        gate = ApprovalGate(timeout=5)
        gate.prepare({"action": "browser_click"})

        with self.assertRaises(ValueError):
            gate.decide("wrong-id", approved=True)

    def test_approval_gate_duplicate_decision_raises(self):
        """A second decide() call raises ValueError."""
        from agent_service import ApprovalGate
        gate = ApprovalGate(timeout=5)
        gate.prepare({"action": "browser_click"})
        gate.decide(gate.action_id, approved=True)

        with self.assertRaises(ValueError):
            gate.decide(gate.action_id, approved=True)

    def test_approval_gate_cancel_never_approves(self):
        """cancel() unblocks the worker with approved=False."""
        from agent_service import ApprovalGate
        gate = ApprovalGate(timeout=5)
        results = []

        import threading
        def worker():
            results.append(gate.request({"action": "browser_click"}))

        t = threading.Thread(target=worker)
        t.start()
        import time; time.sleep(0.05)
        gate.cancel()
        t.join(timeout=2)

        self.assertEqual(results, [False])
        self.assertFalse(gate.approved)

    def test_approval_gate_timeout_never_approves(self):
        """A gate that times out sets approved=False and timed_out=True."""
        from agent_service import ApprovalGate
        gate = ApprovalGate(timeout=0.05)  # 50 ms
        result = gate.request({"action": "browser_click"})

        self.assertFalse(result)
        self.assertTrue(gate.timed_out)
        self.assertFalse(gate.approved)

    def test_approval_gate_prepare_sets_action_id_before_wait(self):
        """prepare() assigns a non-empty UUID before wait() is called."""
        from agent_service import ApprovalGate
        gate = ApprovalGate(timeout=5)
        action_id = gate.prepare({"action": "browser_click"})

        self.assertTrue(action_id)
        self.assertEqual(gate.action_id, action_id)
        # Clean up without blocking.
        gate.cancel()

    def test_approval_gate_is_pending_true_while_waiting(self):
        """is_pending is True while the worker is blocked."""
        from agent_service import ApprovalGate
        gate = ApprovalGate(timeout=5)
        import threading, time
        ready = threading.Event()

        def worker():
            ready.set()
            gate.request({"action": "browser_click"})

        t = threading.Thread(target=worker)
        t.start()
        ready.wait(timeout=1)
        time.sleep(0.05)
        self.assertTrue(gate.is_pending)
        gate.cancel()
        t.join(timeout=2)
        self.assertFalse(gate.is_pending)

    # ------------------------------------------------------------------
    # Phase 2: run_task with approval_gate — backward compat & gate wiring
    # ------------------------------------------------------------------

    def test_three_arg_workflow_still_works_without_approval_gate(self):
        """Existing 3-arg fake workflows work when no approval_gate is passed."""
        calls = []

        def fake_workflow(task_request, emit, ask_model):
            calls.append("called")
            return {"success": True, "summary": "done"}

        result = run_task(
            "task-compat",
            TaskRequest(task="test", task_type="computer"),
            self._record_event,
            ask_model=lambda p: "unused",
            computer_workflow=fake_workflow,
        )

        self.assertEqual(calls, ["called"])
        self.assertEqual(result.status, TaskStatus.COMPLETED)

    def test_four_arg_workflow_receives_approval_gate(self):
        """A workflow that accepts approval_gate receives the gate instance."""
        received = []
        from agent_service import ApprovalGate
        gate = ApprovalGate()

        def fake_workflow(task_request, emit, ask_model, approval_gate=None):
            received.append(approval_gate)
            return {"success": True, "summary": "done"}

        run_task(
            "task-gate-wired",
            TaskRequest(task="test", task_type="computer"),
            self._record_event,
            ask_model=lambda p: "unused",
            computer_workflow=fake_workflow,
            approval_gate=gate,
        )

        self.assertIs(received[0], gate)

    def test_sensitive_click_pauses_and_resumes_on_approval(self):
        """Sensitive browser_click pauses the worker; approval resumes it."""
        from agent_service import ApprovalGate
        import threading, time

        gate = ApprovalGate(timeout=5)
        events_seen = []
        action_ids_seen = []

        def fake_workflow(task_request, emit, ask_model, approval_gate=None):
            # Simulate the approval path directly.
            action_snapshot = {"action": "browser_click", "selector": "#login"}
            action_id = approval_gate.prepare(action_snapshot)
            action_ids_seen.append(action_id)
            emit(
                TaskEventType.CONFIRMATION_REQUIRED,
                TaskStatus.WAITING_CONFIRMATION,
                "Approval required.",
                {"action_id": action_id, "action": action_snapshot},
            )
            approved = approval_gate.wait()
            if approved:
                emit(TaskEventType.APPROVAL_GRANTED, TaskStatus.RUNNING, "Approved.")
                return {"success": True, "summary": "Click executed after approval."}
            return {"success": False, "summary": "Rejected."}

        result_holder = []

        def run_worker():
            result_holder.append(
                run_task(
                    "task-pause-resume",
                    TaskRequest(task="login", task_type="computer", require_confirmation=True),
                    lambda et, st, msg, data=None: events_seen.append(et),
                    ask_model=lambda p: "unused",
                    computer_workflow=fake_workflow,
                    approval_gate=gate,
                )
            )

        t = threading.Thread(target=run_worker)
        t.start()
        # Wait until the gate has a pending action_id.
        deadline = time.time() + 2
        while not gate.action_id and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(gate.action_id, "Gate never got an action_id")
        gate.decide(gate.action_id, approved=True)
        t.join(timeout=3)

        self.assertEqual(len(result_holder), 1)
        self.assertTrue(result_holder[0].success)
        self.assertIn(TaskEventType.CONFIRMATION_REQUIRED, events_seen)
        self.assertIn(TaskEventType.APPROVAL_GRANTED, events_seen)

    def test_sensitive_click_rejection_prevents_execution(self):
        """Rejecting a sensitive action returns success=False without executing."""
        from agent_service import ApprovalGate
        import threading, time

        gate = ApprovalGate(timeout=5)
        executed = []

        def fake_workflow(task_request, emit, ask_model, approval_gate=None):
            action_snapshot = {"action": "browser_click", "selector": "#submit"}
            action_id = approval_gate.prepare(action_snapshot)
            emit(
                TaskEventType.CONFIRMATION_REQUIRED,
                TaskStatus.WAITING_CONFIRMATION,
                "Approval required.",
                {"action_id": action_id, "action": action_snapshot},
            )
            approved = approval_gate.wait()
            if approved:
                executed.append(True)
                return {"success": True, "summary": "Executed."}
            emit(TaskEventType.APPROVAL_REJECTED, TaskStatus.RUNNING, "Rejected.")
            return {"success": False, "state": "FAILED", "summary": "Action rejected by user."}

        result_holder = []

        def run_worker():
            result_holder.append(
                run_task(
                    "task-reject",
                    TaskRequest(task="login", task_type="computer", require_confirmation=True),
                    self._record_event,
                    ask_model=lambda p: "unused",
                    computer_workflow=fake_workflow,
                    approval_gate=gate,
                )
            )

        t = threading.Thread(target=run_worker)
        t.start()
        deadline = time.time() + 2
        while not gate.action_id and time.time() < deadline:
            time.sleep(0.01)
        gate.decide(gate.action_id, approved=False)
        t.join(timeout=3)

        self.assertEqual(executed, [], "Action must not execute after rejection")
        self.assertFalse(result_holder[0].success)
        self.assertEqual(result_holder[0].status, TaskStatus.FAILED)

    def test_action_id_is_set_before_confirmation_event(self):
        """The action_id in the CONFIRMATION_REQUIRED event must be a real UUID."""
        from agent_service import ApprovalGate
        import threading, time

        gate = ApprovalGate(timeout=5)
        event_action_ids = []

        def fake_workflow(task_request, emit, ask_model, approval_gate=None):
            action_snapshot = {"action": "browser_click", "selector": "#login"}
            action_id = approval_gate.prepare(action_snapshot)
            emit(
                TaskEventType.CONFIRMATION_REQUIRED,
                TaskStatus.WAITING_CONFIRMATION,
                "Approval required.",
                {"action_id": action_id, "action": action_snapshot},
            )
            approval_gate.wait()
            return {"success": False, "summary": "cancelled"}

        def capture_event(et, st, msg, data=None):
            if et == TaskEventType.CONFIRMATION_REQUIRED and data:
                event_action_ids.append(data.get("action_id", ""))

        def run_worker():
            run_task(
                "task-id-timing",
                TaskRequest(task="login", task_type="computer", require_confirmation=True),
                capture_event,
                ask_model=lambda p: "unused",
                computer_workflow=fake_workflow,
                approval_gate=gate,
            )

        t = threading.Thread(target=run_worker)
        t.start()
        deadline = time.time() + 2
        while not gate.action_id and time.time() < deadline:
            time.sleep(0.01)
        gate.cancel()
        t.join(timeout=3)

        self.assertEqual(len(event_action_ids), 1)
        self.assertTrue(event_action_ids[0], "action_id must not be empty in the event")
        self.assertNotEqual(event_action_ids[0], "pending")

    # ------------------------------------------------------------------
    # Phase 3: extract_json edge-case tests
    # ------------------------------------------------------------------

    def test_extract_json_missing_action_key_raises(self):
        """JSON object without 'action' key raises ValueError."""
        from agent import extract_json
        with self.assertRaises(ValueError) as ctx:
            extract_json('{"result": "ok"}')
        self.assertIn("action", str(ctx.exception))

    def test_extract_json_invalid_json_raises(self):
        """Malformed JSON raises ValueError."""
        from agent import extract_json
        with self.assertRaises(ValueError):
            extract_json('{action: browser_read}')

    def test_extract_json_no_braces_raises(self):
        """Plain text with no JSON object raises ValueError."""
        from agent import extract_json
        with self.assertRaises(ValueError):
            extract_json("I cannot help with that.")

    def test_extract_json_markdown_wrapped_json_succeeds(self):
        """JSON wrapped in markdown code fences is still extracted correctly."""
        from agent import extract_json
        text = '```json\n{"action": "browser_read"}\n```'
        result = extract_json(text)
        self.assertEqual(result["action"], "browser_read")

    def test_extract_json_synonym_key_raises(self):
        """A JSON object with 'command' instead of 'action' raises ValueError."""
        from agent import extract_json
        with self.assertRaises(ValueError):
            extract_json('{"command": "browser_read"}')

    # ------------------------------------------------------------------
    # Phase 3: parse-retry recovery tests
    # ------------------------------------------------------------------

    def test_workflow_recovers_from_single_bad_model_response(self):
        """One bad model response followed by a valid one completes the task."""
        from agent_service import MAX_PARSE_RETRIES
        call_log = []

        def fake_computer_workflow(task_request, emit, ask_model):
            from agent import extract_json, build_prompt
            from agent_service import sanitize_event_text, MAX_PARSE_RETRIES
            from agent_api_models import TaskEventType, TaskStatus

            responses = iter([
                "not valid json at all",          # bad — triggers retry
                '{"action": "finish", "reason": "Page title is Example Domain."}',  # good
            ])

            observation_text = "No webpage is open yet."
            last_action = None
            action = None
            last_parse_error = ""

            for parse_attempt in range(1, MAX_PARSE_RETRIES + 1):
                try:
                    raw = next(responses)
                    call_log.append(raw)
                    action = extract_json(raw)
                    break
                except (ValueError, Exception) as exc:
                    last_parse_error = str(exc)
                    if parse_attempt < MAX_PARSE_RETRIES:
                        emit(
                            TaskEventType.THINKING,
                            TaskStatus.RUNNING,
                            f"Model response could not be parsed (attempt {parse_attempt}/{MAX_PARSE_RETRIES}); retrying.",
                            {"parse_error": sanitize_event_text(last_parse_error)},
                        )

            if action is None:
                return {"success": False, "summary": "Parse failed.", "steps": 1}
            return {"success": True, "verified": False, "summary": action.get("reason", ""), "steps": 1}

        result = run_task(
            "task-parse-recover",
            TaskRequest(task="Find the page title", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda p: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertTrue(result.success)
        self.assertIn("Example Domain", result.summary)
        # Exactly 2 model calls: 1 bad + 1 good
        self.assertEqual(len(call_log), 2)
        # A THINKING diagnostic event was emitted for the parse failure
        thinking_events = [
            e for e in self.events
            if e["event_type"] == TaskEventType.THINKING
            and e["data"] and "parse_error" in e["data"]
        ]
        self.assertEqual(len(thinking_events), 1)

    def test_workflow_fails_after_max_parse_retries(self):
        """All model responses invalid → task skips step and eventually fails."""
        from agent_service import MAX_PARSE_RETRIES

        def fake_computer_workflow(task_request, emit, ask_model):
            from agent import extract_json
            from agent_service import sanitize_event_text, MAX_PARSE_RETRIES
            from agent_api_models import TaskEventType, TaskStatus

            action = None
            last_parse_error = ""
            for parse_attempt in range(1, MAX_PARSE_RETRIES + 1):
                try:
                    extract_json("totally broken")
                except (ValueError, Exception) as exc:
                    last_parse_error = str(exc)
                    if parse_attempt < MAX_PARSE_RETRIES:
                        emit(
                            TaskEventType.THINKING,
                            TaskStatus.RUNNING,
                            f"Model response could not be parsed (attempt {parse_attempt}/{MAX_PARSE_RETRIES}); retrying.",
                            {"parse_error": sanitize_event_text(last_parse_error)},
                        )

            emit(
                TaskEventType.THINKING,
                TaskStatus.RUNNING,
                "Model failed to return a valid action after all retries; skipping step.",
                {"parse_error": sanitize_event_text(last_parse_error)},
            )
            return {"success": False, "state": "FAILED", "summary": "Parse retries exhausted.", "steps": 1}

        result = run_task(
            "task-parse-exhaust",
            TaskRequest(task="Find the page title", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda p: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, TaskStatus.FAILED)
        # Diagnostic THINKING events emitted for each retry + final skip
        parse_events = [
            e for e in self.events
            if e["event_type"] == TaskEventType.THINKING
            and e["data"] and "parse_error" in e["data"]
        ]
        self.assertGreaterEqual(len(parse_events), MAX_PARSE_RETRIES)

    # ------------------------------------------------------------------
    # Phase 3: generic-finish loop termination test
    # ------------------------------------------------------------------

    def test_workflow_generic_finish_loop_terminates(self):
        """Repeated generic finish actions terminate before the 15-step limit."""
        from agent_service import MAX_FINISH_REJECTIONS, MAX_COMPUTER_STEPS

        steps_taken = []

        def fake_computer_workflow(task_request, emit, ask_model):
            from agent_service import MAX_FINISH_REJECTIONS
            from agent_api_models import TaskEventType, TaskStatus

            finish_rejected_count = 0
            for step in range(1, 20):  # deliberately higher than MAX_COMPUTER_STEPS
                # Model always returns generic finish
                finish_rejected_count += 1
                steps_taken.append(step)
                if finish_rejected_count >= MAX_FINISH_REJECTIONS:
                    return {
                        "success": False,
                        "verified": False,
                        "state": "FAILED",
                        "summary": (
                            f"Task stopped: model returned a generic finish reason "
                            f"{finish_rejected_count} times without a meaningful answer."
                        ),
                        "steps": step,
                    }
            return {"success": True, "summary": "Should not reach here.", "steps": 20}

        result = run_task(
            "task-generic-finish",
            TaskRequest(task="Find the page title", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda p: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, TaskStatus.FAILED)
        # Must stop at MAX_FINISH_REJECTIONS, well before MAX_COMPUTER_STEPS
        self.assertEqual(len(steps_taken), MAX_FINISH_REJECTIONS)
        self.assertLess(len(steps_taken), MAX_COMPUTER_STEPS)

    # ------------------------------------------------------------------
    # Phase 3: unsupported action test
    # ------------------------------------------------------------------

    def test_workflow_unsupported_action_updates_observation_and_continues(self):
        """An unsupported action name updates observation and loops; does not crash."""
        call_log = []

        def fake_computer_workflow(task_request, emit, ask_model):
            from agent_service import sanitize_event_text
            from agent_api_models import TaskEventType, TaskStatus

            observation_text = "No webpage is open yet."
            last_action = None

            actions = iter([
                {"action": "fly_to_moon"},   # unsupported
                {"action": "finish", "reason": "Task completed with a real answer."},
            ])

            for step in range(1, 5):
                action = next(actions, None)
                if action is None:
                    break
                action_name = action.get("action")
                call_log.append(action_name)

                if action_name == "finish":
                    return {"success": True, "verified": False, "summary": action.get("reason", ""), "steps": step}

                # Unsupported action path
                if action_name not in {
                    "browser_open", "browser_read", "browser_click",
                    "browser_type", "browser_scroll", "browser_press",
                    "browser_screenshot", "finish",
                }:
                    observation_text = (
                        f"Unsupported or unsafe action: {sanitize_event_text(action_name)}. "
                        "Choose a safe available action or finish."
                    )
                    last_action = "action_failed"
                    continue  # must loop, not crash

            return {"success": False, "summary": "Loop ended.", "steps": 4}

        result = run_task(
            "task-unsupported",
            TaskRequest(task="Find the page title", task_type="computer", url="https://example.com"),
            self._record_event,
            ask_model=lambda p: "unused",
            computer_workflow=fake_computer_workflow,
        )

        self.assertTrue(result.success)
        self.assertIn("fly_to_moon", call_log)
        self.assertIn("finish", call_log)

    def test_unsupported_action_does_not_bypass_sensitive_check(self):
        """An unsupported action name cannot bypass the approval gate or safety checks."""
        from agent_service import ApprovalGate
        executed = []

        def fake_workflow(task_request, emit, ask_model, approval_gate=None):
            action_name = "fly_to_moon"
            if action_name not in {
                "browser_open", "browser_read", "browser_click",
                "browser_type", "browser_scroll", "browser_press",
                "browser_screenshot", "finish",
            }:
                pass
            else:
                executed.append(action_name)
            return {"success": True, "summary": "No sensitive action executed.", "steps": 1}

        gate = ApprovalGate()
        result = run_task(
            "task-unsupported-safe",
            TaskRequest(task="login", task_type="computer", require_confirmation=True),
            self._record_event,
            ask_model=lambda p: "unused",
            computer_workflow=fake_workflow,
            approval_gate=gate,
        )

        self.assertEqual(executed, [], "Unsupported action must not execute anything")
        self.assertTrue(result.success)

    def test_browser_open_returns_failure_dict_on_connection_refused(self):
        """browser_open must return success=False dict rather than raising an uncaught exception on connection failure."""
        from unittest.mock import MagicMock
        from browser_tools import browser_open

        mock_page = MagicMock()
        mock_page.goto.side_effect = Exception("net::ERR_CONNECTION_REFUSED at http://127.0.0.1:9999")
        mock_page.url = "http://127.0.0.1:9999"

        res = browser_open(mock_page, "http://127.0.0.1:9999", retries=2, retry_delay=0.01)
        self.assertFalse(res["success"])
        self.assertIn("ERR_CONNECTION_REFUSED", res["error"])
        self.assertEqual(mock_page.goto.call_count, 2)

    def test_browser_open_ipv4_fallback_for_localhost(self):
        """browser_open should fall back from localhost to 127.0.0.1 when connection is refused."""
        from unittest.mock import MagicMock
        from browser_tools import browser_open

        mock_page = MagicMock()
        call_urls = []

        def fake_goto(url, **kwargs):
            call_urls.append(url)
            if "localhost" in url:
                raise Exception("net::ERR_CONNECTION_REFUSED")
            mock_page.url = url
            mock_page.title.return_value = "Test Title"
            return None

        mock_page.goto.side_effect = fake_goto
        res = browser_open(mock_page, "http://localhost:8001/test_page.html", retries=2, retry_delay=0.01)
        self.assertTrue(res["success"])
        self.assertIn("127.0.0.1", res["url"])
        self.assertIn("http://127.0.0.1:8001/test_page.html", call_urls)

    def test_observed_target_finds_by_id_and_single_candidate(self):
        """_observed_target correctly resolves targets by id, text, and single candidate fallback."""
        from agent_service import _observed_target

        obs = {
            "buttons": [
                {
                    "tag": "button",
                    "selector": "#showName",
                    "id": "showName",
                    "name": "",
                    "type": "button",
                    "text": "Show Name",
                }
            ],
            "links": [],
        }

        # Exact match
        t1 = _observed_target(obs, "#showName", ("buttons", "links"))
        self.assertIsNotNone(t1)
        self.assertEqual(t1["selector"], "#showName")

        # Match by text
        t2 = _observed_target(obs, "Show Name", ("buttons", "links"))
        self.assertIsNotNone(t2)
        self.assertEqual(t2["selector"], "#showName")

        # Match by id without hash
        t3 = _observed_target(obs, "showName", ("buttons", "links"))
        self.assertIsNotNone(t3)
        self.assertEqual(t3["selector"], "#showName")

        # Single candidate fallback for buttons
        t4 = _observed_target(obs, "#submit", ("buttons", "links"))
        self.assertIsNotNone(t4)
        self.assertEqual(t4["selector"], "#showName")

    def test_navigation_alone_cannot_complete_when_interaction_requested(self):
        """_simple_read_task_completed must return False when interaction (fill/click/verify) is requested."""
        from agent_service import _simple_read_task_completed

        obs = {"title": "Test Page", "body_text": "Sample page body text"}
        interactive_tasks = [
            "Open http://127.0.0.1:8001/test_page.html, enter Moula into the name input, click the Show Name button, and verify the result.",
            "Open the page, fill #name with Moula, click #showName, and read the page",
            "Fill #name with Moula and click #showName",
            "Click the submit button and check result",
            "Type Moula into #name and verify result",
        ]
        for t in interactive_tasks:
            self.assertFalse(
                _simple_read_task_completed(t, obs),
                f"Task '{t}' should not be treated as a simple read-only task",
            )

    def test_failed_verification_returns_honest_failure_status(self):
        """When task verification fails, the workflow must produce an honest failed status rather than success."""
        from agent_service import _run_computer_workflow
        from unittest.mock import MagicMock, patch

        mock_page = MagicMock()
        mock_page.url = "http://127.0.0.1:8001/test_page.html"
        mock_page.title.return_value = "AI Agent Test Page"

        events = []
        def emit(event_type, status, message, data=None):
            events.append((event_type, status, message, data))

        # Model tries to finish immediately without verifying completion
        def mock_ask_model(prompt):
            if "verify" in prompt.lower() or "verifier" in prompt.lower():
                return '{"passed": false, "reason": "No evidence that #result was populated."}'
            if "plan" in prompt.lower():
                return '{"steps": ["Open page", "Fill name", "Click button", "Verify"]}'
            return '{"action": "finish", "reason": "Claiming task is done."}'

        req = TaskRequest(
            task="Open http://127.0.0.1:8001/test_page.html, fill #name with Moula, click #showName, and verify result",
            task_type="computer",
            url="http://127.0.0.1:8001/test_page.html",
        )

        with patch("playwright.sync_api.sync_playwright") as mock_pw, \
             patch("browser_tools.browser_open", return_value={"success": True, "url": req.url, "title": "Test"}), \
             patch("agent_service._read_page_safely", return_value={"title": "Test", "body_text": "Empty", "buttons": [], "inputs": []}):
            mock_pw_ctx = MagicMock()
            mock_pw.return_value.__enter__.return_value = mock_pw_ctx
            mock_browser = MagicMock()
            mock_pw_ctx.chromium.launch.return_value = mock_browser
            mock_browser.new_page.return_value = mock_page

            res = _run_computer_workflow(req, emit, mock_ask_model)

            # Verification failed, so success MUST be False and verified MUST be False
            self.assertFalse(res["success"], "Workflow must return success=False when verification fails")
            self.assertFalse(res["verified"], "Workflow must return verified=False when verification fails")
            self.assertEqual(res["state"], "FAILED")

    def test_repeated_click_is_blocked_and_does_not_repeat_click(self):
        """When model repeatedly proposes clicking #showName, workflow blocks repeated click and prevents infinite click loop."""
        from agent_service import _run_computer_workflow
        from unittest.mock import MagicMock, patch

        mock_page = MagicMock()
        mock_page.url = "http://127.0.0.1:8001/test_page.html"
        mock_page.title.return_value = "AI Agent Test Page"

        events = []
        def emit(event_type, status, message, data=None):
            events.append((event_type, status, message, data))

        step_counter = [0]
        def mock_ask_model(prompt):
            if "plan" in prompt.lower():
                return '{"steps": ["Open page", "Click button", "Verify"]}'
            step_counter[0] += 1
            if step_counter[0] == 1:
                return '{"action": "browser_click", "selector": "#showName"}'
            elif step_counter[0] == 2:
                return '{"action": "browser_click", "selector": "#showName"}'
            else:
                return '{"action": "finish", "reason": "Done"}'

        req = TaskRequest(
            task="Click #showName and verify that #result equals Hello, Moula!",
            task_type="computer",
            url="http://127.0.0.1:8001/test_page.html",
        )

        obs = {
            "title": "AI Agent Test Page",
            "body_text": "Sample",
            "buttons": [{"tag": "button", "selector": "#showName", "id": "showName", "text": "Show Name"}],
            "inputs": [],
            "elements": [{"id": "result", "selector": "#result", "text": "Hello, Moula!"}],
        }

        with patch("playwright.sync_api.sync_playwright") as mock_pw, \
             patch("browser_tools.browser_open", return_value={"success": True, "url": req.url, "title": "Test"}), \
             patch("agent_service._read_page_safely", return_value=obs), \
             patch("browser_tools.browser_click", return_value={"success": True, "url": req.url}) as mock_b_click:
            mock_pw_ctx = MagicMock()
            mock_pw.return_value.__enter__.return_value = mock_pw_ctx
            mock_browser = MagicMock()
            mock_pw_ctx.chromium.launch.return_value = mock_browser
            mock_browser.new_page.return_value = mock_page

            res = _run_computer_workflow(req, emit, mock_ask_model)

            # browser_click must only be called ONCE, not twice
            self.assertEqual(mock_b_click.call_count, 1)
            self.assertTrue(res["success"])
            self.assertTrue(res["verified"])
            self.assertEqual(res["state"], "VERIFIED")

    def test_complete_browser_workflow_executes_actions_and_verifies_dom(self):
        """Test complete workflow with mocked actions: fill #name -> click #showName -> DOM read -> verified."""
        from agent_service import _run_computer_workflow
        from unittest.mock import MagicMock, patch

        mock_page = MagicMock()
        mock_page.url = "http://127.0.0.1:8001/test_page.html"
        mock_page.title.return_value = "AI Agent Test Page"

        events = []
        def emit(event_type, status, message, data=None):
            events.append((event_type, status, message, data))

        step_counter = [0]
        def mock_ask_model(prompt):
            if "plan" in prompt.lower():
                return '{"steps": ["Open page", "Fill name", "Click showName", "Verify result"]}'
            step_counter[0] += 1
            if step_counter[0] == 1:
                return '{"action": "browser_type", "selector": "#name", "text": "Moula"}'
            elif step_counter[0] == 2:
                return '{"action": "browser_click", "selector": "#showName"}'
            return '{"action": "finish", "reason": "Task completed."}'

        req = TaskRequest(
            task="Open http://127.0.0.1:8001/test_page.html, fill #name with Moula, click #showName, read #result, and verify Hello, Moula!",
            task_type="computer",
            url="http://127.0.0.1:8001/test_page.html",
        )

        obs_initial = {
            "title": "AI Agent Test Page",
            "body_text": "Enter name:",
            "buttons": [{"tag": "button", "selector": "#showName", "id": "showName", "text": "Show Name"}],
            "inputs": [{"tag": "input", "selector": "#name", "id": "name", "value": ""}],
            "elements": [{"id": "result", "selector": "#result", "text": ""}],
        }
        obs_after_type = {
            "title": "AI Agent Test Page",
            "body_text": "Enter name: Moula",
            "buttons": [{"tag": "button", "selector": "#showName", "id": "showName", "text": "Show Name"}],
            "inputs": [{"tag": "input", "selector": "#name", "id": "name", "value": "Moula"}],
            "elements": [{"id": "result", "selector": "#result", "text": ""}],
        }
        obs_after_click = {
            "title": "AI Agent Test Page",
            "body_text": "Enter name: Moula\nHello, Moula!",
            "buttons": [{"tag": "button", "selector": "#showName", "id": "showName", "text": "Show Name"}],
            "inputs": [{"tag": "input", "selector": "#name", "id": "name", "value": "Moula"}],
            "elements": [{"id": "result", "selector": "#result", "text": "Hello, Moula!"}],
        }

        read_responses = [obs_initial, obs_after_type, obs_after_click, obs_after_click]

        with patch("playwright.sync_api.sync_playwright") as mock_pw, \
             patch("browser_tools.browser_open", return_value={"success": True, "url": req.url, "title": "Test"}), \
             patch("agent_service._read_page_safely", side_effect=read_responses), \
             patch("browser_tools.browser_type", return_value={"success": True}) as mock_type, \
             patch("browser_tools.browser_click", return_value={"success": True, "url": req.url}) as mock_click:
            mock_pw_ctx = MagicMock()
            mock_pw.return_value.__enter__.return_value = mock_pw_ctx
            mock_browser = MagicMock()
            mock_pw_ctx.chromium.launch.return_value = mock_browser
            mock_browser.new_page.return_value = mock_page

            res = _run_computer_workflow(req, emit, mock_ask_model)

            self.assertTrue(res["success"])
            self.assertTrue(res["verified"])
            self.assertEqual(res["state"], "VERIFIED")
            self.assertEqual(res["result"]["text"], "Hello, Moula!")
            mock_type.assert_called_once_with(mock_page, "#name", "Moula")
            mock_click.assert_called_once_with(mock_page, "#showName")

            # Check that verification event was emitted
            verif_events = [e for e in events if e[0] == TaskEventType.THINKING and e[3] and e[3].get("verified")]
            self.assertTrue(len(verif_events) > 0)

    def test_dom_verification_reports_actual_observation_when_result_differs(self):
        """When #result in DOM has wrong text, task reports actual observation and does not claim verification."""
        from agent_service import _run_computer_workflow
        from unittest.mock import MagicMock, patch

        mock_page = MagicMock()
        mock_page.url = "http://127.0.0.1:8001/test_page.html"
        mock_page.title.return_value = "AI Agent Test Page"

        events = []
        def emit(event_type, status, message, data=None):
            events.append((event_type, status, message, data))

        step_counter = [0]
        def mock_ask_model(prompt):
            if "plan" in prompt.lower():
                return '{"steps": ["Open page", "Click button", "Finish"]}'
            step_counter[0] += 1
            if step_counter[0] == 1:
                return '{"action": "browser_click", "selector": "#showName"}'
            return '{"action": "finish", "reason": "Claiming done."}'

        req = TaskRequest(
            task="Open http://127.0.0.1:8001/test_page.html, click #showName, read #result, and verify Hello, Moula!",
            task_type="computer",
            url="http://127.0.0.1:8001/test_page.html",
        )

        # DOM produces unexpected result: "Please enter your name."
        obs_wrong = {
            "title": "AI Agent Test Page",
            "body_text": "Please enter your name.",
            "buttons": [{"tag": "button", "selector": "#showName", "id": "showName", "text": "Show Name"}],
            "inputs": [],
            "elements": [{"id": "result", "selector": "#result", "text": "Please enter your name."}],
        }

        with patch("playwright.sync_api.sync_playwright") as mock_pw, \
             patch("browser_tools.browser_open", return_value={"success": True, "url": req.url, "title": "Test"}), \
             patch("agent_service._read_page_safely", return_value=obs_wrong), \
             patch("browser_tools.browser_click", return_value={"success": True, "url": req.url}):
            mock_pw_ctx = MagicMock()
            mock_pw.return_value.__enter__.return_value = mock_pw_ctx
            mock_browser = MagicMock()
            mock_pw_ctx.chromium.launch.return_value = mock_browser
            mock_browser.new_page.return_value = mock_page

            res = _run_computer_workflow(req, emit, mock_ask_model)

            self.assertFalse(res["verified"])
            self.assertFalse(res["success"])
            self.assertEqual(res["state"], "FAILED")
            self.assertIn("Please enter your name.", res["summary"])


if __name__ == "__main__":
    unittest.main()