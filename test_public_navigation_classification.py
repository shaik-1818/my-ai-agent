import unittest

from real_online_read_workflow import classify_public_navigation


class PublicNavigationClassificationTests(unittest.TestCase):
    def test_leetcode_problem_description_redirect_is_allowed(self):
        requested = "https://leetcode.com/problems/two-sum/"
        final = "https://leetcode.com/problems/two-sum/description/"

        result = classify_public_navigation(requested, final, {
            "page_title": "Two Sum",
            "body_text": "Given an array of integers, return indices that add to a target.",
            "inputs": [],
        })

        self.assertEqual(result["status"], "CONTINUE")
        self.assertIsNone(result["blocker"])
        self.assertTrue(result["redirected"])
        self.assertEqual(result["requested_url"], requested)
        self.assertEqual(result["final_url"], final)

    def test_same_host_path_redirect_is_allowed(self):
        result = classify_public_navigation(
            "https://example.com/problem/start",
            "https://example.com/problem/description",
            {"title": "Coding problem", "body_text": "Given n, return n + 1.", "inputs": []},
        )

        self.assertEqual(result["status"], "CONTINUE")
        self.assertTrue(result["redirected"])

    def test_accepted_subdomain_redirect_is_allowed(self):
        result = classify_public_navigation(
            "https://example.com/problem/1",
            "https://www.example.com/problem/1",
            {"title": "Problem", "body_text": "Given n, return n.", "inputs": []},
        )

        self.assertEqual(result["status"], "CONTINUE")
        self.assertIsNone(result["blocker"])

    def test_cross_origin_redirect_is_unexpected_not_security_block(self):
        result = classify_public_navigation(
            "https://example.com/problem/1",
            "https://unrelated.example.net/problem/1",
            {"title": "Problem", "body_text": "Given n, return n.", "inputs": []},
        )

        self.assertEqual(result["status"], "UNEXPECTED_NAVIGATION")
        self.assertEqual(result["blocker"], "cross_origin_navigation")

    def test_login_requirement_returns_auth_required(self):
        result = classify_public_navigation(
            "https://example.com/problem/1",
            "https://example.com/login",
            {
                "title": "Sign in to continue",
                "body_text": "Please sign in to continue.",
                "inputs": [{"type": "password"}],
            },
        )

        self.assertEqual(result["status"], "AUTH_REQUIRED")
        self.assertEqual(result["blocker"], "login_required")

    def test_login_path_redirect_returns_auth_required(self):
        result = classify_public_navigation(
            "https://example.com/problem/1",
            "https://example.com/login?next=%2Fproblem%2F1",
            {"title": "Welcome", "body_text": "", "inputs": []},
        )

        self.assertEqual(result["status"], "AUTH_REQUIRED")
        self.assertEqual(result["blocker"], "login_required")

    def test_captcha_returns_security_challenge(self):
        result = classify_public_navigation(
            "https://example.com/problem/1",
            "https://example.com/problem/1",
            {"title": "Verify", "body_text": "Please complete the CAPTCHA.", "inputs": []},
        )

        self.assertEqual(result["status"], "BLOCKED_SECURITY_CHALLENGE")
        self.assertEqual(result["blocker"], "captcha")

    def test_access_denied_returns_security_block(self):
        result = classify_public_navigation(
            "https://example.com/problem/1",
            "https://example.com/problem/1",
            {"title": "Forbidden", "body_text": "Access denied.", "inputs": []},
        )

        self.assertEqual(result["status"], "BLOCKED_SECURITY_CHALLENGE")
        self.assertIn(result["blocker"], {"access_denied", "security_challenge"})

    def test_valid_coding_problem_content_after_redirect_can_continue(self):
        result = classify_public_navigation(
            "https://leetcode.com/problems/two-sum/",
            "https://leetcode.com/problems/two-sum/description/",
            {
                "title": "Two Sum",
                "body_text": "Given nums and target, return indices of two numbers that sum to target. Example: [2,7], 9 -> [0,1].",
                "inputs": [],
            },
        )

        self.assertEqual(result["status"], "CONTINUE")
        self.assertTrue(result["redirected"])

    def test_requested_and_final_urls_are_always_preserved(self):
        requested = "https://example.com/problem/"
        final = "https://www.example.com/problem/description"
        result = classify_public_navigation(
            requested,
            final,
            {"title": "Problem", "body_text": "Statement", "inputs": []},
        )

        self.assertEqual(result["requested_url"], requested)
        self.assertEqual(result["final_url"], final)
        self.assertTrue(result["redirected"])
        self.assertIn("blocker", result)


if __name__ == "__main__":
    unittest.main()
