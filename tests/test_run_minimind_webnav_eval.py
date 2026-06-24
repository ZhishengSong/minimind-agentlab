from __future__ import annotations

import unittest

from scripts.run_minimind_webnav_eval import (
    OracleFirstOpenGenerator,
    clean_completion,
    compact_messages,
    render_prompt,
    summarize,
)


class MiniMindWebNavEvalTests(unittest.TestCase):
    def test_render_prompt_preserves_tool_observation(self) -> None:
        prompt = render_prompt(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {"role": "tool", "content": "CARD(el_abc) name=item"},
            ]
        )

        self.assertIn("<tool_response>\nCARD(el_abc) name=item\n</tool_response>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_clean_completion_extracts_first_tool_block(self) -> None:
        completion = 'prefix <tool_call>{"name":"click","arguments":{"element_id":"el_abc"}}</tool_call> suffix'

        self.assertEqual(
            clean_completion(completion),
            '<tool_call>{"name":"click","arguments":{"element_id":"el_abc"}}</tool_call>',
        )

    def test_compact_messages_keeps_task_and_latest_observation(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "open"},
            {"role": "tool", "content": "large home page"},
            {"role": "assistant", "content": "filter"},
            {"role": "tool", "content": "latest candidates"},
        ]

        compacted = compact_messages(messages)

        self.assertEqual(
            [message["content"] for message in compacted],
            ["system", "task", "filter", "latest candidates"],
        )
        observation_only = compact_messages(messages, include_latest_action=False)
        self.assertEqual(
            [message["content"] for message in observation_only],
            ["system", "task", "latest candidates"],
        )

    def test_summary_groups_v2_templates(self) -> None:
        trajectories = [
            {
                "template": "v2_shopping_name",
                "summary": {
                    "success": True,
                    "termination": "submitted",
                    "invalid_tool_calls": 0,
                    "format_errors": 0,
                    "model_steps": 3,
                },
            },
            {
                "template": "v2_shopping_name",
                "summary": {
                    "success": False,
                    "termination": "max_steps",
                    "invalid_tool_calls": 2,
                    "format_errors": 1,
                    "model_steps": 8,
                },
            },
        ]

        report = summarize(trajectories)

        self.assertEqual(report["success"], 1)
        self.assertEqual(report["submitted"], 1)
        self.assertEqual(report["invalid_tool_calls"], 2)
        self.assertEqual(report["format_errors"], 1)
        self.assertEqual(report["by_template"]["v2_shopping_name"], {"total": 2, "success": 1})

    def test_oracle_first_open_delegates_after_one_action(self) -> None:
        delegated_messages = []

        def delegate(messages: list[dict[str, str]]) -> str:
            delegated_messages.append(messages)
            return "model output"

        generator = OracleFirstOpenGenerator(delegate, "v2_eval_c_shop_home")

        first = generator([])
        second = generator([{"role": "tool", "content": "page"}])

        self.assertEqual(
            first,
            '<tool_call>{"name": "open_page", "arguments": {"page_id": "v2_eval_c_shop_home"}}</tool_call>',
        )
        self.assertEqual(second, "model output")
        self.assertEqual(len(delegated_messages), 1)


if __name__ == "__main__":
    unittest.main()
