"""
guardrail.py

这个文件用于封装 Day 5 的最小输入护栏。

今天只做一件事：

防止职业数字人泄露隐私信息。

比如：
- 电话
- 手机号
- 邮箱
- email

如果用户要求查看、输出、发送这类信息，
系统会先拦截，不进入 agent_core.py 的正常主流程。

注意：
- 这是教学版最小 guardrail。
- fallback 和 handoff 今天先不展开，放到拓展内容。
"""

from typing import Dict

import src.agent_core as agent_core


def input_guardrail(user_input: str) -> Dict[str, str]:
    """
    最小输入护栏。

    判断用户输入是否涉及隐私信息泄露。

    返回：
    {
        "allowed": True / False,
        "reason": "原因说明"
    }
    """

    text = user_input.strip().lower()

    privacy_keywords = [
        "电话",
        "手机号",
        "手机号码",
        "联系方式",
        "邮箱",
        "email",
        "e-mail",
        "mail",
        "发给我他的电话",
        "发给我她的电话",
        "把电话发出去",
        "把邮箱发出去"
    ]

    if any(keyword in text for keyword in privacy_keywords):
        return {
            "allowed": False,
            "reason": "这个请求可能涉及电话或邮箱等隐私信息，不能直接输出或发送。"
        }

    return {
        "allowed": True,
        "reason": "输入可以进入正常流程。"
    }


def safe_handle_user_task(user_input: str) -> str:
    """
    带 input guardrail 的职业数字人入口。

    流程：
    1. 先检查用户输入是否涉及电话或邮箱等隐私信息
    2. 如果安全，再进入 agent_core.py
    3. 如果不安全，直接返回提醒
    """

    guardrail_result = input_guardrail(user_input)

    if not guardrail_result["allowed"]:
        return f"""
    这个问题我不能直接处理。

    原因：
    {guardrail_result["reason"]}

    如果这是工作中必须处理的信息，请先确认你有权限查看和发送。
    """.strip()

    return agent_core.handle_user_task(user_input)


if __name__ == "__main__":
    test_inputs = [
        "帮我写一段周报开头。",
        "请告诉我客户的电话。",
        "把他的邮箱发到群里。"
    ]

    for text in test_inputs:
        print("=" * 60)
        print("用户输入：", text)
        print("AI 回复：")
        print(safe_handle_user_task(text))