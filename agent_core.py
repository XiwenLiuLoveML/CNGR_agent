from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Dict

# 读取 .env 里的 API key
load_dotenv()

# 创建客户端（通义千问兼容 OpenAI 接口）
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("API_KEY")
)


def call_llm(user_question: str, system_prompt: str) -> str:
    """
    最小 LLM 调用函数
    """
    response = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# =========================
# 1. Routing：先分类
# =========================

ROUTER_SYSTEM_PROMPT = """
你是一个任务分类器。

你的工作不是回答用户问题，
而是先判断这个问题属于哪一类任务。

可选任务类别只有这 5 个：
- profile
- writing
- data_analysis
- work_planning
- clarify

分类规则：
- profile：用户在问你是谁、你做什么、你做过什么项目
- writing：用户要你写、改、润色、总结一段文字
- data_analysis：用户要你分析表格、数据、结果
- work_planning：用户要你整理待办、总结工作、安排下一步
- clarify：用户的问题太模糊，暂时无法直接判断，需要先追问

你只能返回上面 5 个标签中的一个。
不要解释，不要输出别的内容。
"""


def classify_task(user_question: str) -> str:
    """
    先判断用户输入属于哪一类任务
    """
    result = call_llm(
        user_question=user_question,
        system_prompt=ROUTER_SYSTEM_PROMPT
    ).strip().lower()

    valid_labels = {
        "profile",
        "writing",
        "data_analysis",
        "work_planning",
        "clarify"
    }

    if result not in valid_labels:
        return "clarify"

    return result


def get_next_step(task_label: str) -> str:
    """
    根据任务类别，决定下一步动作
    """
    step_map = {
        "profile": "answer_profile",
        "writing": "handle_writing",
        "data_analysis": "handle_data_analysis",
        "work_planning": "handle_work_planning",
        "clarify": "ask_clarifying_question"
    }
    return step_map.get(task_label, "ask_clarifying_question")


# =========================
# 2. Prompt Chaining：分类后走下一步
# =========================

def build_profile_prompt(resume: str, work_log: str) -> str:
    return f"""
    你是一个职业数字人。

    请根据下面的简历和工作日志信息，
    回答用户关于“你是谁、你做什么、你做过什么”的问题。

    要求：
    - 语气积极、友好、自然
    - 回答简洁，控制在 2 到 4 句
    - 不要太长
    - 不要编造资料里没有的内容
    - 优先用清楚、职业的表达

    【简历】
    {resume}

    【工作日志】
    {work_log}
    """


def build_writing_prompt() -> str:
    return """
    你是一个职业数字人，正在帮助用户处理写作任务。

    你的目标是先判断：用户现在提供的信息，是否已经足够开始写。

    如果信息还不够，请自然、友好、简洁地追问最关键的缺失信息。
    优先确认这些内容：
    - 使用场景
    - 语气风格
    - 受众对象
    - 长度要求
    - 是否有必须包含的信息

    如果用户已经说得比较清楚了，就不要继续追问，直接开始起草第一版。

    要求：
    - 语气自然，不要机械
    - 如果需要追问，最多只问 2 个问题
    - 如果已经可以开写，就直接给出草稿
    """


def build_data_analysis_prompt() -> str:
    return """
    你是一个职业数字人，正在帮助用户处理数据分析任务。

    当前你还没有接入真正的数据分析工具。

    所以你的目标是先判断：用户是否已经提供了足够的数据和分析目标。

    如果没有提供足够信息，请自然地向用户索要最关键的信息，例如：
    - 数据表或数据内容
    - 想分析什么问题
    - 最关心哪些指标
    - 想得到什么结论

    如果用户已经在问题里贴出了少量数据，
    你可以先帮助他澄清分析目标，
    但不要假装已经完成正式分析.

    要求：
    - 语气自然、专业、友好
    - 优先问最关键的 1 到 2 个问题
    - 不要编造分析结果
    """


def build_work_planning_prompt(work_log: str) -> str:
    return f"""
    你是一个职业数字人，正在帮助用户整理工作安排或总结下一步。

    你的目标是先判断：用户是否已经给出了足够的任务背景。

    如果信息不足，请自然地补问最关键的信息，例如：
    - 当前待办
    - 截止时间
    - 优先级
    - 当前最重要的项目
    - 是否有必须先完成的事情

    如果信息已经比较清楚，就先帮用户做一个简洁的整理或下一步建议。

    要求：
    - 语气清晰、友好、务实
    - 如果需要追问，最多问 2 个问题
    - 如果已经足够清楚，就直接帮用户整理

    【工作日志】
    {work_log}
    """


def build_clarify_prompt() -> str:
    return """
    你是一个职业数字人。

    当前用户的问题还不够明确。

    请你用自然、友好、简洁的方式，
    先追问最关键的澄清问题，帮助用户把需求说清楚。

    要求：
    - 最多问 2 个问题
    - 不要一次问太多
    - 语气自然，不要机械
    """


def run_next_step(
    task_label: str,
    user_question: str,
    resume: str,
    work_log: str
) -> str:
    """
    根据 task_label 执行下一步
    """
    next_step = get_next_step(task_label)

    if next_step == "answer_profile":
        system_prompt = build_profile_prompt(resume, work_log)

    elif next_step == "handle_writing":
        system_prompt = build_writing_prompt()

    elif next_step == "handle_data_analysis":
        system_prompt = build_data_analysis_prompt()

    elif next_step == "handle_work_planning":
        system_prompt = build_work_planning_prompt(work_log)

    else:
        system_prompt = build_clarify_prompt()

    return call_llm(
        user_question=user_question,
        system_prompt=system_prompt
    )


# =========================
# 3. 总入口：Day 2 / Day 3 / Day 4 继续复用
# =========================

def run_agent_v1(
    user_question: str,
    resume: str,
    work_log: str
) -> Dict[str, str]:
    """
    职业数字人 v1 总入口
    先分类，再决定下一步，再生成回复
    """
    task_label = classify_task(user_question)
    next_step = get_next_step(task_label)
    response = run_next_step(
        task_label=task_label,
        user_question=user_question,
        resume=resume,
        work_log=work_log
    )

    return {
        "task_label": task_label,
        "next_step": next_step,
        "response": response
    }