# 5天，让 AI 成为你的生产力

这是一个面向 AI 初学者的实战型课程项目。

本项目不以算法研究为目标，也不要求学习者一开始就具备很强的代码能力。  
它的目标是帮助学习者从零开始，逐步搭建一个能够理解个人工作角色、判断任务流程、调用工具、读取资料，并具备基本边界意识的 **职业数字人 AI 助手**。

先不依赖框架，从第一性原理出发，让学员真正理解一个 AI Agent 系统为什么需要 LLM、workflow、tool、context、RAG、guardrail。等以后再学 LangGraph、CrewAI、OpenAI Agents SDK，才知道这些框架到底在帮我们封装什么。

---

## 项目简介

职业数字人可以理解为：

> 以 LLM 为大脑，理解你的角色、任务、资料和工作习惯，能按流程判断，必要时调用工具，并在边界内帮你完成工作的 AI 助手。

整个项目会用 5 天时间，从最基础的大模型调用开始，逐步搭建一个可运行、可测试、可展示的 AI 工作助手。

这 5 天的主线是：

- Day 1：接上 AI 大脑
- Day 2：让它有流程
- Day 3：让它会用工具
- Day 4：让它懂资料和上下文
- Day 5：让它有边界，并做成可展示 Demo

---

## 这个项目适合谁

本项目适合：

- 没有系统学过 AI 编程，但希望快速上手 AI 应用开发的人
- 公司内部正在参与 AI 转型的业务人员、项目人员、工程师
- 希望把 AI 用到日常工作中的职场人
- 想理解智能体、workflow、tool、RAG、context、guardrail 等核心概念的人
- 希望快速做出一个 AI workflow 原型的人

这不是一门培养算法研究员的课程。  
它更关注：如何借助大模型和现成工具，把真实工作需求做成一个可以运行的 AI 原型。

---

## 项目最终目标

完成本项目后，你将能够搭建一个基础版职业数字人，它可以：

- 根据用户信息进行职业介绍
- 判断不同任务类型，并进入不同处理流程
- 处理写作、总结、规划等基础任务
- 调用工具读取表格、分析数据
- 读取网页或公开信息
- 基于本地资料进行问答
- 结合 history、memory 和 resource 生成更贴合上下文的回答
- 对超出范围或不适合处理的问题进行基本边界判断
- 通过 Gradio 做成一个可展示的 Web App Demo

---

## 课程与项目结构

### Day 1：认识职业数字人，接上 AI 的大脑

目标：让 AI 第一次真正开始替你工作。

主要内容：

- 什么是职业数字人
- 什么是 LLM API
- 如何配置环境
- 如何第一次调用大模型
- prompt 为什么会影响结果
- 做出职业数字人 v0

Day 1 的重点是让学习者跑通第一次 LLM 调用，并做出一个可以进行基础职业介绍的最小助手。

---

### Day 2：让它有流程，知道不同任务走不同路

目标：让职业数字人不再只靠一段 prompt，而是开始具备 workflow。

主要内容：

- 什么是 workflow
- workflow 和 agent 的区别
- 什么是 routing
- 如何判断任务类型
- 如何让不同任务进入不同处理路径

Day 2 的核心是：

> 智能体不是一段 prompt，而是一条工作流。

这一阶段会做出职业数字人 v1：  
一个会先判断任务类型，再决定下一步处理方式的助手。

---

### Day 3：让它会做事，接上最实用的工具

目标：让职业数字人从“会回答”走向“会执行”。

主要内容：

- 什么是 tool
- model 和 tool 有什么区别
- 如何读取表格
- 如何读取网页
- 如何把消息推送到测试群
- 如何根据任务类型自动选择工具

Day 3 的核心句是：

> 模型负责思考和表达，工具负责读取、搜索、分析、通知和执行。

这一阶段会做出职业数字人 v2：  
一个可以调用工具完成基础办公任务的助手。

---

### Day 4：让它懂你的资料，也接得住上下文

目标：让 AI 不只靠脑补，而是真正基于资料和上下文工作。

主要内容：

- 什么是 context
- history、memory、resource 的区别
- 为什么 resource 不能直接全部丢给 AI
- 什么是 RAG
- 如何做最小文件夹式 RAG
- 如何把 history、memory、resource 拼进最终 prompt

Day 4 的核心定义是：

> context = 模型这一次看到的全部信息。

这一阶段会做出职业数字人 v3：  
一个能基于资料、对话历史和长期规则回答问题的助手。

---

### Day 5：让它有边界，并做成可展示 Demo

目标：让职业数字人更适合真实工作场景，并完成最终展示。

主要内容：

- 什么是 guardrail
- 什么是 fallback
- 什么是 handoff
- 为什么真实工作里的 AI 不能乱答
- 什么问题可以继续处理
- 什么问题应该拒绝、提醒或交给人
- 如何给职业数字人加一层最小 guardrail
- 如何用 Gradio 做出一个可展示的 Web App Demo

Day 5 的理论部分会讲清楚三件事：

- Guardrail：判断边界
- Fallback：答不了时如何兜底
- Handoff：AI 不适合继续处理时，如何交给人

但 Day 5 的 Lab 不会一次性把三套机制都做完。  
为了保持初学者友好，Lab 只实操最小版 guardrail：

> 先判断用户输入是否适合进入职业数字人的正常流程。

最终会把前四天的能力接入 Gradio，形成一个可以展示的职业数字人 Demo。

---

## 推荐项目结构

```text
ai-agent-course/
├── lab1.ipynb
├── lab2.ipynb
├── lab3.ipynb
├── lab4.ipynb
├── lab5.ipynb
├── app.py
├── data/
│   ├── CV.txt
│   ├── work_log.txt
│   ├── sales_data.csv
│   └── context/
│       ├── history.txt
│       ├── memory.txt
│       └── resource.txt
├── src/
│   ├── __init__.py
│   ├── llm.py
│   ├── task_router.py
│   ├── tool.py
│   ├── context.py
│   ├── agent_core.py
│   └── guardrail.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

建议把可复用的 `.py` 文件统一放进 `src/` 文件夹，而不是全部写在 Notebook 里。  
这样项目结构更清楚，也更接近真实工程项目的组织方式。

---

## 环境准备

建议使用 Python 3.10 或以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

示例 `requirements.txt`：

```txt
openai
python-dotenv
pandas
matplotlib
requests
beautifulsoup4
scikit-learn
numpy
gradio

# Notebook
notebook
ipykernel
ipython
ipywidgets
```

---

## 环境变量配置

请不要把真实 API Key 上传到 GitHub。

在项目根目录创建 `.env` 文件：

```env
API_KEY=your_api_key_here
BASE_URL=your_base_url_here
MODEL_NAME=your_model_name_here
```

`.gitignore` 中应包含：

```gitignore
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## 运行方式

### 1. 运行 Notebook

从 Day 1 开始，按顺序打开：

```text
lab1.ipynb
lab2.ipynb
lab3.ipynb
lab4.ipynb
lab5.ipynb
```

每个 Notebook 建议按照以下结构学习：

1. Definition：今天的核心概念
2. Why it matters：这个概念为什么重要
3. Build：今天给系统增加什么能力
4. Test：用固定问题测试
5. What this block does：每段代码在做什么
6. Further Reading：扩展阅读

这种结构是为了让学习者不仅能跑代码，也能理解每个模块在系统中的作用。

---

### 2. 运行最终 Web App Demo

如果已经完成前面模块，可以运行：

```bash
python app.py
```

启动后，根据终端提示打开本地链接，即可访问职业数字人 Web App Demo。

---

## 核心概念

### LLM

大语言模型，是职业数字人的“大脑”。

### Prompt

告诉模型应该如何理解任务、以什么角色回答、输出什么格式。

### Workflow

预先设计好的任务流程。  
例如：先判断任务类型，再进入不同处理路径。

### Routing

路由分流。  
让 AI 先判断用户输入属于哪类任务，再决定下一步怎么处理。

### Tool

让 AI 具备外部能力的函数或接口。  
例如：读取表格、分析数据、读取网页、发送消息。

### Context

模型这一次能看到的全部信息。  
包括当前问题、对话历史、长期记忆和检索出来的资料。

### RAG

先从资料中找出相关内容，再基于这些内容回答问题。

### Guardrail

让 AI 在合适范围内工作的边界机制。

### Fallback

当资料不足、工具失败、问题不清楚或结果不确定时，给出安全、诚实、可继续推进的回答。

### Handoff

当 AI 不适合继续处理时，把任务整理清楚，交给人或后续负责人处理。

---

## 教学理念

这个项目不逐行讲代码。

原因不是代码不重要，而是对于 AI 应用入门者来说，更重要的是先建立工程理解力。

学习者需要掌握三层能力：

1. 看懂模块  
   知道这段代码是在接模型、做路由、调工具、读资料，还是做界面。

2. 会改参数和提示词  
   比如换模型、改 system prompt、改角色设定、改工具逻辑。

3. 会让 AI 帮你改错  
   把报错信息发给 AI，让 AI 帮助定位问题和修改代码。

这门课不是带你背代码。  
它是带你学会，怎么和 AI 一起做系统。

---

## 当前版本

当前项目处于课程开发与教学演示阶段。

已覆盖：

- LLM API 调用
- 职业数字人 v0
- Routing 系统
- 工具调用
- 表格分析工具
- 网页读取工具
- 消息推送工具
- Context 结构
- 最小 RAG
- Guardrail 理论与最小实操
- Fallback / Handoff 理论介绍
- Gradio Demo 设计

---

## 后续计划

后续可以继续扩展：

- 更完整的 fallback 机制
- 更完整的 handoff 交接摘要
- 更稳定的 RAG 知识库
- 更复杂的工具调用
- 企业内部权限控制
- 日志与可观测性
- 多智能体协作
- LangGraph
- CrewAI
- OpenAI Agents SDK
- MCP 工具协议
- 更完整的部署方案

---

## 免责声明

本项目主要用于 AI 应用开发教学和原型演示。

请不要在未经过充分测试、权限审核和安全评估的情况下，将其直接用于生产环境或处理敏感业务数据。

---

## License

本项目仅用于学习、教学和内部演示。  
如需商业使用或二次开发，请根据实际情况补充正式授权说明。
