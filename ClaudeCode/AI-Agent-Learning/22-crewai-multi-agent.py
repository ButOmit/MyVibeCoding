"""
CrewAI —— 工业级多 Agent 编排框架
=====================================
对比你之前学过的三种多 Agent 方式：

  项目 15（手写）:
    自己写 Coordinator + Agent 类 + while 循环
    灵活但繁琐，每次都要手写任务分配逻辑

  项目 21（LangGraph）:
    图结构定义工作流，适合固定流程
    Send/conditional_edges/interrupt 很强大，但偏底层

  CrewAI（本项目）:
    为多 Agent 协作专门设计的框架
    只需定义角色+任务，框架自动编排执行

CrewAI 的核心概念：
  Agent  = 角色定义（role + goal + backstory + tools）
  Task   = 任务定义（description + expected_output + agent）
  Crew   = 团队编排（agents + tasks + process）
  Process = 执行策略：
    - sequential:   按顺序执行（研究员写完→作者写→审校改）
    - hierarchical: 有 Manager 动态分配任务

真实世界的应用：
  - 内容创作流水线：研究→写作→审校→发布
  - 客服团队：分流→专业客服→质检
  - 代码开发：需求分析→架构设计→编码→测试

安装依赖：
  pip install crewai
"""

import sys
import os
import warnings

sys.stdout.reconfigure(encoding="utf-8")

# 忽略 CrewAI 的一些兼容性警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pydantic.*")

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process


# ============================================================
# 第一部分：定义 Agent 角色
# ============================================================
# 之前（项目 15）你手写:
#   researcher = SpecialistAgent(name="研究员", role="你是信息研究员...", tool_names=[...])
#
# CrewAI 用更语义化的方式:
#   Agent(role="研究员", goal="查找信息", backstory="你擅长...")
#
# backstory（背景故事）是 CrewAI 的特色——给 Agent 一个"人设"，
# 让它像真人一样思考，产出质量更高。

researcher = Agent(
    role="资深研究员",
    goal="深入调研给定主题，找出最关键的信息和数据",
    backstory="你是拥有十年经验的科技研究员，擅长从海量信息中提炼核心观点。"
              "你的座右铭是'没有找不到的信息，只有不够深的搜索'。"
              "每次调研，你都会列出3-5个关键发现，每条都有数据或事实支撑。",
    llm=f"openai/deepseek-chat",  # CrewAI 通过 LiteLLM 调用，格式: provider/model
    verbose=False,
    allow_delegation=False,
)

writer = Agent(
    role="科技专栏作者",
    goal="把研究结果写成生动有趣、通俗易懂的文章",
    backstory="你是受欢迎的科技专栏作者，你的读者是普通大众。"
              "你擅长用比喻和故事来解释复杂概念。"
              "你的文章总是有吸引人的标题、清晰的结构（引言→正文→总结）和自然的文风。",
    llm="openai/deepseek-chat",
    verbose=False,
    allow_delegation=False,
)

reviewer = Agent(
    role="资深编辑",
    goal="审校文章，确保内容准确、语言流畅、适合发布",
    backstory="你是有十五年经验的科技编辑，对文章质量要求苛刻。"
              "你检查：事实准确性、逻辑连贯性、语言流畅度、标题吸引力。"
              "你会给出具体的修改建议和改进后的版本。",
    llm="openai/deepseek-chat",
    verbose=False,
    allow_delegation=False,
)


# ============================================================
# 第二部分：定义 Task 任务
# ============================================================
# 之前（项目 15）你手写:
#   coordinator_messages.append({"role": "user", "content": f"任务: {task}"})
#   agent.ask(task)
#
# CrewAI 用 Task 对象，明确每个任务的输入输出和依赖关系。

research_task = Task(
    description="""
调研主题：{topic}

请做深入的桌面研究，找出这个主题的：
1. 核心概念和定义
2. 3-5 个关键数据或事实
3. 当前趋势和发展方向
4. 对普通人生活的影响

用中文输出结构化的研究报告。""",
    expected_output="一份结构化研究报告，包含概念定义、关键数据、趋势分析和影响评估，500字以内",
    agent=researcher,
)

writing_task = Task(
    description="""
根据研究员提供的研究报告，写一篇面向大众读者的科普文章。

要求：
- 标题吸引人，让人想点进去看
- 开头用生动的例子或场景引入
- 中间解释核心概念（用类比帮助理解）
- 列出关键发现
- 结尾总结并展望未来

用中文，文风轻松自然，300-500字。""",
    expected_output="一篇结构完整的科普文章，标题吸引人，内容通俗易懂",
    agent=writer,
)

review_task = Task(
    description="""
审校以下文章，检查：
1. 事实是否准确（对照研究报告）
2. 逻辑是否通顺
3. 语言是否流畅易懂
4. 标题是否足够吸引人

用中文输出修改后的最终版本，并附上简短的修改说明。""",
    expected_output="修改后的最终文章 + 修改说明（100字以内）",
    agent=reviewer,
)


# ============================================================
# 第三部分：Crew 编排 —— 把 Agent 和 Task 组装成团队
# ============================================================

def run_sequential(topic: str) -> str:
    """Sequential 模式：研究员 → 作者 → 审校，流水线作业。

    这是最常见的模式，适合有明确步骤的工作流。
    """
    print(f"\n{'=' * 60}")
    print(f"Sequential 模式：研究 → 写作 → 审校")
    print(f"主题: {topic}")
    print(f"{'=' * 60}")

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs={"topic": topic})
    return str(result)


# ============================================================
# 第四部分：对比 —— 手写多 Agent vs CrewAI
# ============================================================

def show_comparison():
    print("=" * 60)
    print("三种多 Agent 方案对比")
    print("=" * 60)
    print("""
┌──────────────┬──────────────────┬──────────────────┬──────────────────┐
│ 维度          │ 手写（项目15）    │ LangGraph（项目21）│ CrewAI（本项目）   │
├──────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 角色定义       │ 手写 System Prompt│ 图节点 + LLM调用  │ Agent(role,goal,  │
│              │ + Tool 列表       │                  │  backstory)       │
│ 任务分配       │ Coordinator 手写  │ 图结构边+条件路由  │ Task 对象自动编排  │
│ 执行顺序       │ 串行 while 循环   │ 图结构控制         │ Process.sequential│
│ 并行支持       │ 需手写多线程      │ Send API 原生支持  │ ⚠ 需额外配置     │
│ 代码量         │ ~300 行          │ ~200 行           │ ~80 行            │
│ 适用场景       │ 完全自定义需求    │ 固定流程+复杂分支  │ 团队角色协作      │
│ 学习曲线       │ 低（理解原理）    │ 中（图结构思维）   │ 低（语义化API）   │
└──────────────┴──────────────────┴──────────────────┴──────────────────┘
""")


# ============================================================
# 第五部分：深入理解 —— CrewAI 底层做了什么
# ============================================================

def explain_under_the_hood():
    """解释 CrewAI 底层原理，让你知道它和你手写的代码本质一样"""
    print("=" * 60)
    print("CrewAI 底层揭秘 —— 和你手写代码的对应关系")
    print("=" * 60)
    print("""
CrewAI 底层做的事情，你项目 15 都手写过：

  1. Agent(role, goal, backstory)
     → 本质就是组建 System Prompt:
       "你是{role}。你的目标是{goal}。背景: {backstory}"

  2. Task(description, expected_output)
     → 本质就是构造 User Message:
       "请完成以下任务: {description}。期望输出: {expected_output}"

  3. Process.sequential
     → 本质就是串行调用:
       result1 = researcher.ask(task1)
       result2 = writer.ask(task2 + result1)     ← 把上一步结果传下去
       result3 = reviewer.ask(task3 + result2)

  4. Crew.kickoff()
     → 本质就是启动循环，逐步执行每个 Task，
       自动管理 Agent 之间的上下文传递。

框架只是帮你把这些重复代码封装了。理解了底层再学框架，
你能写出更好的框架代码，也能更好地 debug。
""")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 设置 API key（CrewAI 通过环境变量读取）
    os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

    print("=" * 60)
    print("CrewAI —— 工业级多 Agent 编排框架")
    print("=" * 60)

    show_comparison()
    explain_under_the_hood()

    # 运行演示
    topic = "AI Agent 在 2026 年的发展趋势"
    print(f"\n🎯 创作主题: {topic}")
    print("组建团队：研究员 + 作者 + 审校（Sequential 流水线）")

    try:
        result = run_sequential(topic)
        print(f"\n{'=' * 60}")
        print("📰 最终输出（经过研究→写作→审校三道工序）:")
        print(f"{'=' * 60}")
        print(result)
    except Exception as e:
        print(f"\n运行出错: {e}")
        print("可能需要设置 DEEPSEEK_API_KEY 环境变量")

    print("\n" + "=" * 60)
    print("面试话术：")
    print('  "我实践过三种多 Agent 架构：')
    print('   1. 手写 Coordinator + SpecialistAgent（理解底层）')
    print('   2. LangGraph 图结构工作流（复杂分支+中断）')
    print('   3. CrewAI 角色驱动编排（快速搭建团队协作）')
    print('   能根据场景选择最合适的方案，而不是只会用一个框架。"')
    print("=" * 60)
