"""
LangGraph 进阶 —— 工业级 Agent 工作流
=========================================
项目 16 学的是 LangGraph 基础：一个 Agent + 工具调用。
现在升级到真实生产环境需要的四个高级特性：

  1. 并行执行 (Send API)       → 多个审查员同时工作，不排队
  2. 条件分支 (Conditional Edge) → 安全风险高→人工审批，低→自动通过
  3. 人工介入 (Human-in-the-Loop) → 关键决策暂停等人类确认
  4. 断点续跑 (Checkpoint)       → 程序挂了恢复状态，不重来

对比项目 15（手写多 Agent）：
  手写: Coordinator 串行调用 Researcher → Coder → 汇总
  LangGraph: 图结构 + Send 并行 → 3 个审查员同时干活 → 汇总

真实场景：
  文档审核工作流：提交代码 → 并行检查(代码质量+安全漏洞+摘要)
  → 安全问题? → 人工审批 → 发布

安装依赖：
  pip install langgraph langchain-openai langchain-core
"""

import sys
import os
from typing import TypedDict, Annotated, Literal
import operator

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send, interrupt
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)


# ============================================================
# 第一部分：定义状态 —— 工作流的"共享内存"
# ============================================================

class ReviewState(TypedDict):
    """整个工作流的状态定义。

    项目 15 的协调员用 self.coordinator_messages 记录状态，
    这里用 TypedDict 统一管理所有节点的输入输出。
    """
    # 输入
    document: str                      # 待审查的文档/代码
    # 并行审查结果（Annotated 累加器：多个节点并发写，operator.add 自动合并）
    code_reviews: Annotated[list[str], operator.add]
    security_reviews: Annotated[list[str], operator.add]
    summary_reviews: Annotated[list[str], operator.add]
    # 汇总结果
    combined_result: str
    # 条件分支：是否需要人工审批
    needs_human_review: bool
    security_risk: str                 # "high" / "medium" / "low"
    # 人工审批结果
    human_approved: bool
    human_feedback: str
    # 最终输出
    final_result: str


# ============================================================
# 第二部分：定义节点 —— 工作流的每一步
# ============================================================

def receive_document(state: ReviewState) -> dict:
    """节点 1: 接收文档，准备分发"""
    doc = state["document"]
    print(f"\n📄 收到文档 ({len(doc)} 字): {doc[:60]}...")
    return {"code_reviews": [], "security_reviews": [], "summary_reviews": []}


def code_review(state: ReviewState) -> dict:
    """节点 2a: 代码质量审查（并行分支 1）"""
    doc = state["document"]
    print("  [代码审查] 检查代码质量...")
    response = llm.invoke(f"""你是代码审查员。审查以下内容的质量、可读性、逻辑正确性。
用中文，50 字以内的简要评价。

内容：
{doc[:500]}""")
    return {"code_reviews": [f"代码审查: {response.content}"]}


def security_review(state: ReviewState) -> dict:
    """节点 2b: 安全漏洞审查（并行分支 2）"""
    doc = state["document"]
    print("  [安全审查] 检查安全漏洞...")
    response = llm.invoke(f"""你是安全审查员。审查以下内容是否有安全风险。
用中文，50 字以内。末尾标注风险等级: [高/中/低]。

内容：
{doc[:500]}""")
    return {"security_reviews": [f"安全审查: {response.content}"]}


def summary_review(state: ReviewState) -> dict:
    """节点 2c: 摘要生成（并行分支 3）"""
    doc = state["document"]
    print("  [摘要生成] 生成文档摘要...")
    response = llm.invoke(f"""你负责生成摘要。把以下内容总结为一句话（30字以内）。

内容：
{doc[:500]}""")
    return {"summary_reviews": [f"摘要: {response.content}"]}


def combine_reviews(state: ReviewState) -> dict:
    """节点 3: 汇总三个并行审查的结果，判断是否需要人工介入"""
    print("\n📊 汇总审查结果...")

    code = state.get("code_reviews", [])
    security = state.get("security_reviews", [])
    summary = state.get("summary_reviews", [])

    all_results = "\n".join(code + security + summary)

    # 让 LLM 判断安全风险等级
    security_text = str(security)
    if "高" in security_text or "高危" in security_text or "严重" in security_text:
        needs_human = True
        risk = "high"
    elif "中" in security_text:
        needs_human = True
        risk = "medium"
    else:
        needs_human = False
        risk = "low"

    combined = f"=== 审查汇总 ===\n{all_results}\n\n风险等级: {risk} | 需人工审批: {'是' if needs_human else '否'}"

    return {
        "combined_result": combined,
        "needs_human_review": needs_human,
        "security_risk": risk,
    }


def human_approval(state: ReviewState) -> dict:
    """节点 4: 🔴 人工介入 —— 暂停等待人类审批

    这就是 human-in-the-loop！
    interrupt() 会让工作流暂停，状态保存到 checkpoint。
    人类确认后，通过 Command(resume=...) 继续执行。
    """
    print("\n🛑 触发人工审批 —— 等待人类决策...")
    print(f"   审查结果: {state['combined_result'][:150]}...")

    # interrupt() 暂停执行，返回提示信息给外部调用者
    decision = interrupt({
        "message": "安全审查发现问题，需要人工审批",
        "risk_level": state["security_risk"],
        "review_summary": state["combined_result"],
        "options": ["approve", "reject", "request_changes"],
    })

    # 人类通过 Command(resume=...) 传入决策
    if isinstance(decision, dict):
        approved = decision.get("action") == "approve"
        feedback = decision.get("feedback", "")
    else:
        approved = (decision == "approve")
        feedback = ""

    print(f"\n👤 人工决定: {'✅ 通过' if approved else '❌ 驳回'} {feedback}")

    return {
        "human_approved": approved,
        "human_feedback": feedback,
    }


def publish(state: ReviewState) -> dict:
    """节点 5a: 发布（审批通过路径）"""
    print("\n✅ 审批通过，发布文档...")
    return {"final_result": f"发布成功！\n{state['combined_result']}"}


def reject(state: ReviewState) -> dict:
    """节点 5b: 驳回（审批驳回路径）"""
    print(f"\n❌ 驳回！原因: {state.get('human_feedback', '安全风险未解决')}")
    return {"final_result": f"审核被驳回。反馈: {state.get('human_feedback', '')}"}


# ============================================================
# 第三部分：路由函数 —— 条件分支和并行分发
# ============================================================

def route_after_review(state: ReviewState) -> Literal["human_approval", END]:
    """条件分支：安全风险高 → 人工审批，否则 → 直接结束"""
    if state.get("needs_human_review", False):
        return "human_approval"
    return END


def route_after_approval(state: ReviewState) -> Literal["publish", "reject"]:
    """人工审批后：通过 → 发布，驳回 → 拒绝"""
    if state.get("human_approved", False):
        return "publish"
    return "reject"


def fanout_reviews(state: ReviewState) -> list[Send]:
    """🎯 核心：并行分发！

    之前（项目 15）你串行调用多个专家:
      result1 = researcher.ask(task1)
      result2 = coder.ask(task2)       ← 等 researcher 完才执行！

    现在用 Send API 并行:
      三个审查员同时干活，总时间 = 最慢那个的时间

    Send(node_name, state_dict) 表示"把这份 state 发到那个节点"
    LangGraph 会并行执行所有 Send。
    """
    print("\n🚀 并行分发：3 个审查员同时开始工作...")
    # Send 会把 state_update 合并到子节点看到的状态里
    shared = {"document": state["document"]}
    return [
        Send("code_review", shared),
        Send("security_review", shared),
        Send("summary_review", shared),
    ]


# ============================================================
# 第四部分：构建图 —— 把节点和边连成工作流
# ============================================================

def build_review_workflow():
    """构建文档审核工作流图"""
    workflow = StateGraph(ReviewState)

    # 添加节点
    workflow.add_node("receive", receive_document)
    workflow.add_node("code_review", code_review)
    workflow.add_node("security_review", security_review)
    workflow.add_node("summary_review", summary_review)
    workflow.add_node("combine", combine_reviews)
    workflow.add_node("human_approval", human_approval)
    workflow.add_node("publish", publish)
    workflow.add_node("reject", reject)

    # 连接边
    workflow.add_edge(START, "receive")

    # 🎯 并行分发：receive → [code_review, security_review, summary_review] 同时执行
    workflow.add_conditional_edges("receive", fanout_reviews)

    # 并行节点都完成后 → 汇总
    workflow.add_edge("code_review", "combine")
    workflow.add_edge("security_review", "combine")
    workflow.add_edge("summary_review", "combine")

    # 🎯 条件分支：combine → 风险高去人工审批，低直接结束
    workflow.add_conditional_edges("combine", route_after_review)

    # 🎯 人工审批后 → 条件分支
    workflow.add_conditional_edges("human_approval", route_after_approval)

    # 最终节点 → 结束
    workflow.add_edge("publish", END)
    workflow.add_edge("reject", END)

    return workflow


# ============================================================
# 第五部分：演示四种高级特性
# ============================================================

def demo_1_parallel():
    """演示 1: 并行执行 —— 3 个审查员同时工作"""
    print("\n" + "=" * 60)
    print("演示 1: ⚡ 并行执行（Send API）")
    print("=" * 60)
    print("3 个审查员（代码/安全/摘要）同时工作，不等排队")

    workflow = build_review_workflow()
    app = workflow.compile()

    test_doc = """
    def login(username, password):
        sql = f"SELECT * FROM users WHERE name='{username}' AND pw='{password}'"
        result = database.execute(sql)
        if result:
            return {"status": "ok", "token": generate_token(username)}
        return {"status": "fail"}
    """

    result = app.invoke({"document": test_doc})
    print(f"\n📋 结果:\n{result.get('combined_result', 'N/A')}")


def demo_2_human_in_loop():
    """演示 2: 人工介入 —— 暂停等人类审批"""
    print("\n" + "=" * 60)
    print("演示 2: 🛑 人工介入（Human-in-the-Loop）")
    print("=" * 60)

    # 编译时带 checkpoint 才能暂停
    workflow = build_review_workflow()
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "review-001"}}

    # 包含安全漏洞的代码 → 必然触发人工审批
    malicious_doc = """
    # 这段代码把用户密码明文存在日志里
    def save_user(username, password):
        logging.info(f"Creating user {username} with password: {password}")
        db.insert("users", {"name": username, "password": password})
    """

    print("\n提交包含安全问题的代码...")
    # 第一次运行 → 会在 human_approval 节点暂停
    result = app.invoke({"document": malicious_doc}, config=config)

    # 检查是否暂停
    state = app.get_state(config)
    if state.next:  # 有 next 说明暂停了，等待人工
        print(f"\n⏸️  工作流在 '{state.next}' 暂停，等待人工介入！")
        print(f"   当前状态保存到 checkpoint: thread_id=review-001")

        # 人类决策：驳回
        print("\n👤 人类操作：检查后决定驳回（密码不能明文记录）")
        app.invoke(
            Command(resume={"action": "reject", "feedback": "密码不能明文存储！请使用 bcrypt 哈希后再存入数据库。"}),
            config=config,
        )

        # 获取最终状态
        final_state = app.get_state(config)
        print(f"\n📋 最终结果: {final_state.values.get('final_result', 'N/A')}")

    # 🎯 断点续跑：可以从暂停点恢复！
    print("\n💡 如果这时候服务器挂了，重启后可以通过 thread_id 恢复状态继续执行！")
    print(f"   app.get_state({{'configurable': {{'thread_id': 'review-001'}}}})")
    print(f"   状态包含全部中间结果，不会丢失进度。")


def demo_3_branching():
    """演示 3: 条件分支 —— 自动路由"""
    print("\n" + "=" * 60)
    print("演示 3: 🔀 条件分支（Conditional Edge）")
    print("=" * 60)

    workflow = build_review_workflow()
    app = workflow.compile()

    # 安全的文档 → 不需要人工审批，直接通过
    safe_doc = """
    这是一个简单的工具函数，用于格式化日期：
    def format_date(year, month, day):
        return f"{year}-{month:02d}-{day:02d}"
    """

    print("\n提交安全代码...")
    result = app.invoke({"document": safe_doc})
    print(f"   风险: {result.get('security_risk', '?')}")
    print(f"   需要人工: {result.get('needs_human_review', '?')}")
    print("   → 自动通过，不触发人工审批 ✅")


def demo_4_checkpoint():
    """演示 4: 断点续跑 —— 中断后恢复"""
    print("\n" + "=" * 60)
    print("演示 4: 💾 断点续跑（Checkpoint）")
    print("=" * 60)

    workflow = build_review_workflow()
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    thread_id = "resume-demo"
    config = {"configurable": {"thread_id": thread_id}}

    # 问题代码 → 触发人工审批
    risky_doc = """
    # API key 硬编码在代码里，可以被 git 追踪到
    API_KEY = "sk-abc123def456ghi789"
    """

    print("\n第一次运行 → 到达人工审批点时暂停...")
    result = app.invoke({"document": risky_doc}, config=config)

    state = app.get_state(config)
    print(f"   当前节点: {state.next}")

    # 模拟"程序挂了"——不 resume，而是重新获取状态
    print(f"\n💥 模拟：程序在这里崩溃了...")
    print(f"   但所有状态已保存到 checkpoint!")

    # "重启"后恢复状态
    print(f"\n🔄 重启后恢复状态...")
    restored = app.get_state(config)
    print(f"   恢复的节点: {restored.next}")
    print(f"   已有的审查结果数量: {len(restored.values.get('code_reviews', []))} 代码审查")
    print(f"   + {len(restored.values.get('security_reviews', []))} 安全审查")
    print(f"   + {len(restored.values.get('summary_reviews', []))} 摘要审查")
    print(f"   无需重新运行前面的节点！")

    # 继续执行（人工批准）
    print(f"\n👤 人工审批：修改建议后批准")
    app.invoke(
        Command(resume={"action": "approve", "feedback": "API key 已移除，改用环境变量"}),
        config=config,
    )
    final_state = app.get_state(config)
    print(f"📋 最终结果: {final_state.values.get('final_result', 'N/A')[:100]}...")


# ============================================================
# 对比总结
# ============================================================

def show_comparison():
    print("=" * 60)
    print("LangGraph 进阶 vs 手写多 Agent（项目 15）")
    print("=" * 60)
    print("""
┌──────────────────────┬──────────────────────────────┐
│ 手写（项目 15）       │ LangGraph 进阶（本项目）       │
├──────────────────────┼──────────────────────────────┤
│ 串行调用专家          │ Send API 并行，3倍速度        │
│ while + if/else 分支  │ add_conditional_edges 自动路由│
│ 暂停 = 手动 break     │ interrupt() 标准暂停点         │
│ 无状态持久化          │ MemorySaver checkpoint 自动存 │
│ 挂了重来              │ 断点续跑，不重复执行           │
│ coordinator 硬编码    │ 图结构可视化，修改节点即可     │
└──────────────────────┴──────────────────────────────┘
""")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 进阶 —— 工业级 Agent 工作流")
    print("=" * 60)
    print("四大特性：并行 × 条件分支 × 人工介入 × 断点续跑")

    show_comparison()

    try:
        demo_1_parallel()
    except Exception as e:
        print(f"演示1 出错: {e}")

    try:
        demo_3_branching()
    except Exception as e:
        print(f"演示3 出错: {e}")

    try:
        demo_2_human_in_loop()
    except Exception as e:
        print(f"演示2 出错: {e}")

    try:
        demo_4_checkpoint()
    except Exception as e:
        print(f"演示4 出错: {e}")

    print("\n" + "=" * 60)
    print("面试话术：")
    print('  "我使用 LangGraph 构建过复杂 Agent 工作流，')
    print('   包括并行执行（Send API）、条件分支、')
    print('   Human-in-the-Loop 人工介入和 Checkpoint 断点续跑。')
    print('   对比手写多 Agent，图结构更灵活，')
    print('   支持状态持久化和工作流可视化。\"')
    print("=" * 60)
