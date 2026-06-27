"""
Prompt Engineering 进阶 —— 让 AI 稳定输出的工程方法
=====================================================
之前你写 prompt 都是 "你是XX助手，用中文回答"。
但真实业务需要 AI 输出可控、格式一致、逻辑正确。

本项目覆盖面试和工作中必会的四大技术：

  1. Zero-shot → Few-shot   用示例教会 AI 你想要什么格式
  2. Chain-of-Thought       让 AI "先思考再回答"，准确率暴增
  3. 结构化输出             强制 JSON，程序可解析
  4. System Prompt 设计    角色+约束+格式+边界

面试高频：
  - "你怎么设计 Prompt 让 Agent 输出更稳定？"
  - "Few-shot 和 Fine-tuning 什么时候用哪个？"
  - "Chain-of-Thought 为什么能提高准确率？"
"""

import sys
import os
import json

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DS = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def test(prompt_name: str, system: str, user: str, temperature: float = 0.3):
    """统一测试函数"""
    response = DS.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    print(f"\n{'─' * 60}")
    print(f"【{prompt_name}】")
    print(f"System: {system[:80]}...")
    print(f"User: {user[:80]}...")
    print(f"结果:")
    print(response.choices[0].message.content)


# ============================================================
# 第一课：Zero-shot → Few-shot
# ============================================================
# Few-shot = 给 AI 看几个例子，它就会照着做。
# 这是提升输出质量最简单最有效的方法。

print("=" * 60)
print("第一课：Zero-shot vs Few-shot")
print("=" * 60)

# 任务：把自然语言转成结构化任务列表
user_input = "帮我整理一下需求：用户要一个登录页面，支持手机号验证码登录和微信登录，密码要加密存储，还需要忘记密码功能"

# ---- Zero-shot（不给例子） ----
print("\n❌ Zero-shot（不给例子）：")
response = DS.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "system",
        "content": "把用户需求整理成任务列表。",
    }, {
        "role": "user",
        "content": user_input,
    }],
    temperature=0.3,
)
print(f"输出:\n{response.choices[0].message.content[:300]}")

# ---- Few-shot（给2个例子） ----
print("\n\n✅ Few-shot（给2个例子）：")
response = DS.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "system",
        "content": """把用户需求整理成 JSON 格式的任务列表。

示例1:
用户: "做一个用户注册功能"
输出: {"tasks": [{"id":1,"title":"数据库建表","priority":"高"},{"id":2,"title":"前端表单","priority":"中"}]}

示例2:
用户: "做一个搜索框"
输出: {"tasks": [{"id":1,"title":"搜索API","priority":"高"},{"id":2,"title":"自动补全","priority":"低"}]}

严格按示例的 JSON 格式输出，不要加额外说明。""",
    }, {
        "role": "user",
        "content": user_input,
    }],
    temperature=0.3,
)
print(f"输出:\n{response.choices[0].message.content}")


# ============================================================
# 第二课：Chain-of-Thought（思维链）
# ============================================================
# 让 AI "一步步想"，而不是直接给答案。
# 数学题、逻辑推理、代码 debug 必备。
# 论文证明：CoT 能把数学推理准确率从 18% 提升到 57%

print("\n" + "=" * 60)
print("第二课：Chain-of-Thought 思维链")
print("=" * 60)

math_problem = "小明有15个苹果，给了小红3个，又买了8个，吃了2个，最后把剩下的一半给了小刚。小明现在还有几个苹果？"

# ---- 直接回答（无 CoT） ----
print("\n❌ 直接回答（无思维链）：")
response = DS.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "user",
        "content": f"{math_problem}\n直接给出答案。",
    }],
    temperature=0,
)
print(f"输出: {response.choices[0].message.content}")

# ---- CoT（思维链） ----
print("\n✅ Chain-of-Thought（一步步想）：")
response = DS.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "user",
        "content": f"""{math_problem}

请一步步推理：
步骤1：初始数量 = ?
步骤2：给小红后 = ?
步骤3：买完后 = ?
步骤4：吃完后 = ?
步骤5：一半给小刚后 = ?
步骤6：最终答案 = ?""",
    }],
    temperature=0,
)
print(f"输出:\n{response.choices[0].message.content}")


# ============================================================
# 第三课：结构化输出（JSON Mode）
# ============================================================
# 让 AI 输出合法 JSON，方便程序解析。
# 关键技巧：明确 JSON Schema + 用 markdown 代码块

print("\n" + "=" * 60)
print("第三课：结构化 JSON 输出")
print("=" * 60)

# ---- 模糊指令 ----
print("\n❌ 模糊指令：")
response = DS.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "user",
        "content": "推荐3本编程入门书，列出书名和价格。",
    }],
    temperature=0.3,
)
print(f"输出:\n{response.choices[0].message.content[:200]}")

# ---- 精确 JSON Schema ----
print("\n✅ 明确 JSON Schema：")
response = DS.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "system",
        "content": """你是书籍推荐系统。输出必须是合法 JSON，格式如下：

```json
{
  "recommendations": [
    {"title": "书名", "author": "作者", "price": 59.0, "difficulty": "入门/进阶/高级", "reason": "一句话推荐理由"}
  ]
}
```

只输出 JSON，不要其他文字。""",
    }, {
        "role": "user",
        "content": "推荐3本 Python 编程入门书。",
    }],
    temperature=0.3,
)
result = response.choices[0].message.content
print(f"输出:\n{result}")

# 验证是否可解析
try:
    data = json.loads(result.strip().strip("`").strip("json").strip("`"))
    print(f"\n✅ JSON 解析成功！推荐了 {len(data['recommendations'])} 本书")
except json.JSONDecodeError:
    print("\n⚠️ JSON 解析失败（但实际项目中加上 retry 和格式修正就能解决）")


# ============================================================
# 第四课：System Prompt 设计
# ============================================================
# 一个好的 System Prompt 包含四要素：
#   ROLE  角色 — 你是谁，做什么
#   RULES 规则 — 必须遵守的约束
#   FORMAT 格式 — 输出长什么样
#   BOUNDARY 边界 — 什么可以做，什么不能做

print("\n" + "=" * 60)
print("第四课：System Prompt 四要素")
print("=" * 60)

# 场景：代码审查 Agent
code_to_review = """
def calc(x, y):
    if y == 0:
        return "err"
    return x / y
"""

# ---- 简陋 System Prompt ----
print("\n❌ 简陋 System Prompt：")
response = DS.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是代码审查员。"},
        {"role": "user", "content": f"审查这段代码：\n{code_to_review}"},
    ],
    temperature=0.3,
)
print(f"输出:\n{response.choices[0].message.content[:200]}")

# ---- 四要素 System Prompt ----
print("\n✅ 四要素 System Prompt：")
response = DS.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "system",
        "content": """[ROLE] 你是资深 Python 代码审查员，有10年经验。

[RULES]
- 检查：命名规范、类型安全、异常处理、边界条件
- 每个问题给出严重等级（严重/一般/建议）
- 提供具体的修改建议和修改后代码

[FORMAT]
## 审查结果
- [等级] 问题描述 → 修改建议
## 建议修改
（修改后的完整代码）

[BOUNDARY]
- 只审查代码质量，不评论业务逻辑
- 不确定的地方标注 [需确认]""",
    }, {
        "role": "user",
        "content": f"审查这段代码：\n{code_to_review}",
    }],
    temperature=0.3,
)
print(f"输出:\n{response.choices[0].message.content}")


# ============================================================
# 第五课：Prompt 模板 —— 像函数一样复用
# ============================================================

print("\n" + "=" * 60)
print("第五课：Prompt 模板化")
print("=" * 60)

# 一个可复用的 Prompt 模板类
class PromptTemplate:
    """像 jinja2 一样，但更简单"""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


# 定义模板
code_review_template = PromptTemplate("""[ROLE] {role}
[RULES] {rules}
[FORMAT] {format_spec}

审查以下代码：
```{language}
{code}
```""")

# 使用模板
prompt = code_review_template.format(
    role="Python 代码审查专家",
    rules="检查命名、安全、性能",
    format_spec="列表形式，每条以 - 开头",
    language="python",
    code="x = [1,2,3]; total = sum(x); print(f'Sum: {total}')",
)

print("模板生成结果:")
print(prompt[:300])

# 实际调用
response = DS.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是代码审查助手。"},
        {"role": "user", "content": prompt},
    ],
    temperature=0.3,
)
print(f"\n审查结果:\n{response.choices[0].message.content}")


# ============================================================
# 总结：Prompt 工程速查表
# ============================================================

print("\n" + "=" * 60)
print("Prompt 工程速查表（面试前过一遍）")
print("=" * 60)
print("""
┌──────────────────────┬──────────────────────────────────────┐
│ 技术                  │ 什么时候用 + 示例                      │
├──────────────────────┼──────────────────────────────────────┤
│ Few-shot（示例驱动）   │ AI 输出格式不稳定时                    │
│                      │ "请看以下2个例子，按相同格式输出"        │
├──────────────────────┼──────────────────────────────────────┤
│ Chain-of-Thought     │ 数学、逻辑、多步骤任务                  │
│                      │ "请一步步推理：步骤1...步骤2..."         │
├──────────────────────┼──────────────────────────────────────┤
│ 结构化输出            │ 需要程序解析 AI 输出时                   │
│                      │ "输出必须是合法 JSON，格式如下..."       │
├──────────────────────┼──────────────────────────────────────┤
│ CoT + Few-shot 组合  │ 复杂任务（最强组合）                    │
│                      │ 给思考过程的示例 + 让 AI 模仿思考        │
├──────────────────────┼──────────────────────────────────────┤
│ System Prompt 四要素  │ 每次定义 Agent 角色时                   │
│                      │ ROLE + RULES + FORMAT + BOUNDARY      │
└──────────────────────┴──────────────────────────────────────┘

面试话术：
  "我在 Agent 开发中系统使用 Prompt Engineering：
   用 Few-shot 控制输出格式，用 Chain-of-Thought 提高推理准确率，
   用结构化 JSON 保证程序可解析，用四要素模板统一 Agent 角色定义。
   选择 Few-shot 而不是 Fine-tuning，因为业务变化快，
   改 Prompt 比重新训练模型快得多。"
""")
