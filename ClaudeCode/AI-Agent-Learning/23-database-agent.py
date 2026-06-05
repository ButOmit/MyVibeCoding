"""
Agent + 数据库 —— 让 AI 自动写 SQL 查数据
============================================
数据库是计科核心技能。本项目做两件事：
  1. 入门 SQLite（Python 自带，零安装的数据库）
  2. 让 Agent 自动把自然语言问题转成 SQL 查询

核心技术：
  自然语言 → Schema 感知 → 生成 SQL → 安全执行 → 格式化答案

真实场景：
  - 运营："上周卖了多少单？" → Agent 自动查数据库
  - CEO："哪个产品利润最高？" → Agent 写 SQL 分析
  - 客服："订单 #1234 的物流状态" → Agent 查表返回

安装依赖：
  无需额外安装！SQLite 是 Python 标准库。
"""

import sys
import os
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DS = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ============================================================
# 第一部分：数据库基础 —— 用 SQLite 建一个书店数据库
# ============================================================
# SQLite 是一个文件型数据库，不需要安装服务器。
# 你电脑上的 .db 文件就是整个数据库，极其轻量。
#
# 数据库核心概念（30秒入门）：
#   表（Table）   = Excel 工作表
#   行（Row）     = Excel 一行数据
#   列（Column）  = Excel 一列
#   SQL          = 对表的操作指令
#
# 最常用的 SQL:
#   SELECT 列 FROM 表 WHERE 条件          → 查数据
#   SELECT 列 FROM 表 GROUP BY 列         → 分组统计
#   SELECT 列 FROM 表 ORDER BY 列 DESC    → 排序
#   SELECT 列 FROM 表A JOIN 表B ON 条件   → 多表关联

DB_PATH = os.path.join(os.path.dirname(__file__), "agent_workspace", "bookstore.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def create_sample_database():
    """创建示例数据库：一家在线书店"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ---- 表 1: 图书 ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        )
    """)

    # ---- 表 2: 顾客 ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            city TEXT,
            join_date TEXT NOT NULL
        )
    """)

    # ---- 表 3: 订单 ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            book_id INTEGER,
            quantity INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    """)

    # 清空旧数据
    for table in ["orders", "customers", "books"]:
        cur.execute(f"DELETE FROM {table}")

    # ---- 插入示例数据 ----

    # 图书
    cur.executemany(
        "INSERT INTO books VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "深入理解Python", "张三", "编程", 59.0, 50),
            (2, "机器学习实战", "李四", "AI", 79.0, 30),
            (3, "深度学习入门", "王五", "AI", 89.0, 20),
            (4, "算法导论", "Thomas", "计算机科学", 99.0, 15),
            (5, "数据库系统概念", "Silberschatz", "计算机科学", 69.0, 25),
            (6, "Python数据处理", "赵六", "编程", 49.0, 40),
            (7, "自然语言处理", "周七", "AI", 85.0, 18),
            (8, "计算机网络", "谢希仁", "计算机科学", 55.0, 35),
        ],
    )

    # 顾客
    cur.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)",
        [
            (1, "小明", "上海", "2026-01-15"),
            (2, "小红", "北京", "2026-02-20"),
            (3, "小刚", "杭州", "2026-03-10"),
            (4, "小美", "上海", "2026-04-05"),
        ],
    )

    # 订单（最近 30 天）
    today = datetime.now()
    order_data = [
        (1, 1, 2, today - timedelta(days=3)),
        (2, 2, 1, today - timedelta(days=5)),
        (3, 3, 1, today - timedelta(days=7)),
        (1, 3, 1, today - timedelta(days=10)),
        (4, 4, 1, today - timedelta(days=12)),
        (2, 5, 1, today - timedelta(days=15)),
        (3, 2, 3, today - timedelta(days=18)),
        (1, 6, 2, today - timedelta(days=20)),
        (4, 7, 1, today - timedelta(days=22)),
        (2, 8, 1, today - timedelta(days=25)),
    ]
    for cid, bid, qty, dt in order_data:
        cur.execute(
            "INSERT INTO orders (customer_id, book_id, quantity, order_date) VALUES (?, ?, ?, ?)",
            (cid, bid, qty, dt.strftime("%Y-%m-%d")),
        )

    conn.commit()
    conn.close()
    print(f"数据库创建完毕：{DB_PATH}")
    print(f"  3 张表：books(8本) | customers(4位) | orders(10笔)")


# ============================================================
# 第二部分：数据库 Schema —— Agent 的"地图"
# ============================================================
# Agent 需要知道数据库有哪些表、各表有哪些列，才能写 SQL。
# 就像你去一个陌生图书馆，需要先看楼层导览图。

SCHEMA = """
数据库 Schema（SQLite）：

表名: books (图书)
  列: id (主键) | title (书名) | author (作者) | category (分类) | price (价格/元) | stock (库存/本)

表名: customers (顾客)
  列: id (主键) | name (姓名) | city (城市) | join_date (注册日期)

表名: orders (订单)
  列: id (主键) | customer_id (顾客ID→customers.id) | book_id (图书ID→books.id) | quantity (数量) | order_date (下单日期)

表关系：orders.customer_id → customers.id，orders.book_id → books.id
"""


# ============================================================
# 第三部分：安全查询执行器
# ============================================================

def safe_execute_sql(sql: str) -> str:
    """安全执行 SQL 查询。

    只允许 SELECT（查询），拒绝 DROP/DELETE/UPDATE/INSERT（修改），
    防止 Agent 误操作破坏数据。
    """
    sql_stripped = sql.strip().upper()

    # 只允许 SELECT
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
    for word in forbidden:
        if sql_stripped.startswith(word):
            return f"❌ 安全拦截：不允许 {word} 操作。只能执行 SELECT 查询。"

    # 防止分号注入（多条语句）
    if ";" in sql.rstrip(";"):
        return "❌ 安全拦截：一次只能执行一条 SQL。"

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 让结果能用列名访问
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()

        if not rows:
            conn.close()
            return "查询成功，但没有找到匹配的数据。"

        # 格式化结果
        columns = [desc[0] for desc in cur.description]
        result = f"查询结果（{len(rows)} 行，{len(columns)} 列）：\n"
        result += " | ".join(columns) + "\n"
        result += "-" * (len(result) - 1) + "\n"
        for row in rows:
            result += " | ".join(str(row[c]) for c in columns) + "\n"

        conn.close()
        return result

    except Exception as e:
        return f"SQL 执行错误：{e}"


# ============================================================
# 第四部分：数据库 Agent
# ============================================================

class DatabaseAgent:
    """能自动查询数据库的 Agent。

    工作流程：
      用户问："AI 类图书的平均价格是多少？"
      → Agent 看 Schema → 生成 SQL → 安全执行 → 解释结果
    """

    def __init__(self):
        # 工具：执行 SQL
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "执行 SQL SELECT 查询。先看 Schema 再写 SQL。查询图书、顾客、订单信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL SELECT 查询语句。只允许 SELECT，不允许修改数据。",
                            },
                        },
                        "required": ["sql"],
                    },
                },
            },
        ]

    def chat(self, question: str) -> str:
        system_prompt = f"""你是数据库分析助手。用户用自然语言提问，你写 SQL 查数据库来回答。

{SCHEMA}

重要规则：
1. 先调用 query_database 工具执行 SQL，再根据结果回答
2. 只允许 SELECT 查询，不要尝试修改数据
3. 查询结果用中文回复用户，简洁清晰
4. SQL 中的字符串用单引号，表名和列名不需要引号
5. 如果问题涉及"最近7天"，用 date('now') 和 date('now', '-7 days')
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # 第一次调用：AI 决定是否需要查数据库
        response = DS.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=self.tools,
        )
        reply = response.choices[0].message

        if reply.tool_calls:
            for tc in reply.tool_calls:
                args = json.loads(tc.function.arguments)
                sql = args["sql"]
                print(f"\n  [SQL 查询] {sql}")

                result = safe_execute_sql(sql)
                print(f"  [查询结果] {result[:100]}...")

                messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [{
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }],
                })
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            # 第二次调用：AI 根据查询结果回答
            response = DS.chat.completions.create(
                model="deepseek-chat", messages=messages,
            )
            reply = response.choices[0].message

        return reply.content


# ============================================================
# 对比：纯 Agent vs Agent + 数据库
# ============================================================

def show_why_database():
    """演示为什么 Agent 需要数据库"""
    print("=" * 60)
    print("为什么 Agent 需要数据库？")
    print("=" * 60)
    print("""
没有数据库的 Agent：
  用户："昨天的销售额多少？"
  Agent："我没有数据，无法回答。"         ← 只能聊，不能查数

有数据库的 Agent（本项目）：
  用户："昨天的销售额多少？"
  Agent → 看 Schema → 写 SQL → 执行 → "昨天销售额 238 元，3 笔订单。"

这就是从"聊天机器人"到"业务助手"的关键一步。
实际工作中，企业的数据全在数据库里，Agent 必须会查数据库。
""")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Agent + 数据库 —— 让 AI 自动写 SQL")
    print("=" * 60)

    show_why_database()

    # 创建示例数据库
    create_sample_database()

    # 创建 Agent
    agent = DatabaseAgent()

    # 测试各种查询
    questions = [
        # 基础查询：筛选 + 计数
        "数据库里有多少本 AI 类的书？列出来。",
        # 聚合查询：平均、最大、最小
        "AI 类图书的平均价格是多少？",
        # 排序
        "最贵的 3 本书是哪些？按价格从高到低排列。",
        # 跨表 JOIN
        "每个顾客买了多少本书？按数量从多到少排列。",
        # 分组统计
        "每种分类的图书各有多少本？总库存是多少？",
    ]

    for q in questions:
        print(f"\n{'─' * 60}")
        print(f"用户: {q}")
        try:
            answer = agent.chat(q)
            print(f"AI: {answer}")
        except Exception as e:
            print(f"出错: {e}")

    print("\n" + "=" * 60)
    print("SQL 速查表（面试前过一遍）：")
    print("=" * 60)
    print("""
  SELECT * FROM books                           → 查全部
  SELECT * FROM books WHERE category = 'AI'     → 筛选
  SELECT AVG(price) FROM books                  → 平均值
  SELECT category, COUNT(*) FROM books GROUP BY category → 分组统计
  SELECT * FROM books ORDER BY price DESC LIMIT 3       → 排序取前3
  SELECT c.name, SUM(o.quantity) FROM customers c        → 多表关联
  JOIN orders o ON c.id = o.customer_id GROUP BY c.name

面试高频："让 AI 操作数据库有什么安全措施？"
你的回答："只允许 SELECT，拦截 DROP/DELETE/UPDATE，
          SQL 执行前做语法检查，防止注入和多语句执行。"
""")
