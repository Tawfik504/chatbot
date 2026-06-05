import operator
from typing import TypedDict, List, Annotated

# 🌟 تفعيل البيئة أولاً لحل مشاكل الصلاحيات والـ Threads
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# =========================
# MODEL
# =========================
model = ChatOpenAI(model="gpt-4o", temperature=0)

# =========================
# STATE
# =========================
class AgentState(TypedDict):
    task: str
    plan: str

    draft: Annotated[List[str], operator.add]
    critique: Annotated[List[str], operator.add]
    content: Annotated[List[str], operator.add]

    revision_number: int
    max_revisions: int
    
    # 💡 قمنا بإزالة المتغير paused اليدوي لأن المخطط سيقاطع التنفيذ تلقائياً

# =========================
# NODES
# =========================
def planner(state):
    res = model.invoke([
        SystemMessage(content="Create a structured plan."),
        HumanMessage(content=state["task"])
    ])
    return {"plan": res.content}


def writer(state):
    # نمرر آخر نقد تم توليده إن وجد لكي يستفيد منه الكاتب أثناء التعديل
    last_critique = state["critique"][-1] if state.get("critique") else "None"
    
    res = model.invoke([
        SystemMessage(content="Write a high quality article."),
        HumanMessage(content=f"Task: {state['task']}\nPlan: {state['plan']}\nLast Critique: {last_critique}")
    ])

    return {
        "draft": [res.content],
        "revision_number": state["revision_number"] + 1
    }


def critic(state):
    res = model.invoke([
        SystemMessage(content="Critique the text. Be concise and point out what needs improvement."),
        HumanMessage(content=state["draft"][-1])
    ])

    return {"critique": [res.content]}

# =========================
# ROUTING
# =========================
def route_after_writer(state):
    # الفحص الوحيد الآن: هل وصلنا للحد الأقصى للمراجعات المطلوبة؟
    if state["revision_number"] >= state["max_revisions"]:
        return END

    # إذا لم ينتهِ، نذهب للنقد فوراً
    return "critic"

# =========================
# GRAPH
# =========================
builder = StateGraph(AgentState)

# إضافة العقد الأساسية فقط (بدون عقدة الـ wait الميتة)
builder.add_node("planner", planner)
builder.add_node("writer", writer)
builder.add_node("critic", critic)

# الروابط والمنافذ
builder.set_entry_point("planner")
builder.add_edge("planner", "writer")

# حافة شرطية واحدة بعد الكاتب لمعرفة هل شارفنا على النهاية أم لا
builder.add_conditional_edges(
    "writer",
    route_after_writer,
    {
        "critic": "critic",
        END: END
    }
)

# بعد الناقد، نعود دائماً إلى الكاتب
builder.add_edge("critic", "writer")

# الذاكرة المؤقتة السريعة والمستقرة للـ UI
memory = InMemorySaver()

# 🌟 السحر الحقيقي هنا 🌟
# نقوم بـ compile للمخطط مع إخبار LangGraph بإيقاف ومقاطعة العمل إجبارياً (Interrupt) 
# "قبل" الدخول في عقدة الكاتب وعقدة الناقد. 
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["writer", "critic"]
)