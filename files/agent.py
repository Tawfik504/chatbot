import operator
import requests
from typing import TypedDict, List, Annotated, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# =========================================================
# 1. الأدوات (Tools)
# =========================================================

@tool
def get_current_weather(location: str) -> str:
    """جلب حالة الطقس الحالية ودرجات الحرارة لمدينة معينة."""
    try:
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            temp = current['temp_C']
            desc = current['lang_ar'][0]['value'] if 'lang_ar' in current else current['weatherDesc'][0]['value']
            humidity = current['humidity']
            return f"🌤️ حالة الطقس في {location}: {desc} \n🌡️ درجة الحرارة الحالية: {temp}°C \n💧 نسبة الرطوبة: {humidity}%"
        return f"⚠️ تعذر جلب بيانات الطقس لمدينة ({location}) حالياً."
    except Exception as e:
        return f"❌ حدث خطأ أثناء جلب الطقس: {str(e)}"


model = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearch(max_results=3)

tools = [search_tool, get_current_weather]
model_with_tools = model.bind_tools(tools)


# =========================================================
# 2. حالة العميل (State)
# =========================================================

class AgentState(TypedDict):
    task: str
    plan: str
    weather_report: str
    content: Annotated[List[str], operator.add]
    draft: Annotated[List[str], operator.add]
    critique: Annotated[List[str], operator.add]
    revision_number: int
    max_revisions: int
    needs_weather: bool
    needs_search: bool
    needs_article: bool
    weather_args: Optional[Dict]
    search_args: Optional[Dict]


# =========================================================
# 3. عقدة التوجيه (Router) — بديل core_agent القديمة
#    مهمتها الوحيدة: تحليل الطلب وتحديد أي عقدة تالية يجب تفعيلها
#    (طقس / بحث / مباشر) دون تنفيذ أي أداة بنفسها
# =========================================================

def router(state: AgentState):
    task = state["task"]

    # تحديد هل المهمة تتضمن كتابة مقال أم هي استعلام مباشر فقط
    needs_article = ("مقال" in task) or ("اكتب" in task)

    messages = [
        SystemMessage(content=(
            "أنت مساعد ذكي تستخدم الأدوات المتاحة بحرية حسب حاجة المهمة:\n"
            "- استخدم أداة الطقس (get_current_weather) إذا كانت المهمة تحتاج بيانات طقس حالية.\n"
            "- استخدم أداة البحث (TavilySearch) إذا كانت المهمة تحتاج معلومات/مصادر لكتابة مقال.\n"
            "- يمكنك استخدام الأداتين معاً إن احتاج الأمر (مثال: مقال عن طقس بلد معين).\n"
            "- إذا لم تكن هناك حاجة لأي أداة، أجب مباشرة بنص قصير."
        )),
        HumanMessage(content=f"المهمة المطلوبة: {task}")
    ]
    response = model_with_tools.invoke(messages)

    needs_weather = False
    needs_search = False
    weather_args = {}
    search_args = {}

    if response.tool_calls:
        for tool_call in response.tool_calls:
            # نقارن بالاسم الفعلي المسجل للأداة بدل افتراض اسم الكلاس
            # (هذا يصحح خطأ شائع كان يمنع تفعيل عقدة البحث في النسخة القديمة)
            if tool_call["name"] == get_current_weather.name:
                needs_weather = True
                weather_args = tool_call["args"]
            elif tool_call["name"] == search_tool.name:
                needs_search = True
                search_args = tool_call["args"]

    update = {
        "needs_weather": needs_weather,
        "needs_search": needs_search,
        "needs_article": needs_article,
        "weather_args": weather_args,
        "search_args": search_args,
    }

    # لا توجد أداة مطلوبة ولا مقال مطلوب => رد مباشر من النموذج وإنهاء المهمة
    if not needs_weather and not needs_search and not needs_article:
        update["draft"] = [response.content]

    return update


def route_after_router(state: AgentState):
    if state.get("needs_weather"):
        return "weather_node"
    if state.get("needs_search"):
        return "search_node"
    if state.get("needs_article"):
        return "planner"
    return END

# =========================================================
# 4. عقدة معالجة الطقس (مستقلة تماماً)
# =========================================================

def weather_node(state: AgentState):
    args = state.get("weather_args", {}) or {"location": state["task"]}
    weather_info = get_current_weather.invoke(args)

    update = {"weather_report": weather_info}

    # إذا لم يكن هناك حاجة لمقال، التقرير نفسه هو المخرج النهائي
    if not state.get("needs_article"):
        update["draft"] = [f"🌤️ تقرير الطقس:\n{weather_info}"]

    return update


def route_after_weather(state: AgentState):
    if state.get("needs_search"):
        return "search_node"
    if state.get("needs_article"):
        return "planner"
    return END

# =========================================================
# 5. عقدة البحث (مستقلة تماماً) — تغذّي عقدة planner بالنتائج
# =========================================================

def search_node(state: AgentState):
    args = state.get("search_args", {}) or {"query": state["task"]}
    raw_results = search_tool.invoke(args)

    # TavilySearch قد يرجع dict فيه مفتاح "results" أو يرجع قائمة مباشرة،
    # نتعامل مع الحالتين لتجنّب كسر الكود
    items = raw_results.get("results", []) if isinstance(raw_results, dict) else (raw_results or [])

    formatted = "\n\n".join(
        f"🔗 المصدر: {res.get('url', 'غير معروف')}\n📝 المحتوى: {res.get('content', '')}"
        for res in items
    ) or "لم يتم العثور على نتائج بحث مفيدة."

    update = {"content": [formatted]}

    if not state.get("needs_article"):
        update["draft"] = [f"🔍 نتائج البحث:\n\n{formatted}"]

    return update


def route_after_search(state: AgentState):
    if state.get("needs_article"):
        return "planner"
    return END

# =========================================================
# 6. عقدة التخطيط — تستقبل مخرجات الطقس و/أو البحث وتبني خطة المقال
# =========================================================

def planner(state: AgentState):
    # إذا كانت الخطة موجودة مسبقاً (مثلاً بعد استئناف الجلسة بخطة معدّلة من المستخدم)
    # لا نعيد توليدها من جديد
    if state.get("plan") and not str(state["plan"]).startswith("⏳"):
        return {}

    content_list = state.get("content", [])
    research_data = "\n\n".join(content_list) if content_list else "لا يوجد بيانات بحث متاحة."

    weather_data = state.get("weather_report", "")
    weather_note = (
        f"\n\nبيانات الطقس المتوفرة (استخدمها فقط إذا كان موضوع المقال متعلقاً بالطقس):\n{weather_data}"
        if weather_data else ""
    )

    prompt = f"""أنت خبير في هيكلة المقالات.
المهمة: {state['task']}

سياق البحث المتاح:
{research_data}{weather_note}

قم بصياغة خطة مقال احترافية ومنظمة (عناوين فرعية ونقاط رئيسية) بناءً على السياق أعلاه.
"""

    res = model.invoke([
        SystemMessage(content="أنت خبير محترف في وضع الخطط الهيكلية للمقالات. ركز على الموضوع الأساسي للمقال."),
        HumanMessage(content=prompt)
    ])

    return {"plan": f"🤖 **الخطة المقترحة:**\n\n{res.content}"}

# =========================================================
# 7. عقدة الكتابة
# =========================================================

def writer(state: AgentState):
    plan = state.get("plan", "")
    last_critique = state["critique"][-1] if state.get("critique") else "لا يوجد نقد سابق."
    content_list = state.get("content", [])
    context = content_list[-1] if content_list else "لا يوجد معلومات إضافية من الأدوات."

    res = model.invoke([
        SystemMessage(content="أنت كاتب مقالات محترف ومبدع. اكتب نصاً مفصلاً وشاملاً ومصاغاً بأسلوب جذاب بناءً على الخطة المعتمدة وملاحظات المدقق."),
        HumanMessage(content=(
            f"المهمة الأساسية: {state['task']}\n\n"
            f"سياق الأدوات المتوفر:\n{context}\n\n"
            f"الخطة الهيكلية المعتمدة:\n{plan}\n\n"
            f"ملاحظات المراجعة الأخيرة:\n{last_critique}"
        ))
    ])
    return {
        "draft": [res.content],
        "revision_number": state.get("revision_number", 0) + 1
    }


# =========================================================
# 8. عقدة التدقيق
# =========================================================

def critic(state: AgentState):
    res = model.invoke([
        SystemMessage(content="أنت مصحح ومدقق لغوي وعلمي محترف. راجع نص المسودة المكتوب واكتب ملاحظاتك بشكل مختصر ومباشر في نقاط لتحسين جودة النص، وإذا كان ممتازاً اكتب كلمة 'مكتمل'."),
        HumanMessage(content=state["draft"][-1])
    ])
    return {"critique": [res.content]}


def route_after_writer(state: AgentState):
    if state["revision_number"] >= state["max_revisions"]:
        return END
    return "critic"

# =========================================================
# 9. بناء المخطط (Graph)
# =========================================================

builder = StateGraph(AgentState)

builder.add_node("router", router)
builder.add_node("weather_node", weather_node)
builder.add_node("search_node", search_node)
builder.add_node("planner", planner)
builder.add_node("writer", writer)
builder.add_node("critic", critic)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "weather_node": "weather_node",
        "search_node": "search_node",
        "planner": "planner",
        END: END,
    }
)

builder.add_conditional_edges(
    "weather_node",
    route_after_weather,
    {
        "search_node": "search_node",
        "planner": "planner",
        END: END,
    }
)

builder.add_conditional_edges(
    "search_node",
    route_after_search,
    {
        "planner": "planner",
        END: END,
    }
)

builder.add_edge("planner", "writer")
builder.add_conditional_edges("writer", route_after_writer, {"critic": "critic", END: END})
builder.add_edge("critic", "writer")

memory = InMemorySaver()
# نوقف قبل "writer" للسماح للمستخدم بمراجعة/تعديل الخطة الناتجة عن planner قبل بدء الكتابة
graph = builder.compile(checkpointer=memory, interrupt_before=["writer"])
