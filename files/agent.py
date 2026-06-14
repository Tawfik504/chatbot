import operator
import requests
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv 

load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# 1. أداة الطقس الحية
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

# دمج الأدوات ليتخذ الـ LLM قراره بحرية تامة بناءً على الموقف
tools = [search_tool, get_current_weather]
model_with_tools = model.bind_tools(tools)

class AgentState(TypedDict):
    task: str
    plan: str
    content: Annotated[List[str], operator.add]
    draft: Annotated[List[str], operator.add]
    critique: Annotated[List[str], operator.add]
    revision_number: int
    max_revisions: int

def core_agent(state):
    task = state["task"]
    messages = [
        SystemMessage(content="أنت باحث ومخطط ذكي. لديك أدوات للبحث في الإنترنت وجلب الطقس الحالي. تفحص طلب المستخدم بدقة: إذا طلب طقس مدينة معينة، استدعي أداة الطقس فوراً. إذا طلب معلومات حديثة عامة أو مقالاً يتطلب بحثاً، استخدم أداة البحث Tavily. إذا كانت لديك المعرفة الكافية دون أدوات، ضع الخطة مباشرة."),
        HumanMessage(content=f"المهمة المطلوبة: {task}")
    ]
    response = model_with_tools.invoke(messages)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        
        if tool_name == "TavilySearch":
            search_results = search_tool.invoke(tool_call["args"])
            formatted_results = "\n---\n".join(
                [f"المصدر: {res.get('url', 'لا يوجد')}\nالمحتوى: {res.get('content', '')}" for res in search_results]
            )
            return {"content": [f"[SEARCH]\n{formatted_results}"]}
            
        elif tool_name == "get_current_weather":
            weather_result = get_current_weather.invoke(tool_call["args"])
            return {"content": [f"[WEATHER]\n{weather_result}"]}
            
    return {"plan": response.content}

def planner(state):
    task_input = state["task"]
    
    # إذا قام المستخدم بتعديل الخطة يدوياً في صندوق العرض، نحافظ على تعديله ولا نقوم بإعادة الكتابة فوقه
    if state.get("plan") and not state["plan"].startswith("⏳"):
        return {}
        
    raw_content = state["content"][-1] if state.get("content") else ""
    
    # المسار الأول: إذا كانت البيانات المجلوبة تخص الطقس (مسار سريع)
    if "[WEATHER]" in raw_content:
        weather_data = raw_content.replace("[WEATHER]\n", "")
        explanation = f"🤖 **قرار الـ LLM الذكي:** تم تحليل طلبك وتحديد أن الموقف يتطلب حالة الطقس الحية؛ لذا قرر النموذج تشغيل أداة الطقس حصراً دون مسار كتابة المقالات المفتوح.\n\n{'-'*40}\n\n📊 **تقرير الأرصاد الجوية المستلم:**\n{weather_data}"
        return {"plan": explanation}
        
    # المسار الثاني: إذا كانت بيانات بحث عام من Tavily (إجبار توليد الخطة مع سياق كامل)
    elif "[SEARCH]" in raw_content:
        actual_data = raw_content.replace("[SEARCH]\n", "")
        
        # 🎯 تم تحسين الـ Prompt وتمرير المهمة الأصلية بوضوح لضمان توليد الخطة كاملة
        res = model.invoke([
            SystemMessage(content="أنت خبير محترف في وضع الخطط الهيكلية للمقالات. بناءً على موضوع المستخدم ومعلومات البحث المرفقة، قم بصياغة خطة مفصلة تحتوي على مقدمة، وعناصر أساسية، وخاتمة مقترحة للمقال بشكل منظم."),
            HumanMessage(content=f"الموضوع المطلوب: {task_input}\n\nمعلومات البحث المستخرجة من الأداة:\n{actual_data}")
        ])
        
        full_plan = f"🤖 **قرار الـ LLM الذكي:** تم رصد طلب بحث ومقال عام، وقرر النموذج تشغيل أداة البحث العالمية (Tavily).\n\n{'-'*40}\n\n📋 **الخطة الهيكلية المقترحة لبناء المقال:**\n{res.content}"
        return {"plan": full_plan}
        
    # المسار الثالث: إذا لم يستدعي النموذج أي أداة ويمتلك المعرفة المسبقة
    res = model.invoke([
        SystemMessage(content="أنت خبير في وضع الخطط الهيكلية للمقالات. قم بصياغة خطة مفصلة تحتوي على الأقسام الرئيسية للمقال المطلوب."),
        HumanMessage(content=f"الموضوع المطلوب: {task_input}")
    ])
    return {"plan": res.content}

def writer(state):
    plan = state.get("plan", "")
    
    # إذا كان مسار طقس سريع، نمرر التقرير مباشرة كمسودة نهائية دون تكرار الكتابة
    if "تقرير الأرصاد الجوية المستلم" in plan:
        return {"draft": [plan], "revision_number": state.get("revision_number", 0) + 1}
        
    last_critique = state["critique"][-1] if state.get("critique") else "لا يوجد نقد سابق."
    context = state["content"][-1] if state.get("content") else "لا يوجد معلومات إضافية من الأدوات."
    
    res = model.invoke([
        SystemMessage(content="أنت كاتب مقالات محترف ومبدع. اكتب نصاً مفصلاً وشاملاً ومصاغاً بأسلوب جذاب بناءً على الخطة المعتمدة وملاحظات المدقق."),
        HumanMessage(content=f"المهمة الأساسية: {state['task']}\n\nسياق الأدوات المتوفر:\n{context}\n\nالخطة الهيكلية المعتمدة:\n{plan}\n\nملاحظات المراجعة الأخيرة:\n{last_critique}")
    ])
    return {
        "draft": [res.content],
        "revision_number": state.get("revision_number", 0) + 1
    }

def critic(state):
    res = model.invoke([
        SystemMessage(content="أنت مصحح ومدقق لغوي وعلمي محترف. راجع نص المسودة المكتوب واكتب ملاحظاتك بشكل مختصر ومباشر في نقاط لتحسين جودة النص، وإذا كان ممتازاً اكتب كلمة 'مكتمل'."),
        HumanMessage(content=state["draft"][-1])
    ])
    return {"critique": [res.content]}

def route_after_agent(state):
    if state.get("content") and not state.get("plan"):
        return "planner"
    return "writer"

def route_after_writer(state):
    plan_content = state.get("plan", "")
    # في مسار الطقس، ننهي الجلسة فوراً عند الوصول للكاتب
    if "تقرير الأرصاد الجوية المستلم" in plan_content:
        return END
        
    if state["revision_number"] >= state["max_revisions"]:
        return END
    return "critic"

builder = StateGraph(AgentState)
builder.add_node("core_agent", core_agent)
builder.add_node("planner", planner)
builder.add_node("writer", writer)
builder.add_node("critic", critic)
builder.set_entry_point("core_agent")

builder.add_conditional_edges("core_agent", route_after_agent, {"planner": "planner", "writer": "writer"})
builder.add_edge("planner", "writer")
builder.add_conditional_edges("writer", route_after_writer, {"critic": "critic", END: END})
builder.add_edge("critic", "writer")

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["writer"])