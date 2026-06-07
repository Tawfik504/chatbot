import operator
from typing import TypedDict, List, Annotated

from dotenv import load_dotenv  
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_community.tools import TavilySearchResults


model = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearchResults(max_results=3)


tools = [search_tool]
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
    """
    عقدة العميل الذكي: يقرر بنفسه هل يبحث في الإنترنت، أم يضع الخطة مباشرة 
    بناءً على فهمه للسؤال ومجريات الحوار.
    """
    task = state["task"]
    

    messages = [
        SystemMessage(content="أنت باحث ومخطط ذكي. لديك أداة بحث في الإنترنت. إذا كان سؤال المستخدم يتطلب معلومات حديثة أو تفاصيل لا تعرفها، استخدم الأداة. إذا كانت لديك المعلومات الكافية، ضع خطة المقال مباشرة دون استخدام الأداة."),
        HumanMessage(content=f"المهمة المطلوبة: {task}")
    ]
    

    response = model_with_tools.invoke(messages)
    

    if response.tool_calls:
        tool_call = response.tool_calls[0]

        search_results = search_tool.invoke(tool_call["args"])
        
        formatted_results = "\n---\n".join(
            [f"المصدر: {res.get('url', 'لا يوجد')}\nالمحتوى: {res.get('content', '')}" for res in search_results]
        )
        return {"content": [formatted_results]}
    

    return {"plan": response.content}


def planner(state):
    """عقدة التخطيط: تعمل فقط في حال تم البحث أولاً وتحتاج لتنسيق النتائج في خطة"""

    if state.get("plan"):
        return {}
        
    context = state["content"][-1] if state.get("content") else "لم يتم العثور على معلومات."
    
    res = model.invoke([
        SystemMessage(content="أنت خبير في وضع الخطط. صمم خطة هيكلية للمقال بناءً على معلومات البحث المتاحة."),
        HumanMessage(content=f"سؤال الزبون: {state['task']}\n\nمعلومات البحث:\n{context}")
    ])
    return {"plan": res.content}


def writer(state):
    """عقدة الكتابة: صياغة المقال"""
    last_critique = state["critique"][-1] if state.get("critique") else "لا يوجد نقد سابق."
    context = state["content"][-1] if state.get("content") else "لا يوجد معلومات بحث إضافية."
    plan = state.get("plan", "اكتب المقال مباشرة برؤيتك الاحترافية.")
    
    res = model.invoke([
        SystemMessage(content="أنت كاتب مقالات محترف. اكتب مقالاً مفصلاً ودقيقاً مستعيناً بالخطة والمعلومات المتاحة."),
        HumanMessage(content=f"المهمة: {state['task']}\n\nالبحث:\n{context}\n\nالخطة:\n{plan}\n\nالنقد السابق:\n{last_critique}")
    ])

    return {
        "draft": [res.content],
        "revision_number": state.get("revision_number", 0) + 1
    }


def critic(state):
    """عقدة النقد: مراجعة النص"""
    res = model.invoke([
        SystemMessage(content="أنت مصحح ومدقق لغوي وعلمي. راجع النص واكتب ملاحظاتك بشكل مختصر ومباشر."),
        HumanMessage(content=state["draft"][-1])
    ])
    return {"critique": [res.content]}


def route_after_agent(state):
    """توجيه ذكي: إذا قرر العميل البحث نذهب للتخطيط، وإذا وضع الخطة مباشرة نذهب للكتابة"""
    if state.get("content") and not state.get("plan"):
        return "planner"
    return "writer"

def route_after_writer(state):
    if state["revision_number"] >= state["max_revisions"]:
        return END
    return "critic"


builder = StateGraph(AgentState)

# إضافة العقد
builder.add_node("core_agent", core_agent)
builder.add_node("planner", planner)
builder.add_node("writer", writer)
builder.add_node("critic", critic)


builder.set_entry_point("core_agent")


builder.add_conditional_edges(
    "core_agent",
    route_after_agent,
    {
        "planner": "planner",
        "writer": "writer"
    }
)

builder.add_edge("planner", "writer")


builder.add_conditional_edges(
    "writer",
    route_after_writer,
    {
        "critic": "critic",
        END: END
    }
)

builder.add_edge("critic", "writer")

memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["critic"]
)