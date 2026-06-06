import operator
from typing import TypedDict, List, Annotated

# تحميل إعدادات البيئة ومفاتيح التشغيل
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 🌟 تم التعديل هنا: الاستيراد المستقر والمضمون دائماً لمنع خطأ الـ ImportError
from langchain_community.tools import TavilySearchResults

# =========================
# إعداد النموذج والأدوات
# =========================
# نستخدم نموذج جي بي تي 4 وتحديد درجة الإبداع إلى صفر لضمان الدقة
model = ChatOpenAI(model="gpt-4o", temperature=0)

# تحديد أداة البحث وجعلها تجلب أفضل 3 نتائج فقط لسرعة الأداء
search_tool = TavilySearchResults(max_results=3)

# =========================
# ذاكرة النظام (حالة العميل)
# =========================
class AgentState(TypedDict):
    task: str           # سؤال أو مهمة الزبون
    plan: str           # خطة العمل المنشأة
    content: Annotated[List[str], operator.add]   # نصوص ومعلومات البحث المسترجعة
    draft: Annotated[List[str], operator.add]     # المسودات والمقالات المكتوبة
    critique: Annotated[List[str], operator.add]  # الملاحظات والنقد
    revision_number: int  # رقم المراجعة الحالية
    max_revisions: int    # الحد الأقصى للمراجعات المطلوب

# =========================
# العقد (خطوات العمل)
# =========================

def researcher(state):
    """1. عقدة البحث: تأخذ سؤال الزبون وتبحث عنه في شبكة الإنترنت"""
    query = state["task"]
    
    # تشغيل أداة البحث لجلب البيانات
    search_results = search_tool.invoke({"query": query})
    
    # ترتيب وتنسيق النتائج المسترجعة في نص واحد واضح
    formatted_results = "\n---\n".join(
        [f"المصدر: {res.get('url', 'لا يوجد')}\nالمحتوى: {res.get('content', '')}" for res in search_results]
    )
    
    # حفظ نتائج البحث في قائمة المحتويات
    return {"content": [formatted_results]}


def planner(state):
    """2. عقدة التخطيط: تضع خطة المقال بناءً على نتائج البحث المسترجعة"""
    context = state["content"][-1] if state.get("content") else "لم يتم العثور على معلومات من الإنترنت."
    
    res = model.invoke([
        SystemMessage(content="أنت خبير في وضع الخطط. صمم خطة هيكلية للمقال بناءً على معلومات البحث المسترجعة من الإنترنت والمرفقة في الرسالة."),
        HumanMessage(content=f"سؤال الزبون: {state['task']}\n\nمعلومات البحث المسترجعة:\n{context}")
    ])
    return {"plan": res.content}


def writer(state):
    """3. عقدة الكتابة: صياغة المقال بالاعتماد على البحث والخطة والنقد السابق"""
    last_critique = state["critique"][-1] if state.get("critique") else "لا يوجد نقد سابق."
    context = state["content"][-1] if state.get("content") else "لم يتم العثور على معلومات من الإنترنت."
    
    res = model.invoke([
        SystemMessage(content="أنت كاتب مقالات محترف. اكتب مقالاً مفصلاً ودقيقاً مستعيناً بمعلومات البحث ومتتبعاً للخطة الهيكلية الموضوعة."),
        HumanMessage(content=f"المهمة: {state['task']}\n\nمعلومات البحث:\n{context}\n\nالخطة:\n{state['plan']}\n\nالنقد السابق:\n{last_critique}")
    ])

    return {
        "draft": [res.content],
        "revision_number": state["revision_number"] + 1
    }


def critic(state):
    """4. عقدة النقد: مراجعة المقال المكتوب وتحديد نقاط التحسين"""
    res = model.invoke([
        SystemMessage(content="أنت مصحح ومدقق لغوي وعلمي. راجع النص واكتب ملاحظاتك بشكل مختصر ومباشر وحدد ما يحتاج إلى تحسين."),
        HumanMessage(content=state["draft"][-1])
    ])
    return {"critique": [res.content]}

# =========================
# توجيه المسار تلقائياً
# =========================
def route_after_writer(state):
    """فحص هل وصلنا إلى الحد الأقصى من التعديلات والمراجعات أم لا"""
    if state["revision_number"] >= state["max_revisions"]:
        return END  # إنهاء العمل
    return "critic"  # الذهاب إلى النقد

# =========================
# بناء المخطط وتوصيله
# =========================
builder = StateGraph(AgentState)

# إضافة الخطوات الأربعة إلى المخطط
builder.add_node("researcher", researcher)
builder.add_node("planner", planner)
builder.add_node("writer", writer)
builder.add_node("critic", critic)

# تحديد مسار التدفق بين الخطوات
builder.set_entry_point("researcher")       # البداية من عقدة البحث أولاً
builder.add_edge("researcher", "planner")   # الانتقال من البحث إلى التخطيط
builder.add_edge("planner", "writer")       # الانتقال من التخطيط إلى الكتابة

# تحديد المسار الشرطي بعد كتابة المقال
builder.add_conditional_edges(
    "writer",
    route_after_writer,
    {
        "critic": "critic",  # إذا لم تنتهِ المراجعات يذهب للناقد
        END: END             # إذا انتهت المراجعات يقف البرنامج وينتهي
    }
)

# بعد الانتهاء من النقد، يعود النظام دائماً إلى الكاتب لتحديث المقال
builder.add_edge("critic", "writer")

# إعداد ذاكرة الحفظ المؤقتة لضمان استقرار جلسات المستخدمين
memory = InMemorySaver()

# تجميع المخطط وتفعيل خاصية المقاطعة التلقائية
# النظام سيتوقف تلقائياً (قبل) مرحلة التخطيط ليعرض للمستخدم نتائج البحث أولاً،
# وكذلك سيتوقف (قبل) مرحلة النقد لعرض المسودة والملاحظات.
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["planner", "critic"]
)