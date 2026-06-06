import uuid
import gradio as gr
from dotenv import load_dotenv

# تحميل ملف البيئة والمفاتيح
load_dotenv()

# استيراد المخطط الذي قمنا ببنائه في الملف السابق
from agent import graph

# =========================
# دالة مساعدة لتحديث البيانات في الواجهة
# =========================
def get_latest_ui_data(config, current_logs):
    """تقوم هذه الدالة بجلب البيانات الحالية من الذاكرة وعرضها في الصناديق المناسبة بشكل آمن"""
    state_now = graph.get_state(config)
    current_state = state_now.values
    
    draft_display = ""
    
    # 1. عرض المحتوى المناسب في صندوق النتائج بناءً على المرحلة الحالية
    if current_state.get("draft") and len(current_state["draft"]) > 0:
        draft_display = f"📝 [المسودة الأخيرة - المراجعة رقم {current_state.get('revision_number', 0)}]:\n\n{current_state['draft'][-1]}"
        if current_state.get("critique") and len(current_state["critique"]) > 0:
            draft_display += f"\n\n{'='*40}\n❌ [ملاحظات ونقد المصحح]:\n\n{current_state['critique'][-1]}"
            
    elif current_state.get("plan"):
        draft_display = f"📋 [الخطة الهيكلية المنشأة للمقال]:\n\n{current_state['plan']}"
        
    elif current_state.get("content") and len(current_state["content"]) > 0:
        draft_display = f"🔍 [نتائج البحث المسترجعة من شبكة الإنترنت]:\n\n{current_state['content'][-1]}"
    else:
        draft_display = "⏳ جاري البحث في الإنترنت أو معالجة البيانات..."

    # 2. تحديث سجل العمليات وإضافة إرشادات التوقف للمستخدم
    updated_logs = current_logs
    if state_now.next:
        updated_logs += f"\n⏸️ توقف النظام تلقائياً قبل الخطوة: {list(state_now.next)}\n"
        if "planner" in state_now.next:
            updated_logs += "💡 [إمكانية التوجيه متاحة]:\n"
            updated_logs += "   - يمكنك رؤية نتائج البحث في الصندوق المقابل لمعاينتها.\n"
            updated_logs += "   - اضغط على زر 'استئناف' لتبدأ عملية التخطيط والكتابة بناءً على هذه المعلومات!\n"
        else:
            updated_logs += "👉 اضغط على زر 'استئناف' للانتقال إلى الخطوة التالية في المخطط.\n"
        updated_logs += "-" * 40 + "\n"
    else:
        updated_logs += "\n✅ انتهت عملية النظام بالكامل وتم الوصول للنهاية!\n"

    return updated_logs, draft_display


# =========================
# دالة تشغيل العميل لأول مرة (Start)
# =========================
def run_agent(task, max_revisions):
    if not task.strip():
        yield "⚠️ الرجاء كتابة السؤال أو المهمة أولاً!", "", ""
        return

    # إنشاء رقم تعريف فريد لهذه الجلسة
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # إعداد البيانات الأولية للنظام
    inputs = {
        "task": task,
        "plan": "",
        "content": [],
        "draft": [],
        "critique": [],
        "revision_number": 0,
        "max_revisions": int(max_revisions)
    }

    logs = "🔄 جاري بدء تشغيل العميل الذكي...\n📡 تم تفعيل أداة البحث في الإنترنت...\n"
    yield logs, "جاري البحث في الإنترنت وحصد المعلومات...", thread_id

    # تشغيل النظام تدريجياً (سيتوقف تلقائياً بعد مرحلة البحث وقبل مرحلة التخطيط)
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        if isinstance(event, dict):
            for node in event.keys():
                logs += f"🔷 اكتملت الخطوة: '{node}'\n"
        yield logs, "جاري تحديث البيانات الحالية...", thread_id

    # قراءة البيانات النهائية وعرضها للزبون في الواجهة
    logs, draft_display = get_latest_ui_data(config, logs)
    yield logs, draft_display, thread_id


# =========================
# دالة استئناف العمل (Resume)
# =========================
def resume_agent(thread_id, current_task_text, current_logs):
    if not thread_id:
        yield "⚠️ لم يتم العثور على جلسة نشطة! يرجى الضغط على زر البدء أولاً.", "", ""
        return

    config = {"configurable": {"thread_id": thread_id}}
    
    # تحديث نص السؤال في الذاكرة في حال قام الزبون بتعديله أثناء التوقف
    graph.update_state(config, {"task": current_task_text})

    # الحفاظ على السجلات القديمة وإضافة السجل الجديد عليها لمنع الاختفاء
    logs = current_logs + f"▶️ جاري استئناف العمل... الانتقال إلى الخطوة التالية.\n" + "-" * 40 + "\n"
    yield logs, "جاري المعالجة والكتابة...", thread_id

    # استئناف الحركة من نقطة التوقف الحالية حتى نقطة التوقف القادمة
    for event in graph.stream(None, config=config, stream_mode="updates"):
        if isinstance(event, dict):
            for node in event.keys():
                logs += f"🔷 اكتملت الخطوة: '{node}'\n"
        
        # بث التحديثات خطوة بخطوة إلى الواجهة مباشرة ليراها المستخدم
        temp_logs, temp_draft = get_latest_ui_data(config, logs)
        yield temp_logs, temp_draft, thread_id

    # التحديث الختامي بعد انتهاء المرحلة الحالية بالكامل
    logs, draft_display = get_latest_ui_data(config, logs)
    yield logs, draft_display, thread_id


def pause_agent(thread_id):
    return "⏸️ النظام يتم إيقافه تلقائياً من خلال البنية البرمجية، لا حاجة للإيقاف اليدوي.", gr.update(), thread_id


# =========================
# تصميم واجهة المستخدم المرئية
# =========================
with gr.Blocks() as app:
    gr.Markdown("# 🚀 نظام كتابة المقالات الذكي المدعوم بالبحث في الإنترنت")

    with gr.Row():
        with gr.Column():
            task = gr.Textbox(label="سؤال أو مهمة الزبون", lines=4, placeholder="مثال: ما هي آخر المستجدات في سباقات الفورمولا 1 لهذا الموسم؟")
            max_rev = gr.Slider(1, 5, value=2, step=1, label="الحد الأقصى لمراجعات المقال")

            with gr.Row():
                start = gr.Button("بدء العمل", variant="primary")
                resume = gr.Button("استئناف", variant="secondary")
                pause = gr.Button("معلومات")

            thread = gr.Textbox(label="رقم تعريف الجلسة الحالية", interactive=False)

        with gr.Column():
            logs = gr.Textbox(label="سجل سير العمليات وتتبع الخطوات", lines=10, interactive=False)
            draft = gr.Textbox(label="صندوق النتائج (البحث / الخطة / المقال النهائي)", lines=15, interactive=False)

    # ربط الأزرار بالدوال البرمجية وتحديد المدخلات والمخرجات لكل زر
    start.click(run_agent, [task, max_rev], [logs, draft, thread])
    pause.click(pause_agent, thread, [logs, draft, thread])
    resume.click(resume_agent, [thread, task, logs], [logs, draft, thread])

# تشغيل الواجهة بتنسيق مريح وبسيط
app.launch(theme=gr.themes.Soft())