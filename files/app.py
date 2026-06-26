import uuid
import gradio as gr
from dotenv import load_dotenv
from agent import graph

load_dotenv()


def update_ui(config, current_logs):
    try:
        state = graph.get_state(config)
        vals = state.values if state else {}
    except Exception:
        vals = {}
        state = None

    logs_out = current_logs
    draft_out = ""

    if not vals:
        return logs_out, "⏳ جاري تحميل البيانات من النظام الذكي..."

    weather_report = vals.get("weather_report", "")
    is_finished = not (state and state.next)

    output_parts = []

    # 1. عرض تقرير الطقس أولاً إن وُجد (مصدره عقدة weather_node المستقلة)
    if weather_report:
        output_parts.append(f"🌤️ تقرير الطقس الحالي:\n{weather_report}\n\n{'=' * 30}\n\n")

    # 2. عرض المسودة النهائية إن وُجدت، وإلا عرض الخطة الحالية
    if vals.get("draft"):
        output_parts.append(vals["draft"][-1])
        if is_finished:
            output_parts.append(f"\n\n{'=' * 50}\n✨ تم التحقق وعرض التقرير الصافي والمناسب للموقف بنجاح! ✨")
    elif vals.get("plan"):
        output_parts.append(vals["plan"])
    elif not weather_report:
        output_parts.append("⏳ جاري معالجة البيانات وتفويض المسار الأنسب للموقف...")

    draft_out = "".join(output_parts)

    # 3. تحديث شاشة الـ Logs
    if weather_report and "🌤️ [نجاح]" not in logs_out:
        logs_out += "🌤️ [نجاح]: تم جلب بيانات حالة الطقس الحية والمباشرة بنجاح تام.\n"

    if state and state.next:
        if "writer" in state.next and vals.get('revision_number', 0) == 0:
            if "⏸️ [توقف مؤقت]" not in logs_out:
                logs_out += (
                    "\n⏸️ [توقف مؤقت]: تم توليد الخطة الهيكلية للمقال بنجاح.\n"
                    "👉 راجع الخطة بالأسفل، ثم اضغط 'استئناف واعتماد التعديلات'.\n"
                    "👉 في حال اردت تعديل المهمه عدلها واضغط معالجه من جديد.\n"
                )

    if not state or not state.next:
        if "✅ تم الانتهاء" not in logs_out:
            logs_out += "\n✅ تم إنتاج وتأكيد المخرجات النهائية بنجاح!\n"

    return logs_out, draft_out


def run_agent(task, max_revisions):
    if not task.strip():
        return "⚠️ يرجى إدخال موضوع المقال أو الاستعلام أولاً!", "", ""

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {
        "task": task,
        "plan": "",
        "weather_report": "",
        "content": [],
        "draft": [],
        "critique": [],
        "revision_number": 0,
        "max_revisions": int(max_revisions),
        "needs_weather": False,
        "needs_search": False,
        "needs_article": False,
    }

    logs = "🔄 تم بدء تشغيل العميل الذكي الموجه للمواقف والمهمات...\n"
    yield logs, "⏳ جاري إرسال النص للـ LLM ليقرر الموقف والمسار البرمجي المناسب...", thread_id

    try:
        for event in graph.stream(inputs, config=config, stream_mode="updates"):
            for node in event.keys():
                if node == "router":
                    logs += "🔷 قام الـ LLM بفحص الموقف وتحديد المسار المناسب (طقس / بحث / مباشر).\n"
                elif node == "weather_node":
                    logs += "🌦️ عقدة الطقس: تم جلب بيانات الطقس الحية بنجاح.\n"
                elif node == "search_node":
                    logs += "🔎 عقدة البحث: تم جلب نتائج المصادر بنجاح.\n"
                elif node == "planner":
                    logs += "🗂️ عقدة التخطيط: تم استلام نتائج الطقس/البحث وبناء هيكل المقال.\n"
            current_logs, current_draft = update_ui(config, logs)
            yield current_logs, current_draft, thread_id
    except Exception as e:
        logs += f"❌ حدث خطأ أثناء المعالجة الذكية: {str(e)}\n"
        yield logs, "خطأ في معالجة المدخلات", thread_id

    logs, draft_disp = update_ui(config, logs)
    yield logs, draft_disp, thread_id


def resume_agent(thread_id, current_logs, max_revisions, edited_plan_or_draft, new_task):
    if not thread_id:
        return "⚠️ لا توجد جلسة نشطة للاستئناف!", "", ""

    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
    except Exception:
        return "❌ تعذر جلب حالة الجلسة الحالية.", "", thread_id

    if not state or not state.next:
        return "⚠️ هذه الجلسة مكتملة ومنتهية بالفعل!\n", edited_plan_or_draft, thread_id

    updated_values = {
        "max_revisions": int(max_revisions),
        "task": new_task,
        "plan": edited_plan_or_draft,
    }

    graph.update_state(config, updated_values, as_node="planner")

    waiting_msg = "⏳ جاري إغلاق الجلسة وإخراج التقرير النهائي الصافي... يرجى الانتظار ثانية."
    yield waiting_msg, waiting_msg, thread_id

    try:
        graph.invoke(None, config=config)
    except Exception as e:
        error_msg = f"❌ حدث خطأ أثناء إغلاق المخطط: {str(e)}"
        yield error_msg, "حدث خطأ في الاستئناف", thread_id
        return

    final_logs, final_draft = update_ui(config, "✅ تم إنتاج وعرض الخرج النهائي بنجاح!\n")
    yield final_logs, final_draft, thread_id


with gr.Blocks() as app:
    gr.Markdown("# 🤖 نظام كتابة المقالات  ")

    # 1. الأعلى: منصة العرض التفاعلية + شاشة مراقبة سير العمليات
    with gr.Row():
        draft = gr.Textbox(label="منصة العرض التفاعلية ", lines=12, interactive=True, scale=2)
        logs = gr.Textbox(label="شاشة مراقبة سير العمليات ", lines=12, interactive=False, scale=1)

    # 2. الوسط: المهمة المطلوبة مع الحد الأقصى لجولات المراجعة
    with gr.Row():
        task = gr.Textbox(label="المهمة المطلوبة أو موضوع المقال", lines=3, scale=3, placeholder="اكتب موضوعك هنا...")
        max_rev = gr.Slider(1, 5, value=3, step=1, label="الحد الأقصى لجولات المراجعة ", scale=1)

    # 3. الأسفل: الأزرار (تمت إزالة مربع الجلسة)
    # نستخدم gr.State لتخزين رقم الجلسة خلف الكواليس
    thread_state = gr.State("") 
    
    with gr.Row():
        start = gr.Button("🔍 بدء المعالجة ", variant="primary")
        resume = gr.Button("✨ استئناف واعتماد التعديلات", variant="secondary")

    # تحديث الدالات لتتعامل مع thread_state بدلاً من مربع النص
    start.click(run_agent, [task, max_rev], [logs, draft, thread_state])
    resume.click(resume_agent, [thread_state, logs, max_rev, draft, task], [logs, draft, thread_state])

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())

