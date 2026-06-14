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

    is_finished = not (state and state.next)

    # عرض الخرج الصافي للمقال أو تقرير الطقس بذكاء
    if vals.get("draft") and isinstance(vals["draft"], list) and len(vals["draft"]) > 0:
        if is_finished:
            draft_out = vals['draft'][-1] + "\n\n" + "="*50 + "\n✨ تم التحقق وعرض التقرير الصافي والمناسب للموقف بنجاح! ✨"
        else:
            draft_out = vals['draft'][-1]
    elif vals.get("plan"):
        draft_out = vals["plan"]
    else:
        draft_out = "⏳ جاري معالجة البيانات وتفويض المسار الأنسب للموقف..."

    # 🔄 التعديل المطلوب هنا: تغيير نص المسار السريع للطقس في شاشة الـ Logs
    plan_text = vals.get("plan", "")
    if "تقرير الأرصاد الجوية المستلم" in plan_text:
        if "🌤️ [نجاح]" not in logs_out:
            logs_out += "🌤️ [نجاح]: تم جلب بيانات حالة الطقس الحية والمباشرة بنجاح تام عبر الـ LLM.\n"
    else:
        if state and state.next:
            if "writer" in state.next and vals.get('revision_number', 0) == 0:
                if "⏸️ [توقف مؤقت]" not in logs_out:
                    logs_out += f"\n⏸️ [توقف مؤقت]: تم توليد الخطة الهيكلية للمقال بنجاح.\n👉 راجع الخطة بالأسفل، ثم اضغط 'استئناف واعتماد التعديلات' لبدء جولات الصياغة والمراجعة.\n"

    if not state or not state.next:
        if "✅ تم الانتهاء" not in logs_out:
            logs_out += "\n✅ تم إنتاج وتأكيد المخرجات النهائية بنجاح تامة وفق الموقف المناسب!\n"
        
    return logs_out, draft_out

def run_agent(task, max_revisions):
    if not task.strip(): 
        return "⚠️ يرجى إدخال موضوع المقال أو الاستعلام أولاً!", "", ""
        
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "task": task, 
        "content": [], 
        "draft": [], 
        "critique": [], 
        "revision_number": 0, 
        "max_revisions": int(max_revisions)
    }

    logs = "🔄 تم بدء تشغيل العميل الذكي الموجه للمواقف والمهمات...\n"
    yield logs, "⏳ جاري إرسال النص للـ LLM ليقرر الموقف والأداة والمسار البرمجي المتوافق...", thread_id

    try:
        for event in graph.stream(inputs, config=config, stream_mode="updates"):
            for node in event.keys():
                if node == "core_agent":
                    logs += "🔷 قام الـ LLM بفحص الموقف واختيار الأداة البرمجية الحرة بذكاء.\n"
                elif node == "planner":
                    logs += "🔷 استجابت الأداة المختارة وجاري تسليم البيانات المبدئية.\n"
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
        "plan": edited_plan_or_draft
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
    
    with gr.Row():
        with gr.Column(scale=1):
            task = gr.Textbox(label="المهمة المطلوبة أو موضوع المقال", lines=4, placeholder="اكتب موضوعك هنا (مثال: اكتب مقال عن الفاكهة، أو ما هو طقس العراق اليوم؟)...")
            max_rev = gr.Slider(1, 5, value=3, step=1, label="الحد الأقصى لجولات المراجعة والتعديل (للمقالات فقط)")
            thread = gr.Textbox(label="رقم جلسة العميل الذكي (Thread ID)", interactive=False)
            with gr.Row():
                start = gr.Button("🔍 بدء المعالجة ", variant="primary")
                resume = gr.Button("✨ استئناف واعتماد التعديلات", variant="secondary")
                
        with gr.Column(scale=2):
            logs = gr.Textbox(label="شاشة مراقبة سير العمليات ", lines=5, interactive=False)
            draft = gr.Textbox(label="منصة العرض التفاعلية ", lines=15, interactive=True)

    start.click(run_agent, [task, max_rev], [logs, draft, thread])
    resume.click(resume_agent, [thread, logs, max_rev, draft, task], [logs, draft, thread])

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())