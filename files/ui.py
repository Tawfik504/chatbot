import uuid
import gradio as gr
from dotenv import load_dotenv

load_dotenv()


from agent import graph


def get_latest_ui_data(config, current_logs):
    """تقوم هذه الدالة بجلب البيانات الحالية من الذاكرة وعرضها في الصناديق المناسبة بشكل آمن"""
    state_now = graph.get_state(config)
    current_state = state_now.values
    
    draft_display = ""
    

    if current_state.get("draft") and len(current_state["draft"]) > 0:
        draft_display = f"📝 [المسودة الأخيرة - المراجعة رقم {current_state.get('revision_number', 0)}]:\n\n{current_state['draft'][-1]}"
        if current_state.get("critique") and len(current_state["critique"]) > 0:
            draft_display += f"\n\n{'='*40}\n❌ [ملاحظات ونقد المصحح]:\n\n{current_state['critique'][-1]}"
            
    elif current_state.get("plan"):
        draft_display = f"📋 [الخطة الهيكلية المنشأة للمقال]:\n\n{current_state['plan']}"
        
    elif current_state.get("content") and len(current_state["content"]) > 0:
        draft_display = f"🔍 [نتائج البحث التي جلبها الـ Agent بنفسه]:\n\n{current_state['content'][-1]}"
    else:
        draft_display = "⏳ الـ Agent يفكر الآن ويحدد حاجته لاستخدام الأدوات..."

    updated_logs = current_logs
    if state_now.next:
        next_step = list(state_now.next)
        updated_logs += f"\n⏸️ توقف النظام تلقائياً قبل الخطوة: {next_step}\n"
        
        if "critic" in next_step:
            updated_logs += "💡 [المراجعة والتدقيق اللغوي جاهز]:\n"
            updated_logs += "   - يمكنك رؤية مسودة المقال المكتوبة في الصندوق المقابل.\n"
            updated_logs += "   - اضغط على زر 'استئناف' ليرسل النظام المسودة إلى المصحح لنقدها وتعديلها.\n"
        else:
            updated_logs += "👉 اضغط على زر 'استئناف' ليتابع الـ Agent رحلته الذكية.\n"
        updated_logs += "-" * 40 + "\n"
    else:
        updated_logs += "\n✅ انتهت عملية النظام بالكامل وصيغ المقال النهائي بنجاح!\n"

    return updated_logs, draft_display


def run_agent(task, max_revisions):
    if not task.strip():
        yield "⚠️ الرجاء كتابة السؤال أو المهمة أولاً!", "", ""
        return

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # تهيئة الحالة الافتراضية
    inputs = {
        "task": task,
        "plan": "",
        "content": [],
        "draft": [],
        "critique": [],
        "revision_number": 0,
        "max_revisions": int(max_revisions)
    }

   
    logs = "🔄 جاري بدء تشغيل العميل الذكي (Agent)...\n🧠 يقوم العميل الآن بتحليل النص لتحديد ما إذا كان يحتاج للبحث في الإنترنت أم لا...\n"
    yield logs, "جاري معالجة طلبك وتحديد المسار ذكياً...", thread_id

    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        if isinstance(event, dict):
            for node in event.keys():
  
                display_node = "تفكير العميل واتخاذ القرار (core_agent)" if node == "core_agent" else node
                logs += f"🔷 اكتملت الخطوة: '{display_node}'\n"
        yield logs, "جاري تحديث البيانات الحالية...", thread_id

    logs, draft_display = get_latest_ui_data(config, logs)
    yield logs, draft_display, thread_id


def resume_agent(thread_id, current_task_text, current_logs):
    if not thread_id:
        yield "⚠️ لم يتم العثور على جلسة نشطة! يرجى الضغط على زر البدء أولاً.", "", ""
        return

    config = {"configurable": {"thread_id": thread_id}}
    
    
    graph.update_state(config, {"task": current_task_text})

    logs = current_logs + f"▶️ جاري استئناف العمل... الانتقال إلى الخطوة التالية.\n" + "-" * 40 + "\n"
    yield logs, "جاري المعالجة والمتابعة...", thread_id
    
    for event in graph.stream(None, config=config, stream_mode="updates"):
        if isinstance(event, dict):
            for node in event.keys():
                display_node = "تفكير العميل واتخاذ القرار (core_agent)" if node == "core_agent" else node
                logs += f"🔷 اكتملت الخطوة: '{display_node}'\n"
        
        temp_logs, temp_draft = get_latest_ui_data(config, logs)
        yield temp_logs, temp_draft, thread_id

    logs, draft_display = get_latest_ui_data(config, logs)
    yield logs, draft_display, thread_id


def pause_agent(thread_id):
    return "⏸️ النظام يتم إيقافه تلقائياً من خلال نقاط التوقف المبرمجة مسبقاً (Interrupts)، لا حاجة للإيقاف اليدوي.", gr.update(), thread_id



# بناء واجهة Gradio

with gr.Blocks() as app:
    gr.Markdown("#  نظام كتابة المقالات الذكي ")

    with gr.Row():
        with gr.Column():
            task = gr.Textbox(label="سؤال أو مهمة الزبون", lines=4, placeholder="مثال: اكتب مقالاً عن أهمية التمارين الرياضية الصباحية.")
            max_rev = gr.Slider(1, 5, value=2, step=1, label="الحد الأقصى لمراجعات المقال")

            with gr.Row():
                start = gr.Button("بدء العمل الذكي", variant="primary")
                resume = gr.Button("استئناف", variant="secondary")
                pause = gr.Button("معلومات")

            thread = gr.Textbox(label="رقم تعريف الجلسة الحالية", interactive=False)

        with gr.Column():
            logs = gr.Textbox(label="سجل تفكير العميل وتتبع الخطوات ", lines=10, interactive=False)
            draft = gr.Textbox(label="صندوق المخرجات الديناميكي (نتائج البحث الذكي / الخطة / المقال المكتوب)", lines=15, interactive=False)

    start.click(run_agent, [task, max_rev], [logs, draft, thread])
    pause.click(pause_agent, thread, [logs, draft, thread])
    resume.click(resume_agent, [thread, task, logs], [logs, draft, thread])

app.launch(theme=gr.themes.Soft())