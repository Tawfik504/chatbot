import uuid
import gradio as gr
from dotenv import load_dotenv

# 1. تحميل مفاتيح البيئة في السطر الأول
load_dotenv()

# استيراد المخطط المستقر
from agent import graph

# =========================
# RUN (START)
# =========================
def run_agent(task, max_revisions):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {
        "task": task,
        "plan": "",
        "content": [],
        "draft": [],
        "critique": [],
        "revision_number": 0,
        "max_revisions": int(max_revisions)
    }

    logs = "🔄 Starting Agent (Step-by-Step Mode)...\n"
    draft_display = ""

    yield logs, draft_display, thread_id

    # تشغيل المخطط لأول مرة لإنشاء الخطة
    for event in graph.stream(inputs, config=config):
        if isinstance(event, dict):
            for node in event.keys():
                # نكتفي بطباعة حركة العقد فقط في الـ Logs لمنع التكرار البصري
                logs += f"🔷 Completed Node: '{node}'\n"
        yield logs, draft_display, thread_id

    # جلب الحالة النهائية بعد انتهاء عقدة التخطيط
    current_state = graph.get_state(config).values
    
    # نضع الخطة داخل خانة الـ Draft وليس الـ Logs
    if current_state.get("plan"):
        draft_display = f"📋 [Generated Plan Overview]:\n\n{current_state['plan']}"

    # فحص التوقف الإجباري وعرض التعليمات في الـ Logs بشكل نظيف
    state_now = graph.get_state(config)
    if state_now.next:
        logs += f"\n⏸️ Agent paused automatically before node: {list(state_now.next)}\n"
        logs += "💡 [Steering Enabled]:\n"
        logs += "   - If you LIKE the plan in the Draft box, just click 'Resume'.\n"
        logs += "   - If you DON'T LIKE it, change the question in 'Task' box now, then click 'Resume'!\n"
        logs += "-" * 40 + "\n"
        
    yield logs, draft_display, thread_id


# =========================
# RESUME 
# =========================
def resume_agent(thread_id, current_task_text):
    if not thread_id:
        yield "⚠️ No active Thread ID found!", "", ""
        return

    config = {"configurable": {"thread_id": thread_id}}

    # تحديث نص المهمة في الذاكرة في حال قام المستخدم بتعديله
    graph.update_state(config, {"task": current_task_text})

    # جلب الحالة والـ Logs السابقة للإضافة عليها وتجنب تصفير الصندوق
    current_state = graph.get_state(config).values
    
    logs = f"▶️ Resuming workflow... Executing next step.\n" + "-" * 40 + "\n"
    draft_display = ""
    yield logs, draft_display, thread_id

    # استئناف المخطط بتمرير None ليتحرك من نقطة التوقف
    for event in graph.stream(None, config=config):
        if isinstance(event, dict):
            for node in event.keys():
                logs += f"🔷 Completed Node: '{node}'\n"
        yield logs, draft_display, thread_id

    # جلب الحالة المحدثة من الذاكرة لعرض المسودة أو النقد في خانة الـ Draft
    current_state = graph.get_state(config).values
    
    if current_state.get("draft"):
        # نعرض آخر مسودة تمت كتابتها في خانة الـ Draft
        draft_display = f"📝 [Latest Draft]:\n\n{current_state['draft'][-1]}"
        
        # 🌟 تم الإصلاح هنا: تحويل علامات الاقتباس إلى مفردة لمنع الـ SyntaxError
        if current_state.get("critique"):
            draft_display += f"\n\n{'='*40}\n❌ [Critic Feedback]:\n\n{current_state['critique'][-1]}"
    else:
        # إذا لم تكن هناك مسودة بعد، نعرض الخطة الحالية
        draft_display = f"📋 [Current Plan]:\n\n{current_state.get('plan', '')}"

    # فحص المقاطعة التالية
    state_now = graph.get_state(config)
    if state_now.next:
        logs += f"\n⏸️ Agent paused automatically before node: {list(state_now.next)}\n"
        logs += "👉 Click 'Resume' to take the next step.\n"
        logs += "-" * 40 + "\n"
    else:
        logs += "\n✅ Agent workflow finished completely! (Reached END)\n"
        
    yield logs, draft_display, thread_id


# =========================
# PAUSE 
# =========================
def pause_agent(thread_id):
    return "⏸️ The agent stops automatically before main nodes. Just wait for it to pause, then use Resume.", gr.update(), thread_id


# =========================
# UI DESIGN
# =========================
with gr.Blocks() as app:

    gr.Markdown("# 🚀 Stable LangGraph Agent (Human-in-the-Loop Mode)")

    with gr.Row():
        with gr.Column():
            task = gr.Textbox(label="Task", lines=4)
            max_rev = gr.Slider(1, 5, value=2)

            start = gr.Button("Start", variant="primary")
            pause = gr.Button("Pause")
            resume = gr.Button("Resume")

            thread = gr.Textbox(label="Thread ID", interactive=False)

        with gr.Column():
            logs = gr.Textbox(label="Logs", lines=20)
            draft = gr.Textbox(label="Draft / Output Results", lines=20)

    # ربط الأزرار
    start.click(run_agent, [task, max_rev], [logs, draft, thread])
    pause.click(pause_agent, thread, [logs, draft, thread])
    resume.click(resume_agent, [thread, task], [logs, draft, thread])

app.launch(theme=gr.themes.Soft())