import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
import gradio as gr
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ==========================================
# 1. تعريف الأدوات (Tools)
# ==========================================
@tool
def get_product_price(product: str) -> str:
    """Look up the price of a product in the catalog. 
    Returns the price as a string, or 'NOT_FOUND' if the product is not in the list."""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    
    # قاموس لربط الكلمات العربية بالاسم البرمجي الصحيح للمنتج
    arabic_synonyms = {
        "laptop": ["laptop", "لابتوب", "اللابتوب", "حاسوب", "كمبيوتر محمول"],
        "headphones": ["headphones", "سماعات", "السماعات", "هيدفون", "سماعه"],
        "keyboard": ["keyboard", "كيبورد", "الكيبورد", "لوحة مفاتيح", "لوحة المفاتيح"]
    }
    
    # تنظيف النص المدخل تماماً
    key_input = product.lower().strip()
    
    # البحث المتقدم باستخدام الكلمات المفتاحية الإنجليزية والعربية
    for product_key, synonyms in arabic_synonyms.items():
        for synonym in synonyms:
            if synonym in key_input:
                return str(prices[product_key])
                
    return "NOT_FOUND"

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold."""
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = [get_product_price, apply_discount]

# ==========================================
# 2. الـ System Prompt الصارم لحماية منطق العمل
# ==========================================
system_prompt = (
    "You are a strict and helpful shopping assistant.\n\n"
    
    "CRITICAL RULE FOR MULTIPLE PRODUCTS:\n"
    "- If the user asks about multiple products at the same time (e.g., 'headphones and laptop'), you MUST make separate tool calls for EACH product individually in a parallel or sequential manner.\n"
    "- NEVER combine multiple products into a single 'product' argument when calling 'get_product_price'.\n\n"
    
    "CRITICAL RULE FOR INVALID PRODUCTS:\n"
    "- If a product is not in the list (laptop, headphones, keyboard), you MUST explicitly state that the product is not available and respond with the EXACT phrase: \"من فضلك تقيد بقائمة المنتجات المتوفرة في الأعلى\".\n"
    "- NEVER use phrases like 'I could not find the price' for products that are not part of the catalog.\n\n"

    "CRITICAL RULE FOR INVALID PRODUCTS:\n"
    "- ONLY trigger the phrase \"من فضلك تقيد بقائمة المنتجات المتوفرة في الأعلى\" if a product lookup via 'get_product_price' genuinely returns 'NOT_FOUND'.\n"
    "- If some products are found and others are not, state the prices of the found products first, and then inform the user about the unavailable ones neutrally.\n\n"
    
    "CRITICAL RULE FOR INVALID DISCOUNTS:\n"
    "- If the user types an invalid, misspelled, or unrecognized discount tier (anything other than bronze, silver, or gold, or their clear Arabic equivalents), "
    "or if they type any text that does not clearly specify a valid tier, you MUST reply with an error message stating that the input is invalid and they must choose from the available tiers (bronze, silver, gold).\n"
    "- NEVER trigger the invalid product phrase (\"من فضلك تقيد بقائمة المنتجات المتوفرة في الأعلى\") for a discount error.\n\n"
    
    "FLOW RULES:\n"
    "1. If the user asks about a product (like laptop, headphones, or keyboard), you MUST immediately call 'get_product_price' for each recognized product.\n"
    "2. Once 'get_product_price' returns a valid price, you MUST state that original price immediately to the user in Arabic. "
    "DO NOT ask the user for any specific laptop names, brands, or details. Just state the price found.\n"
    "3. IMMEDIATELY after stating the price(s), you MUST ask the user if they have a specific discount tier (bronze, silver, gold) to apply.\n"
    "4. If the user answers that they have a discount but DOES NOT explicitly name the tier (e.g., saying 'Yes, I have a discount' or 'apply discount'), "
    "DO NOT guess, DO NOT assume, and NEVER call the 'apply_discount' tool. "
    "Instead, you MUST reply by asking them to supply the exact discount tier name (bronze, silver, or gold).\n"
    "5. ONLY call the 'apply_discount' tool if the user explicitly names one of the valid tiers: bronze, silver, or gold.\n\n"
    
    "STRICT COMPLIANCE:\n"
    "- NEVER invent or assume any values.\n"
    "- NEVER ask follow-up questions other than the discount tier after a successful price lookup.\n"
    "- NEVER use mental math for calculations."
)

# ==========================================
# 3. تهيئة النموذج وربطه بالأدوات مباشرة
# ==========================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model_with_tools = llm.bind_tools(tools)

# ==========================================
# 4. بناء الـ Graph يدوياً (العقد والروابط)
# ==========================================

class AgentState(TypedDict):
    messages: List[BaseMessage]
    
# العقدة الأولى: عقدة الوكيل (Agent Node)
def call_model(state):
    """تقوم بدمج الـ System Prompt يدوياً مع الرسائل الحالية واستدعاء النموذج المربوط بالأدوات."""
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}
#العقدة الثانية:عقدة الأدوات  
def execute_tools(state):
    """عقدة يدوية لتنفيذ الأدوات بناءً على طلبات النموذج."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    
    for tool_call in tool_calls:
        # اختيار الأداة المناسبة بناءً على الاسم
        if tool_call["name"] == "get_product_price":
            result = get_product_price.invoke(tool_call["args"])
        elif tool_call["name"] == "apply_discount":
            result = apply_discount.invoke(tool_call["args"])
        else:
            result = "Error: Tool not found."
            
        # إضافة النتيجة كـ ToolMessage
        results.append(ToolMessage(
            tool_call_id=tool_call["id"],
            content=str(result)
        ))
        
    return {"messages": results}
# الرابط الشرطي (Conditional Edge / Router)
def router(state):
    """تفحص آخر رسالة من الوكيل: إذا طلب أداة تتوجه لـ tools، وإذا انتهى تتوجه للنهاية END."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# إنشاء المخطط الشبكي المبني على الحالة
workflow = StateGraph(MessagesState)

# إضافة العقد يدوياً إلى المخطط
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools) # عقدة تشغيل الأدوات مسبقة الصنع والآمنة

# توصيل العقد بالروابط والمسارات
workflow.add_edge(START, "agent")              # البداية تذهب دائماً للوكيل
workflow.add_conditional_edges("agent", router) # من الوكيل، نطبق شرط الـ Router
workflow.add_edge("tools", "agent")            # بعد تنفيذ الأداة، نعود للوكيل مجدداً

# تجميع المخطط ليصبح جاهزاً للتشغيل بنفس الاسم القديم
agent_executor = workflow.compile()

# ==========================================
# 5. دالة معالجة المحادثة المتوافقة مع Gradio
# ==========================================
def respond(user_message, chat_history):
    if not user_message.strip():
        return "", chat_history

    if chat_history is None:
        chat_history = []

    formatted_messages = []
    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get('role')
            content = msg.get('content')
            if role and content:
                formatted_messages.append((role, content))
    
    formatted_messages.append(("user", user_message))
    
    try:
        # تشغيل الـ Graph المبني يدوياً
        response = agent_executor.invoke({"messages": formatted_messages})
        final_answer = response["messages"][-1].content
        
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": final_answer})
        
    except Exception as e:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": f"حدث خطأ: {str(e)}"})
    
    return "", chat_history

# ==========================================
# 6. واجهة المستخدم الرسومية (Gradio UI)
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ مساعد التسوق الإلكتروني ")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### المنتجات المتاحة")
            gr.Markdown("- laptop\n- headphones\n- keyboard")
            clear_btn = gr.Button("🗑️ مسح المحادثة")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="المحادثة الحية", height=450)
            with gr.Row():
                txt_input = gr.Textbox(show_label=False, placeholder="اسألني عن منتج...", scale=4)
                submit_btn = gr.Button("إرسال", scale=1)

    txt_input.submit(fn=respond, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot])
    submit_btn.click(fn=respond, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot])
    clear_btn.click(fn=lambda: [], inputs=None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=False, theme="soft")