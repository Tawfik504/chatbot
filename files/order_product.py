import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import gradio as gr

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 2. تعريف الأدوات (Tools)
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

# 3. الـ System Prompt
# 3. الـ System Prompt المعدل لفرض الالتزام التام بالسيناريو
# 3. الـ System Prompt المحدث مع معالجة شاملة للأخطاء
system_prompt = (
    "You are a strict and helpful shopping assistant.\n\n"
    
    "CRITICAL RULE FOR INVALID PRODUCTS:\n"
    "- When you call 'get_product_price' and it returns 'NOT_FOUND', you MUST reply to the user with this exact Arabic phrase and nothing else:\n"
    "\"من فضلك تقيد بقائمة المنتجات المتوفرة في الأعلى\"\n\n"
    
    "CRITICAL RULE FOR INVALID DISCOUNTS:\n"
    "- If the user types an invalid, misspelled, or unrecognized discount tier (anything other than bronze, silver, or gold, or their clear Arabic equivalents), "
    "or if they type any text that does not clearly specify a valid tier, you MUST reply with an error message stating that the input is invalid and they must choose from the available tiers (bronze, silver, gold).\n"
    "- NEVER trigger the invalid product phrase (\"من فضلك تقيد بقائمة المنتجات المتوفرة في الأعلى\") for a discount error.\n\n"
    
    "FLOW RULES:\n"
    "1. If the user asks about a product (like laptop, headphones, or keyboard), you MUST immediately call 'get_product_price'.\n"
    "2. Once 'get_product_price' returns a valid price, you MUST state that original price immediately to the user in Arabic. "
    "DO NOT ask the user for any specific laptop names, brands, or details. Just state the price found.\n"
    "3. IMMEDIATELY after stating the price, you MUST ask the user if they have a specific discount tier (bronze, silver, gold) to apply.\n"
    "4. If the user answers that they have a discount but DOES NOT explicitly name the tier (e.g., saying 'Yes, I have a discount' or 'apply discount'), "
    "DO NOT guess, DO NOT assume, and NEVER call the 'apply_discount' tool. "
    "Instead, you MUST reply by asking them to supply the exact discount tier name (bronze, silver, or gold).\n"
    "5. ONLY call the 'apply_discount' tool if the user explicitly names one of the valid tiers: bronze, silver, or gold.\n\n"
    
    "STRICT COMPLIANCE:\n"
    "- NEVER invent or assume any values.\n"
    "- NEVER ask follow-up questions other than the discount tier after a successful price lookup.\n"
    "- NEVER use mental math for calculations."
)
# 4. بناء الـ Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent_executor = create_react_agent(llm, tools, prompt=system_prompt)

# 5. دالة معالجة المحادثة المتوافقة كلياً مع التنسيق الحديث (Messages Format)
def respond(user_message, chat_history):
    if not user_message.strip():
        return "", chat_history

    # تأكيد تهيئة التاريخ كقائمة إذا كان فارغاً
    if chat_history is None:
        chat_history = []

    formatted_messages = []
    # استخراج النصوص بناءً على هيكلية القواميس المطلوبة من Gradio
    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get('role')
            content = msg.get('content')
            if role and content:
                formatted_messages.append((role, content))
    
    # إضافة الرسالة الحالية للمستخدم
    formatted_messages.append(("user", user_message))
    
    try:
        # تشغيل الـ Agent
        response = agent_executor.invoke({"messages": formatted_messages})
        final_answer = response["messages"][-1].content
        
        # حفظ الرسائل بصيغة القواميس الصارمة المتوافقة مع النظام لديك
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": final_answer})
        
    except Exception as e:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": f"حدث خطأ: {str(e)}"})
    
    return "", chat_history

# 6. بناء الواجهة الرسومية
with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ مساعد التسوق الإلكتروني التفاعلي")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### المنتجات المتاحة")
            gr.Markdown("- laptop\n- headphones\n- keyboard")
            clear_btn = gr.Button("🗑️ مسح المحادثة")

        with gr.Column(scale=3):
            # نتركها بدون المعامل type المسبب للمشاكل، وسيقوم الكود في دالة respond بالباقي
            chatbot = gr.Chatbot(label="المحادثة الحية", height=450)
            with gr.Row():
                txt_input = gr.Textbox(show_label=False, placeholder="اسألني عن منتج...", scale=4)
                submit_btn = gr.Button("إرسال 🚀", scale=1)

    # ربط الأحداث
    txt_input.submit(fn=respond, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot])
    submit_btn.click(fn=respond, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot])
    clear_btn.click(fn=lambda: [], inputs=None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=False, theme="soft")