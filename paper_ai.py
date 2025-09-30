import streamlit as st
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, pipeline
import PyPDF2

# تحميل الموديل
MODEL_NAME = "google/mt5-small"
@st.cache_resource
def load_model():
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# دالة لاستخراج النص من PDF
def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

# دالة لاستدعاء الموديل
def ask_model(prompt: str, max_new_tokens=500):
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )
    return outputs[0]["generated_text"]

# واجهة Streamlit
st.title("📄 Paper AI Assistant (Arabic)")
st.write("ارفع ورقة علمية PDF واحصل على التصنيف + الملخص + اسأل أسئلة عنها")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)

    with st.spinner("⏳ جاري التصنيف..."):
        classify_prompt = f"صنّف هذه الورقة العلمية حسب مجالها ونوعها:\n{text[:1500]}"
        classification = ask_model(classify_prompt, max_new_tokens=200)

    with st.spinner("⏳ جاري التلخيص..."):
        summary_prompt = f"لخّص هذه الورقة العلمية بالعربية في نقاط واضحة:\n{text[:4000]}"
        summary = ask_model(summary_prompt, max_new_tokens=400)

    st.subheader("📌 التصنيف")
    st.write(classification)

    st.subheader("📝 الملخص")
    st.write(summary)

    st.subheader("📖 النص المستخرج (جزء)")
    st.text(text[:1000])

    # قسم الأسئلة
    st.subheader("❓ اسأل سؤال عن الورقة")
    question = st.text_input("اكتب سؤالك هنا")
    if st.button("إجابة"):
        if question.strip():
            with st.spinner("⏳ جاري البحث عن الإجابة..."):
                qa_prompt = f"النص التالي من ورقة علمية:\n{text[:2000]}\n\nالسؤال: {question}\nالإجابة:"
                answer = ask_model(qa_prompt, max_new_tokens=300)
            st.success(answer)
        else:
            st.warning("من فضلك اكتب سؤال.")
