import streamlit as st
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, pipeline
import PyPDF2

# ------------------- Model Setup -------------------
MODEL_NAME = "google/mt5-small"

@st.cache_resource
def load_model():
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# ------------------- Helper Functions -------------------
def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

def ask_model(prompt: str, max_new_tokens=500):
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )
    return outputs[0]["generated_text"]

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="Paper AI", layout="wide")
st.title("📄 Paper AI Assistant")
st.write("ارفع ورقة علمية واحصل على تصنيف + ملخص + اسأل أسئلة عنها.")

# اختيار اللغة
language = st.radio("اختر اللغة:", ["العربية", "English"])

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    if language == "العربية":
        classify_prompt = f"صنّف هذه الورقة العلمية حسب مجالها ونوعها:\n{text[:1500]}"
        summary_prompt = f"لخّص هذه الورقة العلمية بالعربية في نقاط واضحة:\n{text[:4000]}"
    else:
        classify_prompt = f"Classify this scientific paper by its field and type:\n{text[:1500]}"
        summary_prompt = f"Summarize this scientific paper in clear bullet points (English):\n{text[:4000]}"

    with st.spinner("⏳ Processing..."):
        classification = ask_model(classify_prompt, max_new_tokens=200)
        summary = ask_model(summary_prompt, max_new_tokens=400)

    st.subheader("📌 Classification")
    st.write(classification)

    st.subheader("📝 Summary")
    st.write(summary)

    st.subheader("📖 Extracted Text (First 2000 chars)")
    st.text(text[:2000])

    # Q&A Section
    st.subheader("❓ Ask a Question")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if language == "العربية":
            qa_prompt = f"النص التالي من ورقة علمية:\n{text[:4000]}\n\nالسؤال: {question}\nالإجابة:"
        else:
            qa_prompt = f"The following text is from a scientific paper:\n{text[:4000]}\n\nQuestion: {question}\nAnswer:"
        
        with st.spinner("⏳ Generating Answer..."):
            answer = ask_model(qa_prompt, max_new_tokens=300)
        st.success(answer)
