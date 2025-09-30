import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, pipeline
import PyPDF2

# ------------------- Model Setup -------------------
MODEL_NAME = "google/mt5-small"

tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

app = FastAPI(
    title="Paper AI Assistant",
    description="📑 تصنيف وتلخيص الأوراق العلمية + أسئلة وأجوبة"
)

# ------------------- Utils -------------------
def extract_text_from_pdf(file) -> str:
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def ask_model(prompt: str, max_new_tokens=500):
    """Generate text using mT5 model"""
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )
    return outputs[0]["generated_text"]

# ------------------- Endpoints -------------------
@app.post("/upload-paper/")
async def upload_paper(file: UploadFile = File(...)):
    """Upload PDF, classify & summarize it"""
    text = extract_text_from_pdf(file.file)

    classify_prompt = f"صنّف هذه الورقة العلمية حسب مجالها ونوعها:\n{text[:1500]}"
    classification = ask_model(classify_prompt, max_new_tokens=200)

    summary_prompt = f"لخّص هذه الورقة العلمية بالعربية في نقاط واضحة:\n{text[:4000]}"
    summary = ask_model(summary_prompt, max_new_tokens=400)

    return {
        "classification": classification,
        "summary": summary,
        "full_text": text[:5000]
    }

@app.post("/ask/")
async def ask_question(
    question: str = Form(...),
    context: str = Form(...)
):
    """Ask a question about the paper's context"""
    qa_prompt = f"النص التالي من ورقة علمية:\n{context}\n\nالسؤال: {question}\nالإجابة:"
    answer = ask_model(qa_prompt, max_new_tokens=300)
    return {"answer": answer}

# ------------------- Run -------------------
if __name__ == "__main__":
    uvicorn.run("paper_ai:app", host="0.0.0.0", port=8000, reload=True)
