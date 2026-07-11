
# 🩺 medispace: Your AI-Powered Medical Companion

**medispace** is an intelligent healthcare assistant designed to bridge the gap between unstructured medical documents and actionable health insights. By integrating **Optical Character Recognition (OCR)** with a **Retrieval-Augmented Generation (RAG)** pipeline, medispace transforms prescription images and natural language queries into reliable, context-aware medical information.

---

### 🚀 Key Features

* **✅ OCR Prescription Analysis**: Automatically extracts medicine names and dosages from uploaded handwritten or printed images.
* **✅ RAG-Powered Q&A**: Combines a curated medical knowledge base with LLMs to provide factual, evidence-based answers.
* **✅ Semantic Medicine Search**: Uses **FAISS** for lightning-fast vector searches across extensive drug databases.
* **✅ Modern Flask Web UI**: A sleek, responsive web interface designed for both patients and healthcare providers.
* **✅ Secure Access**: Integrated basic session-based authentication to ensure user data privacy.
* **✅ Environment Configuration**: Uses dotenv for secure management of API keys and secrets.

---

### 🛠️ The Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | Flask, HTML5/CSS3, Font Awesome |
| **Orchestration** | Python, Flask (Web Framework) |
| **AI/NLP** | Hugging Face Sentence Transformers, Google Gemini, Ollama (Mistral) |
| **OCR Engine** | Tesseract OCR |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **Data Handling** | Pandas, NumPy, RapidFuzz |
| **Environment Management** | python-dotenv |

---

### 📖 System Architecture

MediNova operates on a multi-stage pipeline to ensure accuracy:
1.  **Ingestion**: User uploads an image or types a query.
2.  **Processing**: OCR extracts text; NLP models generate semantic embeddings.
3.  **Retrieval**: The system queries the **FAISS index** to find the most relevant medical context.
4.  **Generation**: An LLM (Gemini or Mistral) synthesizes the retrieved data into a natural, easy-to-understand response.

---

### 🔧 Installation & Setup

#### 1. Clone & Environment
```bash
# Clone the repository
git clone https://github.com/your-username/medinova.git
cd medinova

# Create a virtual environment
python -m venv rag-env

# Activate (Windows)
rag-env\Scripts\activate
# Activate (Mac/Linux)
source rag-env/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Configuration
Create a `.env` file in the root directory (use .env.example as a template):
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
FLASK_SECRET_KEY=your_secure_secret_key_here
```

#### 4. Launch the Application
```bash
python app.py
```
Then, open your browser and navigate to http://localhost:5000

---

### 🎥 How to Use

1. **Login**: Use username: `admin`, password: `admin` (for demo purposes)
2. **Upload Prescription**: Go to "Upload Prescription" and upload an image of your prescription
3. **Chat with AI Assistant**: Go to "Chatbot" and ask any medicine-related questions
4. **Explore Home**: Learn about MediNova's capabilities from the home page

---

### 📬 Contact & Support

**Developer**: Patnam Koushik  
**Email**: [koushikwork31@gmail.com](mailto:koushikwork31@gmail.com)

---
### 📜 License
This project is licensed under the **MIT License**. Feel free to use, modify, and distribute!
