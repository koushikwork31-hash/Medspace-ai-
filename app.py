print("Starting medispace...")
from sentence_transformers import SentenceTransformer
print("Imported SentenceTransformer")
import faiss
import numpy as np
import ollama
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd
from rapidfuzz import fuzz, process
import pytesseract
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

print("Loading environment variables...")
# Load environment variables
load_dotenv()

# Configure API keys
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
tesseract_cmd = os.getenv("TESSERACT_CMD", r'C:\Program Files\Tesseract-OCR\tesseract.exe')
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
print("API keys configured")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-ChangeMeInProduction")

print("Loading medicine data...")
# Load medicine data
medicine_df = pd.read_csv('Medicine_Details.csv')
medicine_names = medicine_df['Medicine Name'].tolist()
print("Medicine data loaded!")

print("Loading SentenceTransformer model...")
# Prepare RAG embedding index
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

medicine_df["full_text"] = medicine_df.apply(lambda row: f"""
Medicine: {row['Medicine Name']}
Uses: {row.get('Uses', 'N/A')}
Side Effects: {row.get('Side_effects', 'N/A')}
Composition: {row.get('Composition', 'N/A')}
""", axis=1)

print("Generating embeddings...")
embeddings = embed_model.encode(medicine_df["full_text"].tolist()).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("Embeddings generated and index created!")

# Initialize TF-IDF vectorizer for keyword search
print("Initializing TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(medicine_df["full_text"].tolist())
print("TF-IDF initialized!")

# Conversation memory (store last 5 messages)
CONVERSATION_MEMORY_SIZE = 5
conversation_memory = {}  # key: session id, value: deque of messages


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chatbot')
def chatbot():
    # Initialize conversation memory for this session
    session_id = request.remote_addr  # Simple session ID based on IP
    if session_id not in conversation_memory:
        conversation_memory[session_id] = deque(maxlen=CONVERSATION_MEMORY_SIZE)
    return render_template('chatbot.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Simple authentication for demo purposes
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('chatbot'))
        else:
            return render_template('login.html', error="Invalid credentials!")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))


def build_context_from_memory(session_id):
    """Build context from conversation memory"""
    memory = conversation_memory.get(session_id, deque(maxlen=CONVERSATION_MEMORY_SIZE))
    context = []
    for msg in memory:
        role = "User" if msg['role'] == 'user' else "Assistant"
        context.append(f"{role}: {msg['content']}")
    return "\n".join(context)


def hybrid_search(user_query, k=5, semantic_weight=0.7):
    """Hybrid search combining semantic (FAISS) and keyword (TF-IDF) search"""
    # Semantic search
    query_embedding = embed_model.encode([user_query])[0].astype("float32")
    distances, semantic_indices = index.search(np.array([query_embedding]), k)
    semantic_scores = 1 / (1 + distances[0])  # Convert distance to similarity (0-1)

    # Keyword search
    query_tfidf = tfidf_vectorizer.transform([user_query])
    keyword_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

    # Combine scores
    combined_scores = {}
    for i, idx in enumerate(semantic_indices[0]):
        combined_scores[idx] = semantic_weight * semantic_scores[i] + (1 - semantic_weight) * keyword_scores[idx]

    # Add top keyword-only results for diversity
    top_keyword_indices = keyword_scores.argsort()[::-1][:k]
    for idx in top_keyword_indices:
        if idx not in combined_scores:
            combined_scores[idx] = keyword_scores[idx] * (1 - semantic_weight)

    # Sort and return top k
    sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]
    return medicine_df.iloc[sorted_indices]["full_text"].tolist()


def build_rag_prompt(user_query, context, conversation_history):
    """Build a structured prompt for RAG"""
    prompt = f"""You are a helpful AI medical assistant for medispace.
Use the following medicine information and conversation history to answer the user's question.

Conversation History:
{conversation_history}

Medicine Data:
{context}

User Question: {user_query}

Answer in simple, clear language. If you don't know the answer, say so clearly.
"""
    return prompt


# Gemini general reply
def get_general_reply_from_gemini(user_input, conversation_history):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""You are a friendly AI medical assistant for medispace.
Conversation History:
{conversation_history}
User: {user_input}
Assistant:"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return "Hey there! 👋 Feel free to ask me anything about medicines."


# RAG answer with Gemini or Ollama
def smart_medicine_answer(user_query, session_id, k=5, use_ollama=False):
    # Get conversation history
    conversation_history = build_context_from_memory(session_id)

    # Hybrid search
    relevant_rows = hybrid_search(user_query, k)
    context = "\n\n".join(relevant_rows)

    # Build prompt
    prompt = build_rag_prompt(user_query, context, conversation_history)

    try:
        if use_ollama:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        else:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text.strip()
    except Exception as e:
        print(f"RAG error: {e}")
        return "I'm sorry, I couldn't generate a response right now. Please try again later."


# Fuzzy medicine detail search
def get_medicine_details(name_query):
    formatted_query = name_query.title()
    match_result = process.extractOne(formatted_query, medicine_names, scorer=fuzz.WRatio)
    if match_result:
        best_match = match_result[0]
        matched_row = medicine_df[medicine_df['Medicine Name'] == best_match].iloc[0]
        reply = f"Medicine: {matched_row['Medicine Name'].strip()}\n"
        reply += f"Uses: {matched_row.get('Uses', 'No data').strip()}\n"
        reply += f"Side Effects: {matched_row.get('Side_effects', 'No data').strip()}\n"
        reply += f"Composition: {matched_row.get('Composition', 'No data').strip()}"
        return reply
    else:
        return "No matching medicine found."


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image:
            return render_template('upload_image.html', response=None, extracted="No image uploaded.")

        try:
            img = Image.open(io.BytesIO(image.read()))
            extracted_text = pytesseract.image_to_string(img)
            formatted_text = extracted_text.title()

            match_result = process.extractOne(formatted_text, medicine_names, scorer=fuzz.WRatio)
            if match_result:
                best_match = match_result[0]
                structured_data = get_medicine_details(best_match)

                if structured_data:
                    return render_template('upload_image.html', response=structured_data, extracted=formatted_text)

            return render_template('upload_image.html', response=None, extracted=formatted_text)
        except Exception as e:
            return render_template('upload_image.html', response=None, extracted=f"Error processing image: {str(e)}")

    return render_template('upload_image.html', response=None)


# Main message route with Gemini fallback
@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'response': "Please enter a valid message."})

    session_id = request.remote_addr  # Simple session ID

    # Initialize conversation memory if needed
    if session_id not in conversation_memory:
        conversation_memory[session_id] = deque(maxlen=CONVERSATION_MEMORY_SIZE)

    # Add user message to memory
    conversation_memory[session_id].append({'role': 'user', 'content': user_message})

    # Handle greetings or general convo with Gemini
    GREETINGS = ['hi', 'hello', 'hey', 'how are you', 'hlo', 'hola', 'yo', 'sup', 'what’s up']
    user_message_lower = user_message.lower()
    conversation_history = build_context_from_memory(session_id)

    if any(greet in user_message_lower for greet in GREETINGS):
        reply = get_general_reply_from_gemini(user_message, conversation_history)
        conversation_memory[session_id].append({'role': 'assistant', 'content': reply})
        return jsonify({'response': reply})

    # Default to RAG-based Gemini (set use_ollama=True if you want to use Ollama)
    reply = smart_medicine_answer(user_message, session_id, use_ollama=False)
    conversation_memory[session_id].append({'role': 'assistant', 'content': reply})
    return jsonify({'response': reply})


if __name__ == '__main__':
    app.run(debug=True)
