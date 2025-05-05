#!/usr/bin/env python
import streamlit as st
import pymongo
import faiss
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import time # Added for potential delays or UI updates

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="ML Legal Advisor Chat")

# --- Resource Loading ---

@st.cache_resource(show_spinner="Connecting to Database...")
def get_mongo_collection():
    """Connects to MongoDB and returns the collection."""
    try:
        # Consider adding authentication if required:
        # client = pymongo.MongoClient("mongodb://user:password@localhost:27017/")
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Verify connection
        db = client["mllegaladvisordb"]
        collection = db["bns_sections"]
        print("Successfully connected to MongoDB.")
        return collection
    except pymongo.errors.ServerSelectionTimeoutError:
        st.error("Connection to MongoDB timed out. Ensure it's running, accessible, and the URI is correct.")
        st.stop()
    except pymongo.errors.ConnectionFailure as e:
        st.error(f"Failed to connect to MongoDB. Ensure it's running. Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during MongoDB connection: {e}")
        st.exception(e) # Log full traceback
        st.stop()

@st.cache_resource(show_spinner="Loading AI models and data...")
def load_models_and_data(_collection):
    """Loads data, Sentence Transformer, spaCy model, and creates FAISS index."""
    try:
        documents = list(_collection.find())
    except Exception as e:
        st.error(f"Failed to fetch documents from MongoDB: {e}")
        st.stop()

    if not documents:
        st.warning("No documents found in the 'bns_sections' collection.")
        st.stop()

    # Filter out documents with missing or empty descriptions crucial for embedding
    valid_docs = [doc for doc in documents if doc.get("Description")]
    if not valid_docs:
        st.warning("No documents with valid 'Description' fields found for analysis.")
        st.stop()
    print(f"Found {len(valid_docs)} documents with descriptions.")

    descriptions = [doc["Description"] for doc in valid_docs]

    # --- Load Sentence Transformer Model ---
    model_path = r"C:\content\fine_tuned_model" # Make sure this path is correct
    try:
        print(f"Loading Sentence Transformer model from: {model_path}")
        model = SentenceTransformer(model_path)
        print("Sentence Transformer model loaded successfully.")
    except Exception as e:
        st.error(f"Fatal Error: Failed to load Sentence Transformer model: {e}")
        st.error(f"Please ensure the model exists at the specified path: {model_path}")
        st.exception(e)
        st.stop()

    # --- Load spaCy Model ---
    spacy_model_name = "en_core_web_sm"
    try:
        print(f"Loading spaCy model: {spacy_model_name}")
        # Check if already loaded (useful if running locally with frequent edits)
        if 'nlp' not in globals() or not isinstance(nlp, spacy.language.Language):
             nlp = spacy.load(spacy_model_name)
        print("spaCy model loaded successfully.")
    except OSError:
        st.error(f"spaCy model '{spacy_model_name}' not found. Please download it first.")
        st.info(f"Try running: python -m spacy download {spacy_model_name}")
        st.stop()
    except Exception as e:
        st.error(f"Fatal Error: Failed to load spaCy model '{spacy_model_name}': {e}")
        st.exception(e)
        st.stop()

    # --- Create Embeddings and FAISS Index ---
    try:
        print(f"Creating embeddings for {len(descriptions)} descriptions...")
        # Use a progress bar for user feedback during encoding
        progress_bar = st.progress(0, text="Generating document embeddings...")
        embeddings = model.encode(
            descriptions,
            convert_to_tensor=False,
            show_progress_bar=False # We use Streamlit's progress bar instead
        )
        progress_bar.progress(1.0, text="Embeddings generated.")
        time.sleep(0.5) # Keep message visible briefly
        progress_bar.empty() # Remove progress bar


        embeddings_np = np.array(embeddings).astype("float32")

        if embeddings_np.ndim != 2 or embeddings_np.shape[0] != len(descriptions):
            st.error(f"Embedding shape mismatch: Got {embeddings_np.shape}, expected ({len(descriptions)}, dimension). Check model output.")
            st.stop()
        if embeddings_np.shape[1] == 0:
             st.error("Embedding dimension is 0. Check the Sentence Transformer model.")
             st.stop()


        d = embeddings_np.shape[1]
        print(f"Embedding dimension: {d}")
        print("Building FAISS index...")
        index = faiss.IndexFlatL2(d) # Using L2 distance (Euclidean)
        index.add(embeddings_np)
        print(f"FAISS index created successfully with {index.ntotal} vectors.")
        st.success(f"AI models and FAISS index ready ({index.ntotal} documents indexed).")

    except Exception as e:
        st.error(f"Fatal Error: Failed to create embeddings or FAISS index: {e}")
        st.exception(e)
        st.stop()

    # Return the documents that correspond to the indexed descriptions and the models/index
    return valid_docs, model, index, nlp

# --- Initialize Application ---
collection = get_mongo_collection()
# Ensure 'nlp' is globally accessible after loading
documents, model, index, nlp = load_models_and_data(collection)

# --- Session State for Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI Legal Advisor. How can I assist you today?"}]

# --- Helper Functions ---

def preprocess_query(query_text):
    """Preprocesses the query: lowercase, remove special chars, lemmatize, remove stop words."""
    if not query_text or not isinstance(query_text, str):
        return ""
    # Remove special characters but keep spaces
    query_text = re.sub(r"[^a-zA-Z0-9\s]", "", query_text.lower())
    # Process with spaCy
    doc = nlp(query_text)
    # Lemmatize and remove stop words and punctuation
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.strip()]
    processed_query = " ".join(lemmas)
    # Return original if processing results in empty string, else return processed
    return processed_query if processed_query.strip() else query_text


def generate_advice(user_query, top_docs, distances):
    """Generates a conversational advice string based on top matching documents."""
    if not top_docs:
        return "I couldn't find specific legal sections closely related to your query based on the available data. Could you please provide more details or rephrase?"

    # Simple approach: Summarize top matches
    advice = f"Based on your query regarding '{user_query}', here are some areas that might be relevant according to the information I have:\n\n"
    for i, (doc, dist) in enumerate(zip(top_docs, distances)):
        bns = doc.get('BNS_Section', 'N/A')
        ipc = doc.get('IPC_Section', 'N/A')
        desc = doc.get("Description", "No description available.")
        # Simple similarity score (higher is better) - inverse of L2 distance
        # Add a small epsilon to avoid division by zero if distance is exactly 0
        similarity_score = 1 / (1 + dist + 1e-6)
        advice += (
            f"{i+1}. **Sections {bns} (BNS) / {ipc} (IPC):** "
            f"This section pertains to: \"{desc[:200]}...\" " # Show a snippet
            f"(Relevance score: {similarity_score:.2f})\n"
        )

    advice += "\n---\n**Disclaimer:** I am an AI assistant. This information is based on matching your query to legal section descriptions and is **not** legal advice. Laws are complex and nuanced. You should consult with a qualified legal professional for advice tailored to your specific situation."
    return advice

    # --- Placeholder for Advanced LLM Integration ---
    # To use a more sophisticated LLM (like Google Gemini):
    # 1. Install the library: `pip install google-generativeai`
    # 2. Import: `import google.generativeai as genai`
    # 3. Configure API Key (use Streamlit secrets for security):
    #    `genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])`
    # 4. Prepare the prompt combining user query and retrieved context:
    #    context = "\n".join([f"- Sec {d.get('BNS_Section')}/{d.get('IPC_Section')}: {d.get('Description')}" for d in top_docs])
    #    prompt = f"""You are a helpful AI legal assistant simulator.
    #    A user asked: "{user_query}"
    #    Based on my analysis, the following legal sections seem potentially relevant:
    #    {context}
    #
    #    Please provide a conversational response based *only* on the information in these sections.
    #    Explain briefly, in simple terms, how these sections might relate to the user's query.
    #    Do *not* invent information beyond the provided sections.
    #    Start your response directly with the explanation.
    #    Conclude with a clear disclaimer: "This is AI-generated information based on provided text and is not legal advice. Consult a qualified legal professional."
    #    """
    # 5. Call the LLM:
    #    try:
    #        llm = genai.GenerativeModel('gemini-1.5-flash') # Or another suitable model
    #        response = llm.generate_content(prompt)
    #        return response.text
    #    except Exception as llm_error:
    #        st.warning(f"Could not generate enhanced advice using LLM: {llm_error}")
    #        # Fallback to the simple advice generated above
    #        return advice # Return the basic advice if LLM fails
    # --- End Placeholder ---


# --- Streamlit UI ---

st.title("– Conversational ML Legal Advisor")
st.caption("Ask a question about a legal situation, and I'll try to identify potentially relevant sections.")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is your legal question?"):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query and generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Analysing your query...")

        try:
            start_time = time.time()

            # 1. Preprocess Query
            cleaned_query = preprocess_query(prompt)
            print(f"Original Query: '{prompt}' | Processed Query: '{cleaned_query}'")

            if not cleaned_query.strip():
                response = "Your query seems empty after processing. Could you please provide more details or rephrase your question?"
                message_placeholder.markdown(response)
            else:
                # 2. Embed Query
                query_embedding = model.encode(cleaned_query).astype("float32")

                # Ensure embedding is 2D for FAISS search
                if query_embedding.ndim == 1:
                    query_embedding = query_embedding.reshape(1, -1)

                # Dimension check
                if query_embedding.shape[1] != index.d:
                    st.error(f"Query embedding dimension ({query_embedding.shape[1]}) doesn't match index dimension ({index.d}). Model configuration issue?")
                    response = "Sorry, there's an internal configuration error. Cannot process the query."
                    message_placeholder.error(response)
                else:
                    # 3. Search FAISS Index
                    k = 3 # Number of relevant sections to retrieve
                    print(f"Searching FAISS index for top {k} matches...")
                    distances, indices = index.search(query_embedding, k=k)
                    print(f"FAISS Search Results - Indices: {indices}, Distances: {distances}")


                    # 4. Retrieve Matching Documents
                    top_docs_data = []
                    top_distances_data = []
                    if indices.size > 0 and distances.size > 0: # Check if results exist
                        for i, idx in enumerate(indices[0]):
                            doc_index = int(idx)
                            # Check for invalid index (-1 means no match found for that position)
                            if doc_index != -1 and 0 <= doc_index < len(documents):
                                top_docs_data.append(documents[doc_index])
                                top_distances_data.append(distances[0][i])
                            else:
                                print(f"Warning: Search returned invalid or out-of-bounds index {doc_index}, skipping.")

                    # 5. Generate Advice
                    print(f"Generating advice based on {len(top_docs_data)} found documents.")
                    response = generate_advice(prompt, top_docs_data, top_distances_data)
                    message_placeholder.markdown(response) # Update the placeholder with the final response

            end_time = time.time()
            print(f"Processing time: {end_time - start_time:.2f} seconds")

            # Add assistant's final response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            # Log the error and inform the user
            print(f"Error during query processing: {e}")
            st.exception(e)
            error_message = f"Sorry, I encountered an error while processing your request: {e}. Please try again."
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.markdown("""
This application uses AI (Sentence Transformers and FAISS) to find potentially relevant legal sections based on your query description.

**How it works:**
1.  Your query is compared to descriptions of known legal sections.
2.  The most similar sections are identified.
3.  The AI provides a summary based on these findings.

**Disclaimer:**
This tool provides informational suggestions **only** and does **not** constitute legal advice. Always consult a qualified legal professional for help with specific legal matters.
""")
st.sidebar.divider()
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How can I help you?"}]
    st.rerun()
