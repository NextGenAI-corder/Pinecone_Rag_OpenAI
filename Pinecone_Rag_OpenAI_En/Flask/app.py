from flask import (
    Flask,
    render_template,
    request,
    jsonify,
)  # Import core Flask modules
import openai  # Library for using OpenAI API
from pinecone import Pinecone  # Pinecone Python SDK
import config  # Custom module containing API keys, etc.
import sys  # Standard library for handling command-line arguments

# ---------------------------
# Initialize Flask application
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Initialize OpenAI and Pinecone settings
# Use API keys defined in config.py
# Pinecone host URL is manually specified (supports self-hosted endpoint)
# ---------------------------
openai.api_key = config.OPENAI_API_KEY
pc = Pinecone(api_key=config.PINECONE_API_KEY)
index = pc.Index(
    name=config.PINECONE_INDEX_NAME,
    host=config.PINECONE_URL,
)

# ---------------------------
# Retrieve namespace from command-line argument at startup
# Exit with error if not specified
# ※ The Flask app uses a fixed namespace throughout
# ---------------------------
if len(sys.argv) < 2:
    print("Usage: python app.py <namespace> is required!")
    exit(1)

NAMESPACE = sys.argv[1]  # Set namespace as global variable used throughout the app

# ---------------------------
# Display index.html when root endpoint ("/") is accessed
# templates/index.html is automatically loaded
# ---------------------------
@app.route("/")
def index_page():
    return render_template("index.html")

# ---------------------------
# API endpoint to handle POST request "/query"
# Performs vector search + answer generation based on user input
# ---------------------------
@app.route("/query", methods=["POST"])
def query():
    data = request.json  # Get JSON sent from frontend
    user_input = data.get("query")  # Extract user query text

    # Generate embedding using OpenAI API
    embedding = (
        openai.embeddings.create(
            model="text-embedding-3-small",  # Lightweight, high-accuracy embedding model
            input=user_input,
        )
        .data[0]
        .embedding
    )

    # Perform vector search against Pinecone (Top 5 results)
    result = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True,  # Return metadata including original text
        namespace=NAMESPACE,  # Use the namespace specified at startup (fixed)
    )

    # Extract match results (handle both dict and object formats from Pinecone)
    matches = result["matches"] if isinstance(result, dict) else result.matches

    if matches:
        # Concatenate matched texts as context
        context = "\n\n".join([m["metadata"]["text"] for m in matches])

        # Generate natural language response using ChatGPT API (with constraints)
        completion = openai.chat.completions.create(
            model="gpt-4o",  # Fast, high-accuracy model
            messages=[
                {
                    "role": "system",
                    "content": "Please answer the user's question in English in a concise and helpful manner, within 50 characters.",
                },
                {"role": "user", "content": f"Question: {user_input}\nInfo:\n{context}"},
            ],
        )

        # Extract and return the generated answer
        answer_text = completion.choices[0].message.content.strip()
        return jsonify({"answer": answer_text})
    else:
        # Error message when no matches are found
        return jsonify({"answer": "No relevant answer found."})

# ---------------------------
# Launch Flask application (debug mode enabled)
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
