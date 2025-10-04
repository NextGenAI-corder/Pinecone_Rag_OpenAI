import os
import requests
import pdfplumber
import docx
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_URL = os.getenv("PINECONE_URL")

# ---------------------------
# テキスト抽出ハンドラ
# ---------------------------
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text

        elif ext == ".docx":
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])

        else:
            # txt, md, py, html, etc. 読めるものはそのまま
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    except Exception as e:
        print(f"[警告] {file_path} からテキスト抽出失敗: {e}")
        return ""


# ---------------------------
# チャンク化
# ---------------------------
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------
# OpenAI埋め込み
# ---------------------------
def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"input": text, "model": "text-embedding-3-small"}
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
    return response.json()["data"][0]["embedding"]


# ---------------------------
# Pineconeアップロード
# ---------------------------
def upload_to_pinecone(vector_id, embedding, metadata, namespace):
    headers = {"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"}
    upsert_url = f"{PINECONE_URL}/vectors/upsert"

    data = {
        "vectors": [
            {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata,
            }
        ],
        "namespace": namespace,
    }

    response = requests.post(upsert_url, headers=headers, json=data)
    if response.status_code != 200:
        print("Pineconeアップロード失敗:", response.status_code, response.text)


# ---------------------------
# ディレクトリ全ファイル処理
# ---------------------------
def process_directory(directory_path, namespace):
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"処理中: {file_path}")

            text = extract_text(file_path)
            if not text.strip():
                continue

            chunks = chunk_text(text)

            for idx, chunk in enumerate(chunks):
                vector_id = f"{os.path.basename(file_path)}-chunk-{idx+1}"
                embedding = get_embedding(chunk)
                metadata = {"source": file_path, "text": chunk}
                upload_to_pinecone(vector_id, embedding, metadata, namespace)

            print(f"アップロード完了: {file_path}")


if __name__ == "__main__":
    # 例: docsフォルダ配下すべてのファイル対象
    process_directory("Flask/PDF", "nextgen-specs")
