import os
import requests
import pdfplumber
import docx
import argparse
from dotenv import load_dotenv

# ---------------------------
# 環境変数読み込み（APIキー・接続先URLを外部ファイルから安全に取得）
# ---------------------------
load_dotenv("config.env.template")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_URL = os.getenv("PINECONE_URL")  # 末尾に /query を含まないこと


# ---------------------------
# ファイル種別に応じたテキスト抽出処理
# PDF → pdfplumber
# Word → python-docx
# その他 → UTF-8でそのまま読み取り（例: .txt, .md, .py）
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
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"[警告] テキスト抽出失敗: {file_path} → {e}")
        return ""


# ---------------------------
# チャンク化処理（長文を指定バイト単位で分割）
# chunk_size: 1チャンクの文字数（例: 1000文字）
# overlap: チャンク間の重複（前後文脈維持のため、重複させる）
# → RAG精度向上に重要な前処理
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
# OpenAI API による埋め込み生成
# モデル：text-embedding-3-small（2024年以降の高精度版）
# 各チャンクをベクトル化し、Pineconeで意味検索に利用
# ---------------------------
def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"input": text, "model": "text-embedding-3-small"}
    response = requests.post(
        "https://api.openai.com/v1/embeddings", headers=headers, json=data
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


# ---------------------------
# Pinecone へのアップロード処理
# ID: ファイル名＋チャンク番号で一意に生成
# namespace: ユーザー指定の論理グループ（用途別に切り替え可能）
# metadata: 検索時に返す元テキストやファイル情報を保持
# ---------------------------
def upload_to_pinecone(vector_id, embedding, metadata, namespace):
    headers = {"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"}
    url = f"{PINECONE_URL}/vectors/upsert"
    data = {
        "vectors": [{"id": vector_id, "values": embedding, "metadata": metadata}],
        "namespace": namespace,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"[エラー] アップロード失敗: {vector_id} → {response.text}")
    else:
        print(f"[成功] アップロード: {vector_id}")


# ---------------------------
# 単一ファイルの全文をチャンク分割 → ベクトル化 → Pinecone登録
# 各チャンクごとにアップロードとログ出力
# ---------------------------
def process_file(file_path, namespace):
    text = extract_text(file_path)
    if not text.strip():
        print(f"[スキップ] 空または抽出不可: {file_path}")
        return
    chunks = chunk_text(text)
    for idx, chunk in enumerate(chunks):
        vector_id = f"{os.path.basename(file_path)}-chunk-{idx+1}"
        try:
            embedding = get_embedding(chunk)
            metadata = {"source": file_path, "text": chunk}
            upload_to_pinecone(vector_id, embedding, metadata, namespace)
        except Exception as e:
            print(f"[エラー] {vector_id} の処理中に失敗: {e}")


# ---------------------------
# ディレクトリ内の全ファイルを走査して順次処理
# サブディレクトリも含め再帰的に処理
# ---------------------------
def process_directory(directory_path, namespace):
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"[処理開始] {file_path}")
            process_file(file_path, namespace)


# ---------------------------
# コマンドライン引数の読み取りと実行
# directory: 処理対象の文書フォルダ（例: Flask/PDF）
# namespace: ベクトルを保存するPinecone上の論理グループ名
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="対象ディレクトリ（例: Flask/PDF）")
    parser.add_argument(
        "namespace", help="Pineconeのネームスペース（例: our-project-specs）"
    )
    args = parser.parse_args()

    # フォルダ存在チェック
    if not os.path.exists(args.directory) or not os.path.isdir(args.directory):
        print(f"[エラー] ディレクトリが存在しません: {args.directory}")
        exit(1)

    # 中身空チェック
    if not any(os.scandir(args.directory)):
        print(f"[エラー] ディレクトリが空です: {args.directory}")
        exit(1)

    # 一括処理開始
    process_directory(args.directory, args.namespace)
