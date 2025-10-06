from dotenv import dotenv_values
import openai
from pinecone import Pinecone

# ---------------------------
# 環境変数読み込み（config.env.template から直接取得）
# ※ システムの環境変数に依存せず、ファイルからキー・設定を読み込む構造
# → セキュリティ性・移植性が高く、テスト環境・本番環境の切替にも適する
# ---------------------------
config = dotenv_values("config.env.template")

# OpenAI APIキーを直接設定（環境変数経由ではない）
openai.api_key = config.get("OPENAI_API_KEY")

# Pinecone初期化（APIキーを直接取得）
pc = Pinecone(api_key=config.get("PINECONE_API_KEY"))

# Pineconeのインデックス名も外部設定から取得
# → 複数プロジェクト・データセットに対応できる柔軟な構造
index = pc.Index(config.get("PINECONE_INDEX_NAME"))


# ---------------------------
# 類似文書検索（Pinecone + OpenAI埋め込み）
# 入力: 質問文（自然言語）, namespace（データセット識別子）
# 出力: 検索されたメタ情報（text）のリスト
# ---------------------------
def get_similar_chunks(question, namespace):
    # OpenAI APIで質問をベクトル化（埋め込みモデルを使用）
    embedding = (
        openai.embeddings.create(
            input=question,
            model="text-embedding-3-small",  # 軽量・高精度の最新埋め込みモデル
        )
        .data[0]
        .embedding
    )

    # PineconeベクトルDBから類似検索を実行
    results = index.query(
        vector=embedding,
        top_k=5,  # 上位5件の類似文書を取得
        include_metadata=True,  # 元テキストを含むメタ情報を含めて返す
        namespace=namespace,  # プロジェクトや用途で論理分離するための識別子
    )

    # メタ情報からテキスト本文のみ抽出して返却
    return [match["metadata"]["text"] for match in results["matches"]]


# ---------------------------
# 応答生成（OpenAI Chatモデル）
# 入力: ユーザーの質問, namespace（検索対象）
# 出力: gpt-4o による自然言語の回答文（str）
# ---------------------------
def ask_direct_answer(question, namespace):
    # 類似文書を取得し、回答のコンテキスト（文脈）として利用
    chunks = get_similar_chunks(question, namespace)
    context = "\n".join(chunks)

    # プロンプト設計：FAQ文書を前提にした回答生成を指示
    prompt = (
        f"以下の情報に基づいて質問に答えてください:\n\n{context}\n\nQ: {question}\nA:"
    )

    # gpt-4o（2024年最新）を使用して応答生成
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )

    # 応答から生成テキストのみ抽出して返す
    return response.choices[0].message.content.strip()
