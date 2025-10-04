import os
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

# .env または config.env.template から読み込む
load_dotenv("config.env.template")

openai.api_key = os.environ.get("OPENAI_API_KEY")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("urata-soft")  # インデックス名を必要に応じて変更

def get_similar_chunks(question, namespace="nextgen-specs", top_k=3):
    # 質問を埋め込み化
    embedding = openai.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Pineconeから類似文書検索
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    return [match["metadata"]["text"] for match in results["matches"]]

def ask_direct_answer(question, namespace="nextgen-specs", top_k=3):
    # 類似チャンク取得
    chunks = get_similar_chunks(question, namespace, top_k)
    context = "\n".join(chunks)

    # OpenAIへ回答生成プロンプト
    prompt = f"以下の情報に基づいて質問に答えてください:\n\n{context}\n\nQ: {question}\nA:"

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()
