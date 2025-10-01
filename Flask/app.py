from flask import Flask, render_template, request, jsonify  # Flaskの主要モジュールをインポート
import openai  # OpenAI API用モジュール
from pinecone import Pinecone  # Pinecone SDK
import config  # 環境変数を読み込む自作モジュール

# Flask初期化
app = Flask(__name__)

# OpenAI APIキー設定
openai.api_key = config.OPENAI_API_KEY

# Pinecone APIキーを使ってインスタンス生成
pc = Pinecone(api_key=config.PINECONE_API_KEY)
#index = pc.Index(config.PINECONE_INDEX_NAME)  # 指定Indexを操作対象にする
index = pc.Index(name=config.PINECONE_INDEX_NAME, host="https://urata-soft-js34rwd.svc.aped-4627-b74a.pinecone.io")
# トップページを表示（index.htmlを返す）
@app.route("/")
def index_page():
    return render_template("index.html")

# ユーザーからのクエリを処理
@app.route("/query", methods=["POST"])
def query():
    user_input = request.json.get("query")  # 入力テキストを取得
    
    # OpenAIで埋め込みベクトルを生成
    embedding = openai.embeddings.create(
        model="text-embedding-3-small",
        input=user_input
    ).data[0].embedding

    # Pineconeでベクトル検索（Top3件、メタデータ付き、指定namespace）
    result = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True,
        namespace="nextgen-specs"
    )

    # 結果が取得できなかった場合のエラーハンドリング
    if result is None:
        return jsonify({"error": "検索結果がありませんでした"}), 500

    # matchesを取り出す（Pineconeの戻り値はオブジェクトまたは辞書）
    matches = result['matches'] if isinstance(result, dict) else result.matches

    if matches:
        # 複数マッチ結果のテキストを結合し、OpenAIに要約させる
        context = "\n\n".join([m['metadata']['text'] for m in matches])

        # GPT-4o-miniで、質問に対する答えのみを抽出
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "ユーザーの質問に対して、以下の情報から答えだけを日本語で簡潔に抜き出して返答してください。"},
                {"role": "user", "content": f"質問: {user_input}\n情報:\n{context}"}
            ]
        )

        # 抽出した答えを返却
        answer_text = completion.choices[0].message.content.strip()
        return jsonify({"answer": answer_text})
    else:
        # マッチが0件の場合の処理
        return jsonify({"answer": "該当する回答が見つかりませんでした"})

# アプリケーションをデバッグモードで起動
if __name__ == "__main__":
    app.run(debug=True)
