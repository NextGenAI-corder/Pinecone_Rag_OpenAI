from flask import (
    Flask,
    render_template,
    request,
    jsonify,
)  # Flaskの主要機能をインポート
import openai  # OpenAI APIを利用するためのライブラリ
from pinecone import Pinecone  # PineconeのPython SDK
import config  # APIキーなどを保持する自作モジュール
import sys  # コマンドライン引数を扱うための標準ライブラリ

# ---------------------------
# Flask アプリケーションの初期化
# ---------------------------
app = Flask(__name__)

# ---------------------------
# OpenAI および Pinecone の初期設定
# config.py に定義された APIキーを使用
# PineconeのホストURLは手動で指定している（self-hosted endpoint対応）
# ---------------------------
openai.api_key = config.OPENAI_API_KEY
pc = Pinecone(api_key=config.PINECONE_API_KEY)
index = pc.Index(
    name=config.PINECONE_INDEX_NAME,
    host="https://urata-soft-js34rwd.svc.aped-4627-b74a.pinecone.io",
)

# ---------------------------
# 起動時に namespace をコマンドライン引数から取得
# 指定がなければエラーメッセージを出して終了
# ※ Flaskアプリ全体で固定のnamespaceを使用する構成
# ---------------------------
if len(sys.argv) < 2:
    print("使用法: python app.py パラメータ<namespace>を指定して下さい！")
    exit(1)

NAMESPACE = sys.argv[1]  # グローバル変数として全体で使用するnamespaceを設定


# ---------------------------
# ルートエンドポイント（"/"）にアクセスされた際に index.html を表示
# templates/index.html が自動的に読み込まれる
# ---------------------------
@app.route("/")
def index_page():
    return render_template("index.html")


# ---------------------------
# POSTリクエスト "/query" を処理するAPIエンドポイント
# ユーザーから送信された質問文をもとに、ベクトル検索＋生成応答を行う
# ---------------------------
@app.route("/query", methods=["POST"])
def query():
    data = request.json  # フロントエンドから送信されたJSONを取得
    user_input = data.get("query")  # ユーザーの質問テキストを抽出

    # OpenAI APIで埋め込み（ベクトル化）を実行
    embedding = (
        openai.embeddings.create(
            model="text-embedding-3-small",  # 軽量かつ精度の高い埋め込み専用モデル
            input=user_input,
        )
        .data[0]
        .embedding
    )

    # Pinecone に対してベクトル検索を実行（Top5件）
    result = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True,  # 元テキストなどのメタ情報を含めて返す
        namespace=NAMESPACE,  # 起動時に指定されたnamespaceを使用（固定）
    )

    # 検索結果からマッチ部分を取得（Pineconeの返却形式に柔軟対応）
    matches = result["matches"] if isinstance(result, dict) else result.matches

    if matches:
        # 複数マッチ結果のテキストを文脈として連結
        context = "\n\n".join([m["metadata"]["text"] for m in matches])

        # ChatGPT APIを使って自然言語で応答を生成（制約付き）
        completion = openai.chat.completions.create(
            model="gpt-4o",  # 高速・高精度モデル
            messages=[
                {
                    "role": "system",
                    "content": "ユーザーの質問に対して、日本語で簡潔に50字以内で返答してください。",
                },
                {"role": "user", "content": f"質問: {user_input}\n情報:\n{context}"},
            ],
        )

        # 応答文を抽出して返却
        answer_text = completion.choices[0].message.content.strip()
        return jsonify({"answer": answer_text})
    else:
        # マッチがない場合のエラーメッセージ
        return jsonify({"answer": "該当する回答が見つかりませんでした"})


# ---------------------------
# Flask アプリケーションの起動（デバッグモード有効）
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
