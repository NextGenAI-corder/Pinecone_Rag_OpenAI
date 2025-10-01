## 🚀 起動手順（Flask版）

1. Python 3.10以上をインストールしておく

2. 必要なライブラリをインストール：

```bash
pip install -r requirements.txt
config.env.template をコピーして .env を作成し、APIキー等を記述：

text
コードをコピーする
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx
PINECONE_ENV=us-east4-gcp
PINECONE_INDEX_NAME=your-index-name
PDFなどの知識ソースを Flask/PDF/ に配置
→ 初回のみベクトル登録を実行：

bash
コードをコピーする
python Flask/upload_embeddings.py
アプリを起動：

bash
コードをコピーする
python Flask/app.py
ブラウザでアクセス：

arduino
コードをコピーする
http://localhost:5000
yaml
コードをコピーする

---