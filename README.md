# Pinecone RAG FAQ Bot 構築キット

RAG と OpenAI API を用いた FAQ ボットの構築テンプレートです。

## ディレクトリ構成

```text
.
├── Flask
│   ├── PDF
│   │   └── sample_specification.pdf
│   ├── app.py
│   ├── config.py
│   ├── templates
│   │   └── index.html
│   └── upload_embeddings.py
├── Scripts
│   ├── index.html
│   ├── openai.web.js
│   └── query_embeddings.py
├── config.env.template
├── docs
│   └── operating_instructions.pdf
└── requirements.txt

起動手順

markdown
コードをコピーする

## 起動手順

1. Python 3.10 以上をインストール
2. ライブラリをインストール：

```bash
pip install -r requirements.txt
.env を作成し、以下を記述：

ini
コードをコピーする
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx
PINECONE_ENV=your-environment
PINECONE_INDEX_NAME=your-index-name
初回のみ、埋め込みを登録：

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
マニュアル
docs/operating_instructions.pdf を参照してください。

ライセンス
教育・研究・PoC 用途での利用は許可します。商用利用・再配布は販売条件に従ってください。

コードをコピーする
