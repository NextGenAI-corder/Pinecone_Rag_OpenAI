# Pinecone+Rag AI FAQ構築セット

RAG と OpenAI API を用いた FAQ ボットの構築テンプレートです。

## 1. ディレクトリ構成

```
├── Flask
│   ├── .env
│   ├── PDF
│   │   └── sample_specification.pdf
│   ├── app.py
│   ├── config.py
│   └── templates
│       └── index.html
├── README.md
├── config.env.template
├── docs
│   ├── operating_instructions.pdf
├── query_embeddings.py
├── requirements.txt
└── upload_embeddings.py

```

## 2. 起動手順
2.1 Python の準備
Python 3.10 以上をインストールしてください。

2.2 ライブラリのインストール
pip install -r requirements.txt

2.3 .env ファイルの作成と設定
config.env.template をコピーして .env を作成し、以下の内容を記述します。

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx
PINECONE_ENV=your-environment
PINECONE_INDEX_NAME=your-index-name
```

2.4 ベクトルの登録（初回のみ）
python upload_embeddings.py

2.5 アプリの起動
python Flask/app.py

2.6 ブラウザでのアクセス
http://localhost:5000

## 3. マニュアル
詳細な使用方法は docs/operating_instructions.pdf を参照してください。

## 4. ライセンス
教育・研究・PoC 用途での利用は許可します。
商用利用・再配布は販売条件に従ってください。