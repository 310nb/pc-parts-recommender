# 用途・予算別 PC構成推薦

Sentence-BERTで入力した用途と登録済みPC構成の意味的な類似度を計算し、
予算内のおすすめ構成を表示するStreamlitプロトタイプです。

## 起動方法

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType none
```

初回起動時は、Sentence-BERTモデルのダウンロードが発生します。

## 操作

- 「構成を検索」では、自然文の用途、予算、候補数を指定します。
- 「性能評価」では、固定した言い換え入力10件のTop-1/Top-3正解率を確認できます。
