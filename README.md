# atmacup9
## これは何か
- atmacup9にて使用したコード類をまとめました。
- ソリューションについては[ぐるぐる](https://www.guruguru.science/competitions/14/discussions/a560bdd2-94a3-4eb2-bc52-972ae700be5e/)にも投稿しておりますが、秘密保持契約が必要なので、solutio.mdにも問題がない程度に記載しておきました。

## コンペ概要
トライアル(24時間営業のスーパー)でのレジカートのデータを使用した購買分析。対象カテゴリ15種をユーザーが来店時に買うか否か予測する。

## 評価指標
Macro AUC (ROC AUC Score) です。

```
from sklearn.metrics import roc_auc_score
scores = []

y_pred = pred_df.values # shape = (n_samples, n_labels)
y_true = pd.read_csv('train.csv').values[:, 1:] # shape = (n_samples, n_labels)

# 各ラベルごとに AUC を計算
for i in range(y_pred.shape[1]):
    score_i = roc_auc_score(y_true[:, i], y_pred[:, i])
    scores.append(score_i)

# 平均をとる
auc_score = sum(scores) / len(scores)

# sklearn.metrics.roc_auc_score には macro option も用意されていますのでこちらが便利だと思います。
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
roc_auc_score(y_true, y_pred, average='macro')
```

## 使用したノートブック
- `./feature_engineering` : 特徴量作成
- `./model` : モデル

## データ
- データについては機密保持契約の都合上、リンクを貼るのは控えておきます。カラム名は問題ないとのことなので、コードはGitHubに上げています。

## 結果
- Private13位(Public27位)
- 上位5%以内、前回より大幅進歩！！
- 自分で色々なモデルを組み立てるいい練習になった