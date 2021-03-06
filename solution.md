# 目標
今回のコンペの目標は下記の通りです。
- 雑でもいいから最速submissionして瞬間1位を狙う(1位ならず2位でした)
- 普段カーネルコピーしてアレンジしていることが多いので、今回は自分でコードを組み立てていく(達成)
- 上位15%に入る(達成)

# Trainの正解データ作成方法
ディスカッションのコメントなどを参考にしながら、なるべくTestデータに近い形で作成しました。
- 10分以上のセッションに絞る
- time_elapsedを[0, 3, 5, 10]からランダムに選択
＊1時間ほど実行に時間がかかってしまいました...

# モデル
- 下記４モデルの加重平均を取りました
    - LightGBM(CV: 0.81494/Public: 0.7704/Private: 0.7693)
    - Catboost(CV: 0.82061)
    - XGBoost(CV: 0.81463)
    - nn(CV: 0.77510/Public: 0.7279/Private: 0.7222)
- lgb * 0.3 + cat * 0.3 + xgb * 0.3 + nn * 0.1
- またLightGBM, Catboost, XGBoostは5seed averaging
- 今回のはシェイクがそこそこありそうだなと思ったので、複数モデルで頑健なものにしようと心がけました。
- nnモデルが金曜から組み始めたのですが、他のモデルほど上げられず...

# バリデーション
- 各ターゲットカテゴリごとにTrue, Falseが同じ数になるようにFalseの方をダウンサンプリング
- 時系列順に並べて、古いセッション8割を学習データに、残りの2割を検証データにした
    - 上記モデルでのベストイテレーション数を取得し、それをもとに全サンプリングデータで再学習
    - 時系列は意識する必要あったか怪しいが、自分の場合はこの切り方にした方が0.02ほどCVとLBの乖離が狭まった
- ここに書くべきか分かりませんが、リークしていないかなどの部分は可能な限り注意しました。

# 特徴量
特徴量は色々作っていって、515個になりました。以下、簡単に記載しておきます。
## メタ情報
- 年代
- 性別
- 来店時間(hour)
- 曜日
- 休日・祝日フラグ
- 前休日フラグ(いわゆる華金、祝日の前日も含む)
- 来店回数
- 月
- 日
- register_number
- time_elapsed

## ラグ及び公開特徴量
ラグ特徴量: 対象セッションよりも前にユーザーが来店していた場合、それらの情報(いわゆる過去のセッション)。主に平均値にして使用。
公開特徴量: 対象セッションのtime_elapsedよりも前に与えられている特徴量
ラグ特徴量で作れるものは大体、公開特徴量でも使用できたため、まとめて記載。
- 対象カテゴリの購入回数
- 対象カテゴリの購入価格
- ユーザーごとの平均購入価格
- ユーザーごとの購入商品数
- ユーザーごとの滞在時間
- 全部門CDごとの購入頻度
- 購入キャンセル回数
- 10分以上買い物をしていたにもかかわらずキャンセルした回数
- クーポンを見た回数
    - クーポンは利用時に見られていることが多かった。
    - 自分の場合は、CV/LBともに変化なし
- targetカテゴリを除くカテゴリごとの購入回数
    - 全カテゴリを確認していき、関連していそうだと個人的に思った物を抜粋
    - また後述するword2vecで対象カテゴリと類似しているカテゴリも入れた
    - ディスカッションに上がっていたground0stateさんの[アソシエーション分析](https://www.guruguru.science/competitions/14/discussions/7e011ee2-544e-4e6a-880c-a32b00dcf0b1/)のも入れました。ありがとうございます。
- 子供持ち家庭特徴量
    - 子供持ちの家庭は子供向けの商品(アンパンマンやポケモン関連のお菓子など)を買った時に、アイスなども一緒に買いやすいのではという仮説のもと作成した
    - 商品名に[アンパンマン、ディズニー、ポケモン]などの子供向けの商品をピックアップした。目で確認していったので漏れや主観あり
- 1人暮らし特徴量
    - 正確には、「料理をしない人」特徴量
    - 野菜や肉などの材料を買わず、弁当や袋麺などを買う人たち
    - 1人暮らしをしていて、カップ麺や間食(お菓子)をしやすいと仮定した。
- 家庭持ち特徴量
    - 1人暮らし特徴量とは逆に、家庭を持っている人たち
    - 正確には、「料理をする人」特徴量
    - 料理をする人は野菜や肉などの材料や調味料を多く買っていると仮定
    - 特に仮説はないけど、1人暮らし特徴量作るならついでに作るかというお気持ちで作成
- 類似ベクトル商品(word2vec)
    - 既にディスカッションに上がっているようなitem2vecです。
    - 自分はカテゴリごとにやりました。JANでやった方が良かったかもしれない。
    - 対象カテゴリの類似カテゴリを抽出してみた。

# 感想
今回、データの扱いが少々面倒で且つ仕事をしながらなので急ぎで書いたら、吐き気がするほどすごいコードが汚くなってしまいました。
次はもう少しコード綺麗に書くのを意識しながら取り組もうかなと思います。