# chiTra ハンズオン on AWS

本リポジトリでは、2022 年 12 月 03 日に開催した[chiTra ハンズオン on AWS](https://worksapplications.connpass.com/event/259016/)にて使用した資料およびリソースを配布します。

## 構成

### `chiTra ハンズオン on AWS.pdf`

当日使用したスライド資料です。
アクセスキーやリファラルコードについては削除されています。

### `notebooks/`

当日使用した Jupyter ノートブックです。
これらを tar アーカイブ化したものが、資料における `notebooks.tar.gz` になります。

### `source/`

当日使用した訓練用資源です。
これらおよび chiTra モデルデータを tar アーカイブ化したものが、資料における `sourcedir.tar.gz` になります。

具体的に `sourcedir.tar.gz` を再構成する場合は、以下のようにします。

```bash
# source ディレクトリ内で作業
cd source

# chiTra モデルのダウンロード
wget https://sudachi.s3.ap-northeast-1.amazonaws.com/chitra/chiTra-1.0.tar.gz
tar -xvf chiTra-1.0.tar.gz
rm chiTra-1.0.tar.gz

# アーカイブの作成
tar -cvf sourcedir.tar.gz ./*
```

## 実行時の注意

### ノートブックの要改変部分

ノートブック `02 chiTraモデルのファインチューニング.ipynb` については、AWS へのアクセス権の関係上、そのまま実行することはできません。
ここでは変更が必要になる部分について概説します。
詳細については以降のセクションも参照してください。

- 定数
  - `s3_handson_bucket` 他
    - アクセス可能な S3 バケットを設定してください。
- データの確認
  - バケット非公開のため、s3 からのデータセットの読み込みはできません。
    - ご自身で前処理を行ったデータを適宜読み込んでください。
- 訓練ジョブの定義
  - 引数 `role`
    - ご自身の IAM ロールを作成・使用してください。
  - 引数 `source_dir`
    - 再構成・アップロードした `sourcedir.tar.gz` の arn へ変更してください。
    - もしくは `source_dir="./source/"` とすることでジョブ実行時にアップロードを行うことも可能です。

### S3

本ハンズオンでは S3 をデータ置き場として使用していますが、当日使用したバケットは非公開となります。
再現される場合はご自身の AWS アカウントにて作成したバケットに変更してください。

### データセット

本ハンズオンでは、[livedoor ニュースコーパス](http://www.rondhuit.com/download.html#ldcc)を文章分類タスクの題材として利用しました。

ハンズオンに際しては、簡単のため事前に前処理を行ったものを使用しています。
具体的には以下の処理を行ったものが、ノートブックにおける `raw` データに相当します。

- データとして以下を抽出
  - 本文：各記事ファイルより取得。URL、日付、タイトルは利用しない。
  - ラベル：各サブディレクトリの名称をラベルとして取得。
- 以下の形式で保存 (Hugging Face datasets を使用)

```json
{
  "sentence1": "本文",
  "label": "ラベル"
}
```

また訓練・開発・評価セットは、生データを 8/1/1 の割合で分割して作成しました（開発セットはハンズオンでは未使用）。

### AWS アカウント

本ハンズオンでは Amazon SageMaker Studio Lab を作業環境、Amazon SageMaker を訓練およびデプロイに使用しました。
Studio Lab から SageMaker API を使用するため、aws CLI による認証情報の設定を行っています。
当日使用した認証情報は非公開となるため、再現のためにはご自身の AWS アカウントにて IAM の作成が必要です。

具体的には以下の二つを当該箇所で代用することができます。

- ターミナルから認証を行う IAM ユーザ
  - ポリシーとして以下を付与
    - `SageMakerFullAccess`
    - `PowerUserAccess`
  - `aws configure` の設定に使用
    - 作成時にアクセスキーを生成する必要があります
- SageMaker API に受け渡す IAM ロール
  - ポリシーとして以下を付与
    - `SageMakerFullAccess`
  - 訓練ジョブ定義時の引数で使用
    - `role=...` に arn を渡す

[こちらの外部資料](https://github.com/aws-samples/aws-ml-enablement-workshop/blob/main/notebooks/scenario_churn/customer_churn_sagemaker.ipynb)も参照してください。

なお、上記に記載のポリシーでは、S3 へのアクセス権付与は一部バケットのみとなるためご注意ください（参考：[SageMaker Roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#sagemaker-roles-create-execution-role) の注）。
