from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk, load_metric
import random
import logging
import sys
import argparse
import os
import torch
import numpy as np
from pathlib import Path

# chitra ライブラリの読み込み
from sudachitra import BertSudachipyTokenizer
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # 訓練用ハイパーパラメータの定義（訓練ジョブの作成時に渡すもの）
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--num_labels", type=int)

    # モデルやデータ、出力先などを指定するパラメータ
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # ロギング
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 訓練・評価用のデータを読み込む
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # モデルの性能評価用の基準を定義する
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # chiTra モデルとトークナイザを読み込む
    model_path = "./chiTra-1.0"
    model = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=args.num_labels)
    tokenizer = BertSudachipyTokenizer.from_pretrained(model_path)

    # 訓練用パラメータを定義する
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )

    # Trainer を定義する
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # 訓練の実行
    trainer.train()

    # テストデータでモデルの性能を評価する
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # 評価結果を後から参照できるよう保存する
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            print(f"{key} = {value}\n")
            writer.write(f"{key} = {value}\n")

    # 訓練済みのモデルをs3に保存する
    trainer.save_model(args.model_dir)

    # デプロイ用のコードもモデルと併せて保存する
    shutil.copytree("./code", f"{args.model_dir}/code")
