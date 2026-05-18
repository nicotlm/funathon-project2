# %%
import os

import dotenv
import mlflow
import polars as pl
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.model.components import AttentionConfig
from torchTextClassifiers.model.components.text_embedder import LabelAttentionConfig
from torchTextClassifiers.tokenizers import WordPieceTokenizer

dotenv.load_dotenv(dotenv_path=".env", override=True)

df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)

n_classes = df.n_unique("code")


def train_val_test_split(
    df: pl.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df[:n_train]
    val = df[n_train : n_train + n_val]
    test = df[n_train + n_val :]
    return train, val, test


from sklearn.preprocessing import LabelEncoder

train, val, test = train_val_test_split(df)

# Fit on all codes so val/test codes are not unknown to the encoder
encoder = LabelEncoder()
encoder.fit(df["code"].to_numpy())

X_train, y_train = (
    train["label"].to_numpy(),
    encoder.transform(train["code"].to_numpy()),
)
X_val, y_val = val["label"].to_numpy(), encoder.transform(val["code"].to_numpy())
X_test, y_test = test["label"].to_numpy(), encoder.transform(test["code"].to_numpy())

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"Example: '{train['label'][0]}' → {encoder.inverse_transform(y_train[:1])[0]}")

tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=125)

tokenizer.train(X_train)

tokenizer.tokenize(X_train[0])
tokenizer.tokenizer.convert_ids_to_tokens(
    tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
)


attention_config = AttentionConfig(
    n_layers=1,
    n_head=4,
    n_kv_head=4,
    sequence_len=tokenizer.output_dim,
)

embedding_dim = 96

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,
    attention_config=attention_config,
    label_attention_config=LabelAttentionConfig(n_head=4, num_classes=n_classes),
)

ttc = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)

config = TrainingConfig(lr=1e-3, num_epochs=1, batch_size=256, num_workers=0)

TRAIN_FRAC = 0.05
VAL_FRAC = 0.05

X_train_small = X_train[: int(len(X_train) * TRAIN_FRAC)]
y_train_small = y_train[: int(len(y_train) * TRAIN_FRAC)]
X_val_small = X_val[: int(len(X_val) * VAL_FRAC)]
y_val_small = y_val[: int(len(y_val) * VAL_FRAC)]

# ttc.train(
#     X_train=X_train_small,
#     y_train=y_train_small,
#     X_val=X_val_small,
#     y_val=y_val_small,
#     training_config=config,
# )

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
RUN_ID = os.environ["MLFLOW_RUN_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# %%
