import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Lambda,
    Softmax,
)
from zenml import step

from steps.train_models import format_results, merge_results_ordered, save_results_json

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# # Make TF deterministic
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# @step(enable_cache=False)
def train_lstm_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    lstm_units: int = 64,
    dropout_rate: float = 0.4,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    seed: int = 42,
    output_dir: str = "./data/lstm_results",
) -> dict:
    # Set seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # tf.config.experimental.enable_op_determinism()

    # Get input dimensions (window_size, num_features)
    input_shape = (X_train.shape[1], X_train.shape[2])

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Masking(mask_value=-1.0),
        tf.keras.layers.LSTM(lstm_units, return_sequences=False, recurrent_dropout=0.2),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    # Compile model
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True
    )

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))

    # Fit model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1,
    )

    model.save(os.path.join(output_dir, "lstm_model.h5"))

    with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    # Evaluate model
    y_pred_probs = model.predict(X_test).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\n✅ Balanced Accuracy: {balanced_acc:.4f}")

    # Metrics
    precision = round(precision_score(y_test, y_pred, average="macro"), 3)
    recall = round(recall_score(y_test, y_pred, average="macro"), 3)
    f1 = round(f1_score(y_test, y_pred, average="macro"), 3)

    report = classification_report(y_test, y_pred, output_dict=True)
    class_0_metrics = report["0"]
    class_1_metrics = report["1"]

    results = {}

    results['lstm'] = {
        "Global Metrics": {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        },
        "Liegen Metrics": class_0_metrics,
        "Stehen Metrics": class_1_metrics,
        "X_test": X_test,
    }

    r_g, c_0, c_1 = format_results(results)

    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\n✅ Balanced Accuracy: {balanced_acc:.4f}")
    merged_results = merge_results_ordered(r_g, c_0, c_1)
    merged_results["balanced_accuracy"] = balanced_acc
    save_results_json(
        os.path.join(output_dir, "results.json"),
        seed,
        merged_results,
        [1, 2, 3]
    )

    return {
        "lstm": {
            "macro_f1": float(f1),
            "balanced_accuracy": float(balanced_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_liegen": float(class_0_metrics["f1-score"]),
            "precision_liegen": float(class_0_metrics["precision"]),
            "recall_liegen": float(class_0_metrics["recall"]),
            "f1_stehen": float(class_1_metrics["f1-score"]),
            "precision_stehen": float(class_1_metrics["precision"]),
            "recall_stehen": float(class_1_metrics["recall"]),
        }
    }



def build_bilstm_seq2seq(input_shape, units=64, dropout=0.4, lr=1e-3):
    inp = tf.keras.Input(shape=input_shape)                  # (T,F)
    x = tf.keras.layers.Masking(mask_value=-1.0)(inp)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)  # (B,T,1)
    model = tf.keras.Model(inp, logits)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.AUC(curve="ROC", from_logits=True)])
    return model




def build_lstm_with_attention(input_shape, lstm_units=64, dropout_rate=0.2):
    inputs = Input(shape=input_shape)

    # LSTM layer (return sequences for attention)
    lstm_out = LSTM(units=lstm_units, return_sequences=True)(inputs)

    # Attention mechanism
    score = Dense(1, activation="tanh")(lstm_out)  # [batch, time, 1]
    attention_weights = Softmax(axis=1)(score)  # [batch, time, 1]

    # context vector as weighted sum of LSTM outputs
    context_vector = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))(
        [attention_weights, lstm_out]
    )

    # Dropout and Dense layers
    x = Dropout(dropout_rate)(context_vector)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs)


def tune_threshold(y_true, probs, metric="balanced_accuracy", grid=None, target_recall=None):
    """
    If target_recall is set (e.g., 0.9), pick the smallest threshold
    that achieves at least that recall using the PR curve.
    Otherwise, maximize the chosen metric over a threshold grid.
    """
    probs = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).ravel()

    if target_recall is not None:
        # pick threshold to satisfy recall >= target
        precision, recall, ths = precision_recall_curve(y_true, probs)
        # ths is length n-1; align with recall[1:]
        idx = np.where(recall[1:] >= target_recall)[0]
        if len(idx) == 0:
            # fallback to best recall available
            j = int(np.argmax(recall[1:]))
        else:
            j = int(idx[-1])  # highest threshold that still meets recall
        return float(ths[j]), {"precision": float(precision[j+1]), "recall": float(recall[j+1])}

    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    scores = []
    for t in grid:
        y_hat = (probs >= t).astype(int)
        if metric == "balanced_accuracy":
            s = balanced_accuracy_score(y_true, y_hat)
        elif metric == "f1":
            s = f1_score(y_true, y_hat)
        else:
            raise ValueError("metric must be 'balanced_accuracy' or 'f1'")
        scores.append(s)

    best_idx = int(np.argmax(scores))
    return float(grid[best_idx]), float(scores[best_idx])


# @step
def train_lstm_with_attention(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    seed: int = 42,
    output_dir: str = "./data/lstm_results",
) -> dict:
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Get input dimensions
    input_shape = (X_train.shape[1], X_train.shape[2])  # (window_size, num_features)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    model = build_lstm_with_attention(
        input_shape, lstm_units=lstm_units, dropout_rate=dropout_rate
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Fit model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=1,
    )

    model.save(os.path.join(output_dir, "lstm_model.h5"))

    # Evaluate model
    y_pred_probs = model.predict(X_test).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\n✅ Balanced Accuracy: {balanced_acc:.4f}")

    # Metrics
    precision = round(precision_score(y_test, y_pred, average="macro"), 3)
    recall = round(recall_score(y_test, y_pred, average="macro"), 3)
    f1 = round(f1_score(y_test, y_pred, average="macro"), 3)

    report = classification_report(y_test, y_pred, output_dict=True)
    class_0_metrics = report["0"]
    class_1_metrics = report["1"]

    results = {}

    results['lstm'] = {
        "Global Metrics": {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        },
        "Liegen Metrics": class_0_metrics,
        "Stehen Metrics": class_1_metrics,
        "X_test": X_test,
    }

    r_g, c_0, c_1 = format_results(results)

    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\n✅ Balanced Accuracy: {balanced_acc:.4f}")
    merged_results = merge_results_ordered(r_g, c_0, c_1)
    merged_results["balanced_accuracy"] = balanced_acc
    save_results_json(
        os.path.join(output_dir, "results.json"),
        seed,
        merged_results,
        [1, 2, 3]
    )

    return {
        "lstm_attention": {
            "macro_f1": float(f1),
            "balanced_accuracy": float(balanced_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_liegen": float(class_0_metrics["f1-score"]),
            "precision_liegen": float(class_0_metrics["precision"]),
            "recall_liegen": float(class_0_metrics["recall"]),
            "f1_stehen": float(class_1_metrics["f1-score"]),
            "precision_stehen": float(class_1_metrics["precision"]),
            "recall_stehen": float(class_1_metrics["recall"]),
        }
    }
