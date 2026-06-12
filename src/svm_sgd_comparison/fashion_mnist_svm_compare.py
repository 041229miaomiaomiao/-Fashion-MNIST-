from __future__ import annotations

import argparse
import gzip
import io
import re
import struct
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


CLASS_NAMES = [
    "T恤/上衣",
    "裤子",
    "套头衫",
    "连衣裙",
    "外套",
    "凉鞋",
    "衬衫",
    "运动鞋",
    "包",
    "短靴",
]

IDX_PATTERNS = {
    "train_images": ("train-images", "idx3"),
    "train_labels": ("train-labels", "idx1"),
    "test_images": ("t10k-images", "idx3"),
    "test_labels": ("t10k-labels", "idx1"),
}


def parse_args() -> argparse.Namespace:
    default_out_dir = Path(__file__).resolve().parent / "fashion_mnist_results"

    parser = argparse.ArgumentParser(
        description=(
            "基于 scikit-learn 的 Fashion MNIST 图像分类与可视化程序。"
            "支持读取已解压的 IDX 数据目录、内层数据压缩包或作业提供的 output.zip。"
        )
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        help=(
            "Fashion MNIST 数据目录或 zip 压缩包路径。"
            "如果省略，程序会自动搜索常见本地位置。"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(default_out_dir),
        help="图表和 CSV 结果文件的保存目录。",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="训练样本数量限制。0 表示使用全部 60000 个训练样本。",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="测试样本数量限制。0 表示使用全部 10000 个测试样本。",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="抽样和模型训练使用的随机种子。",
    )
    parser.add_argument(
        "--pca-samples",
        type=int,
        default=2500,
        help="PCA 降维散点图使用的训练样本数量。",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="跳过 PCA 可视化，用于加快运行速度。",
    )
    return parser.parse_args()


def normalized_name(name: str) -> str:
    return name.replace("\\", "/").lower()


def find_idx_payloads_in_zip(zf: zipfile.ZipFile) -> Dict[str, bytes] | None:
    found: Dict[str, bytes] = {}

    for info in zf.infolist():
        if info.is_dir():
            continue
        lower = normalized_name(info.filename)
        for key, tokens in IDX_PATTERNS.items():
            if key in found:
                continue
            if all(token in lower for token in tokens):
                found[key] = zf.read(info)

    if all(key in found for key in IDX_PATTERNS):
        return found
    return None


def find_idx_payloads_in_nested_zip(zip_path: Path) -> Dict[str, bytes]:
    with zipfile.ZipFile(zip_path) as outer_zip:
        direct = find_idx_payloads_in_zip(outer_zip)
        if direct is not None:
            return direct

        nested_zip_infos = [
            info
            for info in outer_zip.infolist()
            if not info.is_dir() and normalized_name(info.filename).endswith(".zip")
        ]
        for info in nested_zip_infos:
            nested_bytes = outer_zip.read(info)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_zip:
                nested = find_idx_payloads_in_zip(nested_zip)
                if nested is not None:
                    return nested

    raise FileNotFoundError(
        f"压缩包中没有找到 Fashion MNIST 的 IDX 数据文件：{zip_path}"
    )


def find_idx_payloads_in_dir(data_dir: Path) -> Dict[str, bytes]:
    files = [path for path in data_dir.rglob("*") if path.is_file()]
    found: Dict[str, bytes] = {}

    for key, tokens in IDX_PATTERNS.items():
        for path in files:
            lower = normalized_name(path.name)
            if all(token in lower for token in tokens):
                found[key] = path.read_bytes()
                break

    missing = [key for key in IDX_PATTERNS if key not in found]
    if missing:
        raise FileNotFoundError(
            f"数据目录缺少 IDX 文件：{data_dir}，缺少项：{', '.join(missing)}"
        )

    return found


def has_idx_files(data_dir: Path) -> bool:
    if not data_dir.exists() or not data_dir.is_dir():
        return False

    names = [normalized_name(path.name) for path in data_dir.rglob("*") if path.is_file()]
    return all(
        any(all(token in name for token in tokens) for name in names)
        for tokens in IDX_PATTERNS.values()
    )


def candidate_sources() -> Iterable[Path]:
    here = Path(__file__).resolve().parent
    bases = [Path.cwd(), here, here.parent]
    seen: set[Path] = set()

    for base in bases:
        for rel in (
            Path("data"),
            Path("work") / "fashion_data" / "data",
            Path("work") / "output_extracted",
            Path("."),
        ):
            candidate = (base / rel).resolve()
            if candidate not in seen:
                seen.add(candidate)
                yield candidate

    for base in bases:
        if not base.exists():
            continue
        for candidate in base.glob("*.zip"):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved
        work_dir = base / "work"
        if work_dir.exists():
            for candidate in work_dir.rglob("*.zip"):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def resolve_source(source_arg: str | None) -> Path:
    if source_arg:
        source = Path(source_arg).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"数据来源不存在：{source}")
        return source

    for candidate in candidate_sources():
        if candidate.is_dir() and has_idx_files(candidate):
            return candidate
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            try:
                find_idx_payloads_in_nested_zip(candidate)
                return candidate
            except (FileNotFoundError, zipfile.BadZipFile):
                continue

    raise FileNotFoundError(
        "无法自动找到 Fashion MNIST 数据，请通过 --data-source 手动指定数据路径。"
    )


def maybe_decompress(payload: bytes) -> bytes:
    if payload[:2] == b"\x1f\x8b":
        return gzip.decompress(payload)
    return payload


def parse_idx_images(payload: bytes) -> np.ndarray:
    raw = maybe_decompress(payload)
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Unexpected image magic number: {magic}")
    images = np.frombuffer(raw, dtype=np.uint8, offset=16)
    return images.reshape(count, rows, cols)


def parse_idx_labels(payload: bytes) -> np.ndarray:
    raw = maybe_decompress(payload)
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Unexpected label magic number: {magic}")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return labels.reshape(count)


def load_fashion_mnist(source: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if source.is_dir():
        payloads = find_idx_payloads_in_dir(source)
    elif source.is_file() and source.suffix.lower() == ".zip":
        payloads = find_idx_payloads_in_nested_zip(source)
    else:
        raise ValueError(f"不支持的数据来源：{source}")

    x_train = parse_idx_images(payloads["train_images"])
    y_train = parse_idx_labels(payloads["train_labels"])
    x_test = parse_idx_images(payloads["test_images"])
    y_test = parse_idx_labels(payloads["test_labels"])
    return x_train, y_train, x_test, y_test


def stratified_limit(
    x: np.ndarray, y: np.ndarray, limit: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or limit >= len(y):
        return x, y

    indices = np.arange(len(y))
    selected, _ = train_test_split(
        indices,
        train_size=limit,
        stratify=y,
        random_state=random_state,
    )
    selected = np.sort(selected)
    return x[selected], y[selected]


def flatten_and_scale(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def build_models(random_state: int) -> Mapping[str, object]:
    return {
        "SGD逻辑回归": make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                max_iter=40,
                tol=1e-3,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "线性SVM": make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=1.0,
                dual=False,
                max_iter=5000,
                random_state=random_state,
            ),
        ),
        "随机森林": RandomForestClassifier(
            n_estimators=180,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def safe_file_stem(name: str) -> str:
    known_names = {
        "SGD逻辑回归": "sgd_logistic",
        "线性SVM": "linear_svm",
        "随机森林": "random_forest",
    }
    if name in known_names:
        return known_names[name]
    return re.sub(r"[^\w]+", "_", name, flags=re.UNICODE).strip("_").lower()


def plot_sample_grid(
    images: np.ndarray, labels: np.ndarray, out_path: Path, random_state: int
) -> None:
    rng = np.random.default_rng(random_state)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6))
    axes = axes.ravel()

    for class_id, ax in enumerate(axes):
        candidates = np.flatnonzero(labels == class_id)
        chosen = rng.choice(candidates)
        ax.imshow(images[chosen], cmap="gray")
        ax.set_title(CLASS_NAMES[class_id], fontsize=9)
        ax.axis("off")

    fig.suptitle("Fashion MNIST 样本图片", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_class_distribution(
    y_train: np.ndarray, y_test: np.ndarray, out_path: Path
) -> None:
    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)
    x_axis = np.arange(10)
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x_axis - width / 2, train_counts, width, label="训练集", color="#4C78A8")
    ax.bar(x_axis + width / 2, test_counts, width, label="测试集", color="#F58518")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylabel("样本数量")
    ax.set_title("类别数量分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pca_scatter(
    x_train: np.ndarray,
    y_train: np.ndarray,
    out_path: Path,
    sample_count: int,
    random_state: int,
) -> None:
    sample_count = min(sample_count, len(y_train))
    indices, _ = train_test_split(
        np.arange(len(y_train)),
        train_size=sample_count,
        stratify=y_train,
        random_state=random_state,
    )

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(x_train[indices])

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=y_train[indices],
        s=12,
        cmap="tab10",
        alpha=0.75,
        linewidths=0,
    )
    legend = ax.legend(
        handles=scatter.legend_elements()[0],
        labels=CLASS_NAMES,
        title="类别",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    ax.add_artist(legend)
    ax.set_title("训练图像的 PCA 二维可视化")
    ax.set_xlabel("第一主成分")
    ax.set_ylabel("第二主成分")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_model_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    ordered = summary_df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(ordered["model"], ordered["accuracy"] * 100, color="#4C78A8")
    axes[0].set_title("模型准确率对比")
    axes[0].set_ylabel("准确率（%）")
    axes[0].set_ylim(max(0, ordered["accuracy"].min() * 100 - 5), 100)
    axes[0].tick_params(axis="x", rotation=25)

    for idx, value in enumerate(ordered["accuracy"] * 100):
        axes[0].text(idx, value + 0.3, f"{value:.2f}%", ha="center", fontsize=9)

    axes[1].bar(ordered["model"], ordered["total_seconds"], color="#54A24B")
    axes[1].set_title("模型运行时间对比")
    axes[1].set_ylabel("耗时（秒）")
    axes[1].tick_params(axis="x", rotation=25)

    for idx, value in enumerate(ordered["total_seconds"]):
        axes[1].text(idx, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, out_path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    display = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"混淆矩阵 - {model_name}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_misclassified_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Path,
) -> None:
    wrong = np.flatnonzero(y_true != y_pred)
    if len(wrong) == 0:
        return

    count = min(12, len(wrong))
    chosen = wrong[:count]
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    axes = axes.ravel()

    for ax, idx in zip(axes, chosen):
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(
            f"真实: {CLASS_NAMES[int(y_true[idx])]}\n预测: {CLASS_NAMES[int(y_pred[idx])]}",
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes[count:]:
        ax.axis("off")

    fig.suptitle(f"误分类样例 - {model_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def train_and_evaluate(
    models: Mapping[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_test_images: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []

    for model_name, model in models.items():
        print(f"\n正在训练模型：{model_name} ...")
        fit_start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        y_pred = model.predict(x_test)
        pred_seconds = time.perf_counter() - pred_start

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        total_seconds = fit_seconds + pred_seconds

        print(
            f"{model_name}：准确率={accuracy:.4f}，"
            f"宏平均F1={macro_f1:.4f}，耗时={total_seconds:.2f}秒"
        )

        report = classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.rename(
            columns={
                "precision": "精确率",
                "recall": "召回率",
                "f1-score": "F1值",
                "support": "样本数",
            },
            index={
                "accuracy": "准确率",
                "macro avg": "宏平均",
                "weighted avg": "加权平均",
            },
        )
        model_stem = safe_file_stem(model_name)
        report_df.to_csv(
            out_dir / f"classification_report_{model_stem}.csv",
            encoding="utf-8-sig",
        )

        plot_confusion_matrix(
            y_test,
            y_pred,
            model_name,
            out_dir / f"confusion_matrix_{model_stem}.png",
        )
        plot_misclassified_examples(
            x_test_images,
            y_test,
            y_pred,
            model_name,
            out_dir / f"misclassified_{model_stem}.png",
        )

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "fit_seconds": fit_seconds,
                "predict_seconds": pred_seconds,
                "total_seconds": total_seconds,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    summary_output_df = summary_df.rename(
        columns={
            "model": "模型",
            "accuracy": "准确率",
            "macro_f1": "宏平均F1",
            "weighted_f1": "加权平均F1",
            "fit_seconds": "训练耗时（秒）",
            "predict_seconds": "预测耗时（秒）",
            "total_seconds": "总耗时（秒）",
        }
    )
    summary_output_df.to_csv(
        out_dir / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary_df


def main() -> None:
    args = parse_args()
    source = resolve_source(args.data_source)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据来源：{source}")
    print(f"输出目录：{out_dir}")

    x_train_images, y_train, x_test_images, y_test = load_fashion_mnist(source)
    print(
        "已加载 Fashion MNIST："
        f"训练集={x_train_images.shape}，测试集={x_test_images.shape}"
    )

    x_train_images, y_train = stratified_limit(
        x_train_images, y_train, args.train_limit, args.random_state
    )
    x_test_images, y_test = stratified_limit(
        x_test_images, y_test, args.test_limit, args.random_state
    )
    print(
        "实际使用样本："
        f"训练集={x_train_images.shape[0]}，测试集={x_test_images.shape[0]}"
    )

    x_train = flatten_and_scale(x_train_images)
    x_test = flatten_and_scale(x_test_images)

    plot_sample_grid(
        x_train_images,
        y_train,
        out_dir / "sample_images.png",
        args.random_state,
    )
    plot_class_distribution(y_train, y_test, out_dir / "class_distribution.png")
    if not args.skip_pca:
        plot_pca_scatter(
            x_train,
            y_train,
            out_dir / "pca_scatter.png",
            args.pca_samples,
            args.random_state,
        )

    models = build_models(args.random_state)
    summary_df = train_and_evaluate(
        models,
        x_train,
        y_train,
        x_test,
        y_test,
        x_test_images,
        out_dir,
    )
    plot_model_comparison(summary_df, out_dir / "model_comparison.png")

    print("\n最终模型对比：")
    final_df = summary_df[
        [
            "model",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "total_seconds",
        ]
    ].rename(
        columns={
            "model": "模型",
            "accuracy": "准确率",
            "macro_f1": "宏平均F1",
            "weighted_f1": "加权平均F1",
            "total_seconds": "总耗时（秒）",
        }
    )
    final_df["准确率"] = final_df["准确率"].map(lambda value: f"{value * 100:.2f}%")
    final_df["宏平均F1"] = final_df["宏平均F1"].map(lambda value: f"{value * 100:.2f}%")
    final_df["加权平均F1"] = final_df["加权平均F1"].map(lambda value: f"{value * 100:.2f}%")
    final_df["总耗时（秒）"] = final_df["总耗时（秒）"].map(lambda value: f"{value:.2f}")
    print(final_df.to_string(index=False))
    print(f"\n所有结果已保存到：{out_dir}")


if __name__ == "__main__":
    main()
