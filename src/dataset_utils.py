import logging
import os.path as osp
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

logger = logging.getLogger(__name__)


class DfDatasetWithCF(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        target_transform=None,
        is_use_bias: bool = False,
        is_skip_img: bool = False,
    ) -> None:
        super().__init__()
        self.df = df
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform
        self.is_use_bias = is_use_bias
        self.is_skip_img = is_skip_img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row["img_path"]
        pos_img_path = row["pos_img_path"] if "pos_img_path" in row else row["img_path"]
        cf_vector = torch.tensor(row["embs"])
        target = torch.tensor(row["label_vec"])
        is_labeled = row["is_labeled"]
        cf_bias = torch.tensor(row["bias"])
        cf_confidence = torch.tensor(row["cf_confidence"])

        if self.is_use_bias is True:
            cf_vector = torch.hstack((cf_vector, cf_bias)).float()

        if self.is_skip_img is True:
            return cf_vector, target

        # Load image
        image = self.loader(img_path)
        image_pos = self.loader(pos_img_path)

        if self.transform is not None:
            image = self.transform(image)
            image_pos = self.transform(image_pos)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (
            image,
            image_pos,
            cf_vector,
            target,
            is_labeled,
            cf_confidence,
        )


# Transformation as ImageNet training
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize,
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)


def get_loss_based_confidence(cf_based_loss_path: str):
    assert cf_based_loss_path is not None
    cf_based_loss = torch.load(cf_based_loss_path)
    loss_mean = cf_based_loss.mean(axis=1)
    cf_confidence = 1 / loss_mean
    return cf_confidence


def pos_label_loss_based(
    cf_based_loss_path: str, label_vecs: pd.Series
) -> torch.tensor:
    assert cf_based_loss_path is not None
    cf_based_loss = torch.load(cf_based_loss_path)
    label_vecs = torch.tensor(label_vecs.tolist()).bool()

    # Get only loss of GT labels
    loss_mean = torch.tensor(
        [values[mask].mean() for values, mask in zip(cf_based_loss, label_vecs)]
    )
    cf_confidence = 1 / loss_mean

    # For samples without positive labels: set the confidence to 0
    cf_confidence[torch.isnan(cf_confidence)] = 0.0

    return cf_confidence


def clip_confidence(
    cf_conf_train: torch.Tensor, cf_conf_test: torch.Tensor, max_min_ratio: float
) -> np.ndarray:
    lower_limit, upper_limit = torch.quantile(cf_conf_train, torch.tensor([0.25, 0.75]))
    if max_min_ratio is None or lower_limit == upper_limit:
        return cf_conf_train, cf_conf_test

    cf_conf_train_clipped = torch.clip(cf_conf_train, lower_limit, upper_limit)
    cf_conf_test_clipped = torch.clip(cf_conf_test, lower_limit, upper_limit)

    # Normalize to keep min-max value between 1 and max_min_ratio
    min_val, max_val = cf_conf_train_clipped.min(), cf_conf_train_clipped.max()
    cf_conf_train_clipped = 1 + (cf_conf_train_clipped - min_val) * (
        max_min_ratio - 1
    ) / (max_val - min_val)
    cf_conf_test_clipped = 1 + (cf_conf_test_clipped - min_val) * (
        max_min_ratio - 1
    ) / (max_val - min_val)
    # Log
    min_val, max_val = cf_conf_train_clipped.min(), cf_conf_train_clipped.max()
    logger.info(
        f"clip_confidence: {max_min_ratio=} [min max]={min_val:.2f} {max_val:.2f}"
    )
    return cf_conf_train_clipped, cf_conf_test_clipped


def assign_positive_cf(df_train, df_test):
    df_all_set = pd.concat(
        (df_train[["asin", "img_path"]], df_test[["asin", "img_path"]])
    )

    pos_img_path = pd.merge(
        df_train[["asin", "pos_asin"]],
        df_all_set,
        left_on=["pos_asin"],
        right_on=["asin"],
        how="left",
    )["img_path"]
    df_train["pos_img_path"] = pos_img_path

    pos_img_path = pd.merge(
        df_test[["asin", "pos_asin"]],
        df_all_set,
        left_on=["pos_asin"],
        right_on=["asin"],
        how="left",
    )["img_path"]
    df_test["pos_img_path"] = pos_img_path
    return df_train, df_test


def plot_and_save_conf_histogram(
    out_dir: str,
    confidence_type: str,
    cf_conf: np.ndarray,
    cf_conf_clipped: np.ndarray,
):
    mask = np.isfinite(cf_conf)
    value_min, value_max = np.round(
        [np.min(cf_conf_clipped), np.max(cf_conf_clipped)], 2
    )
    _, axs = plt.subplots(2, 1, sharex=False)
    ax = axs[0]
    _, bins, _ = ax.hist(cf_conf[mask], bins=100, alpha=0.5, label="raw", color="C0")
    ax.hist(cf_conf_clipped, bins=bins, alpha=0.5, label="clipped", color="C1")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"Confidence {confidence_type=}")

    ax = axs[1]
    ax.hist(cf_conf_clipped, bins=100, alpha=0.5, label="clipped", color="C1")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Confidnce value. [min max]=[{value_min} {value_max}]")
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, "cf_confidence.jpg"))
    plt.close()


def get_datasets(
    df_train_path: str,
    df_test_path: str,
    cf_vector_df_path: str,
    out_dir: str,
    labeled_ratio: float = 1.0,
    is_use_bias: bool = False,
    is_skip_img: bool = False,
    cf_based_train_loss_path: str = None,
    cf_based_test_loss_path: str = None,
    is_use_cf_embeddings: bool = False,
    cf_embeddings_train_path: str = None,
    cf_embeddings_test_path: str = None,
    confidence_type: str = "uniform",
    conf_max_min_ratio: float = None,
    is_plot_conf_hist: bool = True,
):

    t0 = time.time()
    df_train = pd.read_pickle(df_train_path)
    df_test = pd.read_pickle(df_test_path)
    cf_df = pd.read_pickle(cf_vector_df_path)
    logger.info(
        f"Loaded df in {time.time() -t0 :.2f} sec. {len(df_train)=} {len(df_test)=} {len(cf_df)=}"
    )

    # Add CF vectors
    t0 = time.time()
    cf_df["asin"] = cf_df["asin"].astype(df_train["asin"].dtype)
    df_train = pd.merge(df_train, cf_df, on=["asin"], how="inner")
    df_test = pd.merge(df_test, cf_df, on=["asin"], how="inner")
    logger.info(
        f"merge df in {time.time() -t0 :.2f} sec. {len(df_train)=} {len(df_test)=} {len(cf_df)=}"
    )

    if is_use_cf_embeddings is True:
        t0 = time.time()
        cf_embeddings_train = torch.load(cf_embeddings_train_path)
        cf_embeddings_test = torch.load(cf_embeddings_test_path)
        df_train["embs"] = cf_embeddings_train.tolist()
        df_test["embs"] = cf_embeddings_test.tolist()
        logger.info(f"Override cf vectors in {time.time() -t0 :.2f} sec.")

    # Add positive CF
    if "pos_asin" in df_train.columns:
        df_train, df_test = assign_positive_cf(df_train, df_test)

    # Hide labels
    df_train["is_labeled"] = torch.rand(len(df_train)) > 1.0 - labeled_ratio
    df_test["is_labeled"] = True

    # Define positive weight: Since positives are much less than negatives, increase their weights
    train_labels = np.array(
        df_train[df_train["is_labeled"] == True].label_vec.to_list()
    )
    pos_weight = len(train_labels) / (train_labels.sum(axis=0) + 1e-6)

    # Apply confidence to cf vector
    t0 = time.time()
    if confidence_type == "uniform":
        cf_conf_train = torch.ones(len(df_train))
        cf_conf_test = torch.ones(len(df_test))

    elif confidence_type == "loss_based":
        cf_conf_train = get_loss_based_confidence(cf_based_train_loss_path)
        cf_conf_test = get_loss_based_confidence(cf_based_test_loss_path)

    elif confidence_type == "num_intercations":
        cf_conf_train = torch.from_numpy(np.sqrt(df_train["num_intercations"].values))
        cf_conf_test = torch.from_numpy(np.sqrt(df_test["num_intercations"].values))

    elif confidence_type == "pos_label_loss_based":
        cf_conf_train = pos_label_loss_based(
            cf_based_train_loss_path, df_train["label_vec"]
        )
        cf_conf_test = pos_label_loss_based(
            cf_based_test_loss_path, df_test["label_vec"]
        )

    else:
        raise ValueError(f"{confidence_type} is not supported")
    cf_conf_train_clipped, cf_conf_test_clipped = clip_confidence(
        cf_conf_train, cf_conf_test, conf_max_min_ratio
    )

    df_train["cf_confidence"] = cf_conf_train_clipped
    df_test["cf_confidence"] = cf_conf_test_clipped
    logger.info(f"CF confidence in {time.time() -t0 :.2f} sec.")

    if is_plot_conf_hist is True:
        plot_and_save_conf_histogram(
            out_dir,
            confidence_type,
            cf_conf_train.numpy(),
            cf_conf_train_clipped.numpy(),
        )
        logger.info(f"Plotted CF confidence in {time.time() -t0 :.2f} sec. {out_dir=}")

    # Construct dataset
    train_dataset = DfDatasetWithCF(
        df_train,
        transform=train_transform,
        is_use_bias=is_use_bias,
        is_skip_img=is_skip_img,
    )
    test_dataset = DfDatasetWithCF(
        df_test,
        transform=test_transform,
        is_use_bias=is_use_bias,
        is_skip_img=is_skip_img,
    )

    # Get metadata
    num_classes = len(df_train["label_vec"].iloc[0])
    cf_vector_dim = len(df_train["embs"].iloc[0])
    if is_use_bias is True:
        cf_vector_dim += 1

    dataset_meta = {
        "train_set_size": len(df_train),
        "test_set_size": len(df_test),
        "num_classes": num_classes,
        "cf_vector_dim": cf_vector_dim,
    }

    return train_dataset, test_dataset, dataset_meta, pos_weight
