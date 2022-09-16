import logging
import os
import os.path as osp
import time
from itertools import chain
import imghdr

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.datasets.folder import default_loader as img_loader
from tqdm import tqdm

logger = logging.getLogger(__name__)


label_ignore_list = ["International"]


def merge_label_hierarchy(label_list: list, category: str) -> list:
    # label_list: list of list.

    label_processed_list = []
    for label in label_list:

        # Keep only labels with that belong to the top level category
        if label[0] == category:

            # Remove labels with one letter
            label = [label_i for label_i in label if len(label_i) > 1]
            label_joined = [" + ".join(label[:i]) for i in range(1, len(label))]
            if any(
                [label_ignore in label_joined for label_ignore in label_ignore_list]
            ):
                continue

            # Join hierarchy
            label_processed_list += label_joined
    label_processed_list = np.unique(label_processed_list)
    return label_processed_list


def remove_downlevel_hierarchy(labels: list, label_mapper: dict) -> list:
    outs = []
    for label in labels:
        if label in label_mapper.keys():
            outs.append(label_mapper[label])
        else:
            outs.append(label)
    return outs


def keep_only_exists_images(df: pd.DataFrame) -> pd.DataFrame:
    df["img_exists"] = df["img_path"].apply(
        lambda img_path: imghdr.what(img_path) is not None
        if osp.exists(img_path)
        else False
    )
    logger.info(f"Img exsists {df.img_exists.sum()}/{len(df)}")
    return df[df["img_exists"] == True]


def keep_categories_with_min_samples(
    df_input: pd.DataFrame, num_samples_threshold: int
):
    df = df_input.copy()

    min_num_samples = 0
    n_iter = 0
    while min_num_samples <= num_samples_threshold:
        label_count = pd.value_counts(
            list(chain.from_iterable(df["merged_labels"].tolist()))
        )
        min_num_samples = label_count.min()

        logger.info(
            f"[{n_iter}] {len(label_count)=} {(label_count<num_samples_threshold).sum()=}"
        )
        logger.info(f"\n{label_count[label_count<num_samples_threshold]}")

        label_mapper = {
            label: " + ".join(label.split(" + ")[:-1])
            for label in label_count[label_count <= num_samples_threshold].index
        }

        df["merged_labels"] = df["merged_labels"].apply(
            lambda labels: remove_downlevel_hierarchy(labels, label_mapper=label_mapper)
        )

        # Keep unique categories
        df["merged_labels"] = df["merged_labels"].apply(
            lambda labels: np.unique(labels).tolist()
        )

        n_iter += 1
    return df, label_count


def execute_train_test_split(
    df: pd.DataFrame, train_set_ratio: float, min_label_count_thresh: int
):
    n_iter = 0
    min_label_count = 0
    while min_label_count < min_label_count_thresh:
        df_train, df_test = train_test_split(df, train_size=train_set_ratio)

        train_labels = np.array(df_train.label_vec.to_list())
        test_labels = np.array(df_test.label_vec.to_list())
        min_label_count = min(
            train_labels.sum(axis=0).min(), test_labels.sum(axis=0).min()
        )
        logger.info(f"[{n_iter}] train-test split {min_label_count=}")
        n_iter += 1
    return df_train, df_test


@hydra.main(
    config_path="../configs",
    config_name="process_labels_amazon",
)
def process_labels(cfg: DictConfig):
    out_dir = os.getcwd()
    logger.info(cfg)
    logger.info(os.getcwd())

    # Load df
    t0 = time.time()
    meta_path = osp.join(cfg.data_dir, f"meta_{cfg.category}.pkl")
    meta_df = pd.read_pickle(meta_path)
    logger.info(f"Loadded meta_df in {time.time() -t0:.2f} sec. {len(meta_df)=}")

    t0 = time.time()
    review_path = osp.join(cfg.data_dir, f"reviews_{cfg.category}.pkl")
    reivew_df = pd.read_pickle(review_path)
    logger.info(f"Loadded review_df in {time.time() -t0:.2f} sec. {len(reivew_df)=}")

    # Keep only items with reviews
    t0 = time.time()
    asin = reivew_df.drop_duplicates(subset=["asin"])["asin"].tolist()
    meta_df = meta_df[meta_df["asin"].isin(asin)]
    logger.info(f"Item with reviews {time.time() -t0:.2f} sec. {len(meta_df)=}")

    # Add image paths
    meta_df["img_path"] = meta_df["imUrl"].apply(
        lambda x: osp.join(cfg.data_dir, cfg.category, osp.basename(str(x)))
    )

    # Keep only items with images
    df = keep_only_exists_images(meta_df)[["asin", "img_path", "categories"]]

    # Find top level label name by most ferquent
    toplevel_all_labels = [sublist[0][0] for sublist in df["categories"].tolist()]
    toplevel_label = max(set(toplevel_all_labels), key=toplevel_all_labels.count)

    # Merge label hierarchy:
    # For example: [Clothing, Shoes & Jewelry + Girls + Clothing + Swim] -> [Clothing, Shoes & Jewelry + Girls + Clothing]
    df["merged_labels"] = df["categories"].apply(
        lambda x: merge_label_hierarchy(x, category=toplevel_label)
    )

    # Count number of samples for each category: remove downlevel category if there are not enough samples
    df, label_count = keep_categories_with_min_samples(df, cfg.num_samples_threshold)

    # Remove the top category, it is 1 for all.
    df["merged_labels"] = df["merged_labels"].apply(
        lambda labels: [label for label in labels if label != toplevel_label]
    )

    # Encode to Multilabel vector
    mlb = MultiLabelBinarizer()
    df["label_vec"] = mlb.fit_transform(df["merged_labels"].tolist()).tolist()
    logger.info(f"\n{df.head()}")

    # Save results
    out_path = osp.join(out_dir, "label_count.csv")
    label_count.to_csv(out_path, header=False)

    out_path = osp.join(out_dir, "df_w_labels.pkl")
    df = df.reset_index()
    df.to_pickle(out_path)
    logger.info(f"Save to {out_path}")

    out_path = osp.join(out_dir, "label_mapper.csv")
    pd.DataFrame(mlb.classes_).to_csv(out_path, header=False)
    logger.info(f"Save to {out_path}")

    # Train-test split
    df_train, df_test = execute_train_test_split(
        df, cfg.train_set_ratio, cfg.min_label_count
    )

    # Save train
    out_path = osp.join(out_dir, "df_train.pkl")
    df_train = df_train.reset_index()
    df_train.to_pickle(out_path)
    logger.info(f"Save to {out_path}")
    out_path = osp.join(out_dir, "..", "df_train.pkl")
    df_train.to_pickle(out_path)
    logger.info(f"Save to {out_path}")

    # Save test
    out_path = osp.join(out_dir, "df_test.pkl")
    df_test = df_test.reset_index()
    df_test.to_pickle(out_path)
    logger.info(f"Save to {out_path}")
    out_path = osp.join(out_dir, "..", "df_test.pkl")
    df_test.to_pickle(out_path)
    logger.info(f"Save to {out_path}")

    logger.info("Finish")


if __name__ == "__main__":
    process_labels()
