import logging
import os
import os.path as osp
import time
from itertools import chain

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.datasets.folder import default_loader as img_loader

logger = logging.getLogger(__name__)


def keep_only_exists_images(df: pd.DataFrame) -> pd.DataFrame:
    img_exists = []
    for img_path in df["img_path"]:
        if osp.exists(img_path):
            try:
                img_loader(img_path)
                img_exists.append(True)
            except:
                img_exists.append(False)
        else:
            img_exists.append(False)
    df["img_exists"] = img_exists
    logger.info(f"Img exsists {df.img_exists.sum()}/{len(df)}")
    return df[df["img_exists"] == True]


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
    config_name="process_labels_movielens",
)
def process_labels_movielens(cfg: DictConfig):
    out_dir = os.getcwd()
    logger.info(cfg)
    logger.info(os.getcwd())

    # Load df
    t0 = time.time()
    meta_path = osp.join(cfg.data_dir, f"movies.dat")
    meta_df = pd.read_csv(
        meta_path, delimiter="::", names=["asin", "movie_name", "categories"]
    )
    logger.info(f"Loadded meta_df in {time.time() -t0:.2f} sec. {len(meta_df)=}")

    # Add image paths
    meta_df["img_path"] = meta_df["asin"].apply(
        lambda x: osp.join(cfg.data_dir, cfg.category, str(x) + ".jpg")
    )

    # Keep only items with images
    df = keep_only_exists_images(meta_df)[["asin", "img_path", "categories"]]

    # Find top level label name by most ferquent
    df = df[df["categories"] != "(no genres listed)"]
    df["merged_labels"] = df["categories"].apply(lambda cat_list: cat_list.split("|"))

    # Count number of samples for each category: remove downlevel category if there are not enough samples
    label_count = pd.value_counts(
        list(chain.from_iterable(df["merged_labels"].tolist()))
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
    process_labels_movielens()
