import logging
import os
import os.path as osp
import time

import hydra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from main_process_labels import keep_only_exists_images
from data_utils import DataHelper

logger = logging.getLogger(__name__)


def create_one_hot_vector(df: pd.DataFrame) -> list:
    assert "category" in df.columns  # the int label

    int_labels = df["category"].to_numpy()
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = int_labels.reshape(len(int_labels), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    label_vec = onehot_encoded.tolist()
    return label_vec


def download_data_to_folder(df: pd.DataFrame, data_helper, img_out_dir: str) -> list:
    assert "img_path" in df.columns  # the dst path
    assert "url" in df.columns  # The image url to download from

    # Download data
    dst_exsits = data_helper.list_files_in_dir(img_out_dir)
    to_download_df = df[~df["img_path"].apply(lambda x: osp.basename(x)).isin(dst_exsits)]
    logger.info(f"Post filtering. [Before After]=[{len(df)} {len(to_download_df)}]")
    num_failed = 0
    for _, row in tqdm(to_download_df.iterrows(), total=len(to_download_df)):
        row["url"], row["img_path"]
        try:
            data_helper.download_url(row["url"], row["img_path"])
        except:
            num_failed += 1
    logger.info(f"{len(df)=} {num_failed=}")


def save_train_test_split(df: pd.DataFrame, out_dir: str, train_set_ratio: float):
    df_train, df_test = train_test_split(df, train_size=train_set_ratio)
    out_path = osp.join(out_dir, "df_train.pkl")
    df_train.to_pickle(out_path)
    logger.info(f"df_train to {out_path=}")
    out_path = osp.join(out_dir, "df_test.pkl")
    df_test.to_pickle(out_path)
    logger.info(f"df_test to {out_path=}")


@hydra.main(
    config_path="../configs",
    config_name="download_pinterest_sets",
)
def download_pinterest_data(cfg):
    t0 = time.time()

    logger.info(cfg)
    logger.info(f"{os.getcwd()=}")
    img_out_dir = osp.abspath(osp.join(cfg.data_dir, cfg.category))
    df_out_dir = osp.join(os.getcwd(), "..")

    data_helper = DataHelper(is_debug=cfg.is_debug, is_override=cfg.is_override)
    data_helper.create_dir(cfg.data_dir)
    data_helper.create_dir(img_out_dir)

    # Load url df
    t1 = time.time()
    url_df = pd.read_csv(
        osp.join(cfg.data_dir, cfg.url_file),
        delimiter="|",
        names=["pin_id", "url"],
    )
    repin_df = pd.read_csv(
        osp.join(cfg.data_dir, cfg.repin_file),
        delimiter="|",
        names=["pin_id", "user_id", "category", "board_id"],
    )
    label_name_df = pd.read_csv(
        osp.join(cfg.data_dir, cfg.label_name_file),
        delimiter="|",
        names=["name", "label_id"],
    )
    logger.info(
        f"Loaded dfs. {time.time()-t1:.2f}[s]. {[len(url_df),len(repin_df), len(label_name_df)]=}"
    )

    # Filter by number of intercations
    interaction_count = pd.value_counts(repin_df["pin_id"])
    pin_id_to_keep = interaction_count[
        interaction_count > cfg.num_interactions_min
    ].index.to_numpy()
    df = pd.merge(repin_df, url_df, on=["pin_id"], how="inner")
    repin_df = repin_df[repin_df["pin_id"].isin(pin_id_to_keep)]
    url_df = url_df[url_df["pin_id"].isin(pin_id_to_keep)]
    logger.info(f"Filtered {time.time()-t1:.2f}[s]. {[len(url_df),len(repin_df)]=}")

     # Download data
    t1 = time.time()
    len_init = len(url_df)
    url_df["img_path"] = url_df["pin_id"].apply(
        lambda x: osp.abspath(osp.join(img_out_dir, str(x) + ".jpg"))
    )
    download_data_to_folder(url_df, data_helper, img_out_dir)
    url_df = keep_only_exists_images(url_df)
    logger.info(
        f"Downloaded files in {time.time()-t1:.2f}[s]. {len(url_df)}/{len_init}"
    )

    # Create unified df: use intersection of pin_id
    df = pd.merge(repin_df, url_df, on=["pin_id"], how="inner")

    # Recommender
    rating_path = osp.join(cfg.data_dir, f"rating_{cfg.category}_user_based.txt")
    recommender_df = df[["user_id", "pin_id"]]
    recommender_df["rating"] = 5.0  # Align with other datasets
    recommender_df.to_csv(rating_path, sep="\t", index=False, header=False)

    # Prepare df to vision training
    df_to_vision = df[["pin_id", "img_path", "category"]].drop_duplicates(
        subset="pin_id"
    )
    df_to_vision = df_to_vision.rename(columns={"pin_id": "asin"})
    df_to_vision["label_vec"] = create_one_hot_vector(df_to_vision)

    # Train-test split
    save_train_test_split(df_to_vision, df_out_dir, cfg.train_set_ratio)
    logger.info(f"Finish {cfg.category} in {time.time() -t0:.2f} [s]")


if __name__ == "__main__":
    download_pinterest_data()
