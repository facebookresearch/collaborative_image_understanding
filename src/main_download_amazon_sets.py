import logging
import os
import os.path as osp
import time
from typing import Tuple

import hydra
import pandas as pd
from tqdm import tqdm

from data_utils import DataHelper

logger = logging.getLogger(__name__)


def convert_to_dataframe(
    json_path: str, pkl_path: str, data_downloader: DataHelper
) -> pd.DataFrame:
    if not data_downloader.is_exist(pkl_path):
        logger.info(f"convert_to_dataframe: not exist {pkl_path=}")
        data_downloader.save_df_as_pkl(json_path, pkl_path)
    df = data_downloader.read_pickle(pkl_path)
    return df


def download_categroy(
    url_reviews: str, url_meta: str, dst_dir: str, data_downloader: DataHelper
) -> Tuple[str, str]:

    dst_review, dst_meta = [
        osp.join(dst_dir, osp.basename(url)).replace(".gz", "")
        for url in (url_reviews, url_meta)
    ]

    for url, dst in [(url_reviews, dst_review), (url_meta, dst_meta)]:
        logger.info(f"download_categroy: {url=}")
        data_downloader.download_url(url, f"{dst}.gz")
        data_downloader.ungzip_file(path_src=f"{dst}.gz", path_dst=dst)
    return dst_review, dst_meta


def get_download_url(base_url: str, category: str) -> Tuple[str, str]:
    url_reviews = f"{base_url}/reviews_{category}_5.json.gz"
    url_meta = f"{base_url}/meta_{category}.json.gz"
    return url_reviews, url_meta


def filter_meta_df(df_meta: pd.DataFrame, df_review: pd.DataFrame):
    # Keep only meta with reviews
    meta_df_filterd = df_meta[df_meta["asin"].isin(df_review["asin"].unique())]

    # Keep only items with image
    no_img = meta_df_filterd["imUrl"].str.contains("no-img-sm").astype(bool)
    meta_df_filterd = meta_df_filterd[~no_img]
    meta_df_filterd = meta_df_filterd.dropna(subset=["imUrl"])
    return meta_df_filterd


@hydra.main(
    config_path="../configs",
    config_name="download_amazon_sets",
)
def download_data(cfg):
    logger.info(cfg)
    logger.info(f"{os.getcwd()=}")

    data_helper = DataHelper(is_debug=cfg.is_debug, is_override=cfg.is_override)

    dst_dir = cfg.data_dir
    categories = cfg.category_list
    for i, category in enumerate(categories):
        t0 = time.time()

        # Dowload category data
        t1 = time.time()
        url_reviews, url_meta = get_download_url(cfg.base_url, category)
        dst_review, dst_meta = download_categroy(
            url_reviews, url_meta, dst_dir, data_helper
        )
        logger.info(f"Download category data in {time.time()-t1:.2f} sec")

        # Parse json
        t1 = time.time()
        df_review = convert_to_dataframe(
            dst_review, osp.join(dst_dir, f"reviews_{category}.pkl"), data_helper
        )
        df_meta = convert_to_dataframe(
            dst_meta, osp.join(dst_dir, f"meta_{category}.pkl"), data_helper
        )
        logger.info(f"Load df in {time.time()-t1:.2f} sec")

        # Create output directory for images
        categrory_dir = osp.join(dst_dir, category)
        data_helper.create_dir(categrory_dir)
        logger.info(f"{categrory_dir=}")

        # Filter download images
        meta_df_filt = filter_meta_df(df_meta, df_review)
        img_urls = meta_df_filt.drop_duplicates(subset="imUrl", keep="first")["imUrl"]

        # Download
        logger.info(f"{len(img_urls)=}")

        t1 = time.time()
        dst_exsits = data_helper.list_files_in_dir(categrory_dir)
        dst_imgs = [
            osp.join(categrory_dir, osp.basename(img_url)) for img_url in tqdm(img_urls)
        ]
        logger.info(f"{len(dst_exsits)=} in {time.time()-t1:.2f} sec")

        # Filter lists for exists
        dst_imgs_filt, img_urls_filt = [], []
        for dst, url in zip(dst_imgs, img_urls):
            if osp.basename(dst) not in dst_exsits:
                img_urls_filt.append(url)
                dst_imgs_filt.append(dst)
        assert len(img_urls_filt) == len(dst_imgs_filt)
        logger.info(f"Post filtering. {len(img_urls_filt)=}")

        for img_url, dst_img in tqdm(
            zip(img_urls_filt, dst_imgs_filt), total=len(img_urls_filt)
        ):
            data_helper.download_url(img_url, dst_img)
        logger.info(
            f"[{i}/{len(categories)-1}] Finish {category} in {time.time() -t0:.2f} sec"
        )


if __name__ == "__main__":
    download_data()
