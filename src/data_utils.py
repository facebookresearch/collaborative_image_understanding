import logging
import os
import os.path as osp
from typing import List

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataHelper:
    def __init__(self, is_debug: bool, is_override: bool):
        self.is_debug = is_debug
        self.is_override = is_override

    def is_img_path(self, path: str) -> bool:
        if path.lower().endswith((".jpg", ".png", ".jepg", ".gif", ".tiff")):
            return True
        else:
            return False

    def is_exist(self, path: str):
        is_path_exist = osp.exists(path)
        if self.is_override:
            is_path_exist = False
        return is_path_exist

    def download_url(
        self,
        url: str,
        dst: str = None,
        is_force_download: bool = False,
    ):
        if self.is_debug:
            logger.info(f"download_url: {url=} {dst=} {is_force_download=}")

        if dst is None:
            dst = os.path.basename(url)
        if is_force_download is False and self.is_exist(dst):
            return

        r = requests.get(url)
        with open(dst, "wb") as f:
            f.write(r.content)

    def ungzip_file(self, path_src: str, path_dst: str):
        logger.info(f"ungzip_file: {path_src=} {path_dst=}")
        if self.is_exist(path_dst):
            return
        os.system(f"gzip -dk {path_src}")

    def read_pickle(self, pkl_path: str) -> pd.DataFrame:
        logger.info(f"pd.read_pickle {pkl_path}")
        df = pd.read_pickle(pkl_path)
        return df

    def save_df_as_pkl(self, json_path: str, pkl_path: str):
        logger.info(f"save_df_as_pkl: {json_path=} {pkl_path=}")

        with open(json_path, "r") as fin:
            df = {}
            for i, line in enumerate(tqdm(fin)):
                df[i] = eval(line)
            df = pd.DataFrame.from_dict(df, orient="index")
        df.to_pickle(pkl_path)

    def create_dir(self, dst: str):
        logger.info(f"create_dir {dst=}")
        os.makedirs(dst, exist_ok=True)

    def list_files_in_dir(self, path: str) -> List[str]:
        return os.listdir(path)
