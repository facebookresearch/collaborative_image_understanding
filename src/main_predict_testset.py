import logging
import os
import os.path as osp
import time
from glob import glob

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from hydra.utils import get_original_cwd, to_absolute_path
from dataset_utils import get_datasets
from lit_utils import LitModel
logger = logging.getLogger(__name__)


def load_cfg_file(base_dir: str):
    cfg_path = osp.join(base_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def load_train_model(base_dir: str):
    model_path = glob(osp.join(base_dir, "epoch*.ckpt"))[0]
    haparam_path = glob(osp.join(base_dir, "default", "version_0", "hparams.yaml"))[0]
    model = LitModel.load_from_checkpoint(model_path, hparams_file=haparam_path)
    model.eval()
    return model


@hydra.main(
    config_path="../configs",
    config_name="predict_testset",
)
def predict_testset(cfg: DictConfig):
    os.chdir(get_original_cwd())

    dir_path = cfg.dir_path

    resource_dict = OmegaConf.load(dir_path)
    logger.info(resource_dict)

    resource_dict = {
        key: osp.join(resource_dict["base_path"], value)
        for key, value in resource_dict.items()
        if key != "base_path"
    }

    cfg_file = load_cfg_file(resource_dict["label_ratio_1.0_no_cf"])
    cfg_file.batch_size = cfg.batch_size
    cfg_file.batch_size = cfg.num_workers
    out_dir = "."

    # Load data
    _, test_dataset, dataset_meta, _ = get_datasets(
        cfg_file.train_df_path,
        cfg_file.test_df_path,
        cfg_file.cf_vector_df_path,
        out_dir,
        cfg_file.labeled_ratio,
        cfg_file.is_use_bias,
        cf_based_train_loss_path=cfg_file.cf_based_train_loss_path,
        cf_based_test_loss_path=cfg_file.cf_based_test_loss_path,
        is_use_cf_embeddings=cfg_file.is_use_cf_embeddings,
        cf_embeddings_train_path=cfg_file.cf_embeddings_train_path,
        cf_embeddings_test_path=cfg_file.cf_embeddings_test_path,
        confidence_type=cfg_file.confidence_type,
        is_plot_conf_hist=False,
    )
    logger.info(
        "Sizes [trainset testset num_classes cf_vector_dim]=[{} {} {} {}]".format(
            dataset_meta["train_set_size"],
            dataset_meta["test_set_size"],
            dataset_meta["num_classes"],
            dataset_meta["cf_vector_dim"],
        )
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_grad_enabled(False)

    t0 = time.time()
    for key, base_dir in resource_dict.items():
        logger.info(f'{key}, {base_dir}')
        preds_path = osp.join(base_dir, "preds.npy")
        if osp.exists(preds_path):
            continue

        model = load_train_model(base_dir)
        model = model.to("cuda")
        model.half()

        preds_list, labels = [], []
        for batch in tqdm(testloader):
            imgs, labels_i = batch[0], batch[3]
            preds, _ = model(imgs.to("cuda").half())
            preds_list.append(torch.sigmoid(preds).cpu().numpy())

            labels.append(labels_i.numpy())

        preds = np.vstack(preds_list)
        labels = np.vstack(labels)

        np.save(preds_path, preds)
        np.save(osp.join(base_dir, "labels.npy"), labels)
        logger.info(f'Finish in {time.time()-t0:.2f}. {preds_path}')

if __name__ == "__main__":
    predict_testset()
