import logging
import os
import os.path as osp
import time
from glob import glob

import hydra
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset_utils import get_datasets
from lit_utils import LitModel

logging.basicConfig()
logger = logging.getLogger(__name__)


def get_products(model, dataloader):
    preds_list, labels, embeddings_list = [], [], []
    for batch in tqdm(dataloader):
        imgs, labels_i = batch[0], batch[3]
        preds, _ = model(imgs.to("cuda").half())
        embeddings = model.get_embeddings(imgs.to("cuda").half())

        preds_list.append(torch.sigmoid(preds).cpu())
        embeddings_list.append(embeddings.cpu())
        labels.append(labels_i)

    preds = torch.vstack(preds_list)
    embeddings = torch.vstack(embeddings_list)
    labels = torch.vstack(labels)

    return preds, embeddings, labels


def load_cfg_file(base_dir: str):
    cfg_path = osp.join(base_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def load_trained_model(base_dir: str):
    model_path = glob(osp.join(base_dir, "epoch*.ckpt"))[0]
    haparam_path = glob(osp.join(base_dir, "default", "version_0", "hparams.yaml"))[0]
    model = LitModel.load_from_checkpoint(model_path, hparams_file=haparam_path)
    model.eval()
    return model


@hydra.main(
    config_path="../configs",
    config_name="extract_embeddings",
)
def extract_embeddings(cfg: DictConfig):
    out_dir = os.getcwd()
    os.chdir(get_original_cwd())
    logger.info(os.getcwd())

    dir_path = cfg.dir_path

    resource_dict = {"label_ratio_1.0_no_cf": dir_path}
    logger.info(resource_dict)

    logger.info("load_cfg_file")
    cfg_file = load_cfg_file(resource_dict["label_ratio_1.0_no_cf"])
    cfg_file.batch_size = cfg.batch_size
    cfg_file.batch_size = cfg.num_workers

    # Load data
    logger.info("Load datasets")
    train_dataset, test_dataset, dataset_meta, _ = get_datasets(
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

    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_grad_enabled(False)

    for key, base_dir in resource_dict.items():
        logger.info(f"{key}, {base_dir}")

        model = load_trained_model(base_dir)
        model = model.to("cuda").half()

        dfs = []
        for loader_type, loader in [
            ("test", testloader),
            ("train", trainloader),
        ]:
            t0 = time.time()

            preds, embeddings, labels = get_products(model, loader)

            ap = average_precision_score(labels, preds)
            logger.info(f"Finish {loader_type=} in {time.time()-t0:.2f}. {ap=}")

            df = loader.dataset.df
            df["pred"] = preds.tolist()
            df["image_embedding"] = embeddings.tolist()
            df["set_type"] = loader_type
            df["label_vec_dataloder_output"] = labels.tolist()

            dfs.append(df)

        # Save products
        df = pd.concat(dfs)
        df = df.rename(
            columns={"embs": "cf_vec"},
        )
        df.to_pickle(osp.join(out_dir, f"{cfg.dataset_name}_features.pkl"))
        df.to_csv(osp.join(out_dir, f"{cfg.dataset_name}_features.csv"))


if __name__ == "__main__":
    extract_embeddings()
