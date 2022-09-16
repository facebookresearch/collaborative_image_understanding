import logging
import os
import os.path as osp
import time

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from dataset_utils import get_datasets
from lit_utils import LitModelCFBased

logger = logging.getLogger(__name__)


def get_loader_loss(model_h, dataloader) -> torch.Tensor:
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_list = []
    cf_embeddings_list = []
    with torch.no_grad():
        for cf_vectors, y in dataloader:
            logits = model_h(cf_vectors)
            loss = criterion(logits, y.float())
            loss_list.append(loss.detach().cpu())

            cf_embeddings = model_h.get_embeddings(cf_vectors)
            cf_embeddings_list.append(cf_embeddings.cpu())
    loss_vec = torch.vstack(loss_list)
    cf_embeddings_matrix = torch.vstack(cf_embeddings_list)
    return loss_vec, cf_embeddings_matrix


@hydra.main(
    config_path="../configs",
    config_name="train_model_cf_based",
)
def train_model_cf_based(cfg: DictConfig):
    t_start = time.time()
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(get_original_cwd())
    logger.info(f"{out_dir=}")
    pl.utilities.seed.seed_everything(cfg.seed)
    logger.info(f"{torch.cuda.is_available()=}")

    # Configure logging
    tb_logger = pl_loggers.TensorBoardLogger(out_dir)
    tb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Configure checkpoint saver
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        monitor="ap/val" if cfg.is_debug is False else "ap/train",
        save_top_k=1,
        mode="max",
    )

    # Load data
    t0 = time.time()
    train_dataset, test_dataset, dataset_meta, pos_weight = get_datasets(
        cfg.train_df_path,
        cfg.test_df_path,
        cfg.cf_vector_df_path,
        out_dir,
        cfg.labeled_ratio,
        cfg.is_use_bias,
        is_skip_img=True,
    )

    logger.info(f"Loadded data in {time.time() -t0 :.2f} sec")
    logger.info(
        "Sizes [trainset testset num_classes]=[{} {} {}]".format(
            dataset_meta["train_set_size"],
            dataset_meta["test_set_size"],
            dataset_meta["num_classes"],
        )
    )

    # Create dataloder
    t0 = time.time()
    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Load model
    lit_h = LitModelCFBased(
        dataset_meta["num_classes"], dataset_meta["cf_vector_dim"], cfg, pos_weight
    )
    trainer = pl.Trainer(
        min_epochs=cfg["epochs"],
        max_epochs=cfg["epochs"],
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        fast_dev_run=cfg.is_debug,
        num_sanity_val_steps=0,
        gpus=[cfg.gpu] if torch.cuda.is_available() else None,
    )
    trainer.fit(lit_h, trainloader, testloader)
    logger.info(
        f"Finish training in {time.time() -t_start :.2f} sec. {lit_h.map_best=}"
    )
    logger.info(f"{os.getcwd()=}")

    # Save confidence
    t0 = time.time()
    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    # Save products
    for (loader, set_name) in [(trainloader, "train"), (testloader, "test")]:
        loss_vec, cf_embeddings = get_loader_loss(lit_h, loader)
        out_path = osp.join(out_dir, f"cf_based_{set_name}_loss.pt")
        torch.save(loss_vec, out_path)

        if cfg.save_as_asset is True:
            logger.info(f'{cfg.save_as_asset=}')
            out_path = osp.join(out_dir, "..", f"cf_based_{set_name}_loss.pt")
            torch.save(loss_vec, out_path)

        out_path = osp.join(out_dir, f"cf_embeddings_{set_name}.pt")
        torch.save(cf_embeddings, out_path)

        if cfg.save_as_asset is True:
            logger.info(f'{cfg.save_as_asset=}')
            out_path = osp.join(out_dir, "..", f"cf_embeddings_{set_name}.pt")
            torch.save(cf_embeddings, out_path)

        logger.info(
            f"Finish get_loader_loss in {time.time() -t0 :.2f} sec. {cf_embeddings.shape} {out_path=}"
        )

        plt.hist(loss_vec.mean(axis=1).numpy(), bins=1000)
        plt.xlabel("Loss")
        plt.ylabel("Count")
        plt.savefig(osp.join(out_dir, f"{set_name}_loss_vec_hist.jpg"))
        plt.close()


if __name__ == "__main__":
    train_model_cf_based()
