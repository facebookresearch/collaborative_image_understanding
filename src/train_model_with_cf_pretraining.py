import logging
import os
import time
import os.path as osp
import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
from dataset_utils import get_datasets
from lit_utils import LitModel

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="train_model_with_cf_pretraining",
)
def train_model_with_cf_pretraining(cfg: DictConfig):
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
        cf_based_train_loss_path=cfg.cf_based_train_loss_path,
        cf_based_test_loss_path=cfg.cf_based_test_loss_path,
        is_use_cf_embeddings=cfg.is_use_cf_embeddings,
        cf_embeddings_train_path=cfg.cf_embeddings_train_path,
        cf_embeddings_test_path=cfg.cf_embeddings_test_path,
        confidence_type=cfg.confidence_type,
        conf_max_min_ratio=cfg.conf_max_min_ratio,
    )
    logger.info(f"Loadded data in {time.time() -t0 :.2f} sec")
    logger.info(
        "Sizes [trainset testset num_classes cf_vector_dim]=[{} {} {} {}]".format(
            dataset_meta["train_set_size"],
            dataset_meta["test_set_size"],
            dataset_meta["num_classes"],
            dataset_meta["cf_vector_dim"],
        )
    )

    # Create dataloder
    t0 = time.time()
    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # First training: predict CF vector
    cfg["cf_weight"], cfg["label_weight"] = 1.0, 0.0
    lit_h = LitModel(
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
        precision=16,
    )
    trainer.fit(lit_h, trainloader, testloader)
    logger.info(f"Finish cf training in {time.time() -t_start :.2f} sec")
    logger.info(f"{out_dir=}")
    trainer.save_checkpoint(osp.join(out_dir, "model_pretrained_cf.ckpt"))

    # Second training: predict labels
    cfg["cf_weight"], cfg["label_weight"] = 0.0, 1.0
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
        precision=16,
    )

    trainer.fit(lit_h, trainloader, testloader)
    logger.info(f"Finish label training in {time.time() -t_start :.2f} sec. {lit_h.map_best=:.3f}")
    logger.info(f"{out_dir=}")
    trainer.save_checkpoint(osp.join(out_dir, "model.ckpt"))


if __name__ == "__main__":
    train_model_with_cf_pretraining()
