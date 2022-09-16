import logging
import os
import os.path as osp
import time

import cornac
import hydra
import pandas as pd
import torch
from cornac.eval_methods import RatioSplit
from cornac.metrics import AUC, MAP
from omegaconf import DictConfig

from recommender_utils import RecommendationDataset, VAECFWithBias

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="train_recommender",
)
def train_recommender(cfg: DictConfig):
    out_dir = os.getcwd()
    logger.info(cfg)
    logger.info(os.getcwd())

    # Initalize dataset
    t1 = time.time()
    dataset_h = RecommendationDataset(cfg.data_dir, cfg.category, cfg.user_based)
    dataset = dataset_h.load_feedback()
    rs = RatioSplit(
        data=dataset,
        test_size=cfg.test_size,
        rating_threshold=1.0,
        seed=cfg.seed,
        exclude_unknowns=True,
        verbose=True,
    )
    logger.info(f"Loaded dataset in {time.time()-t1:.2f}")

    # Initalize model
    models = []
    if "most_pop" in cfg.models:
        model = cornac.models.MostPop()
        models.append(model)
    if "bpr" in cfg.models:
        bpr = cornac.models.BPR(
            k=10, max_iter=1000, learning_rate=0.001, lambda_reg=0.001, seed=123
        )
        models.append(bpr)
    if "vae_no_bias" in cfg.models:
        model = cornac.models.VAECF(
            k=cfg.bottleneck_size,
            autoencoder_structure=list(cfg.emb_size),
            act_fn="tanh",
            likelihood="mult",
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            beta=cfg.beta,
            seed=cfg.seed,
            use_gpu=True,
            verbose=True,
        )
        models.append(model)

    if "vae_no_bias" in cfg.models:
        vaecf = VAECFWithBias(
            k=cfg.bottleneck_size,
            autoencoder_structure=list(cfg.emb_size),
            act_fn="tanh",
            likelihood="mult",
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            lr_steps=cfg.lr_steps,
            beta=cfg.beta,
            seed=cfg.seed,
            use_gpu=True,
            verbose=True,
            out_dir=out_dir,
        )
        models.append(vaecf)

    # Run training
    t0 = time.time()
    metrics = [AUC(), MAP()]
    cornac.Experiment(
        eval_method=rs,
        models=models,
        metrics=metrics,
        user_based=False,
    ).run()

    logger.info(f"Finish training in {time.time() -t0:.2f} sec")

    if "bpr" in cfg.models:
        logger.info(bpr)
        
        embs = bpr.i_factors
        bias = bpr.i_biases
        
    if "vae_no_bias" in cfg.models:
        logger.info(vaecf.vae)

        # Save vae model
        out_path = osp.join(out_dir, "vae.pt")
        torch.save(vaecf.vae.state_dict(), out_path)

        embs = vaecf.vae.decoder.fc1.weight.detach().cpu()
        bias = vaecf.vae.item_bias.weight.detach().cpu().squeeze()

    # Create CF data frame
    num_intercations = rs.train_set.csc_matrix.sum(axis=0).tolist()[0]
    df = pd.DataFrame(
        {
            "asin": list(rs.train_set.item_ids),
            "embs": embs.tolist(),
            "bias": bias.tolist(),
            "num_intercations": num_intercations,
        }
    )
    # Save to: out path
    out_path = osp.join(out_dir, "cf_df.pkl")
    logger.info(out_path)
    df.to_pickle(out_path)

    if cfg.test_size == 0.0:
        # Save to: dataset output top dir
        out_path = osp.join(out_dir, "..", "cf_df.pkl")
        logger.info(out_path)
        df.to_pickle(out_path)
        logger.info(f"Finish in {time.time()-t0:.2f} sec")


if __name__ == "__main__":
    train_recommender()
