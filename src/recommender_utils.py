import logging
import os.path as osp
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cornac.data import Reader
from cornac.models import VAECF
from cornac.models.recommender import Recommender
from cornac.models.vaecf.vaecf import VAE, learn
from tqdm.auto import trange

logger = logging.getLogger(__name__)


def learn(
    vae,
    train_set,
    n_epochs,
    batch_size,
    learn_rate,
    lr_steps,
    beta,
    verbose,
    out_dir: str,
    device=torch.device("cpu"),
):
    loss_list, lr_list = [], []
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learn_rate)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps)

    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    for _ in progress_bar:
        sum_loss = 0.0
        count = 0
        for batch_id, u_ids in enumerate(
            train_set.user_iter(batch_size, shuffle=False)
        ):
            u_batch = train_set.matrix[u_ids, :]
            u_batch.data = np.ones(len(u_batch.data))  # Binarize data
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=torch.float32, device=device)

            # Reconstructed batch
            u_batch_, mu, logvar = vae(u_batch)

            loss = vae.loss(u_batch, u_batch_, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))
        schedular.step()
        loss_list.append(sum_loss / count)
        lr_list += schedular.get_last_lr()

    _, axs = plt.subplots(2, 1, sharex=True)
    ax = axs[0]
    ax.plot(loss_list)
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid()
    ax = axs[1]
    ax.plot(lr_list)
    ax.set_ylabel("lr")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.grid()
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, "loss.jpg"))
    plt.close()
    return vae


class VAEWithBias(VAE):
    def __init__(self, z_dim, ae_structure, act_fn, likelihood):
        logger.info("VAEWithBias")
        super().__init__(z_dim, ae_structure, act_fn, likelihood)

        # Add bias
        num_items = ae_structure[0]
        self.item_bias = torch.nn.Embedding(num_items, 1)

    def decode(self, z):
        h = self.decoder(z)
        if self.likelihood == "mult":
            return torch.softmax(h + self.item_bias.weight.T, dim=1)
        else:
            raise NotImplementedError()
            return torch.sigmoid(h)


class VAECFWithBias(VAECF):
    def __init__(
        self,
        name="VAECF",
        k=10,
        autoencoder_structure=[20],
        act_fn="tanh",
        likelihood="mult",
        n_epochs=100,
        batch_size=100,
        learning_rate=0.001,
        lr_steps=[10],
        out_dir=".",
        beta=1.0,
        trainable=True,
        verbose=False,
        seed=None,
        use_gpu=False,
    ):
        super().__init__(
            name=name,
            k=k,
            autoencoder_structure=autoencoder_structure,
            act_fn=act_fn,
            likelihood=likelihood,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            beta=beta,
            trainable=trainable,
            verbose=verbose,
            seed=seed,
            use_gpu=use_gpu,
        )
        self.lr_steps = lr_steps
        self.out_dir = out_dir

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if self.trainable:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "vae"):
                data_dim = train_set.matrix.shape[1]
                self.vae = VAEWithBias(
                    self.k,
                    [data_dim] + self.autoencoder_structure,
                    self.act_fn,
                    self.likelihood,
                ).to(self.device)

            learn(
                self.vae,
                self.train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learn_rate=self.learning_rate,
                lr_steps=self.lr_steps,
                beta=self.beta,
                verbose=self.verbose,
                device=self.device,
                out_dir=self.out_dir,
            )

        elif self.verbose:
            logger.info("%s is trained already (trainable = False)" % (self.name))

        return self


class RecommendationDataset:
    def __init__(
        self,
        data_dir: str,
        category: str = "Clothing_Shoes_and_Jewelry",
        user_based: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.category = category

        self.review_path = osp.join(self.data_dir, f"reviews_{category}.pkl")
        self.rating_path = osp.join(self.data_dir, f"rating_{category}_user_based.txt")
        if not osp.exists(self.rating_path):
            self.convert_review_pkl_to_rating()

    def convert_review_pkl_to_rating(self):
        review_df = pd.read_pickle(
            osp.join(self.data_dir, f"reviews_{self.category}.pkl")
        )
        # Algin to rating.txt format
        review_df = review_df[["reviewerID", "asin", "overall"]]
        review_df.to_csv(self.rating_path, sep="\t", index=False, header=False)

    def load_feedback(self, reader: Reader = None) -> List:
        reader = Reader(bin_threshold=1.0) if reader is None else reader
        return reader.read(self.rating_path, sep="\t")
