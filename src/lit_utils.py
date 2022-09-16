import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import os.path as osp
from architecture_utils import get_backbone, get_cf_predictor, get_classifier

logger = logging.getLogger(__name__)


class LitModel(pl.LightningModule):
    def __init__(
        self,
        num_target_classes: int,
        cf_vector_dim: int,
        cfg,
        pos_weight=None,
        out_dir:str ='.'
    ):
        super().__init__()
        self.cfg = cfg
        self.num_target_classes = num_target_classes
        self.save_hyperparameters()
        self.map_best = 0

        pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        # Define the architecture
        self.backbone, out_feature_num = get_backbone(cfg["is_pretrained"], cfg["arch"])
        self.classifier = get_classifier(out_feature_num, num_target_classes)
        self.cf_layers = get_cf_predictor(out_feature_num, cf_vector_dim)

        # Save best prediction paths
        self.preds_path = osp.join(out_dir, "preds.npy")
        self.labels_path = osp.join(out_dir, "labels.npy")

    def criterion_cf(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cf_topk_loss_ratio: float,
        cf_confidence: torch.tensor,
    ):
        # Dim is as the len of the batch. the mean is for the cf vector dimentsion (64).
        loss_reduction_none = nn.functional.mse_loss(
            pred, target, reduction="none"
        ).mean(dim=-1) + nn.functional.l1_loss(pred, target, reduction="none").mean(
            dim=-1
        )
        loss_reduction_none = torch.exp(loss_reduction_none) - 1.0
        loss_reduction_none = (
            cf_confidence * loss_reduction_none / (cf_confidence.sum())
        )

        # Take item_num with lowest loss
        if cf_topk_loss_ratio < 1.0:
            num_cf_items = int(cf_topk_loss_ratio * len(loss_reduction_none))
            loss = torch.topk(
                loss_reduction_none, k=num_cf_items, largest=False, sorted=False
            )[0].sum()
        else:
            loss = loss_reduction_none.sum()

        return loss

    def criterion_cf_triplet(self, cf_hat: torch.Tensor, cf_hat_pos: torch.Tensor):
        cf_hat_neg = torch.roll(cf_hat_pos, shifts=1, dims=0)
        loss = nn.functional.triplet_margin_loss(
            cf_hat, cf_hat_pos, cf_hat_neg, margin=0.2, p=2
        )
        return loss

    def forward(self, x):
        z = self.backbone(x).flatten(1)
        y_hat = self.classifier(z)
        cf_hat = self.cf_layers(z)
        return y_hat, cf_hat

    def get_embeddings(self, x):
        z = self.backbone(x).flatten(1)
        return z

    def _loss_helper(self, batch, phase: str):
        assert phase in ["train", "val"]

        (
            imgs,
            imgs_pos,
            cf_vectors,
            labels,
            is_labeled,
            cf_confidence,
        ) = batch
        y_hat, cf_hat = self(imgs)

        # Compute calssification loss
        loss_calssification = self.criterion(y_hat.squeeze(), labels.float().squeeze())
        loss_calssification = loss_calssification[is_labeled].mean()

        # Compute CF loss
        cf_topk_loss_ratio = self.cfg["cf_topk_loss_ratio"] if phase == "train" else 1.0
        if self.cfg.cf_loss_type == "exp":
            loss_cf = self.criterion_cf(
                cf_hat,
                cf_vectors,
                cf_topk_loss_ratio,
                cf_confidence,
            )
        elif self.cfg.cf_loss_type == "triplet":
            _, cf_hat_pos = self(imgs_pos)
            loss_cf = self.criterion_cf_triplet(cf_hat.squeeze(), cf_hat_pos.squeeze())
        else:
            raise ValueError(f"{self.cfg.cf_loss_type=}")

        # Combine loss
        loss = (
            self.cfg["label_weight"] * loss_calssification
            + self.cfg["cf_weight"] * loss_cf
        )

        res_dict = {
            f"loss/{phase}": loss.detach(),
            f"loss_classification/{phase}": loss_calssification.detach(),
            f"loss_cf/{phase}": loss_cf.detach(),
        }
        self.log_dict(
            res_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        preds = torch.sigmoid(y_hat)
        res_dict["labels"] = labels.cpu().detach().numpy()
        res_dict["preds"] = preds.cpu().detach().numpy()
        res_dict["loss"] = loss
        return res_dict

    def epoch_end_helper(self, outputs, phase: str):
        assert phase in ["train", "val"]
        loss = torch.mean(torch.stack([out["loss"] for out in outputs])).item()

        preds = np.vstack([out["preds"] for out in outputs])
        labels = np.vstack([out["labels"] for out in outputs])

        # Metrics
        ap = average_precision_score(labels, preds)

        self.log_dict(
            {
                f"ap/{phase}": ap,
            },
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )

        loss, ap = np.round([loss, ap], 3)
        logger.info(
            f"[{self.current_epoch}/{self.cfg['epochs'] - 1}] {phase} epoch end. {[loss, ap]=}"
        )

        if phase == "val" and ap > self.map_best:
            self.map_best = ap

            np.save(self.preds_path, preds)
            np.save(self.labels_path, labels)

    def training_step(self, batch, batch_idx):
        return self._loss_helper(batch, phase="train")

    def training_epoch_end(self, outputs):
        self.epoch_end_helper(outputs, "train")

    def validation_step(self, batch, batch_idx, stage=None):
        return self._loss_helper(batch, phase="val")

    def validation_epoch_end(self, outputs):
        self.epoch_end_helper(outputs, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg["milestones"]
        )
        return [optimizer], [lr_scheduler]


class LitModelCFBased(LitModel):
    def __init__(
        self,
        num_target_classes: int,
        cf_vector_dim: int,
        cfg,
        pos_weight=None,
    ):
        # pl.LightningModule().__init__()
        cfg["is_pretrained"] = False
        cfg["arch"] = "resnet18"
        super().__init__(num_target_classes, cf_vector_dim, cfg, pos_weight)

        # Define the backbone
        layer_dims = torch.linspace(cf_vector_dim, num_target_classes, 3).int()
        self.model = nn.Sequential(
            nn.BatchNorm1d(layer_dims[0]),
            nn.Linear(layer_dims[0], layer_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(layer_dims[1], layer_dims[2]),
        )
        logger.info(self)

    def get_embeddings(self, x):
        z = self.model[0](x)
        z = self.model[1](z)
        return z

    def forward(self, x):
        return self.model(x)

    def _loss_helper(self, batch, phase: str):
        assert phase in ["train", "val"]

        cf_vectors, labels = batch
        y_hat = self(cf_vectors)

        # Compute calssification loss
        loss = self.criterion(y_hat.squeeze(), labels.float().squeeze()).mean()

        res_dict = {
            f"loss/{phase}": loss.detach(),
        }
        self.log_dict(
            res_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        preds = torch.sigmoid(y_hat)
        res_dict["labels"] = labels.cpu().detach().numpy()
        res_dict["preds"] = preds.cpu().detach().numpy()
        res_dict["loss"] = loss

        return res_dict
