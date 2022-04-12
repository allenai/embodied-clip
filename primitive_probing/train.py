import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as MF

from constants import target_objects, max_forward_steps
from data import THOREmbeddingsDataModule


class LinearEncoder(pl.LightningModule):
    def __init__(self, embedding_type, prediction_type, batch_size, lr):
        super().__init__()
        self.save_hyperparameters()

        if prediction_type in ['object_presence', 'reachability', 'free_space']:
            assert embedding_type in ['imagenet_avgpool', 'clip_avgpool', 'clip_attnpool']

            if embedding_type in ['imagenet_avgpool', 'clip_avgpool']:
                input_dim = 2048
            elif embedding_type == 'clip_attnpool':
                input_dim = 1024

            if prediction_type == 'object_presence':
                output_dim = len(target_objects)
                act_fn = nn.Sigmoid()
            elif prediction_type == 'reachability':
                output_dim = 110
                act_fn = nn.Sigmoid()
            elif prediction_type == 'free_space':
                output_dim = max_forward_steps + 1
                act_fn = nn.Softmax(dim=1)

            self.model = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                act_fn
            )

        elif prediction_type == 'object_localization':
            assert embedding_type in ['imagenet_avgpool', 'clip_avgpool']
            self.model = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),
                nn.Conv2d(2048, len(target_objects), kernel_size=1),
                nn.Flatten(start_dim=2),
                nn.Sigmoid()
            )

        else: raise NotImplementedError()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch, eval=False):
        x, y = batch

        if self.hparams.prediction_type == 'object_localization':
            y = y.flatten(start_dim=1)
        elif self.hparams.prediction_type == 'reachability':
            obj_idx, y = y
            obj_idx = obj_idx.tolist()
        elif self.hparams.prediction_type == 'free_space':
            y[y > max_forward_steps] = max_forward_steps

        y_pred = self.forward(x)

        if self.hparams.prediction_type == 'object_localization':
            y_pred = y_pred.permute(0, 2, 1).flatten(start_dim=1)
        elif self.hparams.prediction_type == 'reachability':
            y_pred = y_pred[range(len(obj_idx)), obj_idx]

        # compute loss
        if self.hparams.prediction_type in ['object_presence', 'object_localization', 'reachability']:
            loss = F.binary_cross_entropy(y_pred, y.float())
        elif self.hparams.prediction_type == 'free_space':
            loss = F.cross_entropy(y_pred, y)

        if eval is False:
            return loss

        # compute metrics
        metrics = {}
        if self.hparams.prediction_type in ['object_presence', 'object_localization']:
            metrics['accuracy'] = MF.f1(y_pred, y)
        elif self.hparams.prediction_type == 'reachability':
            metrics['accuracy'] = ((y_pred > 0.5) == y).float().mean()
        elif self.hparams.prediction_type == 'free_space':
            metrics['accuracy'] = (torch.argmax(y_pred, dim=1) == y).float().mean()

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.compute_loss(batch, eval=True)
        self.log("val_loss", loss)
        self.log("val_acc", metrics['accuracy'])
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self.compute_loss(batch, eval=True)
        self.log("test_loss", loss)
        self.log("test_acc", metrics['accuracy'])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


if __name__ == '__main__':
    pl.seed_everything(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, dest='data_dir',
                        default='data',
                        help='Path to data directory')
    parser.add_argument('--log-dir', type=str, dest='log_dir',
                        default='logs/',
                        help='Path to log directory')
    parser.add_argument('--embedding-type', dest='embedding_type', type=str,
                        choices=['imagenet_avgpool', 'clip_attnpool', 'clip_avgpool'],
                        help='Which encoder features to evaluate')
    parser.add_argument('--prediction-type', dest='prediction_type', type=str,
                        choices=['object_presence', 'object_localization', 'reachability', 'free_space'],
                        help='Which task to evaluate')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    args = parser.parse_args()

    batch_size = 128
    lr = 0.001

    logger = pl.loggers.TensorBoardLogger(
        args.log_dir,
        name=f'{args.prediction_type}',
        version=f'{args.embedding_type}'
    )

    dm = THOREmbeddingsDataModule(
        args.data_dir,
        args.embedding_type, args.prediction_type,
        batch_size=batch_size, num_workers=16
    )

    model = LinearEncoder(args.embedding_type, args.prediction_type, batch_size, lr)

    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        logger=logger,
        gpus=args.gpus,
        val_check_interval=0.5,
        max_epochs=250,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                filename="{epoch:02d}-{val_loss:.2f}",
                mode="min"
            )
        ],
    )

    trainer.fit(model, dm)

    trainer.test(
        model=model,
        datamodule=dm,
        ckpt_path='best'
    )
