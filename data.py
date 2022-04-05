import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class THOREmbeddingsDataset(Dataset):
    def __init__(self, data_dir, split, embedding_type, prediction_type):

        assert embedding_type in ['rn50_imagenet_conv', 'rn50_imagenet_avgpool', 'rn50_clip_conv', 'rn50_clip_attnpool', 'rn50_clip_avgpool']
        assert prediction_type in ['object_presence', 'object_presence_grid', 'valid_moves_forward', 'reachable_objects']

        if prediction_type == 'reachable_objects':
            image_features = torch.load(os.path.join(data_dir, f"reachable_image_features.pt"))
            data = pickle.load(open(os.path.join(data_dir, f"reachable_{split}.pkl"), 'rb'))
            self.embeddings = []
            self.predictions = []

            for image, obj, reachable in data:
                self.embeddings.append(image_features[image][embedding_type])
                self.predictions.append((
                    obj,
                    torch.tensor(reachable, dtype=int)
                ))
        else:
            data = torch.load(os.path.join(data_dir, f"thor_{split}.pt"))

            if prediction_type == 'valid_moves_forward_cls':
                prediction_type = 'valid_moves_forward'

            self.embeddings = []
            self.predictions = []
            for scene_name, frames in data.items():
                for frame_features in frames:
                    self.embeddings.append(frame_features[embedding_type])
                    self.predictions.append(frame_features[prediction_type])

    def __getitem__(self, index):
        return self.embeddings[index], self.predictions[index]

    def __len__(self):
        return len(self.embeddings)


class THOREmbeddingsDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, embedding_type, prediction_type, batch_size=1, num_workers=0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_dataset = THOREmbeddingsDataset(
            self.hparams.data_dir, 'train',
            self.hparams.embedding_type, self.hparams.prediction_type
        )
        self.val_dataset = THOREmbeddingsDataset(
            self.hparams.data_dir, 'val',
            self.hparams.embedding_type, self.hparams.prediction_type
        )
        self.test_dataset = THOREmbeddingsDataset(
            self.hparams.data_dir, 'test',
            self.hparams.embedding_type, self.hparams.prediction_type
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=int(0.8 * self.hparams.num_workers)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=int(0.2 * self.hparams.num_workers)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=int(self.hparams.num_workers)
        )
