import torch
from pytorch_lightning.metrics.functional import accuracy
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .TrainingBase import TrainingBase, getClassWeight


class SimpleTrainModule(TrainingBase):
    def training_step(self, batch, batch_idx):
        x, y = batch

        # Forward propagation, loss calculation
        outputs = self.forward(x)
        loss = cross_entropy(outputs, y, weight=getClassWeight(y, self.num_cls).to(self.device))

        # Accuracy calculation
        _, labels_hat = torch.max(outputs, 1)
        train_acc = accuracy(labels_hat, y) * 100

        self.log('tr_loss', loss)
        self.log('tr_acc', train_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = CosineAnnealingLR(optimizer, 25, eta_min=self.lr / self.lrRatio)
        return [optimizer], [scheduler]
