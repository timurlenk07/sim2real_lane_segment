from collections import OrderedDict

import torch
from torch.nn.functional import cross_entropy
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.FCDenseNet.tiramisu import grad_reverse
from .TrainingBase import TrainingBase


def adentropy(output, lamda=1.0):
    return lamda * torch.mean(torch.sum(output * (torch.log(output + 1e-5)), 1))


class MMETrainingModule(TrainingBase):
    def configure_optimizers(self):
        optimizerF = AdamW(self.parameters(), lr=self.lr, weight_decay=self.decay)
        optimizerG = SGD([
            {'params': self.featureExtractor.parameters(), 'lr': self.lr / 3},
            {'params': self.classifier.parameters(), 'lr': self.lr}
        ], lr=self.lr, weight_decay=self.decay, momentum=0.9, nesterov=True)
        lr_schedulerF = CosineAnnealingLR(optimizerF, T_max=25, eta_min=self.lr * 1e-3)
        lr_schedulerG = CosineAnnealingLR(optimizerG, T_max=25, eta_min=self.lr * 1e-3)
        return [optimizerG, optimizerF], [lr_schedulerG, lr_schedulerF]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x_labelled, x_unlabelled, labels, _ = batch

        if optimizer_idx == 0:  # We are unlabelled optimizer -> maximize entropy
            outputs = self.featureExtractor(x_unlabelled)
            outputs = grad_reverse(outputs)
            outputs = self.classifier(outputs)
            loss = adentropy(outputs, lamda=0.1)
        if optimizer_idx == 1:  # We are labelled optimizer -> minimize entropy
            outputs = self.featureExtractor(x_labelled)
            outputs = self.classifier(outputs)
            loss = cross_entropy(outputs, labels)

        output = OrderedDict({
            'loss': loss
        })
        return output
