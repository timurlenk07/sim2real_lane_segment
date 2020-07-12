import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

if __name__ == '__main__':
    model = nn.Sequential(
        nn.Conv2d(1, 20, 5),
        nn.ReLU(),
        nn.Conv2d(20, 64, 5),
        nn.ReLU()
    )
    sgd = SGD(model.parameters(), lr=1, momentum=0, weight_decay=0)
    adam = AdamW(model.parameters(), lr=1, weight_decay=0)

    cos_ann_lr = CosineAnnealingLR(sgd, T_max=25, eta_min=0.1)
    step_lr = StepLR(sgd, step_size=1, gamma=0.98)
    cos_ann_wr = CosineAnnealingWarmRestarts(sgd, T_0=25, T_mult=2, eta_min=0)

    epochs = 200
    steps_per_epoch = 370
    lr_s = [0 for _ in range(epochs + 1)]
    for epoch in range(epochs + 1):
        lr_s[epoch] = sgd.param_groups[0]['lr']
        sgd.step()
        step_lr.step()
        # sgd.param_groups[0]['lr'] = 1 * (1 + 1e-4 * steps_per_epoch * epoch) ** (-0.75)

    plt.plot(lr_s)
    plt.yscale('log')
    plt.savefig('lr_over_epochs.png')
    print(f"Last LR was: {lr_s[-1]:e}")
