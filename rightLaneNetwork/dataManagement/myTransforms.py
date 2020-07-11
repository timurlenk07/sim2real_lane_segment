from albumentations import Compose, Resize, ToGray, NoOp, Normalize
from albumentations.pytorch import ToTensorV2


def testTransform(img, label=None, width=160, height=120, gray=False):
    aug = Compose([
        Resize(height=height, width=width, always_apply=True),
        ToGray(always_apply=True) if gray else NoOp(always_apply=True),
        Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
    ])

    if label is not None and len(label.shape) >= 2:
        # Binarize label
        label[label != 0] = 1

        augmented = aug(image=img, mask=label)
        img = augmented['image']
        label = augmented['mask'].squeeze().long()
    else:
        augmented = aug(image=img)
        img = augmented['image']

    return img, label
