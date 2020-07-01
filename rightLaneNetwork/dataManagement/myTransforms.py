from albumentations import Compose, Resize, ToGray, NoOp
from albumentations.pytorch import ToTensor


def testTransform(img, label=None, width=160, height=120, gray=False):
    aug = Compose([
        Resize(height=height, width=width, always_apply=True),
        ToGray(always_apply=True) if gray else NoOp(always_apply=True),
        ToTensor(),
    ])

    if label is not None and len(label.shape) >= 2:
        # Binarize label
        label[label > 0] = 255

        augmented = aug(image=img, mask=label)
        img = augmented['image']
        label = augmented['mask'].squeeze().long()
    else:
        augmented = aug(image=img)
        img = augmented['image']

    return img, label
