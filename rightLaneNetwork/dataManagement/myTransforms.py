from albumentations import Compose, Resize, ToGray, NoOp, Normalize, GaussNoise, HueSaturationValue, RandomSizedCrop, \
    OneOf, MotionBlur
from albumentations.pytorch import ToTensorV2


class MyTransform:
    def __init__(self, width=160, height=120, gray=False, augment=False):
        augmentations = Compose([
            HueSaturationValue(always_apply=True),
            RandomSizedCrop(min_max_height=(height // 2, height * 4), height=height, width=width,
                            w2h_ratio=width / height, always_apply=True),
            OneOf([MotionBlur(p=0.5), GaussNoise(p=0.5)], p=1),
        ])

        self.transform = Compose([
            augmentations if augment else Resize(height=height, width=width, always_apply=True),
            ToGray(always_apply=True) if gray else NoOp(always_apply=True),
            Normalize(always_apply=True),
            ToTensorV2(),
        ])

    def __call__(self, img, label=None):
        if label is not None and len(label.shape) >= 2:
            # Binarize label
            label[label != 0] = 1

            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask'].squeeze().long()
        else:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label
