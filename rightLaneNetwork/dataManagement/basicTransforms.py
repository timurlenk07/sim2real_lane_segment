import cv2
import numpy as np
import torch


class SavedTransform:
    def __init__(self, grayscale: bool, newRes: tuple = None):
        self.grayscale = grayscale
        self.newRes = newRes

    def __call__(self, img, label):
        if img is not None:
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.newRes is not None:
                img = cv2.resize(img, self.newRes)

        if label is not None:
            if self.newRes is not None:
                label = cv2.resize(label, self.newRes)

        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LoadedTransform:
    def __init__(self, grayscale: bool, newRes: tuple = None):
        self.grayscale = grayscale
        self.newRes = newRes

    def __call__(self, img, label):
        if img is not None:
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.newRes is not None:
                img = cv2.resize(img, self.newRes, interpolation=cv2.INTER_LANCZOS4)

            if self.grayscale:
                img = np.expand_dims(img, -1)

            img = torch.from_numpy(img.transpose((2, 0, 1)))
            img = img.float() / 255

        if label is not None:
            if self.newRes is not None:
                label = cv2.resize(label, self.newRes, interpolation=cv2.INTER_LANCZOS4)

                # y should be binary
                _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

            label = torch.from_numpy(label).long() / 255

        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '()'
