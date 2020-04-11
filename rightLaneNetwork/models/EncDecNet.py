import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, inCh, oCh, kernelSize=3, stride=1, activation=nn.ReLU(), bNorm=True, dropOut=0.3):
        super(Conv, self).__init__()

        if dropOut >= 1 or dropOut < 0:
            raise ValueError(f"A Conv konstuktor dropOut paramétere a [0,1) intervallumban valid!\
                                Kapott érték: {dropOut}")

        # A konvolúciós réteg
        self.conv = nn.Conv2d(inCh, oCh, kernelSize, padding=kernelSize // 2, stride=stride)

        # Kiegészítések: aktiváció, batch normalization, dropout
        self.activation = activation

        if bNorm:
            self.bn = nn.BatchNorm2d(oCh)
        else:
            self.bn = nn.Identity()

        if dropOut == 0:
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout(p=dropOut)

    def forward(self, x):
        # A modulok sorrendje (conv után bármelyik lehet identitás, azaz nem használt):
        # Konvolúció -> aktiváció -> batch normalization -> dropout
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        x = self.drop(x)

        return x


# Aktivációk definiálása
activationDict = nn.ModuleDict({
    'relu': nn.ReLU(),
    'prelu': nn.PReLU(),
    'leakyRelu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'none': nn.Identity(),
})
activationTypes = list(activationDict.keys())


class EncDecNet(nn.Module):
    def __init__(self, nFeat: int, nLevels: int, kernelSize: int = 3,
                 nLinType: str = 'relu', bNorm: bool = True, dropOut: int = 0.3, inFeat=3):
        super(EncDecNet, self).__init__()

        # Kapott paraméterek ellenőrzése
        if nFeat < 1:
            raise ValueError(f"A csatornák számának legalább 1-nek kell lennie! Kapott érték: {nFeat}")
        if nLevels < 1:
            raise ValueError(f"A szintek számának legalább 1-nek kell lennie! Kapott érték: {nLevels}")
        if nLinType not in activationTypes:
            raise ValueError(f"A kapott aktiváció: {nLinType} nincs megvalósítva!\
                                Válasszon egyet a következők közül: {activationTypes}")

        # Aktivációs függvény a paraméter sztringből
        self.activation = activationDict[nLinType]
        self.pool = nn.MaxPool2d(kernelSize, stride=2, padding=kernelSize//2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Az első szint bemenete 3 csatornás (RGB kép)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Első szint csatornáinak számolása
        oFeat = nFeat

        # Minden enkóder hozzáadása
        for _ in range(nLevels):
            layer = Conv(inFeat, oFeat, kernelSize=kernelSize, activation=self.activation, bNorm=bNorm, dropOut=dropOut)
            self.encoders.append(layer)
            inFeat = oFeat
            oFeat = 2 * oFeat


        # Csatornák újraszámolása
        oFeat = oFeat//2

        # Minden enkóder hozzáadása
        for _ in range(nLevels):
            layer = Conv(inFeat, oFeat, kernelSize=kernelSize, activation=self.activation, bNorm=bNorm, dropOut=dropOut)
            self.decoders.append(layer)
            inFeat = oFeat
            oFeat = oFeat // 2

        # Pixelenként következtetés, Sigmoid aktiváció
        self.classifier = Conv(inFeat, 2, kernelSize=1, activation=nn.Softmax(dim=-3), bNorm=False, dropOut=0)

    def forward(self, x):
        # A bemenetet végigvezetjük az összes szinten
        for encoder in self.encoders:
            x = encoder(x)
            x = self.pool(x)

        for decoder in self.decoders:
            x = decoder(x)
            x = self.upsample(x)

        # Klasszifikáció
        x = self.classifier(x)

        return x

    # Segédfüggvény a paraméterek számának meghatározására
    def getNParams(self):
        return sum([p.numel() for p in self.parameters()])


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    import os

    print(f"Usable GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    net = EncDecNet(64, 3, 7)
    print(f"Net number of parameters: {net.getNParams()}")
    img = torch.ones(1, 3, 120, 160)
    out = net(img)
    print(img.shape, out.shape)
