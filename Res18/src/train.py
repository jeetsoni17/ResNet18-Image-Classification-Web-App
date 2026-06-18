import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

import pytorch_lightning as pl


# -------------------------
# 1. DATA PREPROCESSING
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


# -------------------------
# DATASET
# -------------------------
train_dataset = torchvision.datasets.CIFAR10(
    root="./dataT",
    train=True,
    download=True,
    transform=transform,
)

val_dataset = torchvision.datasets.CIFAR10(
    root="./dataT",
    train=False,
    download=True,
    transform=transform,
)


# -------------------------
# DATA LOADERS (FIXED FOR WINDOWS)
# -------------------------
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,   # ✅ FIX: prevents Windows crash
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,   # ✅ FIX: prevents spawn error
)


# -------------------------
# MODEL
# -------------------------
class ResNet18Model(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)

        self.lossFun = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.lossFun(outputs, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.lossFun(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


# -------------------------
# TRAINING (CRITICAL FIX)
# -------------------------
if __name__ == "__main__":   # ✅ FIX: REQUIRED ON WINDOWS

    model = ResNet18Model(num_classes=10)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.model.state_dict(), "resnet18_model.pth")