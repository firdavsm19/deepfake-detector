import os
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter
from config import Config


# ── Transforms ──────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Dataset ─────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, max_samples=None):
        self.transform = transform
        self.samples   = []

        real_images = [f for f in os.listdir(real_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        fake_images = [f for f in os.listdir(fake_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Limit number of images if max_samples is set
        if max_samples:
            real_images = real_images[:max_samples]
            fake_images = fake_images[:max_samples]

        for img in real_images:
            self.samples.append((os.path.join(real_dir, img), 0))

        for img in fake_images:
            self.samples.append((os.path.join(fake_dir, img), 1))

        real  = sum(1 for _, l in self.samples if l == 0)
        fake  = sum(1 for _, l in self.samples if l == 1)
        print(f"    Real: {real:,}  |  Fake: {fake:,}  |  Total: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), 0)
        if self.transform:
            image = self.transform(image)
        return image, label

# ── DataLoaders ──────────────────────────────────────────────────

def get_dataloaders():
    print("\n📂 Loading datasets...")

    print("  [Train]")
    train_dataset = DeepfakeDataset(
        real_dir    = os.path.join(Config.DATA_ROOT, "train", "real"),
        fake_dir    = os.path.join(Config.DATA_ROOT, "train", "fake"),
        transform   = train_transform,
        max_samples = 1000          # 5000 real + 5000 fake = 10,000 total
    )
    print("  [Validation]")
    val_dataset = DeepfakeDataset(
        real_dir    = os.path.join(Config.DATA_ROOT, "valid", "real"),
        fake_dir    = os.path.join(Config.DATA_ROOT, "valid", "fake"),
        transform   = val_test_transform,
        max_samples = 100          # 2000 real + 2000 fake = 4,000 total
    )
    print("  [Test]")
    test_dataset = DeepfakeDataset(
        real_dir    = os.path.join(Config.DATA_ROOT, "test", "real"),
        fake_dir    = os.path.join(Config.DATA_ROOT, "test", "fake"),
        transform   = val_test_transform,
        max_samples = 100          # 2000 real + 2000 fake = 4,000 total
    )

    labels       = [s[1] for s in train_dataset.samples]
    class_counts = Counter(labels)
    weights      = [1.0 / class_counts[l] for l in labels]
    sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                              sampler=sampler, num_workers=Config.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=Config.BATCH_SIZE,
                              shuffle=False,  num_workers=Config.NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=Config.BATCH_SIZE,
                              shuffle=False,  num_workers=Config.NUM_WORKERS,
                              pin_memory=True)

    return train_loader, val_loader, test_loader