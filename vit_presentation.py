from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
from torch.utils.data import DataLoader


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_chans=3, embed_dim=48):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        output_dim = (img_size - patch_size) // stride + 1
        self.n_patches = output_dim ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_p=0.1, proj_p=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError
        if self.dim != self.head_dim*self.n_heads: # make sure dim is divisible by n_heads
            raise ValueError(f"Input & Output dim should be divisible by number of heads")
        qkv = self.qkv(x).reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        dp = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(dp.softmax(dim=-1))
        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2).flatten(2)
        return self.proj_drop(self.proj(weighted_avg))


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(p)

    # TODO: cleanup
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        # x = self.drop(self.act(self.fc1(x)))
        # return self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.1, attn_p=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5) # use default eps
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5) # use default eps
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, stride=4, in_chans=3, n_classes=10, embed_dim=256, depth=8, n_heads=6, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride=stride, in_chans=in_chans)
        flattened_patch_dim = (in_chans * img_size ** 2) // self.patch_embed.n_patches
        self.project_flat_patch = nn.Linear(flattened_patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_embed.n_patches, embed_dim))
        """czy jest tu potrzebny dropout? i w forward tez"""
        # self.pos_drop = nn.Dropout(p=p)
        enc_list = [Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p) for _ in range(depth)]
        self.enc_blocks = nn.Sequential(*enc_list)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5) # use default eps
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        x = self.project_flat_patch(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        # x = self.pos_drop(x)
        x = self.enc_blocks(x)
        return self.head(self.norm(x[:, 0]))


def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct_predictions, total_samples = 0, 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct_predictions / total_samples:.2f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100 * correct_predictions / total_samples
    print(f"Epoch Loss: {epoch_loss:.4f}, Epoch Accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy


# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    running_loss = 0.0
    correct_predictions, total_samples = 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Evaluation Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct_predictions / total_samples:.2f}")

    eval_loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * correct_predictions / total_samples
    print(f"Test Loss: {eval_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return eval_loss, accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "device": device,
        "dataset": "CIFAR10",
        "transforms": {
            "resize": (32, 32),
            "to tensor": None,
            "normalize": {
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5)
            }
        },
        "batch_size": 128,
        "num_workers": 16,
        "img_size": 32,
        "patch_size": 4,
        "stride": 4,
        "n_classes": 10,
        "embed_dim": 384,
        "depth": 8,
        "n_heads": 8,
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 5e-5,
        "dropout": 0.01,
        "attention_dropout": 0.01,
        "loss_func": nn.CrossEntropyLoss(),
        "num_epochs": 100,
        "checkpoint_every_num_epochs": 10,
    }

    wandb.init(entity="mikolajgrycz-mikorg", project="fsk", config=config, name=f"ViT Training from scratch weight decay={config['weight_decay']}, without dropout in vit")

    transform = transforms.Compose([
        transforms.Resize(config["transforms"]["resize"]),
        transforms.ToTensor(),
        transforms.Normalize(config["transforms"]["normalize"]["mean"], config["transforms"]["normalize"]["std"])
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    model = VisionTransformer(img_size=config["img_size"], patch_size=config["patch_size"], stride=config["stride"], n_classes=config["n_classes"], embed_dim=config["embed_dim"], depth=config["depth"], n_heads=config["n_heads"], p=config["dropout"], attn_p=config["attention_dropout"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=config["betas"], eps=config["eps"], weight_decay=config["weight_decay"])
    criterion = config["loss_func"]

    num_epochs = config["num_epochs"]
    checkpoint_every_num_epochs = config["checkpoint_every_num_epochs"]
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
        test_loss, test_accuracy = evaluate(model, test_loader)
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.2f}%")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy})
        if epoch % checkpoint_every_num_epochs == 0:
            torch.save(model.state_dict(), checkpoint_dir / f'vit-regular-{epoch}-epoch.pth')

    model.load_state_dict(torch.load(checkpoint_dir / f'vit-regular-{num_epochs}-epoch.pth'))
    model.eval()

    img, label = test_dataset[0]
    img = img.unsqueeze(0).to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=-1)
        top_prob, top_class = probs.topk(1, dim=-1)

    print(f"Predicted class: {top_class.item()}, Probability: {top_prob.item():.4f}")
