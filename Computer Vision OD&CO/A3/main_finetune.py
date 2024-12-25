import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
from model_factory import ModelFactory
import timm

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="Folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vit",
        metavar="MOD",
        help="Name of the model for model and transform instantiation (default: vit)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=5,
        metavar="SS",
        help="Step size for learning rate scheduler (default: 5 epochs)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        metavar="G",
        help="Learning rate decay factor (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="Random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="How many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="Folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="Number of workers for data loading",
    )
    args = parser.parse_args()
    return args


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Training Loop."""
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        logits = output.logits  # Extract logits from the Hugging Face model output
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        optimizer.step()
        pred = logits.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    print(
        f"\nTrain set: Accuracy: {correct}/{len(train_loader.dataset)} "
        f"({100.0 * correct / len(train_loader.dataset):.0f}%)\n"
    )


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Validation Loop."""
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            logits = output.logits  # Extract logits from the Hugging Face model output
            validation_loss += nn.CrossEntropyLoss()(logits, target).item()
            pred = logits.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print(
        f"\nValidation set: Average loss: {validation_loss:.4f}, "
        f"Accuracy: {correct}/{len(val_loader.dataset)} "
        f"({100.0 * correct / len(val_loader.dataset):.0f}%)\n"
    )
    return validation_loss


def main():
    """Main Function."""
    args = opts()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Create experiment folder
    os.makedirs(args.experiment, exist_ok=True)

    # # Load model and transform
    model, data_transform = ModelFactory(args.model_name).get_all()

    # # Load the number of classes
    num_classes = 500
    model.config.num_labels = num_classes
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    model.load_state_dict(torch.load(f"{args.experiment}best_model.pth"))

    for param in model.parameters():
        param.requires_grad = True
    # Ensure the classifier head requires gradients
    for param in model.classifier.parameters():
        param.requires_grad = True

    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    dataset_train = datasets.ImageFolder(args.data + "/train_images", transform=data_transform)
    dataset_val = datasets.ImageFolder(args.data + "/val_images", transform=data_transform)

    merge = torch.utils.data.ConcatDataset([dataset_train, dataset_val])

    train_set, val_set = torch.utils.data.random_split(merge, [int(0.8 * len(merge)), int(0.2 * len(merge))], generator=torch.Generator().manual_seed(args.seed))

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Fine-tuning loop
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, use_cuda, epoch, args)
        val_loss = validate(model, val_loader, use_cuda)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.experiment}/best_model.pth")
            print(f"Saved best model to {args.experiment}/best_model.pth")

        scheduler.step()

    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()
