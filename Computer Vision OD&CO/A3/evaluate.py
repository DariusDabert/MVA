import argparse
import os
from PIL import Image
import torch
from tqdm import tqdm
from collections import Counter
from model_factory import ModelFactory
import torch.nn as nn

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. test_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        metavar="M",
        help="list of model files to be evaluated. Provide paths like model_1.pth model_2.pth",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs='+',
        metavar="MOD",
        help="Names of the models corresponding to each file, e.g., basic_cnn transformer_vit",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="experiment/kaggle.csv",
        metavar="D",
        help="name of the output csv file",
    )
    args = parser.parse_args()
    return args


def pil_loader(path):
    """Load an image and convert to RGB."""
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def main() -> None:
    """Main Function."""
    # options
    args = opts()
    test_dir = os.path.join(args.data, "test_images", "mistery_category")

    # cuda
    use_cuda = torch.cuda.is_available()

    # Load multiple models and their transforms
    models = []
    transforms = []
    for model_file, model_name in zip(args.models, args.model_names):
        print(f"Loading model: {model_file} with architecture {model_name}")
        state_dict = torch.load(model_file, map_location="cuda" if use_cuda else "cpu")
        model, data_transform = ModelFactory(model_name).get_all()
        model.classifier = nn.Linear(model.classifier.in_features, 500)

        # Load the weights
        model.load_state_dict(state_dict)
        model.eval()

        if use_cuda:
            model.cuda()

        models.append(model)
        transforms.append(data_transform)

    # Prepare the output file
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as output_file:
        output_file.write("Id,Category\n")

        # Process test images
        for f in tqdm(os.listdir(test_dir)):
            if f.endswith(".jpeg"):
                # Load and preprocess the image
                img_path = os.path.join(test_dir, f)
                image = pil_loader(img_path)

                # Collect predictions from all models
                predictions = []
                for model, transform in zip(models, transforms):
                    # Apply the transformation
                    if callable(transform):
                        data = transform(image)
                        if len(data.shape) == 3:  # Handle case where batch dimension is missing
                            data = data.unsqueeze(0)
                    else:
                        raise ValueError("The transform must be a callable function.")

                    # Send data to the appropriate device
                    if use_cuda:
                        data = data.cuda()

                    # Get the output and prediction
                    with torch.no_grad():
                        output = model(data)
                        if hasattr(output, "logits"):  # For Hugging Face models
                            logits = output.logits
                        else:
                            logits = output
                        pred = logits.argmax(dim=1).item()
                        predictions.append(pred)

                # Perform majority voting
                final_pred = Counter(predictions).most_common(1)[0][0]
                output_file.write(f"{f[:-5]},{final_pred}\n")

    print(
        f"Successfully wrote {args.outfile}. You can upload this file to the Kaggle competition website."
    )


if __name__ == "__main__":
    main()
