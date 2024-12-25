from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers.models.auto import AutoConfig
from data import data_transforms  # Assumes data_transforms includes augmentations
from model import Net  # Custom basic CNN implementation


class ModelFactory:
    def __init__(self, model_name: str):
        """
        Factory class for initializing models and their associated transforms.

        Args:
        - model_name (str): Name of the model to load. Options include:
          - 'basic_cnn': A simple custom CNN model.
          - 'vit', 'resnet', 'convnext', 'deit': Vision-based models (NOT pretrained).
          - 'clip': OpenAI's CLIP model (not ImageNet-specific).
          - 'dinov2', 'beit': Self-supervised models (not ImageNet-specific).
        """
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        """Initialize the model architecture without pretrained weights."""
        if self.model_name == "basic_cnn":
            # Custom CNN model
            return Net()
        elif self.model_name in { "vit", "resnet", "convnext", }:
            # Load the model architecture from config without pretrained weights
            config = AutoConfig.from_pretrained(self.get_hf_model_name(self.model_name))
            return AutoModelForImageClassification.from_config(config)
        elif self.model_name == "deit":
            return AutoModelForImageClassification.from_pretrained("facebook/deit-base-patch16-224")
        elif self.model_name == "clip":
            return AutoModelForImageClassification.from_pretrained("openai/clip-vit-base-patch32")
        elif self.model_name == "dinov2":
            return AutoModelForImageClassification.from_pretrained("facebook/dinov2-large")
        elif self.model_name == "beit":
            return AutoModelForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
        else:
            raise NotImplementedError(f"Model '{self.model_name}' is not supported.")

    def init_transform(self):
        """Initialize data transformation pipeline."""
        if self.model_name == "basic_cnn":
            # Use standard data augmentations for custom CNN
            return data_transforms
        elif self.model_name in {"vit", "resnet", "convnext", "deit", "clip", "dinov2", "beit"}:
            processor = AutoImageProcessor.from_pretrained(self.get_hf_model_name(self.model_name))
            # Combine processor with additional augmentations
            return lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            raise NotImplementedError(f"Transform for '{self.model_name}' is not supported.")

    def get_hf_model_name(self, model_name):
        """Map model_name to Hugging Face pretrained models."""
        hf_models = {
            "vit": "google/vit-base-patch16-224",
            "resnet": "microsoft/resnet-50",
            "convnext": "facebook/convnext-tiny-224",
            "deit": "facebook/deit-base-distilled-patch16-224",
            "clip": "openai/clip-vit-base-patch32",
            "dinov2": "facebook/dinov2-base",
            "beit": "microsoft/beit-base-patch16-224",
        }
        return hf_models.get(model_name)

    def get_model(self):
        """Return the model instance."""
        return self.model

    def get_transform(self):
        """Return the transformation pipeline."""
        return self.transform

    def get_all(self):
        """Return both model and transform."""
        return self.model, self.transform
