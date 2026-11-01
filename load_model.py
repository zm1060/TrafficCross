#!/usr/bin/env python3
"""Load a trained multimodal traffic classifier for single-sample or batch inference."""

import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
import json
from typing import Dict, List, Tuple, Optional
import numpy as np

# Import the model class.
from train_multimodal import MultiModalTrafficNet
from utils.config import load_config
from utils.logging_config import setup_logging


class TrafficModelLoader:
    """Load and run a multimodal traffic classification model."""

    def __init__(self, model_path: str, config_path: str = 'config.yaml', device: Optional[str] = None):
        """Initialize the model loader.

        Args:
            model_path: Path to the model checkpoint.
            config_path: Path to the configuration file.
            device: Compute device (cuda or cpu); selected automatically by default.
        """
        self.model_path = Path(model_path)
        self.config = load_config(config_path)

        # Select the compute device.
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Configure logging.
        self.logger = setup_logging(
            model_name="TrafficModelLoader",
            dataset_name=self.config['data']['dataset_name'],
            log_dir=self.config['paths']['log_dir']
        )

        self.logger.info(f"Using device: {self.device}")

        # Load the model.
        self.model, self.categories, self.num_classes = self._load_model()
        self.model.eval()

        # Initialize image transforms.
        image_size = tuple(self.config['image']['size'])
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.image_size = image_size

        self.logger.info(f"Model loaded successfully with {self.num_classes} classes")
        self.logger.info(f"Classes: {self.categories}")

    def _load_model(self) -> Tuple[nn.Module, List[str], int]:
        """Load the model and class metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load the checkpoint.
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Read class metadata.
        dataset_info = checkpoint.get('dataset_info', {})
        categories = dataset_info.get('categories', [])
        num_classes = dataset_info.get('num_classes', len(categories))

        if not categories:
            raise ValueError("No categories found in checkpoint!")

        # Initialize the model.
        model = MultiModalTrafficNet(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        self.logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        return model, categories, num_classes

    def predict_batch(self,
                     images: torch.Tensor,
                     flow_features: torch.Tensor,
                     sequence_features: Dict[str, torch.Tensor],
                     return_probs: bool = True) -> Tuple[List[str], np.ndarray]:
        """Run batch inference.

        Args:
            images: Image tensor [batch_size, 1, H, W].
            flow_features: Flow features [batch_size, 11].
            sequence_features: Dictionary of sequence feature tensors.
            return_probs: Whether to return probabilities.

        Returns:
            predictions: List of predicted class names.
            probabilities: Probabilities [batch_size, num_classes], or None.
        """
        with torch.no_grad():
            images = images.to(self.device)
            flow_features = flow_features.to(self.device)

            # Move sequence features to the compute device.
            sequence_features = {
                k: v.to(self.device) for k, v in sequence_features.items()
            }

            # Forward pass.
            outputs = self.model(
                image=images,
                flow_features=flow_features,
                sequence_features=sequence_features
            )

            # Get predictions.
            probs = torch.softmax(outputs, dim=1)
            pred_indices = torch.argmax(probs, dim=1)

            # Convert indices to class names.
            predictions = [self.categories[idx] for idx in pred_indices.cpu().numpy()]

            if return_probs:
                return predictions, probs.cpu().numpy()
            else:
                return predictions, None

    def predict_single(self,
                      image: torch.Tensor,
                      flow_features: torch.Tensor,
                      sequence_features: Dict[str, torch.Tensor]) -> Dict:
        """Run inference for a single sample.

        Args:
            image: Image tensor [1, H, W].
            flow_features: Flow features [11].
            sequence_features: Dictionary of sequence feature tensors.

        Returns:
            A dictionary containing the predicted class, probabilities, and confidence.
        """
        # Add a batch dimension.
        image = image.unsqueeze(0)
        flow_features = flow_features.unsqueeze(0)
        sequence_features = {
            k: v.unsqueeze(0) for k, v in sequence_features.items()
        }

        predictions, probs = self.predict_batch(
            image, flow_features, sequence_features, return_probs=True
        )

        pred_class = predictions[0]
        pred_idx = self.categories.index(pred_class)
        confidence = float(probs[0, pred_idx])

        # Get the top three predictions.
        top_3_indices = np.argsort(probs[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': self.categories[idx],
                'probability': float(probs[0, idx])
            }
            for idx in top_3_indices
        ]

        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_probabilities': {
                self.categories[i]: float(probs[0, i])
                for i in range(self.num_classes)
            },
            'top_3_predictions': top_3_predictions
        }

    def get_feature_representations(self,
                                   images: torch.Tensor,
                                   flow_features: torch.Tensor,
                                   sequence_features: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Extract intermediate feature representations for visualization.

        Args:
            images: Image tensor.
            flow_features: Flow feature tensor.
            sequence_features: Sequence feature tensors.

        Returns:
            A dictionary of intermediate layer features.
        """
        with torch.no_grad():
            images = images.to(self.device)
            flow_features = flow_features.to(self.device)
            sequence_features = {
                k: v.to(self.device) for k, v in sequence_features.items()
            }

            batch_size = images.size(0)

            # 1. Image features.
            img_features = self.model.image_encoder(images)
            channel_weights = self.model.image_channel_attention(img_features)
            img_features = img_features * channel_weights
            img_features_flat = img_features.view(batch_size, -1)

            # 2. Flow features.
            flow_feat = self.model.flow_encoder(flow_features)
            flow_gate = self.model.flow_gate(flow_feat)
            flow_feat_gated = flow_feat * flow_gate

            # 3. Sequence features.
            sequence_input = torch.stack([
                sequence_features['directions'],
                sequence_features['packet_sizes'],
                sequence_features['relative_times'],
                sequence_features['payloads'],
                sequence_features['flags']
            ], dim=2)

            sequence_output, _ = self.model.sequence_encoder(sequence_input)
            sequence_attn_output, _ = self.model.sequence_attention(
                sequence_output, sequence_output, sequence_output
            )
            sequence_feat = sequence_attn_output[:, -1]
            sequence_feat = self.model.sequence_projection(sequence_feat)

            # 4. Cross-modal features.
            img_seq = img_features_flat.view(batch_size, -1, 128)
            flow_seq = flow_feat_gated.unsqueeze(1)
            seq_seq = sequence_feat.unsqueeze(1)
            all_modalities = torch.cat([img_seq, flow_seq, seq_seq], dim=1)

            cross_attn_output, _ = self.model.cross_modal_attention(
                all_modalities, all_modalities, all_modalities
            )
            cross_modal_feat = cross_attn_output.mean(dim=1)

            # 5. Fused features.
            combined_features = torch.cat([
                img_features_flat,
                flow_feat_gated,
                sequence_feat,
                cross_modal_feat
            ], dim=1)

            combined_features = self.model.fusion_norm(combined_features)

            return {
                'image_features': img_features_flat.cpu().numpy(),
                'flow_features': flow_feat_gated.cpu().numpy(),
                'sequence_features': sequence_feat.cpu().numpy(),
                'cross_modal_features': cross_modal_feat.cpu().numpy(),
                'combined_features': combined_features.cpu().numpy()
            }

    def save_model_info(self, output_path: str):
        """Save model metadata to a JSON file."""
        info = {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'num_classes': self.num_classes,
            'categories': self.categories,
            'image_size': self.image_size,
            'config': {
                'image': self.config['image'],
                'data': {
                    'batch_size': self.config['data']['batch_size'],
                    'dataset_name': self.config['data']['dataset_name']
                }
            }
        }

        with open(output_path, 'w') as f:
            json.dump(info, f, indent=4)

        self.logger.info(f"Model info saved to {output_path}")


def main():
    """Load a model from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Load and test traffic classification model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--save-info', type=str, default=None, help='Save model info to JSON')

    args = parser.parse_args()

    # Load the model.
    loader = TrafficModelLoader(args.model, args.config, args.device)

    # Save model metadata.
    if args.save_info:
        loader.save_model_info(args.save_info)

    print(f"\n✓ Model loaded successfully!")
    print(f"  Device: {loader.device}")
    print(f"  Classes: {loader.categories}")
    print(f"  Num classes: {loader.num_classes}")


if __name__ == '__main__':
    main()
