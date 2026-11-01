#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
from pathlib import Path
import json
import pickle
from PIL import Image
import numpy as np
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
import time
import argparse
from utils.config import load_config
from utils.logging_config import setup_logging
from utils.metrics_logger import MetricsLogger
from utils.reproducibility import configure_training_seed

class MultiModalTrafficDataset(Dataset):
    """Multimodal traffic dataset loaded from packed files."""
    def __init__(self, data_dir: Path, split: str, transform=None, image_size=(40, 40)):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size

        # Load all dataset chunks.
        self.data = self._load_all_chunks()

        # Collect all classes.
        self.categories = sorted(list(set(item['category'] for item in self.data['mapping'])))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        # Count samples per class.
        self.category_counts = {}
        for item in self.data['mapping']:
            category = item['category']
            self.category_counts[category] = self.category_counts.get(category, 0) + 1

        # Log statistics.
        total_samples = len(self.data['mapping'])
        logging.info(f"\n{'-'*50}")
        logging.info(f"Dataset Split: {split}")
        logging.info(f"Total samples: {total_samples}")
        logging.info("\nCategory distribution:")
        for category in self.categories:
            count = self.category_counts[category]
            percentage = (count / total_samples) * 100
            logging.info(f"{category:20s}: {count:5d} samples ({percentage:6.2f}%)")
        logging.info(f"{'-'*50}\n")

    def _load_all_chunks(self):
        """Load and merge all dataset chunks."""
        base_path = self.data_dir
        prefix = f"{self.split}_packed"  # Get the filename prefix without the chunk index.

        # Initialize the merged data structure.
        merged_data = {'mapping': [], 'images': {}, 'flows': {}}

        # Find all chunk files in the directory.
        chunk_files = []
        for file_path in base_path.glob(f"{prefix}_*.pkl"):
            chunk_files.append(file_path)

        # Sort by filename.
        chunk_files.sort()

        if not chunk_files:
            raise RuntimeError(f"No {self.split} chunk files found!")

        logging.info(f"Found {len(chunk_files)} {self.split} chunk files")

        # Load and merge all chunks.
        for chunk_path in chunk_files:
            logging.info(f"Loading chunk: {chunk_path}")
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)

            # Merge data.
            merged_data['mapping'].extend(chunk_data['mapping'])
            merged_data['images'].update(chunk_data['images'])
            merged_data['flows'].update(chunk_data['flows'])

        if not merged_data['mapping']:
            raise RuntimeError("No valid dataset chunks found!")

        return merged_data

    def __len__(self):
        return len(self.data['mapping'])

    def _load_image(self, image_bytes):
        """Load an image with fallback handling."""
        try:
            # Ensure image_bytes is a bytes object.
            if not isinstance(image_bytes, bytes):
                raise ValueError(f"Expected bytes, got {type(image_bytes)}")

            # Create an in-memory file with BytesIO.
            image_buffer = io.BytesIO(image_bytes)

            # Open the image.
            image = Image.open(image_buffer)

            # Convert the image to grayscale.
            if image.mode != 'L':
                image = image.convert('L')

            # Resize the image.
            if image.size != self.image_size:
                image = image.resize(self.image_size, Image.Resampling.LANCZOS)

            # Apply transforms.
            if self.transform:
                image = self.transform(image)
            else:
                # Convert to a tensor by default.
                image = transforms.ToTensor()(image)

            return image

        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            # Return a blank image on failure.
            return torch.zeros((1, *self.image_size))

    def __getitem__(self, idx):
        try:
            # Get the sample mapping.
            item = self.data['mapping'][idx]
            flow_id = item['flow_id']
            category = item['category']

            # 1. Load the image.
            try:
                image_bytes = self.data['images'][flow_id]
                image = self._load_image(image_bytes)
            except Exception as e:
                logging.warning(f"Error loading image for {flow_id}: {str(e)}")
                image = torch.zeros((1, *self.image_size))

            # 2. Load flow data.
            try:
                flow_data = self.data['flows'][flow_id]
            except Exception as e:
                logging.warning(f"Error loading flow data for {flow_id}: {str(e)}")
                flow_data = None

            # 3. Extract flow features.
            flow_features = self._extract_flow_features(flow_data)

            # 4. Extract sequence features.
            sequence_features = self._extract_sequence_features(flow_data)

            # 5. Get the label.
            label = self.category_to_idx[category]

            # 6. Validate data shapes.
            if not isinstance(image, torch.Tensor):
                logging.error(f"Image is not a tensor: {type(image)}")
                image = torch.zeros((1, *self.image_size))
            if image.shape != (1, *self.image_size):
                logging.error(f"Incorrect image shape: {image.shape}")
                image = torch.zeros((1, *self.image_size))

            return {
                'image': image,
                'flow_features': flow_features,
                'sequence_features': sequence_features,
                'label': label
            }

        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            # Return a fallback sample.
            return {
                'image': torch.zeros((1, *self.image_size)),
                'flow_features': torch.zeros(11, dtype=torch.float32),
                'sequence_features': {
                    'directions': torch.zeros(128, dtype=torch.float32),
                    'packet_sizes': torch.zeros(128, dtype=torch.float32),
                    'relative_times': torch.zeros(128, dtype=torch.float32),
                    'payloads': torch.zeros(128, dtype=torch.float32),
                    'flags': torch.zeros(128, dtype=torch.float32)
                },
                'label': 0
            }

    def _extract_flow_features(self, flow_data):
        """Extract statistical flow features from JSON flow records."""
        if flow_data is None:
            return torch.zeros(11, dtype=torch.float32)

        try:
            # Read statistics from flow_info.
            flow_info = flow_data.get('flow_info', {})
            stats = flow_info.get('statistical_info', {})

            # Extract the 11 available features.
            features = [
                # Basic flow features.
                flow_info.get('flow_duration', 0.0),      # Flow duration.
                flow_info.get('packet_count', 0),         # Packet count.
                flow_info.get('total_bytes', 0),          # Total bytes.

                # Packet size statistics.
                stats.get('avg_packet_size', 0.0),        # Mean packet size.
                stats.get('min_packet_size', 0.0),        # Minimum packet size.
                stats.get('max_packet_size', 0.0),        # Maximum packet size.
                stats.get('std_packet_size', 0.0),        # Packet size standard deviation.

                # Inter-arrival time statistics.
                stats.get('avg_iat', 0.0),                # Mean inter-arrival time.
                stats.get('min_iat', 0.0),                # Minimum inter-arrival time.
                stats.get('max_iat', 0.0),                # Maximum inter-arrival time.
                stats.get('std_iat', 0.0)                 # Inter-arrival time standard deviation.
            ]

            # Normalize features.
            norm_features = [
                min(features[0] / 10.0, 10.0),            # Scale flow_duration by 10 and cap at 10.
                min(features[1] / 1000.0, 1.0),          # Scale packet_count by 1000 and cap at 1.
                min(features[2] / 10000.0, 10.0),        # Scale total_bytes by 10000 and cap at 10.
                features[3] / 1500.0,                     # Scale avg_packet_size by the MTU.
                features[4] / 1500.0,                     # Scale min_packet_size by the MTU.
                features[5] / 1500.0,                     # Scale max_packet_size by the MTU.
                features[6] / 1500.0,                     # Scale std_packet_size by the MTU.
                min(features[7] / 1.0, 10.0),            # Cap avg_iat at 10 seconds.
                min(features[8] / 1.0, 10.0),            # Cap min_iat at 10 seconds.
                min(features[9] / 1.0, 10.0),            # Cap max_iat at 10 seconds.
                min(features[10] / 1.0, 10.0)            # Cap std_iat at 10 seconds.
            ]

            return torch.tensor(norm_features, dtype=torch.float32)

        except Exception as e:
            logging.warning(f"Error extracting flow features: {str(e)}")
            return torch.zeros(11, dtype=torch.float32)

    def _extract_sequence_features(self, flow_data):
        """Extract packet direction, size, timing, payload, and flag features."""
        max_seq_len = 128

        if flow_data is None:
            return {
                'directions': torch.zeros(max_seq_len, dtype=torch.float32),
                'packet_sizes': torch.zeros(max_seq_len, dtype=torch.float32),
                'relative_times': torch.zeros(max_seq_len, dtype=torch.float32),
                'payloads': torch.zeros(max_seq_len, dtype=torch.float32),
                'flags': torch.zeros(max_seq_len, dtype=torch.float32)
            }

        try:
            # Get sequence metadata.
            sequence_info = flow_data.get('flow_info', {}).get('sequence_info', {})

            # Extract sequence features and convert them to tensors.
            directions = sequence_info.get('directions', [])
            packet_sizes = sequence_info.get('packet_sizes', [])
            relative_times = sequence_info.get('relative_times', [])
            payloads = sequence_info.get('payloads', [])
            flags = sequence_info.get('flags', [])

            # Convert to tensors and adjust sequence lengths.
            directions = torch.tensor(directions[:max_seq_len], dtype=torch.float32)
            packet_sizes = torch.tensor(packet_sizes[:max_seq_len], dtype=torch.float32)
            relative_times = torch.tensor(relative_times[:max_seq_len], dtype=torch.float32)
            payloads = torch.tensor(payloads[:max_seq_len], dtype=torch.float32)
            flags = torch.tensor(flags[:max_seq_len], dtype=torch.float32)

            # Normalize.
            packet_sizes = packet_sizes / 1500.0  # Normalize by the MTU.
            payloads = payloads / 1500.0  # Normalize by the MTU.
            flags = flags / 255.0  # Normalize TCP flags.

            # Process relative timestamps.
            if len(relative_times) > 1:
                max_time = relative_times[-1]
                relative_times = relative_times / max_time if max_time > 0 else relative_times

            # Pad sequences to a fixed length.
            if len(directions) < max_seq_len:
                pad_length = max_seq_len - len(directions)
                directions = torch.cat([directions, torch.zeros(pad_length)])
                packet_sizes = torch.cat([packet_sizes, torch.zeros(pad_length)])
                relative_times = torch.cat([relative_times, torch.zeros(pad_length)])
                payloads = torch.cat([payloads, torch.zeros(pad_length)])
                flags = torch.cat([flags, torch.zeros(pad_length)])

            return {
                'directions': directions,
                'packet_sizes': packet_sizes,
                'relative_times': relative_times,
                'payloads': payloads,
                'flags': flags
            }

        except Exception as e:
            logging.warning(f"Error extracting sequence features: {str(e)}")
            return {
                'directions': torch.zeros(max_seq_len, dtype=torch.float32),
                'packet_sizes': torch.zeros(max_seq_len, dtype=torch.float32),
                'relative_times': torch.zeros(max_seq_len, dtype=torch.float32),
                'payloads': torch.zeros(max_seq_len, dtype=torch.float32),
                'flags': torch.zeros(max_seq_len, dtype=torch.float32)
            }

class MultiModalTrafficNet(nn.Module):
    """Multimodal traffic classifier combining images, flow statistics, and sequences."""
    def __init__(self, num_classes):
        super(MultiModalTrafficNet, self).__init__()

        # 1. Extract image features with a CNN and channel attention.
        self.image_encoder = nn.Sequential(
            # First layer.
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second layer.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third layer.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Adaptive pooling for a fixed output size.
            nn.AdaptiveAvgPool2d((2, 2))
        )

        # Image channel attention.
        self.image_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid()
        )

        # 2. Encode flow features with gating.
        self.flow_encoder = nn.Sequential(
            nn.Linear(11, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True)
        )

        # Flow feature gating.
        self.flow_gate = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )

        # 3. Encode sequences with a bidirectional GRU and self-attention.
        self.sequence_encoder = nn.GRU(
            input_size=5,  # [direction, size, time, payload, flags]
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # Sequence self-attention.
        self.sequence_attention = nn.MultiheadAttention(
            embed_dim=128,  # Bidirectional GRU output dimension.
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Sequence feature projection.
        self.sequence_projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True)
        )

        # 4. Cross-modal attention.
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 5. Feature fusion and classification.
        # Compute the total feature dimension.
        self.image_features_dim = 128 * 2 * 2  # Image features (512).
        self.flow_features_dim = 128           # Flow features.
        self.sequence_features_dim = 128       # Sequence features.
        self.cross_modal_features_dim = 128    # Cross-modal features.

        total_features = (
            self.image_features_dim +
            self.flow_features_dim +
            self.sequence_features_dim +
            self.cross_modal_features_dim
        )

        # Normalize features before fusion.
        self.fusion_norm = nn.LayerNorm(total_features)

        # Feature fusion network.
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 384),
            nn.LayerNorm(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(192, num_classes)
        )

    def forward(self, image, flow_features, sequence_features):
        batch_size = image.size(0)

        # 1. Extract image features.
        img_features = self.image_encoder(image)  # [batch, 128, 2, 2]

        # Apply channel attention.
        channel_weights = self.image_channel_attention(img_features)
        img_features = img_features * channel_weights

        img_features = img_features.view(batch_size, -1)  # [batch, 512]

        # 2. Encode flow statistics.
        flow_features = self.flow_encoder(flow_features)  # [batch, 128]

        # Apply gating.
        flow_gate = self.flow_gate(flow_features)
        flow_features = flow_features * flow_gate

        # 3. Sequence features.
        # Stack all sequence features.
        sequence_input = torch.stack([
            sequence_features['directions'],
            sequence_features['packet_sizes'],
            sequence_features['relative_times'],
            sequence_features['payloads'],
            sequence_features['flags']
        ], dim=2)  # [batch_size, seq_len, 5]

        # Encode with the GRU.
        sequence_output, _ = self.sequence_encoder(sequence_input)  # [batch, seq_len, 128]

        # Apply self-attention.
        sequence_attn_output, _ = self.sequence_attention(
            sequence_output, sequence_output, sequence_output
        )

        # Take the output at the last time step.
        sequence_features = sequence_attn_output[:, -1]  # [batch, 128]
        sequence_features = self.sequence_projection(sequence_features)  # [batch, 128]

        # 4. Cross-modal attention.
        # Reshape features into sequences.
        img_seq = img_features.view(batch_size, -1, 128)  # [batch, 4, 128]
        flow_seq = flow_features.unsqueeze(1)  # [batch, 1, 128]
        seq_seq = sequence_features.unsqueeze(1)  # [batch, 1, 128]

        # Concatenate sequences from all modalities.
        all_modalities = torch.cat([img_seq, flow_seq, seq_seq], dim=1)  # [batch, 6, 128]

        # Apply cross-modal attention.
        cross_attn_output, _ = self.cross_modal_attention(
            all_modalities, all_modalities, all_modalities
        )

        # Average attention outputs to obtain the final feature vector.
        cross_modal_features = cross_attn_output.mean(dim=1)  # [batch, 128]

        # 5. Feature fusion and classification.
        combined_features = torch.cat([
            img_features,         # [batch, 512]
            flow_features,        # [batch, 128]
            sequence_features,    # [batch, 128]
            cross_modal_features  # [batch, 128]
        ], dim=1)  # [batch, 896]

        # Normalize features.
        combined_features = self.fusion_norm(combined_features)

        # Classify.
        output = self.fusion(combined_features)  # [batch, num_classes]

        return output

def test_model(model_path, config_path=None, data_dir=None):
    """Load a checkpoint and evaluate it on the test set."""
    # Select the matplotlib backend.
    import matplotlib
    matplotlib.use('Agg')

    # Load configuration.
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config('config.yaml')

    # Configure logging.
    logger = setup_logging(
        model_name="MultiModalTrafficNet",
        dataset_name=config['data']['dataset_name'],
        log_dir=config['paths']['log_dir']
    )

    logger.info(f"Testing model: {model_path}")

    try:
        # Select the compute device.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load the model checkpoint.
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        # Get dataset metadata.
        dataset_info = checkpoint.get('dataset_info', {})
        categories = dataset_info.get('categories', [])
        num_classes = dataset_info.get('num_classes', len(categories))

        if not categories:
            logger.error("No categories found in checkpoint!")
            return

        logger.info(f"Model was trained on {num_classes} classes: {categories}")

        # Image transforms.
        image_size = tuple(config['image']['size'])
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Create the test dataset.
        if data_dir:
            test_data_dir = Path(data_dir)
        else:
            test_data_dir = Path(config['data']['processed_dir'])

        logger.info(f"Loading test dataset from: {test_data_dir}")

        # Create the test dataset.
        test_dataset = MultiModalTrafficDataset(
            test_data_dir,
            'test',
            transform=transform,
            image_size=image_size
        )

        # Check that classes match.
        if set(test_dataset.categories) != set(categories):
            logger.warning("Categories mismatch between model and dataset!")
            logger.warning(f"Model categories: {categories}")
            logger.warning(f"Dataset categories: {test_dataset.categories}")

            # Handle class mismatches.
            model_categories_set = set(categories)
            dataset_categories_set = set(test_dataset.categories)

            # Find classes present only in the dataset.
            new_categories = dataset_categories_set - model_categories_set
            # Find classes present only in the model.
            missing_categories = model_categories_set - dataset_categories_set

            if new_categories:
                logger.warning(f"New categories in dataset (will be filtered out): {new_categories}")
            if missing_categories:
                logger.warning(f"Missing categories in dataset: {missing_categories}")

            # Keep only classes supported by the model.
            valid_categories = list(model_categories_set.intersection(dataset_categories_set))
            valid_categories.sort()  # Preserve class order.

            if len(valid_categories) == 0:
                logger.error("No common categories between model and dataset! Cannot proceed.")
                return

            logger.info(f"Using {len(valid_categories)} common categories: {valid_categories}")

            # Create a filtered dataset.
            filtered_indices = []
            for idx, item in enumerate(test_dataset.data['mapping']):
                if item['category'] in valid_categories:
                    filtered_indices.append(idx)

            if len(filtered_indices) == 0:
                logger.error("No valid samples found after filtering! Cannot proceed.")
                return

            logger.info(f"Filtered dataset: {len(filtered_indices)} samples out of {len(test_dataset)}")

            # Create a filtered dataset.
            class FilteredDataset(Dataset):
                def __init__(self, original_dataset, valid_indices, valid_categories):
                    self.original_dataset = original_dataset
                    self.valid_indices = valid_indices
                    self.valid_categories = valid_categories
                    self.category_to_idx = {cat: idx for idx, cat in enumerate(sorted(valid_categories))}
                    self.categories = sorted(valid_categories)

                    # Count samples after filtering.
                    self.category_counts = {}
                    for idx in valid_indices:
                        category = original_dataset.data['mapping'][idx]['category']
                        self.category_counts[category] = self.category_counts.get(category, 0) + 1

                def __len__(self):
                    return len(self.valid_indices)

                def __getitem__(self, idx):
                    original_idx = self.valid_indices[idx]
                    item = self.original_dataset[original_idx]

                    # Remap labels to the new class indices.
                    original_category = self.original_dataset.data['mapping'][original_idx]['category']
                    new_label = self.category_to_idx[original_category]

                    return {
                        'image': item['image'],
                        'flow_features': item['flow_features'],
                        'sequence_features': item['sequence_features'],
                        'label': new_label
                    }

            # Create a filtered dataset.
            test_dataset = FilteredDataset(test_dataset, filtered_indices, valid_categories)
            categories = valid_categories  # Update the class list.

        # Create the test data loader.
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )

        # Initialize the model.
        model = MultiModalTrafficNet(num_classes=len(categories))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully")

        # Loss function.
        criterion = nn.CrossEntropyLoss()

        # Evaluate on the test set.
        logger.info("Evaluating on test set...")
        test_metrics = evaluate(model, test_loader, criterion, device, generate_cm=True)

        # Log detailed test results.
        logger.info("\n" + "="*60)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Overall Metrics:")
        logger.info(f"Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Recall: {test_metrics['recall']:.4f}")
        logger.info(f"F1-Score: {test_metrics['f1']:.4f}")

        logger.info("\nPer-Class Metrics:")
        logger.info("-" * 60)
        logger.info(f"{'Category':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        logger.info("-" * 60)

        for i, category in enumerate(categories):
            precision = test_metrics['class_metrics']['precision'][i]
            recall = test_metrics['class_metrics']['recall'][i]
            f1 = test_metrics['class_metrics']['f1'][i]
            logger.info(f"{category:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")

        logger.info("-" * 60)

        # Save evaluation results.
        model_dir = Path(model_path).parent
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save the confusion matrix.
        cm_path = model_dir / f'confusion_matrix_{timestamp}.png'
        plt.savefig(cm_path)
        plt.close('all')
        logger.info(f"Confusion matrix saved to: {cm_path}")

        # Save detailed evaluation results.
        def convert_to_native_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [float(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native_types(x) for x in obj]
            return obj

        eval_results = {
            'model_path': str(model_path),
            'test_metrics': convert_to_native_types(test_metrics),
            'class_metrics': {
                category: {
                    'precision': float(test_metrics['class_metrics']['precision'][i]),
                    'recall': float(test_metrics['class_metrics']['recall'][i]),
                    'f1': float(test_metrics['class_metrics']['f1'][i])
                }
                for i, category in enumerate(categories)
            },
            'timestamp': timestamp,
            'dataset_path': str(test_data_dir),
            'categories': categories
        }

        # Write through a temporary file.
        eval_path = model_dir / f'evaluation_results_{timestamp}.json'
        temp_eval_path = model_dir / f'temp_evaluation_results_{timestamp}.json'

        try:
            with open(temp_eval_path, 'w') as f:
                json.dump(eval_results, f, indent=4)

            # Replace the original file after a successful write.
            if temp_eval_path.exists():
                if eval_path.exists():
                    eval_path.unlink()  # Remove the existing file.
                temp_eval_path.rename(eval_path)
                logger.info(f"Evaluation results saved to: {eval_path}")
        except Exception as save_error:
            logger.error(f"Error saving evaluation results: {str(save_error)}")
            if temp_eval_path.exists():
                try:
                    temp_eval_path.unlink()  # Clean up the temporary file.
                except:
                    pass

        logger.info("="*60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Error during model testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Release resources.
        plt.close('all')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    with tqdm(train_loader, desc="Training") as pbar:
        for batch in pbar:
            # Get batch data.
            images = batch['image'].to(device)
            flow_features = batch['flow_features'].to(device)
            sequence_features = {
                'directions': batch['sequence_features']['directions'].to(device),
                'packet_sizes': batch['sequence_features']['packet_sizes'].to(device),
                'relative_times': batch['sequence_features']['relative_times'].to(device),
                'payloads': batch['sequence_features']['payloads'].to(device),
                'flags': batch['sequence_features']['flags'].to(device)
            }
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(
                image=images,
                flow_features=flow_features,
                sequence_features=sequence_features
            )
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute aggregate metrics.
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='weighted',
        zero_division=0
    )

    # Compute per-class metrics.
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        zero_division=0
    )

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': {
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1
        }
    }

def evaluate(model, val_loader, criterion, device, generate_cm=False):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            # Get batch data.
            images = batch['image'].to(device)
            flow_features = batch['flow_features'].to(device)
            sequence_features = {
                'directions': batch['sequence_features']['directions'].to(device),
                'packet_sizes': batch['sequence_features']['packet_sizes'].to(device),
                'relative_times': batch['sequence_features']['relative_times'].to(device),
                'payloads': batch['sequence_features']['payloads'].to(device),
                'flags': batch['sequence_features']['flags'].to(device)
            }
            labels = batch['label'].to(device)

            outputs = model(
                image=images,
                flow_features=flow_features,
                sequence_features=sequence_features
            )
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute aggregate metrics.
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='weighted',
        zero_division=0
    )

    # Compute per-class metrics.
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        zero_division=0
    )

    # Generate the confusion matrix only when requested.
    if generate_cm:
        try:
            # Compute the confusion matrix.
            cm = confusion_matrix(all_labels, all_preds)
            class_totals = cm.sum(axis=1)
            cm_percentage = np.nan_to_num(cm / class_totals[:, np.newaxis] * 100, 0)

            # Create a confusion matrix heatmap.
            plt.figure(figsize=(12, 8))
            annot = np.empty_like(cm, dtype='<U20')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    percentage = cm_percentage[i, j]
                    annot[i, j] = f'{int(count)}\n({percentage:.1f}%)'

            # Get the corresponding class labels.
            if hasattr(val_loader.dataset, 'valid_categories'):
                # Filtered dataset.
                category_labels = val_loader.dataset.valid_categories
            else:
                # Original dataset.
                category_labels = val_loader.dataset.categories

            sns.heatmap(cm_percentage,
                        annot=annot,
                        fmt='',
                        cmap='Blues',
                        xticklabels=category_labels,
                        yticklabels=category_labels,
                        annot_kws={'size': 10, 'weight': 'bold'},
                        vmin=0,
                        vmax=100)

            plt.title('Confusion Matrix\n(Count and Percentage of True Label)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            # Save the confusion matrix image.
            cm_path = Path('confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close('all')

            if wandb is not None and wandb.run is not None:
                wandb.log({"confusion_matrix": wandb.Image(str(cm_path))})

        except Exception as e:
            logging.error(f"Error generating confusion matrix: {str(e)}")
            plt.close('all')

    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': {
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1
        }
    }

def main():
    """Main training function"""
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='MultiModal Traffic Classification Training/Testing')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Execution mode: train or test')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to the model checkpoint (test mode only)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Dataset directory (test mode only; defaults to the configured path)')

    parser.add_argument('--seed', type=int, default=None,
                       help='Training seed (overrides training.seed; paper seeds: 17, 29, 43, 71, 97)')

    args = parser.parse_args()

    # Run checkpoint evaluation in test mode.
    if args.mode == 'test':
        if not args.model:
            print("Error: test mode requires --model")
            return

        if not os.path.exists(args.model):
            print(f"Error: model file does not exist: {args.model}")
            return

        print(f"Testing model: {args.model}")
        test_model(args.model, args.config, args.data_dir)
        return

    # Train the model.
    # Use offline mode for Weights & Biases.
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"

    # Initialize the run variable.
    run = None

    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        logging.warning("wandb not found. Running without logging to Weights & Biases.")

    # Record the start time.
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    start_timestamp = time.time()


    # Select the matplotlib backend.
    import matplotlib
    matplotlib.use('Agg')

    # Load configuration.
    config = load_config(args.config)
    training_seed = configure_training_seed(config, args.seed)

    # Configure logging.
    logger = setup_logging(
        model_name="MultiModalTrafficNet",
        dataset_name=config['data']['dataset_name'],
        log_dir=config['paths']['log_dir'],
        seed=training_seed
    )

    logger.info(f"Training started at: {start_time}")

    try:
        # Create output directories.
        model_dir = Path(config['paths']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)

        # Build an experiment ID from the model, dataset, and timestamp.
        model_name = "MultiModalTrafficNet"
        dataset_name = Path(config['data']['processed_dir']).name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{model_name}_{dataset_name}_seed_{training_seed}_{timestamp}"
        experiment_dir = model_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the metrics logger.
        metrics_logger = MetricsLogger(
            base_dir=Path(config['paths']['log_dir']) / 'metrics',
            model_name=model_name,
            dataset_name=dataset_name,
            seed=training_seed
        )

        # Save the full configuration in the experiment directory.
        config_path = experiment_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # Select the compute device.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Image transforms.
        image_size = tuple(config['image']['size'])
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Create datasets.
        data_dir = Path(config['data']['processed_dir'])

        logger.info("\nInitializing datasets...")

        # Create the training set and count samples.
        train_dataset = MultiModalTrafficDataset(data_dir, 'train', transform=transform, image_size=image_size)

        # Create the validation set and count samples.
        val_dataset = MultiModalTrafficDataset(data_dir, 'val', transform=transform, image_size=image_size)

        # Create the test set and count samples.
        test_dataset = MultiModalTrafficDataset(data_dir, 'test', transform=transform, image_size=image_size)

        # Check class consistency across train, validation, and test splits.
        train_categories = set(train_dataset.categories)
        val_categories = set(val_dataset.categories)
        test_categories = set(test_dataset.categories)

        logger.info("\nChecking category consistency across splits...")
        logger.info(f"Train categories: {len(train_categories)}")
        logger.info(f"Val categories: {len(val_categories)}")
        logger.info(f"Test categories: {len(test_categories)}")

        # Check that class sets match exactly.
        if train_categories != val_categories or train_categories != test_categories:
            logger.warning("Categories mismatch detected between train/val/test sets!")

            # Find classes that differ between splits.
            train_only = train_categories - val_categories - test_categories
            val_only = val_categories - train_categories - test_categories
            test_only = test_categories - train_categories - val_categories

            if train_only:
                logger.warning(f"Categories only in train set: {train_only}")
            if val_only:
                logger.warning(f"Categories only in val set: {val_only}")
            if test_only:
                logger.warning(f"Categories only in test set: {test_only}")

            # Find classes shared by all splits.
            common_categories = train_categories.intersection(val_categories).intersection(test_categories)

            if len(common_categories) == 0:
                logger.error("No common categories found across all splits! Cannot proceed with training.")
                return

            logger.info(f"Using {len(common_categories)} common categories: {sorted(common_categories)}")

            # Filter datasets to keep only shared classes.
            def create_filtered_dataset(original_dataset, valid_categories):
                filtered_indices = []
                for idx, item in enumerate(original_dataset.data['mapping']):
                    if item['category'] in valid_categories:
                        filtered_indices.append(idx)

                if len(filtered_indices) == 0:
                    raise RuntimeError(f"No valid samples found in {original_dataset.split} set after filtering!")

                logger.info(f"Filtered {original_dataset.split} set: {len(filtered_indices)} samples out of {len(original_dataset)}")

                class FilteredDataset(Dataset):
                    def __init__(self, original_dataset, valid_indices, valid_categories):
                        self.original_dataset = original_dataset
                        self.valid_indices = valid_indices
                        self.valid_categories = valid_categories
                        self.category_to_idx = {cat: idx for idx, cat in enumerate(sorted(valid_categories))}
                        self.categories = sorted(valid_categories)

                        # Count samples after filtering.
                        self.category_counts = {}
                        for idx in valid_indices:
                            category = original_dataset.data['mapping'][idx]['category']
                            self.category_counts[category] = self.category_counts.get(category, 0) + 1

                    def __len__(self):
                        return len(self.valid_indices)

                    def __getitem__(self, idx):
                        original_idx = self.valid_indices[idx]
                        item = self.original_dataset[original_idx]

                        # Remap labels to the new class indices.
                        original_category = self.original_dataset.data['mapping'][original_idx]['category']
                        new_label = self.category_to_idx[original_category]

                        return {
                            'image': item['image'],
                            'flow_features': item['flow_features'],
                            'sequence_features': item['sequence_features'],
                            'label': new_label
                        }

                return FilteredDataset(original_dataset, filtered_indices, valid_categories)

            # Create a filtered dataset.
            train_dataset = create_filtered_dataset(train_dataset, common_categories)
            val_dataset = create_filtered_dataset(val_dataset, common_categories)
            test_dataset = create_filtered_dataset(test_dataset, common_categories)

            logger.info("Successfully created filtered datasets with consistent categories.")
        else:
            logger.info("All splits have consistent categories. Proceeding with original datasets.")

        # Log aggregate statistics.
        logger.info("\nTotal Dataset Statistics:")
        logger.info(f"{'='*60}")
        logger.info(f"{'Category':20s} {'Train':>10s} {'Val':>10s} {'Test':>10s} {'Total':>10s}")
        logger.info(f"{'-'*60}")

        for category in train_dataset.categories:
            train_count = train_dataset.category_counts.get(category, 0)
            val_count = val_dataset.category_counts.get(category, 0)
            test_count = test_dataset.category_counts.get(category, 0)
            total_count = train_count + val_count + test_count

            logger.info(f"{category:20s} {train_count:10d} {val_count:10d} {test_count:10d} {total_count:10d}")

        logger.info(f"{'-'*60}")
        train_total = len(train_dataset)
        val_total = len(val_dataset)
        test_total = len(test_dataset)
        total = train_total + val_total + test_total
        logger.info(f"{'Total':20s} {train_total:10d} {val_total:10d} {test_total:10d} {total:10d}")
        logger.info(f"{'Percentage':20s} {train_total/total*100:9.1f}% {val_total/total*100:9.1f}% {test_total/total*100:9.1f}% {100:9.1f}%")
        logger.info(f"{'='*60}\n")

        # Create data loaders.
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )

        # Initialize the model.
        model = MultiModalTrafficNet(
            num_classes=len(train_dataset.categories)
        )
        model = model.to(device)

        # Log model information.
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model initialized with {len(train_dataset.categories)} classes")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

        # Loss function and optimizer.
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Use separate learning rates for different model components.
        param_groups = [
            {'params': model.image_encoder.parameters(), 'lr': config['training']['learning_rate'] * 0.1},
            {'params': model.flow_encoder.parameters(), 'lr': config['training']['learning_rate']},
            {'params': model.sequence_encoder.parameters(), 'lr': config['training']['learning_rate']},
            {'params': model.fusion.parameters(), 'lr': config['training']['learning_rate'] * 1.5}
        ]

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )

        # Use the OneCycleLR scheduler.
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group['lr'] for group in param_groups],
            epochs=config['training']['max_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )

        # Initialize Weights & Biases if available.
        if WANDB_AVAILABLE:
            try:
                run = wandb.init(
                    project="traffic-classification-multimodal",
                    name=experiment_id,
                    config={
                        "architecture": "MultiModalTrafficNet",
                        "dataset": "traffic_multimodal",
                        "learning_rate": config['training']['learning_rate'],
                        "batch_size": config['data']['batch_size'],
                        "num_classes": len(train_dataset.categories),
                        "optimizer": "AdamW",
                        "scheduler": "OneCycleLR",
                        "label_smoothing": 0.1
                    },
                    mode="offline"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                WANDB_AVAILABLE = False

        # Training loop.
        best_val_acc = 0.0

        try:
            for epoch in range(config['training']['max_epochs']):
                logger.info(f"\nEpoch {epoch+1}/{config['training']['max_epochs']}")

                # Train.
                train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

                # Validate.
                val_metrics = evaluate(model, val_loader, criterion, device)

                # Record training and validation metrics.
                metrics_logger.log_training_metrics(epoch, train_metrics, val_metrics)

                # Update the learning rate.
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                # Log training metrics.
                logger.info(f"\nTraining Metrics:")
                logger.info(f"Loss: {train_metrics['loss']:.4f}")
                logger.info(f"Accuracy: {train_metrics['accuracy']:.4f}")
                logger.info(f"Precision: {train_metrics['precision']:.4f}")
                logger.info(f"Recall: {train_metrics['recall']:.4f}")
                logger.info(f"F1-Score: {train_metrics['f1']:.4f}")

                # Log validation metrics.
                logger.info(f"\nValidation Metrics:")
                logger.info(f"Loss: {val_metrics['loss']:.4f}")
                logger.info(f"Accuracy: {val_metrics['accuracy']:.4f}")
                logger.info(f"Precision: {val_metrics['precision']:.4f}")
                logger.info(f"Recall: {val_metrics['recall']:.4f}")
                logger.info(f"F1-Score: {val_metrics['f1']:.4f}")

                # Log the learning rate.
                logger.info(f"Learning Rate: {current_lr:.6f}")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

                # Log to Weights & Biases.
                if run is not None:
                    wandb.log({
                        "train_loss": train_metrics['loss'],
                        "train_accuracy": train_metrics['accuracy'],
                        "train_precision": train_metrics['precision'],
                        "train_recall": train_metrics['recall'],
                        "train_f1": train_metrics['f1'],
                        "val_loss": val_metrics['loss'],
                        "val_accuracy": val_metrics['accuracy'],
                        "val_precision": val_metrics['precision'],
                        "val_recall": val_metrics['recall'],
                        "val_f1": val_metrics['f1'],
                        "learning_rate": current_lr
                    })

                # Save the best model.
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']

                    # Save the model and metadata.
                    model_path = experiment_dir / 'best_multimodal_model.pth'
                    categories_path = experiment_dir / 'categories.json'
                    temp_model_path = experiment_dir / 'temp_best_multimodal_model.pth'

                    try:
                        # Convert NumPy values to native Python types for serialization.
                        save_dict = {
                            'epoch': int(epoch),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'val_accuracy': float(val_metrics['accuracy']),
                            'val_loss': float(val_metrics['loss']),
                            'config': {k: (float(v) if isinstance(v, (np.float32, np.float64)) else
                                          (int(v) if isinstance(v, (np.int32, np.int64)) else v))
                                      for k, v in config.items()},
                            'dataset_info': {
                                'name': dataset_name,
                                'num_classes': len(train_dataset.categories),
                                'categories': train_dataset.categories
                            },
                            'training_info': {
                                'seed': training_seed,
                                'split_seed': config['data'].get('split_seed', 128),
                                'batch_size': config['data']['batch_size'],
                                'learning_rate': config['training']['learning_rate'],
                                'max_epochs': config['training']['max_epochs'],
                                'weight_decay': config['training']['weight_decay']
                            }
                        }

                        torch.save(save_dict, str(temp_model_path))

                        # Rename the file after a successful save.
                        if temp_model_path.exists():
                            if model_path.exists():
                                model_path.unlink()  # Remove the existing file first.
                            temp_model_path.rename(model_path)

                        # Save the class mapping.
                        with open(categories_path, 'w') as f:
                            json.dump(train_dataset.categories, f)

                        logger.info(f"Saved best model with validation accuracy: {val_metrics['accuracy']:.4f}")
                    except Exception as save_error:
                        logger.error(f"Error saving model: {str(save_error)}")
                        if temp_model_path.exists():
                            try:
                                temp_model_path.unlink()  # Remove the temporary file on failure.
                            except:
                                pass
                else:
                    pass

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save a checkpoint.
            checkpoint_path = experiment_dir / 'interrupted_checkpoint.pth'
            temp_checkpoint_path = experiment_dir / 'temp_interrupted_checkpoint.pth'

            try:
                save_dict = {
                    'epoch': int(epoch),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': float(best_val_acc),
                    'config': {k: (float(v) if isinstance(v, (np.float32, np.float64)) else
                                  (int(v) if isinstance(v, (np.int32, np.int64)) else v))
                              for k, v in config.items()}
                }

                torch.save(save_dict, str(temp_checkpoint_path))

                if temp_checkpoint_path.exists():
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    temp_checkpoint_path.rename(checkpoint_path)
                    logger.info(f"Saved interrupt checkpoint to {checkpoint_path}")
                else:
                    logger.error("Failed to save interrupt checkpoint")
            except Exception as save_error:
                logger.error(f"Error saving interrupt checkpoint: {str(save_error)}")
                if temp_checkpoint_path.exists():
                    try:
                        temp_checkpoint_path.unlink()
                    except:
                        pass

        finally:
            # Run final evaluation on the test set.
            logger.info("Performing final evaluation on test set...")
            try:
                # Load the best model.
                best_model_path = experiment_dir / 'best_multimodal_model.pth'
                if best_model_path.exists():
                    checkpoint = torch.load(str(best_model_path), map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])

                # Evaluate the test set.
                test_metrics = evaluate(model, test_loader, criterion, device, generate_cm=True)

                # Record test metrics.
                metrics_logger.log_test_metrics(test_metrics)

                # Log detailed test results.
                logger.info("\nTest Set Results:")
                logger.info(f"Overall Metrics:")
                logger.info(f"Loss: {test_metrics['loss']:.4f}")
                logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
                logger.info(f"Precision: {test_metrics['precision']:.4f}")
                logger.info(f"Recall: {test_metrics['recall']:.4f}")
                logger.info(f"F1-Score: {test_metrics['f1']:.4f}")

                logger.info("\nPer-Class Metrics:")
                for i, category in enumerate(test_loader.dataset.categories):
                    logger.info(f"\n{category}:")
                    logger.info(f"Precision: {test_metrics['class_metrics']['precision'][i]:.4f}")
                    logger.info(f"Recall: {test_metrics['class_metrics']['recall'][i]:.4f}")
                    logger.info(f"F1-Score: {test_metrics['class_metrics']['f1'][i]:.4f}")

                # Log to Weights & Biases.
                if run is not None:
                    wandb.log({
                        "test_loss": test_metrics['loss'],
                        "test_accuracy": test_metrics['accuracy'],
                        "test_precision": test_metrics['precision'],
                        "test_recall": test_metrics['recall'],
                        "test_f1": test_metrics['f1']
                    })

                    # Record per-class metrics.
                    for i, category in enumerate(test_loader.dataset.categories):
                        wandb.log({
                            f"test_{category}_precision": test_metrics['class_metrics']['precision'][i],
                            f"test_{category}_recall": test_metrics['class_metrics']['recall'][i],
                            f"test_{category}_f1": test_metrics['class_metrics']['f1'][i]
                        })

                # Save the confusion matrix image.
                cm_path = experiment_dir / 'confusion_matrix.png'
                plt.savefig(cm_path)
                plt.close('all')

                # Save detailed evaluation results.
                def convert_to_native_types(obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return [float(x) for x in obj]
                    elif isinstance(obj, dict):
                        return {k: convert_to_native_types(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_to_native_types(x) for x in obj]
                    return obj

                eval_results = {
                    'test_metrics': convert_to_native_types(test_metrics),
                    'class_metrics': {
                        category: {
                            'precision': float(test_metrics['class_metrics']['precision'][i]),
                            'recall': float(test_metrics['class_metrics']['recall'][i]),
                            'f1': float(test_metrics['class_metrics']['f1'][i])
                        }
                        for i, category in enumerate(test_loader.dataset.categories)
                    },
                    'timestamp': timestamp,
                    'dataset': dataset_name,
                    'config': convert_to_native_types(config)
                }

                # Write through a temporary file.
                eval_path = experiment_dir / 'evaluation_results.json'
                temp_eval_path = experiment_dir / 'temp_evaluation_results.json'

                try:
                    with open(temp_eval_path, 'w') as f:
                        json.dump(eval_results, f, indent=4)

                    # Replace the original file after a successful write.
                    if temp_eval_path.exists():
                        if eval_path.exists():
                            eval_path.unlink()  # Remove the existing file.
                        temp_eval_path.rename(eval_path)
                        logger.info(f"Successfully saved evaluation results to {eval_path}")
                except Exception as save_error:
                    logger.error(f"Error saving evaluation results: {str(save_error)}")
                    if temp_eval_path.exists():
                        try:
                            temp_eval_path.unlink()  # Clean up the temporary file.
                        except:
                            pass

            except Exception as e:
                logger.error(f"Error during test set evaluation: {str(e)}")

            # Release resources.
            plt.close('all')
            if run is not None:
                run.finish()

            # Shut down data loaders.
            if 'train_loader' in locals():
                train_loader._iterator = None
            if 'val_loader' in locals():
                val_loader._iterator = None
            if 'test_loader' in locals():
                test_loader._iterator = None

            # Release GPU memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        # Record the end time and total runtime.
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        end_timestamp = time.time()
        total_time = end_timestamp - start_timestamp
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        logger.info(f"\nTraining ended at: {end_time}")
        logger.info(f"Total training time: {hours}h {minutes}m {seconds}s")

        # Add timing information to the metrics.
        if 'metrics_logger' in locals():
            metrics_logger.metrics_data.update({
                "training_start_time": start_time,
                "training_end_time": end_time,
                "total_training_time": f"{hours}h {minutes}m {seconds}s"
            })
            metrics_logger._save_metrics()

        # Release resources.
        plt.close('all')
        if run is not None:
            run.finish()
        if 'train_loader' in locals():
            train_loader._iterator = None
        if 'val_loader' in locals():
            val_loader._iterator = None
        if 'test_loader' in locals():
            test_loader._iterator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
