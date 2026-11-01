#!/usr/bin/env python3
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional
import argparse
import sys
from dataclasses import dataclass, field

from utils.config import load_config
from utils.flow_visualizer import TrafficImageGenerator
from utils.flow_processor import PcapProcessor


@dataclass
class DatasetStats:
    """Dataset statistics."""
    total_flows: int = 0
    total_images: int = 0
    categories: Dict[str, Dict[str, int]] = field(default_factory=lambda: {})


class TrafficClassifier:
    """Coordinate traffic dataset preprocessing."""

    def __init__(self, config_path: str):
        """Initialize the traffic classifier."""
        self.config = load_config(config_path)
        self.image_generator: Optional[TrafficImageGenerator] = None
        self.processor: Optional[PcapProcessor] = None
        self.setup_logging()

    @property
    def processed_dir(self) -> Path:
        """Return the processed data directory."""
        return Path(self.config['data']['processed_dir'])

    @property
    def pcap_dir(self) -> Path:
        """Return the PCAP directory."""
        return Path(self.config['data']['pcap_dir'])

    def setup_logging(self) -> None:
        """Configure logging."""
        log_level = getattr(logging, self.config['logging']['level'].upper())
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = self.config['data']['dataset_name']
        log_file = log_dir / f'traffic_classification_{dataset_name}.log'

        logging.basicConfig(
            level=log_level,
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def initialize_components(self) -> None:
        """Initialize preprocessing components."""
        if self.config['data']['generation']['generate_images']:
            self._init_image_generator()
        self._init_processor()

    def _init_image_generator(self) -> None:
        """Initialize the image generator."""
        self.image_generator = TrafficImageGenerator(
            image_size=tuple(self.config['image']['size']),
            rows_per_packet=self.config['image']['rows_per_packet'],
            header_rows=self.config['image']['header_rows'],
            bytes_per_row=self.config['image']['bytes_per_row'],
            use_color=self.config['image']['use_color'],
            ip_identifier_mode=self.config['image'].get('ip_identifier_mode', 'role'),
            random_seed=self.config['image'].get('random_seed', 0)
        )
        logging.info("Image generator initialized")

    def _init_processor(self) -> None:
        """Initialize the PCAP processor."""
        self.processor = PcapProcessor(
            config=self.config,
            image_generator=self.image_generator
        )
        logging.info("PCAP processor initialized")

    def process_pcap_files(self) -> None:
        """Process PCAP files."""
        if not self.pcap_dir.exists():
            raise FileNotFoundError(f"PCAP directory does not exist: {self.pcap_dir}")

        logging.info(f"Processing PCAP directory: {self.pcap_dir}")
        self.processor.process_directory(self.pcap_dir)
        logging.info("PCAP processing complete!")
        logging.info(f"Results saved to: {self.processed_dir}")

        self._display_dataset_stats()

    def pack_datasets(self) -> None:
        """Pack dataset splits."""
        logging.info("\nPacking datasets...")

        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed data directory does not exist: {self.processed_dir}")

        if not self.processor:
            self._init_processor()

        self.processor.dataset_processor.split_and_save_dataset()

        logging.info("Dataset packing complete!")

    def _collect_dataset_stats(self) -> Dict[str, DatasetStats]:
        """Collect dataset statistics."""
        stats = {}
        for split in ['train', 'val', 'test']:
            if not any(self.processed_dir.glob(f'{split}_packed_*.pkl')):
                continue

            stats[split] = self._collect_split_stats(split)
        return stats

    def _collect_split_stats(self, split: str) -> DatasetStats:
        """Collect statistics from packed files for a dataset split."""
        split_stats = DatasetStats()
        for chunk_path in sorted(self.processed_dir.glob(f'{split}_packed_*.pkl')):
            with chunk_path.open('rb') as file:
                chunk = pickle.load(file)
            for entry in chunk['mapping']:
                category = entry['category']
                counts = split_stats.categories.setdefault(category, {'flows': 0, 'images': 0})
                has_image = entry['flow_id'] in chunk['images']
                counts['flows'] += 1
                counts['images'] += int(has_image)
                split_stats.total_flows += 1
                split_stats.total_images += int(has_image)

        return split_stats

    def _display_dataset_stats(self) -> None:
        """Display dataset statistics."""
        stats = self._collect_dataset_stats()

        for split, split_stats in stats.items():
            logging.info(f"\n{split} dataset statistics:")
            self._display_split_stats(split_stats)

    def _display_split_stats(self, split_stats: DatasetStats) -> None:
        """Display statistics for a single dataset split."""
        for category, counts in split_stats.categories.items():
            logging.info(f"Class {category}:")
            logging.info(f"  - Flows: {counts['flows']}")
            logging.info(f"  - Images: {counts['images']}")

        logging.info(f"\nTotals:")
        logging.info(f"  - Total flows: {split_stats.total_flows}")
        logging.info(f"  - Total images: {split_stats.total_images}")

    def run(self) -> int:
        """Run traffic dataset preprocessing."""
        try:
            return self._execute_processing()
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}", exc_info=True)
            return 1

    def _execute_processing(self) -> int:
        """Execute the preprocessing pipeline."""
        process_pcap = self.config['data']['mode']['process_pcap']
        pack_dataset = self.config['data']['mode']['pack_dataset']

        if not process_pcap and not pack_dataset:
            logging.error("Enable at least one option: PCAP processing or dataset packing")
            return 1

        if process_pcap:
            logging.info("Processing PCAP files...")
            self.initialize_components()
            self.process_pcap_files()

        if pack_dataset and not process_pcap:
            self.pack_datasets()

        return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Traffic Classification Data Processing')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Configuration file path (default: config.yaml)')
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    try:
        classifier = TrafficClassifier(args.config)
        return classifier.run()
    except Exception as e:
        logging.error(f"Program failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
