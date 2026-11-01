from typing import List, Dict, Optional, Union
import json
import os
import logging
from tqdm import tqdm

from pathlib import Path
from utils.flow_visualizer import TrafficImageGenerator

from utils.pcap_reader import PcapReader
from utils.flow_analyzer import FlowAnalyzer
from utils.dataset_processor import DatasetProcessor



class IPLabelManager:
    """Manage IP-based class labels."""
    def __init__(self, label_file: str):
        """Initialize the IP label manager.

        Args:
            label_file: Path to the IP-to-label mapping file.
        """
        self.ip_labels = {}
        self.ip_pair_labels = {}
        self._load_label_file(label_file)

    def _load_label_file(self, label_file: str):
        """Load an IP-to-label mapping file and detect its format."""
        if not label_file or not os.path.exists(label_file):
            raise FileNotFoundError(f"IP label file does not exist: {label_file}")

        with open(label_file, 'r') as f:
            data = json.load(f)

        # Treat keys containing '->' or ',' as IP pairs.
        for key, label in data.items():
            if '->' in key or ',' in key:
                # Parse IP pair keys.
                if '->' in key:
                    src_ip, dst_ip = map(str.strip, key.split('->'))
                else:
                    src_ip, dst_ip = map(str.strip, key.split(','))
                # Store both directions of each IP pair.
                self.ip_pair_labels[(src_ip, dst_ip)] = label
                self.ip_pair_labels[(dst_ip, src_ip)] = label  # Add the reverse pair.
            else:
                # Single-IP labels.
                self.ip_labels[key.strip()] = label

    def get_label(self, src_ip: str, dst_ip: str) -> Optional[str]:
        """Find a label for an IP pair in either direction."""
        # Look up the IP pair; both directions were stored during loading.
        pair_label = self.ip_pair_labels.get((src_ip, dst_ip))
        if pair_label:
            return pair_label

        # Fall back to individual IP labels.
        return self.ip_labels.get(src_ip) or self.ip_labels.get(dst_ip)


class PcapProcessor:
    """Process PCAP files."""
    def __init__(
        self,
        config: Dict,
        image_generator: Optional[TrafficImageGenerator] = None
    ):
        # Read configuration values.
        self.output_dir = Path(config['data']['processed_dir'])
        self.split_ratio = config['data']['split_ratio']
        self.batch_size = config['data']['batch_size']
        self.pack_dataset = config['data']['mode']['pack_dataset']
        self.chunk_size = config['data']['chunk_size']
        self.split_seed = config['data'].get('split_seed', 0)

        # Labeling settings.
        self.label_mode = config['data']['labeling']['mode']
        self.ip_label_manager = None
        if self.label_mode == 'ip':
            self.ip_label_manager = IPLabelManager(
                config['data']['labeling']['ip_label_file']
            )

        # Data generation options.
        self.generation_config = config['data']['generation']

        # Flow processing settings.
        self.flow_config = config['flow']
        self._pcap_root: Optional[Path] = None

        # Initialize components.
        self.image_generator = image_generator
        self.pcap_reader = PcapReader()
        self.flow_analyzer = FlowAnalyzer(self.flow_config)
        self.dataset_processor = DatasetProcessor(
            self.output_dir,
            self.split_ratio,
            self.chunk_size,
            self.pack_dataset,
            random_seed=self.split_seed
        )

        # Processing error counts.
        self.error_files = {
            'no_packets': [],    # Files with no readable packets.
            'no_valid_packets': [], # Files with no valid labeled flows.
            'no_flows': [],      # Files with no valid flows.
            'parse_error': []    # Files with parsing errors.
        }

        # Ensure the output directory exists.
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_flow_category(self, pcap_path: Path, first_packet: Dict) -> str:
        """Determine the traffic class.

        Args:
            pcap_path: Path to the PCAP file.
            first_packet: First packet of the flow.

        Returns:
            The traffic class.
        """
        if self.label_mode == 'directory':
            # Use the parent directory name as the class.
            return pcap_path.parent.name
        else:
            # Use IP labels.
            src_ip = first_packet['src_ip']
            dst_ip = first_packet['dst_ip']
            label = self.ip_label_manager.get_label(src_ip, dst_ip)
            if not label:
                logging.debug(f"No IP label found: {src_ip} -> {dst_ip}")
                return 'unknown'
            return label

    def process_directory(self, pcap_dir: Union[str, Path]):
        """Process a directory of PCAP files in batches to limit memory use."""
        pcap_dir = Path(pcap_dir)
        if not pcap_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {pcap_dir}")
        self._pcap_root = pcap_dir.resolve()

        # Collect all PCAP files.
        pcap_files = []
        for ext in ['*.pcap', '*.pcapng']:
            pcap_files.extend(pcap_dir.rglob(ext))
        pcap_files.sort()

        if not pcap_files:
            raise ValueError(f"No PCAP files found in {pcap_dir}")

        logging.info(f"Found {len(pcap_files)} PCAP files")

        # Process files through a generator.
        def flow_generator():
            for pcap_file in tqdm(pcap_files, desc="Processing PCAP files"):
                for flow in self.process_pcap(pcap_file):
                    if flow:  # Keep valid flows only.
                        yield flow

        # Process flows in batches.
        batch_size = self.batch_size
        current_batch = []

        for flow in flow_generator():
            current_batch.append(flow)

            if len(current_batch) >= batch_size:
                self._process_and_save_batch(current_batch)
                current_batch = []

        # Process remaining flows.
        if current_batch:
            self._process_and_save_batch(current_batch)

        # Split and save the dataset.
        self.dataset_processor.split_and_save_dataset()

        # Log processing error statistics.
        self._print_error_summary()

        logging.info("Processing complete!")

    def _print_error_summary(self):
        """Log a summary of files with processing errors."""
        logging.info("\n=== Files with processing errors ===")

        if not any(self.error_files.values()):
            logging.info("All files processed successfully!")
            return

        for error_type, files in self.error_files.items():
            if files:
                logging.warning(f"\nFiles with {error_type} errors:")
                for f in files:
                    logging.warning(f"- {f}")
                logging.warning(f"Total: {len(files)} files")

    def _process_and_save_batch(self, flows: List[Dict]):
        """Process and save a batch of flows."""
        self.dataset_processor.process_and_save_batch(
            flows,
            self.image_generator,
            generate_flows=self.generation_config['generate_flows'],
            generate_images=self.generation_config['generate_images']
        )

    def process_pcap(self, pcap_path: Union[str, Path]) -> List[Dict]:
        """Process a single PCAP file."""
        try:
            pcap_path = Path(pcap_path)

            # 1. Read and parse the PCAP file.
            processed_packets = self.pcap_reader.read_pcap_file(pcap_path)

            if not processed_packets:
                logging.warning(f"No valid packets read from PCAP file: {pcap_path}")
                self.error_files['no_packets'].append(str(pcap_path))
                return []

            # 2. Analyze packets and group them into flows.
            processed_flows = self.flow_analyzer.analyze_flows(processed_packets)

            if not processed_flows:
                logging.warning(f"No valid flows extracted from packets: {pcap_path}")
                self.error_files['no_flows'].append(str(pcap_path))
                return []

            # 3. Assign a class label to each flow.
            labeled_flows = []
            for flow_index, flow in enumerate(processed_flows):
                if flow and flow['packets']:
                    # Store the relative capture path and flow index to distinguish recurring five-tuples.
                    resolved_pcap = pcap_path.resolve()
                    if self._pcap_root is not None:
                        source_pcap = resolved_pcap.relative_to(self._pcap_root).as_posix()
                    else:
                        source_pcap = resolved_pcap.as_posix()
                    flow['_source_pcap'] = source_pcap
                    flow['_flow_index'] = flow_index
                    # Determine the traffic class.
                    category = self._get_flow_category(pcap_path, flow['packets'][0])
                    # Update the flow class.
                    flow['category'] = category
                    labeled_flows.append(flow)

            if not labeled_flows:
                logging.warning(f"No valid labeled flows extracted: {pcap_path}")
                self.error_files['no_valid_packets'].append(str(pcap_path))
            else:
                logging.info(f"Processed PCAP file: {pcap_path}; extracted {len(labeled_flows)} valid flows")

            return labeled_flows

        except Exception as e:
            logging.error(f"Failed to process PCAP file: {str(e)}")
            self.error_files['parse_error'].append(str(pcap_path))
            return []
