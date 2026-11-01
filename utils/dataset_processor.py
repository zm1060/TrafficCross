import pickle
from typing import List, Dict, Union, Any
import logging
from tqdm import tqdm
import random
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from utils.anonymizer import anonymize_flow_info


class DatasetProcessor:
    """Process and save traffic datasets."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        split_ratio: List[float],
        chunk_size: int,
        pack_dataset: bool = True,
        random_seed: int = 0
    ):
        """Initialize the dataset processor.

        Args:
            output_dir: Output directory.
            split_ratio: Dataset split ratios, such as [0.7, 0.15, 0.15].
            chunk_size: Number of flows per packed chunk.
            pack_dataset: Whether to pack the dataset.
            random_seed: Seed for deterministic dataset splitting.

        Raises:
            ValueError: If split_ratio is invalid.
            OSError: If directory creation fails.
        """
        if not (len(split_ratio) == 3 and sum(split_ratio) == 1.0):
            raise ValueError("split_ratio must be a three-element list that sums to 1")

        self.output_dir = Path(output_dir)
        self.split_ratio = split_ratio
        self.chunk_size = chunk_size
        self.pack_dataset = pack_dataset
        self.random_seed = random_seed

        # In-memory data storage.
        self.flows_by_category: Dict[str, List[Any]] = defaultdict(list)
        self.images_by_category: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.total_flows: int = 0

        # Temporary storage counter.
        self.temp_storage_count: int = 0
        self.temp_storage_limit: int = chunk_size  # Temporary storage limit.
        self.temp_chunk_id: int = 0  # Temporary chunk ID.

        try:
            # Create temporary and output directories.
            self.temp_dir = self.output_dir / 'temp'
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directory: {str(e)}")
            raise

    def process_and_save_batch(self, flows: List[Dict], image_generator=None, generate_flows: bool = True, generate_images: bool = True) -> None:
        """Process a batch of flows and store the results in memory.

        Args:
            flows: List of flows.
            image_generator: Traffic image generator.
            generate_flows: Whether to generate flow data.
            generate_images: Whether to generate image data.

        Raises:
            RuntimeError: If batch processing fails.
        """
        processed_count = 0
        error_count = 0

        for flow in flows:
            try:
                if not flow or 'flow_info' not in flow:
                    continue

                # Get original flow metadata.
                original_info = flow['flow_info']['original_info']
                category = flow['category']

                # Get the complete packet list.
                all_packets = flow.get('packets', [])
                all_directions = flow.get('directions', [])

                if not all_packets:
                    continue

                # Compute sequence metadata.
                packet_sizes = [p.get('length', 0) for p in all_packets]  # Use IP total length.
                relative_times = [float(p['timestamp'] - all_packets[0]['timestamp']) for p in all_packets]
                inter_arrival_times = [t2 - t1 for t1, t2 in zip(relative_times[:-1], relative_times[1:])]
                payloads = [len(p.get('payload', b'')) for p in all_packets]
                flags = [p.get('flags', 0) if p.get('protocol') == 'tcp' else 0 for p in all_packets]
                window_sizes = [p.get('window', 0) if p.get('protocol') == 'tcp' else 0 for p in all_packets]

                # Validate packet sizes.
                if any(size <= 0 for size in packet_sizes):  # Apply stricter validation.
                    logging.warning(f"Invalid packet sizes detected: {packet_sizes}")
                    logging.debug(f"Original lengths: {[p.get('raw_length', 0) for p in all_packets]}")  # Log original lengths for debugging.
                    continue

                # Limit the number of packets used to generate each image.
                model_packets = all_packets[:5]
                model_directions = all_directions[:5]

                # Generate anonymized metadata.
                anonymized_info = anonymize_flow_info(original_info)

                # Build an instance ID from the source capture, first timestamp, and flow index.
                # Five-tuples can recur across captures or sessions and cannot uniquely identify flow instances.
                source_pcap = flow.get('_source_pcap', '')
                flow_index = flow.get('_flow_index', 0)
                first_timestamp = all_packets[0].get('timestamp', 0)
                identity = '|'.join([
                    str(source_pcap),
                    str(first_timestamp),
                    str(flow_index),
                    str(flow['flow_info']['protocol']),
                    str(original_info['src_ip']),
                    str(original_info['src_port']),
                    str(original_info['dst_ip']),
                    str(original_info['dst_port'])
                ])
                flow_id = (
                    f"{flow['flow_info']['protocol']}_"
                    f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"
                )

                # Generate image data.
                if generate_images and image_generator and model_packets:
                    try:
                        image_data = image_generator.generate_image_data(
                            model_packets,
                            identifier_mode=image_generator.ip_identifier_mode,
                            identifier_key=(
                                f"{flow_id}:{all_packets[0].get('timestamp', 0)}"
                            ),
                            first_src_ip=original_info['first_src_ip'],
                            directions=model_directions
                        )
                        if image_data:
                            self.images_by_category[category][flow_id] = image_data
                        else:
                            logging.warning('Skipping flow with invalid image input: %s', flow_id)
                            continue
                    except Exception as e:
                        logging.warning(f"Failed to generate image: {str(e)}")
                        continue

                # Save flow metadata.
                if generate_flows:
                    flow_info = {
                        'category': category,
                        'flow_info': {
                            'protocol': flow['flow_info']['protocol'],
                            'proto_num': flow['flow_info']['proto_num'],
                            'packet_count': len(all_packets),
                            'relative_start': 0.0,
                            'relative_end': flow['flow_info']['relative_end'],
                            'flow_duration': flow['flow_info']['flow_duration'],
                            'total_bytes': sum(packet_sizes),  # Compute total bytes using IP packet sizes.
                            'original_info': original_info,
                            'instance_info': {
                                'source_pcap': source_pcap,
                                'first_packet_timestamp': first_timestamp,
                                'flow_index': flow_index
                            },
                            'anonymized_info': anonymized_info,
                            'sequence_info': {
                                'packet_sizes': packet_sizes,
                                'relative_times': relative_times,
                                'inter_arrival_times': inter_arrival_times,
                                'directions': all_directions,
                                'payloads': payloads,
                                'flags': flags,
                                'window_sizes': window_sizes
                            }
                        }
                    }

                    # Add statistics.
                    flow_info['flow_info']['statistical_info'] = {
                        'avg_packet_size': float(np.mean(packet_sizes)),
                        'min_packet_size': min(packet_sizes),
                        'max_packet_size': max(packet_sizes),
                        'std_packet_size': float(np.std(packet_sizes)),
                        'avg_iat': float(np.mean(inter_arrival_times)) if inter_arrival_times else 0,
                        'min_iat': min(inter_arrival_times) if inter_arrival_times else 0,
                        'max_iat': max(inter_arrival_times) if inter_arrival_times else 0,
                        'std_iat': float(np.std(inter_arrival_times)) if inter_arrival_times else 0
                    }

                    # Add to the in-memory data structures.
                    self.flows_by_category[category].append((flow_id, flow_info))
                    self.total_flows += 1
                    self.temp_storage_count += 1
                    processed_count += 1

                # Check whether a temporary save is needed.
                if self.temp_storage_count >= self.temp_storage_limit:
                    self._save_temp_chunk()

            except Exception as e:
                error_count += 1
                logging.error(f"Failed to process flow: {str(e)}", exc_info=True)
                continue

        logging.info(f"Batch complete: {processed_count} succeeded, {error_count} failed")

        if error_count > processed_count * 0.1:  # Warn if the failure rate exceeds 10%.
            logging.warning("Warning: high processing failure rate")

    def _save_temp_chunk(self) -> None:
        """Save a temporary data chunk and clear memory.

        Raises:
            OSError: If saving the temporary file fails.
        """
        if self.total_flows == 0:
            return

        logging.info(f"Saving temporary chunk {self.temp_chunk_id} ({self.temp_storage_count} flows)...")

        # Prepare data.
        temp_data = {
            'flows_by_category': dict(self.flows_by_category),
            'images_by_category': dict(self.images_by_category),
            'total_flows': self.temp_storage_count
        }

        # Save the temporary file.
        temp_file = self.temp_dir / f"temp_chunk_{self.temp_chunk_id:03d}.pkl"
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(temp_data, f)
        except Exception as e:
            logging.error(f"Failed to save temporary file: {str(e)}")
            raise OSError(f"Failed to save temporary file: {str(e)}")

        # Clear memory.
        self.flows_by_category.clear()
        self.images_by_category.clear()
        self.temp_storage_count = 0
        self.temp_chunk_id += 1

        logging.info(f"Temporary chunk saved: {temp_file}")

    def split_and_save_dataset(self) -> None:
        """Split the dataset and save it directly in packed format.

        Raises:
            RuntimeError: If dataset splitting fails.
        """
        logging.info("Splitting and saving dataset...")

        try:
            # Save the final temporary chunk.
            if self.temp_storage_count > 0:
                self._save_temp_chunk()

            # Collect all temporary files.
            temp_files = sorted(self.temp_dir.glob("temp_chunk_*.pkl"))
            if not temp_files:
                raise RuntimeError("No temporary data files found")

            # Count total flows.
            total_flows_by_category: Dict[str, List[Any]] = defaultdict(list)
            total_images_by_category: Dict[str, Dict[str, Any]] = defaultdict(dict)

            # Read and merge temporary files.
            for temp_file in tqdm(temp_files, desc="Merging temporary data"):
                try:
                    with open(temp_file, 'rb') as f:
                        temp_data = pickle.load(f)

                        # Merge data.
                        for category, flows in temp_data['flows_by_category'].items():
                            total_flows_by_category[category].extend(flows)

                        for category, images in temp_data['images_by_category'].items():
                            total_images_by_category[category].update(images)
                except Exception as e:
                    logging.error(f"Failed to read temporary file {temp_file}: {str(e)}")
                    continue

            # Check global ID uniqueness before splitting to avoid overwrites across chunks.
            all_flow_ids = [
                flow_id
                for category in sorted(total_flows_by_category)
                for flow_id, _ in total_flows_by_category[category]
            ]
            id_counts = Counter(all_flow_ids)
            duplicate_ids = {
                flow_id for flow_id, count in id_counts.items()
                if count > 1
            }
            if duplicate_ids:
                examples = ', '.join(sorted(duplicate_ids)[:5])
                raise ValueError(
                    f"Detected {len(duplicate_ids)} duplicate flow IDs; packing stopped to prevent overwrites."
                    f"Examples: {examples}"
                )

            # Split data by class.
            splits = {
                'train': defaultdict(list),
                'val': defaultdict(list),
                'test': defaultdict(list)
            }

            # Split each class independently.
            split_rng = random.Random(self.random_seed)
            for category in sorted(total_flows_by_category):
                flows = total_flows_by_category[category]
                flows.sort(key=lambda item: item[0])
                split_rng.shuffle(flows)  # Shuffle using a fixed seed.
                n_flows = len(flows)

                # Compute split boundaries.
                train_end = int(n_flows * self.split_ratio[0])
                val_end = train_end + int(n_flows * self.split_ratio[1])

                # Split the data.
                splits['train'][category] = flows[:train_end]
                splits['val'][category] = flows[train_end:val_end]
                splits['test'][category] = flows[val_end:]

                logging.info(f"Split results for class {category}:")
                logging.info(f"- Training: {len(splits['train'][category])} flows")
                logging.info(f"- Validation: {len(splits['val'][category])} flows")
                logging.info(f"- Test: {len(splits['test'][category])} flows")

            # Save each split.
            for split_name, split_data in splits.items():
                self._save_split(split_name, split_data, total_images_by_category)

            # Clean up the temporary file.
            self._cleanup_temp_files()

        except Exception as e:
            logging.error(f"Dataset splitting failed: {str(e)}")
            raise RuntimeError(f"Dataset splitting failed: {str(e)}")

    def _save_split(self, split_name: str, split_data: Dict[str, List[Any]], total_images_by_category: Dict[str, Dict[str, Any]]) -> None:
        """Save a single dataset split.

        Args:
            split_name: Split name (train/val/test).
            split_data: Data for this split.
            total_images_by_category: All image data grouped by class.

        Raises:
            OSError: If saving the data fails.
        """
        total_flows = sum(len(flows) for flows in split_data.values())
        num_chunks = (total_flows + self.chunk_size - 1) // self.chunk_size

        logging.info(f"\nSaving {split_name} dataset ({total_flows} flows, {num_chunks} chunks)...")

        # Flatten all flows and shuffle using the same fixed seed.
        all_flows = []
        for category, flows in split_data.items():
            all_flows.extend(flows)
        all_flows.sort(key=lambda item: item[0])
        random.Random(self.random_seed).shuffle(all_flows)

        # Save in chunks.
        for chunk_id in range(num_chunks):
            start_idx = chunk_id * self.chunk_size
            end_idx = min((chunk_id + 1) * self.chunk_size, total_flows)
            chunk_flows = all_flows[start_idx:end_idx]

            chunk_data = {
                'flows': {},
                'images': {},
                'mapping': []
            }

            # Populate chunk data.
            for flow_id, flow_info in chunk_flows:
                if flow_id in chunk_data['flows']:
                    raise ValueError(f"Duplicate flow ID detected; refusing to overwrite: {flow_id}")
                category = flow_info['category']
                chunk_data['flows'][flow_id] = flow_info

                # Include matching image data when available.
                if flow_id in total_images_by_category[category]:
                    chunk_data['images'][flow_id] = total_images_by_category[category][flow_id]

                # Add sample mappings.
                chunk_data['mapping'].append({
                    'flow_id': flow_id,
                    'category': category,
                    'has_image': flow_id in chunk_data['images']
                })

            # Save the chunk.
            chunk_file = self.output_dir / f"{split_name}_packed_{chunk_id:03d}.pkl"
            try:
                with open(chunk_file, 'wb') as f:
                    pickle.dump(chunk_data, f)
            except Exception as e:
                logging.error(f"Failed to save chunk {chunk_file}: {str(e)}")
                raise OSError(f"Failed to save chunk: {str(e)}")

            logging.info(f"Saved chunk {chunk_id + 1}/{num_chunks}: {len(chunk_flows)} flows")

    def _cleanup_temp_files(self) -> None:
        """Remove temporary files and directories.

        Raises:
            OSError: If deleting a file fails.
        """
        logging.info("Cleaning up temporary files...")
        try:
            for temp_file in self.temp_dir.glob("temp_chunk_*.pkl"):
                try:
                    temp_file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")
            self.temp_dir.rmdir()
            logging.info("Temporary file cleanup complete")
        except Exception as e:
            logging.error(f"Temporary file cleanup failed: {str(e)}")
            raise OSError(f"Temporary file cleanup failed: {str(e)}")
