#!/usr/bin/env python3
from pathlib import Path
import logging
from typing import List, Dict
import subprocess
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

class PcapMerger:
    """Merge small PCAP files into larger files."""

    def __init__(self, min_file_size: int = 1 * 1024 * 1024):  # Default to 1 MB for datasets with many small files.
        """Initialize the PCAP merger.

        Args:
            min_file_size: Minimum size threshold; files below this size are merged.
        """
        self.min_file_size = min_file_size
        self.max_group_size = 500 * 1024 * 1024  # Cap each merged file at 500 MB.
        self.num_threads = min(32, multiprocessing.cpu_count() * 2)  # Use at most 32 threads.

    def merge_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, List[Path]]:
        """Merge PCAP files in a directory.

        Args:
            input_dir: Input directory containing class subdirectories.
            output_dir: Output directory.

        Returns:
            A mapping from class names to lists of merged file paths.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all label directories.
        label_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        logging.info(f"Found {len(label_dirs)} label directories")

        # Process label directories using a thread pool.
        merged_files = {}
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks.
            future_to_label = {
                executor.submit(self._merge_label_directory, label_dir, output_dir): label_dir.name
                for label_dir in label_dirs
            }

            # Collect results.
            for future in tqdm(as_completed(future_to_label), total=len(label_dirs), desc="Processing label directories"):
                label = future_to_label[future]
                try:
                    result = future.result()
                    if result:
                        merged_files[label] = result
                except Exception as e:
                    logging.error(f"Error processing directory {label}: {str(e)}")

        return merged_files

    def _merge_label_directory(self, label_dir: Path, output_dir: Path) -> List[Path]:
        """Process all PCAP files in a single label directory.

        Args:
            label_dir: Path to the label directory.
            output_dir: Output directory.

        Returns:
            A list of merged file paths.
        """
        # Create the output directory.
        label_output_dir = output_dir / label_dir.name
        label_output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all PCAP files in the directory.
        pcap_files = list(label_dir.glob("*.pcap"))
        if not pcap_files:
            logging.warning(f"No PCAP files found in directory {label_dir}")
            return []

        # Group files by size.
        current_group = []
        current_size = 0
        group_number = 1
        merged_files = []

        # Show progress for each label directory with tqdm.
        for pcap_file in tqdm(pcap_files, desc=f"Merging files for {label_dir.name}", leave=False):
            file_size = pcap_file.stat().st_size

            # Finish the current group before adding a file that would exceed the size limit.
            if current_size + file_size > self.max_group_size:
                if current_group:
                    try:
                        merged_file = self._merge_files(
                            current_group,
                            label_output_dir,
                            f"{label_dir.name}_merged_{group_number}.pcap"
                        )
                        merged_files.append(merged_file)
                        group_number += 1
                    except Exception as e:
                        logging.error(f"Failed to merge group {group_number}: {str(e)}")

                current_group = []
                current_size = 0

            current_group.append(pcap_file)
            current_size += file_size

        # Process the final group.
        if current_group:
            try:
                merged_file = self._merge_files(
                    current_group,
                    label_output_dir,
                    f"{label_dir.name}_merged_{group_number}.pcap"
                )
                merged_files.append(merged_file)
            except Exception as e:
                logging.error(f"Failed to merge the final group: {str(e)}")

        logging.info(f"Directory {label_dir.name} processed; created {len(merged_files)} merged files")
        return merged_files

    def _merge_files(self, files: List[Path], output_dir: Path, output_name: str) -> Path:
        """Merge PCAP files.

        Args:
            files: List of files to merge.
            output_dir: Output directory.
            output_name: Output filename.

        Returns:
            The merged file path.
        """
        output_file = output_dir / output_name

        # Merge files using mergecap.
        try:
            # Force PCAP output format with -F pcap.
            cmd = ["mergecap", "-w", str(output_file), "-F", "pcap"]
            cmd.extend(str(f) for f in files)

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            if output_file.exists():
                logging.info(f"Merged {len(files)} files into {output_file} ({output_file.stat().st_size / 1024 / 1024:.2f} MB)")
                return output_file
            else:
                raise RuntimeError(f"Merged file does not exist: {output_file}")

        except subprocess.CalledProcessError as e:
            error_msg = f"File merging failed: {e.stderr}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error merging files: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
