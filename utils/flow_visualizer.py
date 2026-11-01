import numpy as np
from PIL import Image
import ipaddress
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Union, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import hashlib
import random
from utils.packet_utils import extract_ip_packet

class TrafficImageGenerator:
    """Generate traffic images.

    Supports multichannel header, payload, and direction representations,
    batch processing, parallel processing, and output directory creation.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        image_size: Tuple[int, int] = (40, 40),
        rows_per_packet: int = 8,
        header_rows: int = 2,
        bytes_per_row: int = 40,
        use_color: bool = False,
        ip_identifier_mode: str = 'role',
        random_seed: int = 0
    ):
        """Initialize the traffic image generator.

        Args:
            config: Configuration dictionary; overrides explicit parameters if provided.
            image_size: Output image size, default (40, 40).
            rows_per_packet: Rows per packet, default 8.
            header_rows: Header rows per packet, default 2.
            bytes_per_row: Bytes per row, default 40.
            use_color: Whether to generate color images.
            ip_identifier_mode: IP identifier mode: original, random, role, or zero.
            random_seed: Deterministic seed for random mode.
        """
        if config:
            try:
                image_config = config.get('image', {})
                self.image_size = tuple(image_config['size'])
                self.rows_per_packet = image_config['rows_per_packet']
                self.header_rows = image_config['header_rows']
                self.bytes_per_row = image_config['bytes_per_row']
                self.use_color = image_config['use_color']
                self.max_workers = image_config['max_workers']
                self.ip_identifier_mode = image_config.get('ip_identifier_mode', ip_identifier_mode)
                self.random_seed = image_config.get('random_seed', random_seed)
            except KeyError as e:
                logging.error(f"Missing required image configuration parameter: {e}")
                logging.info("Using default parameters")
                self.image_size = image_size
                self.rows_per_packet = rows_per_packet
                self.header_rows = header_rows
                self.bytes_per_row = bytes_per_row
                self.use_color = use_color
                self.max_workers = 4
                self.ip_identifier_mode = ip_identifier_mode
                self.random_seed = random_seed
        else:
            self.image_size = image_size
            self.rows_per_packet = rows_per_packet
            self.header_rows = header_rows
            self.bytes_per_row = bytes_per_row
            self.use_color = use_color
            self.max_workers = 4
            self.ip_identifier_mode = ip_identifier_mode
            self.random_seed = random_seed

        if self.ip_identifier_mode not in {'original', 'random', 'role', 'zero'}:
            raise ValueError(
                "ip_identifier_mode must be original, random, role, or zero"
            )

        self.payload_rows = self.rows_per_packet - self.header_rows

        # Compute matrix dimensions.
        self.matrix_height = self.image_size[0]
        self.matrix_width = self.image_size[1]

    def _random_identifier_pair(self, identifier_key: Optional[str] = None) -> Tuple[bytes, bytes]:
        """Generate stable per-flow IPv4 endpoints without exposing real IP identities."""
        key = '' if identifier_key is None else str(identifier_key)
        seed_material = f'{self.random_seed}:{key}'.encode('utf-8')
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], 'big')
        rng = random.Random(seed)

        def next_ip() -> bytes:
            # Use the private 10/8 range and avoid broadcast, network, or all-zero addresses.
            return bytes([10, rng.randrange(0, 256), rng.randrange(0, 256), rng.randrange(1, 255)])

        src = next_ip()
        dst = next_ip()
        while dst == src:
            dst = next_ip()
        return src, dst

    @staticmethod
    def _ip_header_offsets(raw_data: bytes) -> Optional[Tuple[int, int, int]]:
        """Return address offsets for validated IP-layer bytes only."""
        try:
            packet = extract_ip_packet(raw_data)
        except ValueError:
            return None
        return (0, 12, 4) if packet[0] >> 4 == 4 else (0, 8, 16)

    def _apply_ip_identifier(
        self,
        raw_data: bytes,
        mode: str,
        direction: int,
        identifier_key: Optional[str] = None,
        link_type: str = 'raw_ip'
    ) -> bytes:
        """Sanitize endpoint fields while preserving all other IP packet bytes."""
        if mode not in {'original', 'random', 'role', 'zero'}:
            raise ValueError(f'Invalid IP identifier mode: {mode}')
        if direction not in (-1, 1):
            raise ValueError('Packet direction must be 1 or -1')
        raw_data = extract_ip_packet(raw_data, link_type)
        if mode == 'original':
            return raw_data

        offsets = self._ip_header_offsets(raw_data)
        if offsets is None:
            raise ValueError('Unable to locate IP address fields')

        _, src_offset, address_length = offsets
        raw_list = list(raw_data)
        if mode == 'random':
            random_src, random_dst = self._random_identifier_pair(identifier_key)
            if address_length == 4:
                src_address, dst_address = random_src, random_dst
            else:
                # Use the IPv6 documentation prefix and derive the suffix from per-flow random endpoints.
                src_address = b'\x20\x01\x0d\xb8' + random_src + bytes(8)
                dst_address = b'\x20\x01\x0d\xb8' + random_dst + bytes(8)
        elif mode == 'zero':
            # Zero all source and destination IPs, removing endpoint identities and direction roles.
            src_address = bytes(address_length)
            dst_address = bytes(address_length)
        else:
            if address_length == 4:
                src_address = bytes([10, 0, 0, 1])
                dst_address = bytes([10, 0, 0, 2])
            else:
                src_address = bytes.fromhex('20010db8000000000000000000000001')
                dst_address = bytes.fromhex('20010db8000000000000000000000002')

        dst_offset = src_offset + address_length
        if direction == 1:
            raw_list[src_offset:src_offset + address_length] = src_address
            raw_list[dst_offset:dst_offset + address_length] = dst_address
        else:
            raw_list[src_offset:src_offset + address_length] = dst_address
            raw_list[dst_offset:dst_offset + address_length] = src_address

        if mode == 'role':
            transport_offset = self._transport_header_offset(raw_data, offsets[0])
            if transport_offset is not None:
                raw_list[transport_offset:transport_offset + 4] = bytes(4)
        return bytes(raw_list)

    @staticmethod
    def _transport_header_offset(raw_data: bytes, ip_offset: int) -> Optional[int]:
        """Return the TCP/UDP header offset for clearing ports in role mode."""
        if ip_offset < 0 or len(raw_data) <= ip_offset:
            return None
        version = raw_data[ip_offset] >> 4
        if version == 4:
            if len(raw_data) < ip_offset + 20:
                return None
            header_length = (raw_data[ip_offset] & 0x0f) * 4
            if header_length < 20 or len(raw_data) < ip_offset + header_length + 4:
                return None
            protocol = raw_data[ip_offset + 9]
            # Noninitial fragments contain continuation payload, not a transport header.
            fragment_offset = int.from_bytes(raw_data[ip_offset + 6:ip_offset + 8], 'big') & 0x1fff
            if fragment_offset:
                return None
            return ip_offset + header_length if protocol in (6, 17) else None

        if version != 6 or len(raw_data) < ip_offset + 44:
            return None

        next_header = raw_data[ip_offset + 6]
        offset = ip_offset + 40
        while next_header not in (6, 17):
            if next_header in (0, 43, 60, 135):
                if len(raw_data) < offset + 2:
                    return None
                next_header, extension_length = raw_data[offset], (raw_data[offset + 1] + 1) * 8
            elif next_header == 44:
                if len(raw_data) < offset + 8:
                    return None
                fragment_offset = int.from_bytes(raw_data[offset + 2:offset + 4], 'big') & 0xfff8
                if fragment_offset:
                    return None
                next_header, extension_length = raw_data[offset], 8
            elif next_header == 51:
                if len(raw_data) < offset + 2:
                    return None
                next_header, extension_length = raw_data[offset], (raw_data[offset + 1] + 2) * 4
            else:
                return None
            offset += extension_length
            if len(raw_data) < offset + 4:
                return None
        return offset

    def flow_to_matrix(
        self,
        packets,
        anonymize=False,
        first_src_ip=None,
        directions=None,
        identifier_mode=None,
        identifier_key=None,
        link_type='raw_ip'
    ):
        """Construct the paper's 40x40 matrix from up to five IP packets.

        Args:
            packets: List of packets.
            anonymize: Whether to anonymize IP addresses.
            first_src_ip: Forward endpoint; defaults to the first packet source.
            directions: Explicit directions (1 or -1), overriding IP-based inference.
            identifier_mode: Override the default IP identifier mode; role also zeros ports.
            identifier_key: Key for stable per-flow endpoints in random mode.
            link_type: Input framing: raw_ip, ethernet, or sll. Packet dictionaries
                may override it with raw_link_type. The image always excludes framing.

        Raises:
            ValueError: If packet bytes, framing, or directions are invalid.
        """
        mode = identifier_mode or ('role' if anonymize else self.ip_identifier_mode)
        if mode not in {'original', 'random', 'role', 'zero'}:
            raise ValueError(f'Invalid IP identifier mode: {mode}')
        sampled_packets = list(itertools.islice(packets, 5))
        explicit_directions = directions is not None and len(directions) > 0
        if explicit_directions and len(directions) < len(sampled_packets):
            raise ValueError('Directions must cover all sampled packets')
        reference_src = None
        if not explicit_directions and first_src_ip is not None:
            reference_src = ipaddress.ip_address(first_src_ip).packed
        segments = []
        for i, packet in enumerate(sampled_packets):
            packet_link_type = link_type
            if isinstance(packet, dict):
                if packet.get('raw') is None:
                    raise ValueError('Image generation requires complete IP headers, not payload-only records')
                raw_data = packet['raw']
                packet_link_type = packet.get('raw_link_type', link_type)
            elif hasattr(packet, 'raw'):
                raw_data = packet.raw
            else:
                raw_data = bytes(packet)

            raw_data = extract_ip_packet(raw_data, packet_link_type)
            _, src_offset, address_length = self._ip_header_offsets(raw_data)
            src = raw_data[src_offset:src_offset + address_length]
            dst = raw_data[src_offset + address_length:src_offset + 2 * address_length]
            if explicit_directions:
                direction = directions[i]
            else:
                if reference_src is None:
                    reference_src = src
                if src == reference_src:
                    direction = 1
                elif dst == reference_src:
                    direction = -1
                else:
                    raise ValueError('Packet endpoints do not match the flow reference address')

            sanitized = self._apply_ip_identifier(raw_data, mode, direction, identifier_key)
            # Section 3.2.2: 80 bytes followed by 240 bytes, with zero padding.
            segments.append(sanitized[:320].ljust(320, b'\x00'))

        data = b''.join(segments).ljust(1600, b'\x00')
        return np.frombuffer(data, dtype=np.uint8).reshape(40, 40).copy()

    def generate_image_data(
        self,
        packets: List[dict],
        anonymize: bool = False,
        first_src_ip=None,
        directions=None,
        identifier_mode=None,
        identifier_key=None,
        link_type='raw_ip'
    ) -> bytes:
        """Generate image bytes without saving a file.

        Args:
            packets: List of packets.
            anonymize: Whether to anonymize endpoints.
            first_src_ip: Source IP of the first packet.
            directions: Packet directions (1 for forward, -1 for reverse).
            link_type: Explicit input framing; all output pixels start at the IP layer.

        Returns:
            Image data as bytes.
        """
        try:
            matrix = self.flow_to_matrix(
                packets,
                anonymize=anonymize,
                first_src_ip=first_src_ip,
                directions=directions,
                identifier_mode=identifier_mode,
                identifier_key=identifier_key,
                link_type=link_type
            )

            # Create the image with fromarray.
            image = Image.fromarray(matrix)

            # Encode the image as bytes.
            from io import BytesIO
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG', optimize=True)
            return img_byte_arr.getvalue()

        except Exception as e:
            logging.error(f"Failed to generate image data: {str(e)}")
            return None

    def generate_image(self, packets: List[dict], save_path: Optional[Union[str, Path]] = None, anonymize: bool = False, first_src_ip=None, directions=None, link_type='raw_ip', identifier_key=None) -> Image.Image:
        """Generate an image and optionally save it."""
        try:
            # Get image data.
            image_data = self.generate_image_data(
                packets, anonymize, first_src_ip, directions,
                identifier_key=identifier_key, link_type=link_type
            )
            if image_data is None:
                raise ValueError("Failed to generate image data")

            # Create an image from bytes.
            from io import BytesIO
            image = Image.open(BytesIO(image_data))

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(image_data)

            return image

        except Exception as e:
            logging.error(f"Failed to generate image: {str(e)}")
            raise

    def process_flow(self, flow_data: dict, output_dir: Path, anonymize: bool = False, link_type='raw_ip') -> bool:
        """Process a single flow."""
        try:
            # Get flow metadata.
            flow_info = flow_data['flow_info']
            original_info = flow_info['original_info']

            # Get the class label.
            label = flow_data.get('category', 'unknown')

            # Build a flow ID from original metadata.
            flow_id = f"{flow_info['protocol']}_{original_info['src_ip']}_{original_info['src_port']}_{original_info['dst_ip']}_{original_info['dst_port']}"

            # Build the image filename.
            filename = f"{label}_{flow_id}.png"
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")

            # Get the first source IP to determine direction.
            first_src_ip = original_info['first_src_ip']

            # Get direction metadata.
            directions = flow_data.get('directions', [])

            # Generate and save the image.
            image_path = output_dir / filename
            self.generate_image(
                flow_data['packets'],
                image_path,
                anonymize=anonymize,
                first_src_ip=first_src_ip,
                directions=directions,
                link_type=link_type,
                identifier_key=flow_id
            )

            logging.debug(f"Generated image: {image_path}")
            return True

        except Exception as e:
            logging.error(f"Failed to process flow: {str(e)}")
            return False

    def process_flows(self, flows: List[dict], output_dir: Union[str, Path], max_workers: int = 4, anonymize: bool = False, link_type='raw_ip'):
        """Process a batch of flows.

        Args:
            flows: List of flow records.
            output_dir: Output directory.
            max_workers: Maximum number of worker threads.
            anonymize: Whether to anonymize IP addresses and ports.
            link_type: Explicit input framing for each flow's packets.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(flows), desc="Generating traffic images") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for flow in flows:
                    future = executor.submit(self.process_flow, flow, output_dir, anonymize, link_type)
                    future.add_done_callback(lambda p: pbar.update(1))
                    futures.append(future)

                # Wait for all tasks to complete.
                completed = 0
                failed = 0
                for future in as_completed(futures):
                    if future.result():
                        completed += 1
                    else:
                        failed += 1

        logging.info(f"Image generation complete! Succeeded: {completed}, failed: {failed}")
