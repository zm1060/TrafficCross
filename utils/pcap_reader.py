import dpkt
from dpkt.ethernet import Ethernet
from dpkt.ip import IP as DPKT_IP
from dpkt.ip6 import IP6
from dpkt.tcp import TCP as DPKT_TCP
from dpkt.udp import UDP as DPKT_UDP
from dpkt.sll import SLL
import socket
import logging
import mmap
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils.packet_utils import extract_ip_packet

def process_packet_chunk(chunk_data):
    """Process a packet chunk in a multiprocessing worker."""
    valid_packets = []

    for ts, buf in chunk_data:
        try:
            # Get the captured packet size, including link-layer headers.
            original_size = len(buf)

            # Try supported encapsulation formats.
            ip_data = None
            packet_format = None

            # 1. Try Ethernet.
            try:
                eth = Ethernet(buf)
                if isinstance(eth.data, (DPKT_IP, IP6)):
                    ip_data = eth.data
                    packet_format = 'ethernet'
            except:
                pass

            # 2. Try SLL.
            if not ip_data:
                try:
                    sll = SLL(buf)
                    if isinstance(sll.data, (DPKT_IP, IP6)):
                        ip_data = sll.data
                        packet_format = 'sll'
                except:
                    pass

            # 3. Try raw IP.
            if not ip_data:
                try:
                    if len(buf) >= 20:
                        version = (buf[0] & 0xf0) >> 4
                        if version == 4:
                            ip_data = DPKT_IP(buf)
                            packet_format = 'raw_ip'
                except:
                    pass

            if ip_data and isinstance(ip_data.data, (DPKT_TCP, DPKT_UDP)):
                proto = ip_data.p if isinstance(ip_data, DPKT_IP) else ip_data.nxt
                if proto in (6, 17):  # TCP or UDP
                    # Get IP total length.
                    if isinstance(ip_data, DPKT_IP):
                        length = ip_data.len  # IPv4 total length
                    else:  # IPv6
                        length = ip_data.plen + 40  # IPv6 payload length + 40 byte header

                    transport_data = ip_data.data
                    if isinstance(transport_data, bytes):
                        if proto == 6:
                            transport_data = DPKT_TCP(transport_data)
                        else:
                            transport_data = DPKT_UDP(transport_data)

                    packet_info = {
                        'timestamp': ts,
                        'ip_version': 6 if isinstance(ip_data, IP6) else 4,
                        'src_ip': socket.inet_ntop(socket.AF_INET6 if isinstance(ip_data, IP6) else socket.AF_INET, ip_data.src),
                        'dst_ip': socket.inet_ntop(socket.AF_INET6 if isinstance(ip_data, IP6) else socket.AF_INET, ip_data.dst),
                        'protocol': 'tcp' if proto == 6 else 'udp',
                        'src_port': transport_data.sport,
                        'dst_port': transport_data.dport,
                        'payload': transport_data.data,
                        'length': length,  # Use IP total length.
                        'raw_length': original_size,  # Keep the original captured length for reference.
                        'packet_format': packet_format,
                        # Keep raw IP-layer bytes so image generation can locate IP header fields.
                        'raw': extract_ip_packet(buf, packet_format),
                        'raw_link_type': 'raw_ip'
                    }

                    if proto == 6:  # TCP-specific fields.
                        packet_info.update({
                            'flags': transport_data.flags,
                            'window': transport_data.win
                        })

                    valid_packets.append(packet_info)

        except Exception as e:
            continue

    return valid_packets


class PcapReader:
    """Read and extract packets from PCAP files."""

    def __init__(self, num_processes=None):
        """Initialize the PCAP reader.

        Args:
            num_processes: Worker process count; defaults to CPU cores minus one.
        """
        self.num_processes = num_processes or max(1, cpu_count() - 1)

    def read_pcap_file(self, pcap_path: Path) -> List[Dict]:
        """Read a PCAP file and extract valid packets.

        Args:
            pcap_path: Path to the PCAP file.

        Returns:
            A list of processed packets.
        """
        logging.info(f"Reading PCAP file: {pcap_path}")

        # Check that the file exists.
        if not pcap_path.exists():
            logging.error(f"PCAP file does not exist: {pcap_path}")
            return []

        # Check the file size.
        file_size = os.path.getsize(pcap_path)
        if file_size == 0:
            logging.error(f"PCAP file is empty: {pcap_path}")
            return []

        logging.info(f"PCAP file size: {file_size} bytes")

        # Read packet chunks.
        chunks = self._read_pcap_chunks(pcap_path)

        if not chunks:
            logging.error(f"Unable to read chunks from PCAP file: {pcap_path}")
            return []

        logging.info(f"Read {len(chunks)} chunks")

        # Process chunks using a process pool.
        processed_packets = []
        total_chunks = len(chunks)

        # Show progress with tqdm.
        with Pool(processes=self.num_processes) as pool:
            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                for result in pool.imap(process_packet_chunk, chunks):
                    processed_packets.extend(result)
                    pbar.update(1)

        if not processed_packets:
            logging.error(f"No valid packets extracted from PCAP file: {pcap_path}")
            return []

        # Sort all packets by timestamp.
        processed_packets.sort(key=lambda x: x['timestamp'])

        logging.info(f"Processed {len(processed_packets)} packets")
        return processed_packets

    def _read_pcap_chunks(self, pcap_path: Path, chunk_size: int = 10000) -> List[List[Tuple[float, bytes]]]:
        """Read and chunk a PCAP file using mmap.

        Args:
            pcap_path: Path to the PCAP file.
            chunk_size: Packets per chunk, default 10000.

        Returns:
            A list of packet chunks.
        """
        chunks = []
        current_chunk = []

        try:
            with open(pcap_path, 'rb') as f:
                # Check the file size.
                file_size = os.path.getsize(pcap_path)
                if file_size == 0:
                    logging.warning(f"PCAP file is empty: {pcap_path}")
                    return chunks

                # Read the file using mmap.
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

                try:
                    # Detect the file format.
                    magic = mm.read(4)
                    mm.seek(0)  # Reset the file pointer.

                    if magic == b'\xa1\xb2\xc3\xd4' or magic == b'\xd4\xc3\xb2\xa1':
                        # Standard PCAP format.
                        pcap_reader = dpkt.pcap.Reader(mm)
                        logging.info(f"Detected standard PCAP format")
                    elif magic == b'\x0a\x0d\x0d\x0a':
                        # PCAP-NG format.
                        logging.info(f"Detected PCAP-NG format")
                        pcap_reader = dpkt.pcapng.Reader(mm)
                    else:
                        logging.warning(f"Unknown PCAP format; magic number: {magic.hex()}")
                        return chunks

                    packet_count = 0
                    for ts, buf in pcap_reader:
                        packet_count += 1
                        current_chunk.append((ts, buf))
                        if len(current_chunk) >= chunk_size:
                            chunks.append(current_chunk)
                            current_chunk = []

                    # Append the final partial chunk.
                    if current_chunk:
                        chunks.append(current_chunk)

                    logging.info(f"Read {packet_count} packets")

                except dpkt.dpkt.NeedData:
                    logging.warning(f"Incomplete PCAP data: {pcap_path}")
                except Exception as e:
                    logging.error(f"Error reading PCAP file: {str(e)}")
                finally:
                    mm.close()
        except Exception as e:
            logging.error(f"Failed to open PCAP file: {str(e)}")

        return chunks
