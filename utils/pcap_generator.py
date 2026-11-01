#!/usr/bin/env python3
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Raw
import random
from pathlib import Path
import logging
from typing import List

class PcapGenerator:
    """Generate PCAP files containing synthetic network traffic."""

    def __init__(self, output_dir: str = "generated_pcaps"):
        """Initialize the PCAP generator.

        Args:
            output_dir: Output directory.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set the random seed.
        random.seed(42)

        # Configure logging.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def generate_tcp_handshake(self, src_ip: str, dst_ip: str,
                             src_port: int, dst_port: int) -> List[scapy.Packet]:
        """Generate packets for a TCP three-way handshake.

        Args:
            src_ip: Source IP address.
            dst_ip: Destination IP address.
            src_port: Source port.
            dst_port: Destination port.

        Returns:
            A list of packets.
        """
        # SYN
        syn = (Ether()/IP(src=src_ip, dst=dst_ip)/
               TCP(sport=src_port, dport=dst_port, flags='S'))

        # SYN-ACK
        syn_ack = (Ether()/IP(src=dst_ip, dst=src_ip)/
                   TCP(sport=dst_port, dport=src_port, flags='SA'))

        # ACK
        ack = (Ether()/IP(src=src_ip, dst=dst_ip)/
               TCP(sport=src_port, dport=dst_port, flags='A'))

        return [syn, syn_ack, ack]

    def generate_tcp_teardown(self, src_ip: str, dst_ip: str,
                            src_port: int, dst_port: int) -> List[scapy.Packet]:
        """Generate packets for a TCP four-way close."""
        # FIN
        fin = (Ether()/IP(src=src_ip, dst=dst_ip)/
               TCP(sport=src_port, dport=dst_port, flags='F'))

        # ACK
        ack1 = (Ether()/IP(src=dst_ip, dst=src_ip)/
                TCP(sport=dst_port, dport=src_port, flags='A'))

        # FIN
        fin2 = (Ether()/IP(src=dst_ip, dst=src_ip)/
                TCP(sport=dst_port, dport=src_port, flags='F'))

        # ACK
        ack2 = (Ether()/IP(src=src_ip, dst=dst_ip)/
                TCP(sport=src_port, dport=dst_port, flags='A'))

        return [fin, ack1, fin2, ack2]

    def generate_http_traffic(self, src_ip: str, dst_ip: str,
                            src_port: int, dst_port: int = 80) -> List[scapy.Packet]:
        """Generate HTTP request and response traffic."""
        packets = []

        # Add the TCP handshake.
        packets.extend(self.generate_tcp_handshake(src_ip, dst_ip, src_port, dst_port))

        # HTTP GET request.
        http_get = (
            Ether()/
            IP(src=src_ip, dst=dst_ip)/
            TCP(sport=src_port, dport=dst_port, flags='PA')/
            Raw(load="GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
        )
        packets.append(http_get)

        # HTTP response.
        http_response = (
            Ether()/
            IP(src=dst_ip, dst=src_ip)/
            TCP(sport=dst_port, dport=src_port, flags='PA')/
            Raw(load="HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n" + "X"*1000)
        )
        packets.append(http_response)

        # Add the TCP close sequence.
        packets.extend(self.generate_tcp_teardown(src_ip, dst_ip, src_port, dst_port))

        return packets

    def generate_large_file_transfer(self, src_ip: str, dst_ip: str,
                                   src_port: int, dst_port: int,
                                   file_size: int = 1000000) -> List[scapy.Packet]:
        """Generate traffic for a large file transfer."""
        packets = []

        # Add the TCP handshake.
        packets.extend(self.generate_tcp_handshake(src_ip, dst_ip, src_port, dst_port))

        # Send the large file in segments.
        mss = 1460  # Maximum segment size.
        remaining = file_size
        seq = 0

        while remaining > 0:
            chunk_size = min(mss, remaining)
            data_packet = (
                Ether()/
                IP(src=src_ip, dst=dst_ip)/
                TCP(sport=src_port, dport=dst_port, flags='PA', seq=seq)/
                Raw(load="X" * chunk_size)
            )
            packets.append(data_packet)

            # ACK
            ack_packet = (
                Ether()/
                IP(src=dst_ip, dst=src_ip)/
                TCP(sport=dst_port, dport=src_port, flags='A',
                    ack=seq+chunk_size)
            )
            packets.append(ack_packet)

            seq += chunk_size
            remaining -= chunk_size

        # Add the TCP close sequence.
        packets.extend(self.generate_tcp_teardown(src_ip, dst_ip, src_port, dst_port))

        return packets

    def generate_interactive_session(self, src_ip: str, dst_ip: str,
                                   src_port: int, dst_port: int) -> List[scapy.Packet]:
        """Generate interactive session traffic, such as SSH."""
        packets = []

        # Add the TCP handshake.
        packets.extend(self.generate_tcp_handshake(src_ip, dst_ip, src_port, dst_port))

        # Simulate exchanges of small packets.
        for _ in range(10):
            # Client command.
            cmd_packet = (
                Ether()/
                IP(src=src_ip, dst=dst_ip)/
                TCP(sport=src_port, dport=dst_port, flags='PA')/
                Raw(load="ls -l\n")
            )
            packets.append(cmd_packet)

            # Server response.
            resp_packet = (
                Ether()/
                IP(src=dst_ip, dst=src_ip)/
                TCP(sport=dst_port, dport=src_port, flags='PA')/
                Raw(load="total 0\n-rw-r--r-- 1 user user 0 Jan 1 00:00 file.txt\n")
            )
            packets.append(resp_packet)

        # Add the TCP close sequence.
        packets.extend(self.generate_tcp_teardown(src_ip, dst_ip, src_port, dst_port))

        return packets

    def generate_dns_traffic(self, src_ip: str, dst_ip: str,
                           src_port: int) -> List[scapy.Packet]:
        """Generate DNS query traffic."""
        # DNS query.
        dns_query = (
            Ether()/
            IP(src=src_ip, dst=dst_ip)/
            UDP(sport=src_port, dport=53)/
            Raw(load=b"\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
                     b"\x03www\x07example\x03com\x00\x00\x01\x00\x01")
        )

        # DNS response.
        dns_response = (
            Ether()/
            IP(src=dst_ip, dst=src_ip)/
            UDP(sport=53, dport=src_port)/
            Raw(load=b"\x00\x01\x81\x80\x00\x01\x00\x01\x00\x00\x00\x00"
                     b"\x03www\x07example\x03com\x00\x00\x01\x00\x01"
                     b"\xc0\x0c\x00\x01\x00\x01\x00\x00\x04\x00\x00\x04"
                     b"\x0a\x00\x00\x0a")
        )

        return [dns_query, dns_response]

    def generate_video_stream(self, src_ip: str, dst_ip: str,
                            src_port: int, dst_port: int,
                            duration: int = 10) -> List[scapy.Packet]:
        """Generate video streaming traffic."""
        packets = []

        # Simulate RTP packets.
        packet_interval = 0.04  # 25fps
        packet_size = 1200  # Typical RTP packet size.

        for _ in range(int(duration/packet_interval)):
            rtp_packet = (
                Ether()/
                IP(src=src_ip, dst=dst_ip)/
                UDP(sport=src_port, dport=dst_port)/
                Raw(load="X" * packet_size)
            )
            packets.append(rtp_packet)

        return packets

    def generate_all_traffic(self, output_file: str) -> None:
        """Generate all traffic types and save a PCAP file.

        Args:
            output_file: Output filename.
        """
        all_packets = []
        src_ip = "192.168.1.100"
        dst_ip = "192.168.1.200"
        base_port = 10000

        # Generate different traffic types.
        traffic_generators = [
            (self.generate_http_traffic, (src_ip, dst_ip, base_port)),
            (self.generate_large_file_transfer, (src_ip, dst_ip, base_port+1)),
            (self.generate_interactive_session, (src_ip, dst_ip, base_port+2)),
            (self.generate_dns_traffic, (src_ip, dst_ip, base_port+3)),
            (self.generate_video_stream, (src_ip, dst_ip, base_port+4, 5004))
        ]

        for generator, args in traffic_generators:
            try:
                packets = generator(*args)
                all_packets.extend(packets)
                logging.info(f"Generated {generator.__name__} traffic: {len(packets)} packets")
            except Exception as e:
                logging.error(f"Failed to generate {generator.__name__} traffic: {str(e)}")

        # Sort by timestamp.
        all_packets.sort(key=lambda p: p.time)

        # Save to a file.
        output_path = self.output_dir / output_file
        scapy.wrpcap(str(output_path), all_packets)
        logging.info(f"Saved {len(all_packets)} packets to {output_path}")

def main():
    """Run the command-line entry point."""
    generator = PcapGenerator("test_pcaps")

    # Generate all traffic types.
    generator.generate_all_traffic("mixed_traffic.pcap")

    # Generate individual traffic types for testing.
    src_ip = "192.168.1.100"
    dst_ip = "192.168.1.200"
    src_port = 12345

    # HTTP traffic.
    packets = generator.generate_http_traffic(src_ip, dst_ip, src_port)
    scapy.wrpcap("test_pcaps/http_traffic.pcap", packets)

    # Large file transfer.
    packets = generator.generate_large_file_transfer(src_ip, dst_ip, src_port, 80)
    scapy.wrpcap("test_pcaps/file_transfer.pcap", packets)

    # Interactive session.
    packets = generator.generate_interactive_session(src_ip, dst_ip, src_port, 22)
    scapy.wrpcap("test_pcaps/interactive.pcap", packets)

    # DNS traffic.
    packets = generator.generate_dns_traffic(src_ip, dst_ip, src_port)
    scapy.wrpcap("test_pcaps/dns_traffic.pcap", packets)

    # Video stream.
    packets = generator.generate_video_stream(src_ip, dst_ip, src_port, 5004)
    scapy.wrpcap("test_pcaps/video_stream.pcap", packets)

if __name__ == "__main__":
    main()
