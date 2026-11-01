from typing import Dict, Optional, Any
import scapy.all as scapy
import ipaddress
import logging
from dataclasses import dataclass


@dataclass
class AnonymizedAddresses:
    """Anonymized endpoint address settings."""
    ipv4_src: str = "10.0.0.1"
    ipv4_dst: str = "10.0.0.2"
    ipv6_src: str = "2001:db8::1"
    ipv6_dst: str = "2001:db8::2"


class AddressAnonymizer:
    """IP address anonymizer."""

    def __init__(self, addresses: Optional[AnonymizedAddresses] = None):
        """Initialize the address anonymizer.

        Args:
            addresses: Optional anonymized address settings.
        """
        self.addresses = addresses or AnonymizedAddresses()
        self._validate_addresses()

    def _validate_addresses(self) -> None:
        """Validate anonymized addresses.

        Raises:
            ValueError: If an address is invalid.
        """
        try:
            # Validate IPv4 addresses.
            ipaddress.IPv4Address(self.addresses.ipv4_src)
            ipaddress.IPv4Address(self.addresses.ipv4_dst)

            # Validate IPv6 addresses.
            ipaddress.IPv6Address(self.addresses.ipv6_src)
            ipaddress.IPv6Address(self.addresses.ipv6_dst)
        except ValueError as e:
            raise ValueError(f"Invalid anonymized address: {str(e)}")


def anonymize_flow_info(original_info: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize flow endpoints while preserving direction.

    Args:
        original_info: Original flow metadata.

    Returns:
        Flow metadata with anonymized IP addresses and ports.

    Raises:
        KeyError: If required fields are missing.
    """
    try:
        required_fields = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'first_src_ip']
        missing_fields = [field for field in required_fields if field not in original_info]
        if missing_fields:
            raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")

        return {
            'src_ip': "10.0.0.1",
            'dst_ip': "10.0.0.2",
            'src_port': 0,
            'dst_port': 0,
            'first_src_ip': original_info['first_src_ip']  # Keep the original first_src_ip to determine direction.
        }
    except Exception as e:
        logging.error(f"Failed to anonymize flow metadata: {str(e)}")
        raise


def anonymize_packet(packet: scapy.Packet, first_src_ip: Optional[str] = None) -> scapy.Packet:
    """Anonymize packet IP addresses and ports while preserving direction.

    Args:
        packet: Packet to anonymize.
        first_src_ip: Source IP of the first packet, used to determine direction.
            If None, use the source IP of the current packet as the reference.

    Returns:
        An anonymized copy of the packet.

    Raises:
        ValueError: If the packet is invalid.
    """
    if not isinstance(packet, scapy.Packet):
        raise ValueError("Invalid packet type")

    try:
        anonymized_packet = packet.copy()

        # Get the current packet's IP layer.
        if scapy.IP in anonymized_packet:
            current_src = anonymized_packet[scapy.IP].src

            # Determine whether the packet is in the reverse direction.
            if first_src_ip is not None:
                is_reverse = (current_src != first_src_ip)
            else:
                is_reverse = False

            # Set anonymized IP addresses according to direction.
            if not is_reverse:
                # Forward: source to destination.
                anonymized_packet[scapy.IP].src = "10.0.0.1"
                anonymized_packet[scapy.IP].dst = "10.0.0.2"
            else:
                # Reverse: destination to source.
                anonymized_packet[scapy.IP].src = "10.0.0.2"
                anonymized_packet[scapy.IP].dst = "10.0.0.1"

        elif scapy.IPv6 in anonymized_packet:
            current_src = anonymized_packet[scapy.IPv6].src

            # Determine whether the packet is in the reverse direction.
            if first_src_ip is not None:
                is_reverse = (current_src != first_src_ip)
            else:
                is_reverse = False

            # Set anonymized IP addresses according to direction.
            if not is_reverse:
                # Forward.
                anonymized_packet[scapy.IPv6].src = "2001:db8::1"
                anonymized_packet[scapy.IPv6].dst = "2001:db8::2"
            else:
                # Reverse.
                anonymized_packet[scapy.IPv6].src = "2001:db8::2"
                anonymized_packet[scapy.IPv6].dst = "2001:db8::1"
        else:
            logging.warning("Packet contains neither an IP nor an IPv6 layer")
            return anonymized_packet

        # Zero TCP/UDP ports.
        if scapy.TCP in anonymized_packet:
            anonymized_packet[scapy.TCP].sport = 0
            anonymized_packet[scapy.TCP].dport = 0
        elif scapy.UDP in anonymized_packet:
            anonymized_packet[scapy.UDP].sport = 0
            anonymized_packet[scapy.UDP].dport = 0

        return anonymized_packet

    except Exception as e:
        logging.error(f"Failed to anonymize packet: {str(e)}")
        raise ValueError(f"Failed to anonymize packet: {str(e)}")


def is_private_ip(ip: str) -> bool:
    """Check whether an IP address is private.

    Args:
        ip: IP address string.

    Returns:
        Whether the address is private.

    Raises:
        ValueError: If the IP address is invalid.
    """
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private
    except ValueError as e:
        logging.error(f"Invalid IP address {ip}: {str(e)}")
        raise
