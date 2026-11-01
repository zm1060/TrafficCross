"""Locate original IP bytes without reserializing protocol headers."""


def extract_ip_packet(raw_data: bytes, link_type: str = 'raw_ip') -> bytes:
    """Strip explicit link framing and validate the captured IP header.

    Truncated payloads are allowed when the complete IP header is available.
    Ethernet padding is excluded using the length in the IP header. No checksums
    or other protocol fields are recalculated.
    """
    raw_data = bytes(raw_data)
    offset = 0
    ether_type = None
    if link_type == 'ethernet':
        if len(raw_data) < 14:
            raise ValueError('Truncated Ethernet header')
        ether_type = int.from_bytes(raw_data[12:14], 'big')
        offset = 14
    elif link_type == 'sll':
        if len(raw_data) < 16:
            raise ValueError('Truncated Linux cooked capture header')
        ether_type = int.from_bytes(raw_data[14:16], 'big')
        offset = 16
    elif link_type != 'raw_ip':
        raise ValueError(f'Unsupported link type: {link_type}')

    if ether_type is not None:
        while ether_type in (0x8100, 0x88a8, 0x9100):
            if len(raw_data) < offset + 4:
                raise ValueError('Truncated VLAN tag')
            ether_type = int.from_bytes(raw_data[offset + 2:offset + 4], 'big')
            offset += 4
        if ether_type not in (0x0800, 0x86dd):
            raise ValueError('Frame does not contain IPv4 or IPv6')

    packet = raw_data[offset:]
    if not packet:
        raise ValueError('Missing IP header')
    version = packet[0] >> 4
    if ether_type is not None and version != (4 if ether_type == 0x0800 else 6):
        raise ValueError('IP version does not match link-layer protocol')
    if version == 4:
        header_length = (packet[0] & 0x0f) * 4
        if header_length < 20 or len(packet) < header_length:
            raise ValueError('Invalid or truncated IPv4 header')
        packet_length = int.from_bytes(packet[2:4], 'big')
        if packet_length < header_length:
            raise ValueError('Invalid IPv4 total length')
    elif version == 6:
        if len(packet) < 40:
            raise ValueError('Truncated IPv6 header')
        payload_length = int.from_bytes(packet[4:6], 'big')
        if payload_length == 0 and len(packet) > 40:
            raise ValueError('IPv6 jumbograms are not supported')
        packet_length = 40 + payload_length
    else:
        raise ValueError('Input does not start with an IPv4 or IPv6 header')
    return packet[:packet_length]
