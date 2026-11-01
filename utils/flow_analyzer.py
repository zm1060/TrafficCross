from typing import List, Dict, Tuple, Any, DefaultDict, Optional
from collections import defaultdict
import logging
import numpy as np
from dataclasses import dataclass


@dataclass
class FlowConfig:
    """Flow configuration settings."""
    flow_key: str = '5-tuple'
    min_packets: int = 5
    inactive_timeout: float = 60.0
    max_packets_per_flow: int = 500
    bidirectional: bool = True


class FlowAnalyzer:
    """Aggregate packets into flows and compute traffic statistics."""

    def __init__(self, flow_config: Dict[str, Any]):
        """Initialize the flow analyzer.

        Args:
            flow_config: Flow configuration dictionary.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        # Validate and initialize configuration.
        self.config = FlowConfig(
            flow_key=flow_config.get('flow_key', FlowConfig.flow_key),
            min_packets=flow_config.get('min_packets', FlowConfig.min_packets),
            inactive_timeout=float(flow_config.get('inactive_timeout', FlowConfig.inactive_timeout)),
            max_packets_per_flow=flow_config.get('max_packets_per_flow', FlowConfig.max_packets_per_flow),
            bidirectional=flow_config.get('bidirectional', FlowConfig.bidirectional)
        )

        # Validate the configuration.
        self._validate_config()

        # Initialize statistics.
        self.stats = {
            'total_packets': 0,
            'valid_flows': 0,
            'invalid_flows': 0,
            'tcp_flows': 0,
            'udp_flows': 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        if self.config.min_packets < 1:
            raise ValueError("min_packets must be greater than 0")
        if self.config.inactive_timeout <= 0:
            raise ValueError("inactive_timeout must be greater than 0")
        if self.config.max_packets_per_flow < self.config.min_packets:
            raise ValueError("max_packets_per_flow must be greater than or equal to min_packets")
        if self.config.flow_key not in ['5-tuple', '3-tuple']:
            raise ValueError("flow_key must be 5-tuple or 3-tuple")

    def get_flow_key(self, packet: Dict[str, Any], is_forward: bool = True) -> Tuple:
        """Generate a flow identifier using the configured grouping method.

        Args:
            packet: Packet metadata.
            is_forward: Whether to use the forward direction.

        Returns:
            A flow identifier tuple.

        Raises:
            KeyError: If required packet fields are missing.
        """
        try:
            if is_forward:
                if self.config.flow_key == '5-tuple':
                    return (packet['protocol'],
                            packet['src_ip'],
                            packet['src_port'],
                            packet['dst_ip'],
                            packet['dst_port'])
                else:  # 3-tuple
                    return (packet['protocol'],
                            packet['src_ip'],
                            packet['dst_ip'])
            else:
                if self.config.flow_key == '5-tuple':
                    return (packet['protocol'],
                            packet['dst_ip'],
                            packet['dst_port'],
                            packet['src_ip'],
                            packet['src_port'])
                else:  # 3-tuple
                    return (packet['protocol'],
                            packet['dst_ip'],
                            packet['src_ip'])
        except KeyError as e:
            raise KeyError(f"Packet is missing required fields: {str(e)}")

    def analyze_flows(self, packets: List[Dict[str, Any]], category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Analyze packets and group them into flows.

        Args:
            packets: List of packets.
            category: Optional traffic class.

        Returns:
            A list of processed flows.

        Raises:
            ValueError: If the input packet list is empty.
        """
        if not packets:
            raise ValueError("Input packet list is empty")

        logging.info(f"Grouping {len(packets)} packets")
        self.stats['total_packets'] = len(packets)

        # Create flow containers.
        flow_dict: DefaultDict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
        flow_times: Dict[Tuple, float] = {}  # Track the last timestamp for each flow.
        processed_flows: List[Dict[str, Any]] = []  # Store completed flows.

        # Sort packets by timestamp.
        packets.sort(key=lambda x: x['timestamp'])

        # Process each packet.
        for packet in packets:
            try:
                current_time = packet['timestamp']

                # Get the flow identifier.
                forward_key = self.get_flow_key(packet, True)
                reverse_key = self.get_flow_key(packet, False) if self.config.bidirectional else None

                # Check for an existing flow.
                flow_key = None
                if self.config.bidirectional:
                    # Check forward and reverse flows.
                    if forward_key in flow_dict:
                        flow_key = forward_key
                    elif reverse_key in flow_dict:
                        flow_key = reverse_key
                else:
                    # Check only forward flows.
                    if forward_key in flow_dict:
                        flow_key = forward_key

                # Check for an inactive flow.
                if flow_key and flow_key in flow_times:
                    # If the gap exceeds the timeout, finish the current flow and start another.
                    if current_time - flow_times[flow_key] > self.config.inactive_timeout:
                        if len(flow_dict[flow_key]) >= self.config.min_packets:
                            flow = self._process_flow(flow_key, flow_dict[flow_key], category)
                            if flow:
                                processed_flows.append(flow)
                                self.stats['valid_flows'] += 1
                                if flow['flow_info']['protocol'] == 'tcp':
                                    self.stats['tcp_flows'] += 1
                                else:
                                    self.stats['udp_flows'] += 1
                        else:
                            self.stats['invalid_flows'] += 1
                        flow_dict[flow_key] = []
                        flow_key = forward_key

                # Initialize new flows.
                if not flow_key:
                    flow_key = forward_key

                # Add the packet to the flow.
                flow_dict[flow_key].append(packet)
                flow_times[flow_key] = current_time

                # When the packet limit is reached, finish the flow and start another.
                if len(flow_dict[flow_key]) >= self.config.max_packets_per_flow:
                    if len(flow_dict[flow_key]) >= self.config.min_packets:
                        flow = self._process_flow(flow_key, flow_dict[flow_key], category)
                        if flow:
                            processed_flows.append(flow)
                            self.stats['valid_flows'] += 1
                            if flow['flow_info']['protocol'] == 'tcp':
                                self.stats['tcp_flows'] += 1
                            else:
                                self.stats['udp_flows'] += 1
                    else:
                        self.stats['invalid_flows'] += 1
                    flow_dict[flow_key] = []
                    flow_times[flow_key] = current_time

            except Exception as e:
                logging.debug(f"Error processing packet: {str(e)}")
                continue

        # Process remaining flows.
        for flow_key, flow_packets in flow_dict.items():
            if len(flow_packets) >= self.config.min_packets:
                try:
                    flow = self._process_flow(flow_key, flow_packets, category)
                    if flow:
                        processed_flows.append(flow)
                        self.stats['valid_flows'] += 1
                        if flow['flow_info']['protocol'] == 'tcp':
                            self.stats['tcp_flows'] += 1
                        else:
                            self.stats['udp_flows'] += 1
                except Exception as e:
                    logging.debug(f"Error processing flow {flow_key}: {str(e)}")
                    self.stats['invalid_flows'] += 1
                    continue
            else:
                self.stats['invalid_flows'] += 1

        # Compute and log statistics.
        self._log_flow_stats()

        return processed_flows

    def _process_flow(self, flow_key: Tuple, flow_packets: List[Dict[str, Any]], category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Process a single flow."""
        if not flow_packets:
            raise ValueError("Flow packet list is empty")

        try:
            # Sort by timestamp.
            flow_packets.sort(key=lambda x: x['timestamp'])
            first_packet = flow_packets[0]
            last_packet = flow_packets[-1]

            # Compute flow duration.
            flow_duration = last_packet['timestamp'] - first_packet['timestamp']

            # Compute packet and payload sizes.
            total_bytes = 0
            total_payload_bytes = 0
            packet_sizes = []
            payloads = []

            for packet in flow_packets:
                # Use IP total length.
                packet_size = packet['length']  # IP total length.
                payload_size = len(packet.get('payload', b''))

                # Record size metadata.
                packet_sizes.append(packet_size)
                total_bytes += packet_size
                payloads.append(payload_size)
                total_payload_bytes += payload_size

                # Debug logging.
                logging.debug(f"Flow packet details:")
                logging.debug(f"  - Protocol: {packet['protocol']}")
                logging.debug(f"  - IP Total Length: {packet_size}")
                logging.debug(f"  - Raw Length: {packet.get('raw_length', 0)}")
                logging.debug(f"  - Payload size: {payload_size}")

            # Build flow metadata.
            flow_info = {
                'protocol': first_packet['protocol'],
                'proto_num': 6 if first_packet['protocol'] == 'tcp' else 17,
                'packet_count': len(flow_packets),
                'relative_start': 0.0,
                'relative_end': flow_duration,
                'flow_duration': flow_duration,
                'total_bytes': total_bytes,
                'total_payload_bytes': total_payload_bytes,
                'original_info': {
                    'src_ip': first_packet['src_ip'],
                    'dst_ip': first_packet['dst_ip'],
                    'src_port': first_packet['src_port'],
                    'dst_port': first_packet['dst_port'],
                    'first_src_ip': first_packet['src_ip']
                }
            }

            # Compute directions.
            directions = [1 if p['src_ip'] == first_packet['src_ip'] else -1 for p in flow_packets]

            # Compute timing metadata.
            relative_times = [float(p['timestamp'] - first_packet['timestamp']) for p in flow_packets]
            inter_arrival_times = [t2 - t1 for t1, t2 in zip(relative_times[:-1], relative_times[1:])]

            # Get TCP-specific fields.
            flags = [p.get('flags', 0) if p.get('protocol') == 'tcp' else 0 for p in flow_packets]
            window_sizes = [p.get('window', 0) if p.get('protocol') == 'tcp' else 0 for p in flow_packets]

            # Add sequence metadata.
            flow_info['sequence_info'] = {
                'packet_sizes': packet_sizes,
                'relative_times': relative_times,
                'inter_arrival_times': inter_arrival_times,
                'directions': directions,
                'payloads': payloads,
                'flags': flags,
                'window_sizes': window_sizes
            }

            # Add statistics.
            flow_info['statistical_info'] = {
                'avg_packet_size': float(np.mean(packet_sizes)),
                'min_packet_size': min(packet_sizes),
                'max_packet_size': max(packet_sizes),
                'std_packet_size': float(np.std(packet_sizes)),
                'avg_iat': float(np.mean(inter_arrival_times)) if inter_arrival_times else 0.0,
                'min_iat': min(inter_arrival_times) if inter_arrival_times else 0.0,
                'max_iat': max(inter_arrival_times) if inter_arrival_times else 0.0,
                'std_iat': float(np.std(inter_arrival_times)) if inter_arrival_times else 0.0
            }

            return {
                'flow_info': flow_info,
                'packets': flow_packets,
                'directions': directions,
                'category': category
            }

        except Exception as e:
            logging.error(f"Failed to process flow: {str(e)}")
            return None

    def _log_flow_stats(self) -> None:
        """Log flow statistics."""
        logging.info(f"Traffic statistics:")
        logging.info(f"- Total packets: {self.stats['total_packets']}")
        logging.info(f"- TCP flows: {self.stats['tcp_flows']}")
        logging.info(f"- UDP flows: {self.stats['udp_flows']}")
        logging.info(f"- Valid flows: {self.stats['valid_flows']}")
        logging.info(f"- Invalid flows: {self.stats['invalid_flows']}")
        if self.stats['total_packets'] > 0:
            avg_packets = self.stats['total_packets'] / (self.stats['valid_flows'] or 1)
            logging.info(f"- Average packets per flow: {avg_packets:.2f}")
        logging.info(f"- Flow grouping method: {self.config.flow_key}")
