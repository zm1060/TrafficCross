# TrafficCross

TrafficCross is a multimodal network traffic classification project. It converts
PCAP captures into traffic images, statistical flow features, and packet sequences,
then trains classifiers using individual modalities or their combination.

## Installation

Use Python 3.10 or later and install the dependencies:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

Training uses PyTorch and selects CUDA when available. For optional Weights &
Biases logging, install `wandb` with `python -m pip install wandb`; logging runs
in offline mode. The optional PCAP merger requires the Wireshark `mergecap`
executable on `PATH`.

## Prepare a dataset

For directory-based labels, place captures in class subdirectories:

```text
datasets/
└── ISCXVPN-Service/
    ├── chat/
    │   ├── capture1.pcap
    │   └── capture2.pcapng
    └── video/
        └── capture3.pcap
```

Edit `config.yaml` to set the dataset name, input and output directories, split
ratios, image representation, and training settings. `${dataset_name}` is expanded
from `data.dataset_name`. The preprocessing pipeline groups TCP and UDP packets
into flows and splits flows within each class using `data.split_seed`.

For IP-based labels, set `data.labeling.mode` to `ip` and provide a JSON mapping
through `data.labeling.ip_label_file`. Keys may be individual IP addresses or IP
pairs separated by `->` or `,`; pairs match both directions.

Generate packed datasets:

```bash
python main.py --config config.yaml
```

The default output directory is `processed/ISCXVPN-Service`, with
files named `train_packed_000.pkl`, `val_packed_000.pkl`, and
`test_packed_000.pkl`. Additional chunks receive increasing numeric suffixes.
Each file contains `flows`, `images`, and `mapping` entries. Training scripts
merge all chunks for the requested split.

### IP identifier modes

`image.ip_identifier_mode` controls the IP addresses represented in traffic images:

| Mode | Representation |
| --- | --- |
| `original` | Preserve the original IP addresses. |
| `random` | Use deterministic random endpoints per flow, controlled by `image.random_seed`. |
| `role` | Use fixed addresses for endpoint roles and zero TCP/UDP ports. |
| `zero` | Zero source and destination IP addresses. |

These modes control endpoint fields in images; payload bytes are retained.

### Preprocessing

The default configuration uses the following processing steps:

- Bidirectional five-tuple flows, a 60-second inactivity timeout, and at most
  500 packets per flow. Flows with fewer than five packets are excluded.
- `image.ip_identifier_mode: role`: fixed endpoint-role addresses and zero
  TCP/UDP source and destination ports. The first observed source endpoint defines
  the forward role.
- Image bytes start at the IP layer. Ethernet headers, MAC addresses, and VLAN
  tags are excluded. The first five packets contribute 320 bytes each (the first
  80 bytes followed by the next 240), padded as needed and reshaped to 40 x 40.
- Flows are split within each class at an 8:1:1 target ratio using split seed 128.

Change `image.ip_identifier_mode` and `flow.min_packets` to select a different
endpoint representation or minimum flow length. Regenerate packed datasets after
changing preprocessing settings, and use separate output directories for
different configurations.

For direct Python calls, `TrafficImageGenerator` expects IP-layer bytes by default.
Pass `link_type='ethernet'` or `link_type='sll'` for framed input; framing is stripped
before image construction. Packet dictionaries can specify `raw_link_type`.
Explicit `directions` take precedence; otherwise direction is inferred from
`first_src_ip`, or from the first packet source when that argument is absent.
Malformed input and payload-only records are rejected instead of being emitted
as unsanitized image bytes. Noninitial fragments retain their payload unchanged.

## Train and evaluate

Train the multimodal classifier:

```bash
python train_multimodal.py --config config.yaml
```

Run individual modality experiments:

```bash
python train_image_model.py --config config.yaml
python train_flow_model.py --config config.yaml
python train_sequence_model.py --config config.yaml
```

`training.seed` defaults to 17. Override it with `--seed`; Python, NumPy, and
PyTorch use the resolved seed. The data split seed is configured separately through
`data.split_seed` and defaults to 128. To run multiple training seeds on the same
processed dataset in PowerShell:

```powershell
foreach ($seed in 17, 29, 43, 71, 97) {
    python train_multimodal.py --config config.yaml --seed $seed
}
```

The same `--seed` option is supported by all four training scripts. The resolved
seed is recorded in saved configurations and checkpoint training metadata, and
experiment directories, log files, and metrics paths identify the training seed.
Seeding controls random draws; exact numerical reproducibility also depends on
the PyTorch version, hardware, and device operations.

The multimodal network combines a CNN image encoder with channel attention, a
gated MLP for 11 flow statistics, a bidirectional GRU with self-attention for packet
sequences, and cross-modal attention before classification. Model architectures
are defined in the training scripts; `model.sequence.max_seq_length` configures
the sequence-only dataset length. The multimodal dataset uses a length of 128.

Evaluate a checkpoint using its corresponding training script:

```bash
python train_multimodal.py --mode test --config config.yaml --model models/<experiment>/best_multimodal_model.pth --data_dir processed/ISCXVPN-Service
```

Replace the checkpoint path with an actual saved model. Checkpoints and experiment
configuration files are saved under `paths.model_dir`; logs and metrics are saved
under `paths.log_dir`. Training also produces confusion matrices and evaluation
reports. Dataset files and pretrained checkpoints are not included.

## Load a trained model

Load a multimodal checkpoint and optionally export its metadata:

```bash
python load_model.py --model models/<experiment>/best_multimodal_model.pth --config config.yaml --save-info model_info.json
```

`TrafficModelLoader` provides `predict_single`, `predict_batch`, and
`get_feature_representations` for Python callers. Inputs must use the same
preprocessing as the training dataset.

## Included modules

| File | Purpose |
| --- | --- |
| `main.py` | PCAP preprocessing entry point. |
| `train_multimodal.py` | Multimodal dataset, model, training, and evaluation. |
| `train_image_model.py` | Image-only classifier. |
| `train_flow_model.py` | Flow-statistics classifier. |
| `train_sequence_model.py` | Packet-sequence classifier. |
| `load_model.py` | Multimodal checkpoint loading and inference. |
| `utils/pcap_reader.py` | PCAP/PCAP-NG parsing and packet extraction. |
| `utils/flow_analyzer.py` | Flow grouping and statistics. |
| `utils/flow_processor.py` | Preprocessing orchestration and class labeling. |
| `utils/dataset_processor.py` | Deterministic splits and packed dataset writing. |
| `utils/flow_visualizer.py` | Traffic image generation. |
| `utils/anonymizer.py` | Endpoint anonymization helpers. |
| `utils/pcap_generator.py` | Synthetic traffic generation. |
| `utils/pcap_merger.py` | PCAP merging using `mergecap`. |
| `utils/config.py` | Configuration loading and validation. |
| `utils/logging_config.py` | Log setup. |
| `utils/metrics_logger.py` | JSON metrics logging. |
| `utils/packet_utils.py` | Explicit link framing removal and IP header validation. |
| `utils/reproducibility.py` | Training seed configuration. |

## License

[MIT License](LICENSE)
