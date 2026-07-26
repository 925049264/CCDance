# Graph-Transformer / X-DANCENET Baseline

**Reference:** Han et al., "Explainable Real Time Sensor Graph Transformer for Dance Recognition," Scientific Reports, 2026.

## Original X-DANCENET Architecture (IMU-based)

The original X-DANCENET was designed for real-time dance recognition from wearable IMU sensors. Its full pipeline includes:

1. **Sensor Normalization (ASN):** Automatic sensor normalization converting raw quaternion/accelerometer/gyroscope data to a standardized reference frame, followed by Kalman filtering and Butterworth low-pass filtering for denoising.

2. **Multi-Scale Motion Feature Extraction (MMFE):** Tempo-conditioned dilated convolutional bank extracting features at multiple temporal scales, guided by beat/downbeat detection.

3. **Spatio-Temporal Graph Attention Core:**
   - Dynamic graph construction: learns adjacency between sensor nodes per timestep via attention
   - Spatial attention: self-attention across sensor nodes at each timestep
   - Temporal attention: self-attention across timesteps for each sensor node
   - Residual connections + LayerNorm around each sub-layer

4. **Explainable Decision Layer:**
   - Prototype classification: compares embedding to learned prototype vectors per class
   - Confidence estimation: outputs prediction confidence score
   - Saliency mapping: highlights which sensor nodes/timesteps drive the decision
   - Combined loss: cross-entropy + prototype clustering + confidence regularization

## Adaptation for CCDance (SMPL Pose Input)

Since the CCDance dataset provides clean SMPL-H pose parameters (69-D axis-angle vectors) rather than raw IMU signals, several simplifications are made:

### Simplifications

| X-DANCENET Component | Our Adaptation | Rationale |
|---|---|---|
| Sensor Normalization (ASN) | **Removed** | SMPL data is already clean; no sensor artifacts to filter |
| Multi-Scale Feature Extraction (MMFE) | **Removed** | SMPL pose operates at 30fps; tempo-conditioned dilation not needed for pose sequences; the transformer's self-attention captures multi-scale temporal patterns natively |
| Dynamic Graph Construction | **Removed** | SMPL joint topology is fixed (23 joints, SMPL skeletal structure); no need to learn adjacency per timestep |
| Spatial Attention (cross-sensor) | **Kept** (cross-joint) | Applied across the 23 SMPL joints per timestep using standard multi-head attention |
| Temporal Attention (cross-time) | **Kept** | Applied across T=300 timesteps per joint |
| Prototype Classification | **Replaced with MLP** | Prototype-based classification adds complexity; standard MLPClassifier (256->128->3) is used instead |
| Confidence/Saliency Output | **Removed** | These are explainability features not required for the grade classification task |
| Training Loss | **CE only** | Original uses CE + prototype clustering + confidence regularization; our adaptation uses standard cross-entropy |

### Architecture (This Baseline)

```
SMPL Pose (B, T, 69)
    |
    v
Reshape to (B, T, 23, 3)              # Per-joint axis-angle
    |
    v
Joint Embedding: Linear(3 -> 256)      # Project each joint to d_model
    |
    v
Positional Encoding                    # Sinusoidal position encodings
    |
    v
[4 layers of dual attention]:
    |-- Spatial Attention (across 23 joints, per timestep)
    |-- Temporal Attention (across T timesteps, per joint)
    |-- Feed-Forward (d_model -> 4*d_model -> d_model)
    |-- Residual + LayerNorm around each sub-layer
    |
    v
Global Mean Pooling (over joints and time) -> (B, 256)
    |
    v
MLPClassifier(256 -> 128 -> 3)         # Grade logits (A, B, C)
```

### Hyperparameters

All hyperparameters follow the unified baseline config unless specified:

| Parameter | Value | Notes |
|---|---|---|
| d_model | 256 | Transformer embedding dimension |
| nhead | 8 | Attention heads |
| num_layers | 4 | Dual attention layers |
| dropout | 0.1 | Dropout rate |
| lr | 1e-3 | AdamW learning rate |
| batch_size | 16 | Training batch size |
| epochs | 50 (classification) / 100 (generation) | Unified across baselines |
| weight_decay | 1e-4 | AdamW weight decay |
| scheduler | Cosine annealing | T_max = n_epochs |
| early_stop | 10 patience | Based on validation accuracy/loss |

## File Structure

```
graph_transformer/
├── README.md
├── classification/
│   ├── model.py          # GraphTransformerModel (encoder + classifier)
│   ├── train.py          # Classification training script (5 seeds)
│   └── test.py           # Classification evaluation script
├── generation/
│   ├── model.py          # GraphTransformerGenerator (encoder + LSTM decoder + tokenizer)
│   └── train.py          # Generation training script (5 seeds)
├── results/              # Classification results (per seed)
├── logs/                 # Training logs
├── checkpoints/          # Model checkpoints
└── generation_results/   # Generation results (per seed)
```

## Usage

```bash
# Classification
cd experiments/baselines/graph_transformer/classification/
python train.py --gpu 0
python test.py --gpu 0 --seed 42

# Generation
cd experiments/baselines/graph_transformer/generation/
python train.py --gpu 0
```

## Dependencies

- PyTorch 2.x, numpy, scikit-learn
- Shared infrastructure: `shared/models.py` (GraphTransformerEncoder, MLPClassifier, LSTMDecoder)

## Key Differences from Other Baselines

- **vs. ST-GCN:** Graph-Transformer uses attention (learned relationships) rather than fixed-graph convolution. ST-GCN has a predefined adjacency matrix; Graph-Transformer lets the attention mechanism discover joint relationships.
- **vs. Pose Transformer:** The standard Transformer encoder (used by the Transformer AQA baseline) operates on the full 69-D pose vector as flat tokens. The Graph-Transformer explicitly preserves the joint structure (23 joints x 3 DoF) and applies spatial attention across joints, which is more parameter-efficient (256-dim per joint vs 512-dim per timestep).
- **vs. USDL/GDLT:** These are score-distribution or group-aware methods; Graph-Transformer is purely attention-based with explicit joint-structure modeling.
