# CycleGAN — CT ↔ MRI Translation

Unpaired image-to-image translation between CT scans and MRI brain scans, trained on the [darren2020 CT-to-MRI dataset](https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan) using a standard CycleGAN objective with LSGAN discriminators.

---

## Architecture

### Generator (`G_AB` / `G_BA`)

Both generators share the same architecture. Each is a ResNet-based encoder-decoder operating on single-channel (grayscale) 128×128 inputs.

**Encoder** — three convolutional stages progressively compress the spatial resolution while expanding feature depth:

| Stage | Operation | Channels | Stride | Output size |
|-------|-----------|----------|--------|-------------|
| 1 | Conv2d 7×7 + InstanceNorm + ReLU | 1 → 64 | 1 | 128×128 |
| 2 | Conv2d 3×3 + InstanceNorm + ReLU | 64 → 128 | 2 | 64×64 |
| 3 | Conv2d 3×3 + InstanceNorm + ReLU | 128 → 256 | 2 | 32×32 |

**Bottleneck** — six residual blocks at 256 channels. Each block applies two 3×3 convolutions with InstanceNorm and a skip connection (`x + F(x)`), preserving fine-grained structural detail across the translation.

**Decoder** — two transposed convolutional stages upsample back to the original resolution, followed by a Tanh output:

| Stage | Operation | Channels | Output size |
|-------|-----------|----------|-------------|
| 1 | ConvTranspose2d 3×3 + InstanceNorm + ReLU | 256 → 128 | 64×64 |
| 2 | ConvTranspose2d 3×3 + InstanceNorm + ReLU | 128 → 64 | 128×128 |
| Out | Conv2d 7×7 + Tanh | 64 → 1 | 128×128 |

> **Why InstanceNorm?** Batch statistics are noisy for small GAN batches (bs=8). InstanceNorm normalises per-image, which stabilises training and preserves domain-specific contrast — critical for medical imaging where HU values (CT) and T1/T2 intensities (MRI) carry diagnostic meaning.

---

### Discriminator (`D_A` / `D_B`) — PatchGAN

Rather than classifying an entire image as real or fake, the PatchGAN discriminator produces a spatial map of real/fake decisions. Each output cell covers a receptive field of the input — penalising local texture mismatches.

| Layer | Channels | Kernel | Stride | Norm |
|-------|----------|--------|--------|------|
| Conv2d | 1 → 64 | 4×4 | 2 | — |
| Conv2d | 64 → 128 | 4×4 | 2 | InstanceNorm |
| Conv2d | 128 → 256 | 4×4 | 2 | InstanceNorm |
| Conv2d | 256 → 512 | 4×4 | 1 | InstanceNorm |
| Conv2d | 512 → 1 | 4×4 | 1 | — |

All hidden layers use `LeakyReLU(0.2)`. The output is an unnormalised patch map; the GAN loss is computed against all-ones (real) or all-zeros (fake) targets of matching shape.

---

## Loss Functions

The total training objective combines three terms:

```
L_total = L_GAN + λ_cycle × L_cycle + λ_identity × L_identity
```

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| **GAN (LSGAN)** | MSE vs. 1/0 targets | 1× | Push generated images into the target domain |
| **Cycle consistency** | L1(G_BA(G_AB(A)), A) + L1(G_AB(G_BA(B)), B) | λ=10 | Prevent mode collapse; preserve content |
| **Identity** | L1(G_AB(B), B) + L1(G_BA(A), A) | λ=5 | Preserve colour/intensity when input is already in target domain |

> **LSGAN vs BCE**: MSE loss on discriminator outputs (LSGAN) provides smoother gradients than binary cross-entropy and is less prone to vanishing gradients when the discriminator becomes too strong — a common failure mode in medical image GANs.

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Image size | 128×128, grayscale (`L` mode) |
| Normalisation | `(x − 0.5) / 0.5` → range [−1, 1] |
| Batch size | 8 |
| Epochs | 50 |
| Optimizer | Adam, lr=2e-4, β=(0.5, 0.999) |
| Precision | Mixed (AMP + GradScaler) |
| Hardware | Kaggle Tesla T4 |
| Checkpoint interval | Every 5 epochs → HuggingFace Hub |

Domains are split independently (80/20) and shuffled separately to maintain the unpaired nature of the dataset. The training loop updates generators first, then each discriminator independently with detached fakes.

---

## Data

- **Source**: [darren2020/ct-to-mri-cgan](https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan)
- **Domain A**: CT brain scans
- **Domain B**: MRI brain scans
- **Split**: 80% train / 20% validation, per domain, `random_state=42`
- **Preprocessing**: Resize to 128×128 (BILINEAR), convert to grayscale, normalise to [−1, 1]

---
## Results
<img width="1117" height="385" alt="image" src="https://github.com/user-attachments/assets/a6cbcdbd-0edb-470e-b743-caeb1b666a14" />
<img width="1134" height="388" alt="image" src="https://github.com/user-attachments/assets/418a993f-d1a0-42d6-8a0b-90cd365391df" />
<img width="1134" height="388" alt="image" src="https://github.com/user-attachments/assets/103fc5e5-e93b-4f15-a587-6bfda5e01b74" />



## Evaluation Metrics

Post-training evaluation computes pixel-level similarity between real CT inputs and their generated MRI counterparts on the validation set:

- **SSIM** (Structural Similarity Index) — captures luminance, contrast, and structural similarity
- **PSNR** (Peak Signal-to-Noise Ratio) — measures reconstruction fidelity in dB

> Note: because this is unpaired translation, SSIM/PSNR are computed against the source domain input as a proxy metric, not against a ground-truth paired target. They reflect structural preservation rather than perceptual realism.

---

## Checkpoints

Model checkpoints are uploaded to HuggingFace Hub every 5 epochs:

```
Awan8754/cycle-gan
└── checkpoints/
    ├── G_AB_epoch5.pth   … G_AB_epoch50.pth
    └── G_BA_epoch5.pth   … G_BA_epoch50.pth
```

---

## Quick Start

```python
import torch
from model import Generator  # your Generator class

device = "cuda" if torch.cuda.is_available() else "cpu"

G_AB = Generator(num_residual_blocks=6).to(device)
G_AB.load_state_dict(torch.load("G_AB_epoch50.pth", map_location=device))
G_AB.eval()

# ct_tensor: [1, 1, 128, 128], values in [-1, 1]
with torch.no_grad():
    mri_fake = G_AB(ct_tensor)
```

---

## Project Structure

```
├── cycle-gan.ipynb      # full training notebook (Kaggle)
├── README.md
└── checkpoints/         # synced to HuggingFace Hub
```

---

## References

- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017
- Mao et al., *Least Squares Generative Adversarial Networks*, ICCV 2017
