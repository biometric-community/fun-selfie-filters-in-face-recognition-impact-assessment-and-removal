# REPORT — Fun Selfie Filters in Face Recognition

**Protocol:** `full`

Lab twin: synthetic selfie filters (coverage × placement) on LFW,
impact on verification, and mask-guided removal recovery.

## Training

- Epochs: 8
- Train images: 141610
- Identities: 7124
- Elapsed (s): 1118.8

![Training](outputs/figures/full/train_loss.svg)

## Clean verification

| Dataset | Accuracy | EER | N |
|---------|----------|-----|---|
| calfw | 52.68% | 47.32% | 6000 |
| cplfw | 54.70% | 45.30% | 6000 |
| lfw | 65.22% | 34.78% | 6000 |

## Filter impact (LFW)

| Condition | Accuracy | EER | Intensity |
|-----------|----------|-----|-----------|
| high/eyes | 55.50% | 44.50% | 0.197 |
| high/mouth | 64.25% | 35.75% | 0.113 |
| high/nose | 64.78% | 35.22% | 0.105 |
| high/total | 54.30% | 45.70% | 0.491 |
| low/eyes | 57.12% | 42.88% | 0.096 |
| low/mouth | 64.73% | 35.27% | 0.099 |
| low/nose | 64.97% | 35.03% | 0.099 |
| low/total | 63.93% | 36.07% | 0.098 |
| medium/eyes | 55.50% | 44.50% | 0.197 |
| medium/mouth | 64.25% | 35.75% | 0.113 |
| medium/nose | 64.78% | 35.22% | 0.105 |
| medium/total | 54.60% | 45.40% | 0.277 |

![Impact](outputs/figures/full/filter_impact.svg)

## Removal

| Condition | EER before | EER after | Acc Δ |
|-----------|------------|-----------|-------|
| high/eyes | 44.50% | 43.82% | 0.68pp |
| high/total | 45.70% | 45.10% | 0.60pp |

![Removal](outputs/figures/full/removal.svg)

## Notes

- Paper uses FRGC/FERET app filters + GAN removal; we use CelebA/LFW + synthetic overlays.
- See `DEVIATIONS.md`.
