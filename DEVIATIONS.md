# DEVIATIONS — Fun Selfie Filters

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Source faces | FRGCv2 / FERET | CelebA train + LFW family | Local public payloads |
| D2 | Filters | 10 mobile-app filters | Synthetic tint/sticker overlays | App assets unavailable |
| D3 | Removal | Segmentation + GAN inpainting | Mask-guided blur fill | Training-free surrogate |
| D4 | Detectors / COTS | dlib/RetinaFace/COTS | Omitted | Scope: recognition impact |
