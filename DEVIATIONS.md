# DEVIATIONS — Fun Selfie Filters

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Source faces | FRGCv2 / FERET | CelebA train + LFW family | Local public payloads |
| D2 | Filters | 10 mobile-app filters | Solid synthetic stickers grown around each placement until their Eq. 1 coverage intensity reaches the Table IV band | App assets unavailable |
| D3 | Removal | Segmentation + GAN inpainting | Known filter mask + diffusion inpainting from the surrounding face | Training-free surrogate |
| D4 | Detectors / COTS | dlib/RetinaFace/COTS | Omitted | Scope: recognition impact |
