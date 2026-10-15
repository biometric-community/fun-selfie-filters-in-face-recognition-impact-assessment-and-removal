# IMPLEMENTATION_PLAN — Fun Selfie Filters

## Lab twin

| Component | Choice |
|-----------|--------|
| Backbone | ResNet-18 CE on CelebA |
| Filters | Synthetic overlays (coverage × placement) |
| Impact | LFW EER/acc under filter conditions |
| Removal | Mask-guided blur inpainting surrogate |
