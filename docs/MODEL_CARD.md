# Model card — untrained source release

| Field | Value |
| --- | --- |
| Task | Multi-class produce image classification |
| Architecture | VGG16 + pooling/dense/batch-normalization/dropout head |
| Initialization | ImageNet by default |
| Inputs | RGB images resized to 64 × 64 and VGG16-preprocessed |
| Classes | Discovered from Training folders and saved with the model |
| Training data | Fruits-360 original-size educational mirror from source notebook |
| Evaluation | Full supplied Test split after validation-based model selection |
| Measured performance | Not available; this package has not been trained |
| Intended use | Learning, reproducible experimentation and portfolio demonstration |
| Unsupported claims | Production reliability, field robustness, unknown-class detection |

Before releasing trained weights, record the archive hash/version, actual dataset counts, environment, training configuration, test metrics and known failure cases. Include whether fine-tuning improved validation loss. Softmax scores are uncalibrated. Object-level leakage and field-domain generalization have not been audited.
