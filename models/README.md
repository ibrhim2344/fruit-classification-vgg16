# Model bundles

No trained weights are included. Training creates `run_vgg16/` by default. For inference keep these three files together:

- `best_model.keras`
- `class_names.json`
- `config.json`

The run also contains stage checkpoints, environment metadata, dataset counts, histories and the validation-based selection decision. Raw RGB images must be resized with bilinear interpolation and processed with VGG16 `preprocess_input`, exactly as `src/predict.py` implements.

To distribute a trained model, package the three required files plus metadata/model card as a separate downloadable artifact after checking applicable pretrained-weight terms. Explain the training dataset, measured results, limitations and environment. Model files are excluded from ordinary Git commits by `.gitignore`.
