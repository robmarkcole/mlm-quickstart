# mlm-quickstart
- https://github.com/stac-extensions/mlm/blob/main/README_STAC_MODEL.md

## Run on codespace
Complete `metadata.yaml` then run:
```
pip install -r requirements.txt

python create.py # generates the ftw.pt2 file, gitignored

python describe.py

Loading PT2 archive from: ftw.pt2
/usr/local/python/3.12.1/lib/python3.12/site-packages/torch/export/pt2_archive/_package.py:682: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1581.)
  tensor = torch.frombuffer(
PT2 archive loaded
Exported programs: ['model', 'transforms']
Loading MLM metadata
Metadata keys: ['$schema', 'properties']

--- Model ---
Model class: GraphModule
Model mode: train
Total parameters: 13,161,123
Trainable parameters: 13,161,123
WARNING: Model is in TRAIN mode (model.training == True)

--- Graph Signature (summary) ---
Input spec counts by kind:
  PARAMETER: 370
  BUFFER: 264
  USER_INPUT: 1
Number of output specs: 1
Example non-parameter input spec:
  b_encoder__bn0_running_mean: BUFFER target='encoder._bn0.running_mean' persistent=True

--- Transforms ---
Transforms class: GraphModule
Transforms mode: train
```
