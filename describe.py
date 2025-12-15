import yaml
import torch
from torch.export.pt2_archive._package import load_pt2

archive_path = "ftw.pt2"

print(f"Loading PT2 archive from: {archive_path}")
pt2 = load_pt2(archive_path)

print("PT2 archive loaded")
print("Exported programs:", list(pt2.exported_programs.keys()))

# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------
print("Loading MLM metadata")
metadata = yaml.safe_load(pt2.extra_files["mlm-metadata"])
print("Metadata keys:", list(metadata.keys()))

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
print("\n--- Model ---")
ep = pt2.exported_programs["model"]
model = ep.module()

print("Model class:", model.__class__.__name__)
print("Model mode:", "train" if model.training else "eval")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

if model.training:
    print("WARNING: Model is in TRAIN mode (model.training == True)")

# ------------------------------------------------------------------
# Graph signature (summary)
# ------------------------------------------------------------------
print("\n--- Graph Signature (summary) ---")

if hasattr(ep, "graph_signature"):
    sig = ep.graph_signature

    input_specs = getattr(sig, "input_specs", [])
    output_specs = getattr(sig, "output_specs", [])

    # Count input kinds
    kind_counts = {}
    for spec in input_specs:
        kind = spec.kind.name
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    print("Input spec counts by kind:")
    for kind, count in kind_counts.items():
        print(f"  {kind}: {count}")

    print("Number of output specs:", len(output_specs))

    # Show first non-parameter input, if any
    non_param_inputs = [
        spec for spec in input_specs
        if spec.kind.name != "PARAMETER"
    ]

    if non_param_inputs:
        print("Example non-parameter input spec:")
        print(" ", non_param_inputs[0])
    else:
        print("No non-parameter inputs (all parameters are explicit)")
else:
    print("No graph_signature available")


# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
print("\n--- Transforms ---")
if "transforms" in pt2.exported_programs:
    t_ep = pt2.exported_programs["transforms"]
    transforms = t_ep.module()
    print("Transforms class:", transforms.__class__.__name__)
    print("Transforms mode:", "train" if transforms.training else "eval")
else:
    print("No transforms program found")

print("\nPT2 archive load complete")
