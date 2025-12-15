## https://github.com/stac-extensions/mlm/blob/main/README_STAC_MODEL.md#exporting-and-packaging-pytorch-models-transforms-and-model-metadata
import torch
import torchvision.transforms.v2 as T
from torchgeo.models import Unet_Weights, unet
from stac_model.torch.export import save

weights = Unet_Weights.SENTINEL2_3CLASS_FTW
transforms = torch.nn.Sequential(
  T.Resize((256, 256)),
  T.Normalize(mean=[0.0], std=[3000.0])
)
model = unet(weights=weights)

save(
    output_file="ftw.pt2",
    model=model,  # Must be an nn.Module
    transforms=transforms,  # Must be an nn.Module
    metadata="metadata.yaml",  # Can be a metadata yaml or stac_model.schema.MLModelProperties object
    input_shape=[-1, 8, -1, -1],  # -1 indicates a dynamic shaped dimension
    device="cpu",
    dtype=torch.float32,
    aoti_compile_and_package=False,  # True for AOTInductor compile otherwise use torch.export
)