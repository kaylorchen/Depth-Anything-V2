import cv2
import onnx.checker
import onnx.shape_inference
import torch
from io import BytesIO
import onnx
from onnx import shape_inference
from depth_anything_v2.dpt import DepthAnythingV2
import onnxsim

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}

encoder = "vits"  # or 'vits', 'vitb', 'vitg'
model_path = f"checkpoints/depth_anything_v2_{encoder}"

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f"{model_path}.pth", map_location="cpu"))
model = model.to(DEVICE).eval()
fake_input = torch.randn([1, 3, 518, 518]).to(DEVICE)
with BytesIO() as f:
    torch.onnx.export(model=model, 
                      args=fake_input, 
                      f=f, 
                      opset_version=18,
                      input_names=['rgb_image'],
                      output_names=['depth_image'],
                      )
    f.seek(0)
    onnx_model = onnx.load(f)
onnx.checker.check_model(onnx_model)
try:
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "assertion check failed"
except Exception as e:
    print(f"Error simplifier failed: {e}")

onnx.save(onnx.shape_inference.infer_shapes(onnx_model), f"{model_path}.onnx")
