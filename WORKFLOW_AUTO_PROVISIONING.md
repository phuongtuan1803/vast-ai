# ComfyUI Workflow Auto-Provisioning

Hệ thống tự động tải và cài đặt tất cả các model, tools cần thiết cho ComfyUI workflow.

## Tính năng

### 1. Workflow Parser (`workflow_parser.py`)
Parse file workflow JSON của ComfyUI và trích xuất:
- **Models**: Checkpoints, VAE, CLIP, LoRA, ControlNet, IP-Adapter, Upscale models, v.v.
- **Custom Nodes**: Phát hiện các node tùy chỉnh cần thiết

Hỗ trợ:
- ComfyUI workflow spec v1.0 (mảng `models`)
- Phân tích node để tự động phát hiện dependencies
- 20+ loại model loader khác nhau

### 2. Model Downloader (`download_models.py`)
Tải models từ nhiều nguồn:
- **HuggingFace Hub**: Tải từ repositories
- **Direct URL**: Tải trực tiếp với xác thực hash
- **CivitAI**: Hỗ trợ API (sẵn sàng mở rộng)

Tính năng:
- Kiểm tra file đã tồn tại
- Xác thực hash (SHA256, MD5, v.v.)
- Báo cáo tiến trình tải
- Tự động tạo thư mục

### 3. Setup Script (`setup_comfyui.py`)
Script cài đặt chính với 2 chế độ:

#### A. Model Type Provisioning (Cũ)
```bash
python setup_comfyui.py --workspace /workspace --model-type flux;sdxl
```

#### B. Workflow-based Provisioning (Mới)
```bash
python setup_comfyui.py --workspace /workspace --workflow my_workflow.json
```

Tự động:
- Parse workflow
- Phát hiện models cần thiết
- Kiểm tra models đã có
- Tải models còn thiếu từ HuggingFace hoặc URL trong workflow
- Báo cáo custom nodes cần cài đặt thủ công

### 4. Workflow Validator (`workflow_validator.py`)
Kiểm tra workflow trước khi chạy:
```bash
python workflow_validator.py my_workflow.json --comfyui-dir /workspace/ComfyUI
```

Báo cáo:
- Models có sẵn vs còn thiếu
- Custom nodes cần cài đặt
- Warnings và errors
- Output JSON hoặc human-readable

## Cách sử dụng

### Cài đặt cơ bản
```bash
# Clone repo
cd vast-ai

# Cài đặt với model types
python setup_comfyui.py --workspace /workspace --model-type flux;controlnet;ip_adapter

# Hoặc cài đặt từ workflow
python setup_comfyui.py --workspace /workspace --workflow my_workflow.json
```

### Validate workflow trước khi chạy
```bash
python workflow_validator.py workflow.json --comfyui-dir /workspace/ComfyUI
```

### Parse workflow để xem dependencies
```bash
python workflow_parser.py workflow.json --verbose
```

### Tải model thủ công
```bash
python download_models.py \
  --comfyui-dir /workspace/ComfyUI \
  --model-name flux1-dev-fp8.safetensors \
  --model-type checkpoints \
  --repo-id kijai/flux-fp8 \
  --filename flux1-dev-fp8.safetensors
```

## Cấu trúc Models Config

File `models_config.json` định nghĩa các model sets:

```json
{
  "flux": {
    "name": "Flux.1 Dev",
    "description": "Flux.1 Dev model with FP8 optimization",
    "checkpoints": [
      {
        "repo_id": "kijai/flux-fp8",
        "filename": "flux1-dev-fp8.safetensors",
        "local_dir": "models/checkpoints"
      }
    ],
    "clip": [...],
    "vae": [...]
  }
}
```

Hỗ trợ các loại model:
- `checkpoints`: Main diffusion models
- `vae`: Variational autoencoders
- `clip`: Text encoders
- `loras`: LoRA adapters
- `controlnet`: ControlNet models
- `ipadapter`: IP-Adapter models
- `clip_vision`: CLIP vision encoders
- `upscale_models`: Upscaling models
- `unet`: UNET models
- `embeddings`: Textual inversions

## Workflow JSON Structure

### V1.0 Spec (với models array)
```json
{
  "models": [
    {
      "name": "flux1-dev-fp8.safetensors",
      "url": "https://huggingface.co/...",
      "directory": "models/checkpoints",
      "hash": "abc123...",
      "hash_type": "sha256"
    }
  ],
  "nodes": [...]
}
```

### Legacy (parse từ nodes)
```json
{
  "nodes": [
    {
      "id": 1,
      "type": "CheckpointLoaderSimple",
      "widgets_values": ["flux1-dev-fp8.safetensors"]
    }
  ]
}
```

## Kiến trúc

```
workflow.json
     ↓
WorkflowParser → WorkflowDependencies
     ↓                    ↓
     ├─→ ModelDependency (name, type, url, hash)
     └─→ CustomNodeDependency (node_type)
                ↓
         ModelDownloader
                ↓
         Download & Verify
```

## Node Type Mapping

Hệ thống tự động nhận diện 20+ loại loader nodes:

| Node Type | Model Type | Widget Index |
|-----------|------------|--------------|
| CheckpointLoaderSimple | checkpoints | 0 |
| VAELoader | vae | 0 |
| CLIPLoader | clip | 0 |
| ControlNetLoader | controlnet | 0 |
| LoraLoader | loras | 0 |
| DualCLIPLoader | clip | [0, 1] |
| UpscaleModelLoader | upscale_models | 0 |
| IPAdapterModelLoader | ipadapter | 0 |

## Custom Nodes Detection

Tự động phát hiện custom nodes qua:
- Known prefixes: `CR`, `WAS`, `ImpactPack`, `AnimateDiff`, v.v.
- Pattern matching: `Ultimate`, `Advanced`, `Pack`, `Suite`
- Case patterns: Mixed case hoặc underscores

## Xử lý lỗi

1. **Model không tìm thấy**: Báo cáo model cần tải thủ công
2. **Custom node thiếu**: Liệt kê và gợi ý cài qua ComfyUI-Manager
3. **Download thất bại**: Cleanup partial downloads, log errors
4. **Hash mismatch**: Xóa file lỗi, báo cáo

## Best Practices

1. **Luôn validate trước**: Chạy validator trước khi provision
2. **Sử dụng hash**: Đảm bảo model integrity
3. **Check models_config.json**: Thêm models thường dùng vào config
4. **Custom nodes**: Cài qua ComfyUI-Manager cho tính tương thích

## Mở rộng

### Thêm model source mới
Extend `ModelDownloader` class:
```python
def download_from_custom_source(self, ...):
    # Implementation
    pass
```

### Thêm node type mapping
Update `WorkflowParser.NODE_TYPE_MAPPING`:
```python
NODE_TYPE_MAPPING = {
    'YourCustomLoader': ('model_type', widget_index),
    ...
}
```

### Thêm model vào config
Edit `models_config.json`:
```json
{
  "your_model_set": {
    "name": "Display Name",
    "description": "Description",
    "checkpoints": [...]
  }
}
```

## Examples

### Example 1: Auto-provision from workflow
```bash
# Parse workflow để xem cần gì
python workflow_parser.py flux_workflow.json -v

# Validate
python workflow_validator.py flux_workflow.json --comfyui-dir ./ComfyUI

# Setup tự động
python setup_comfyui.py --workspace . --workflow flux_workflow.json
```

### Example 2: Mixed provisioning
```bash
# Setup base models
python setup_comfyui.py --model-type flux;controlnet

# Sau đó provision thêm từ workflow
python setup_comfyui.py --workflow custom_workflow.json
```

### Example 3: Validate before deployment
```python
from workflow_validator import WorkflowValidator

validator = WorkflowValidator('/workspace/ComfyUI')
result = validator.validate_workflow('workflow.json')

if not result.valid:
    print("Missing models:", result.missing_models)
    # Auto-download or alert user
```

## Troubleshooting

**Q: Model không tải được?**
- Kiểm tra URL trong workflow hoặc models_config.json
- Xác nhận HuggingFace repo_id và filename
- Thử tải thủ công với download_models.py

**Q: Custom node không nhận diện?**
- Cài ComfyUI-Manager
- Thêm prefix vào KNOWN_CUSTOM_NODE_PREFIXES trong workflow_parser.py

**Q: Hash không khớp?**
- Kiểm tra file nguồn có bị corrupt
- Xác nhận hash_type (sha256, md5, etc.)
- Re-download với --hash parameter

## API Reference

### WorkflowParser
```python
parser = WorkflowParser('workflow.json')
deps = parser.parse()  # Returns WorkflowDependencies

# Access dependencies
for model in deps.models:
    print(f"{model.name} ({model.model_type})")
```

### ModelDownloader
```python
downloader = ModelDownloader(comfyui_dir)

# Generic download
downloader.download_model(
    model_name='model.safetensors',
    model_type='checkpoints',
    repo_id='user/repo',
    filename='model.safetensors'
)

# Check existence
exists = downloader.model_exists('model.safetensors', 'checkpoints')
```

### WorkflowValidator
```python
validator = WorkflowValidator(comfyui_dir)
result = validator.validate_workflow('workflow.json')

print(f"Valid: {result.valid}")
print(f"Missing: {len(result.missing_models)}")
```

## License

Part of local-ai project by phuongtuan1803.
