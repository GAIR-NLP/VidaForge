# AutoModel uv patch

This directory is for uv project files that should be copied into the root of
an AutoModel checkout before installing its training dependencies.

Expected files:

- `pyproject.toml`
- `uv.lock`

Usage:

```bash
cd /path/to/AutoModel
cp /path/to/VidaForge/vidaforge_adapters/automodel/uv_patch/pyproject.toml ./pyproject.toml
cp /path/to/VidaForge/vidaforge_adapters/automodel/uv_patch/uv.lock ./uv.lock
uv sync
```

AutoModel training configs should point to the VidaForge adapter dataloader:

```yaml
_target_: vidaforge_adapters.automodel.build_video_multiresolution_dataloader
```
