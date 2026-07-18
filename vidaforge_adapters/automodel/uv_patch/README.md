# AutoModel uv patch

This directory contains the dependency definition used by the VidaForge
NeMo-AutoModel training environment.

Expected files:

- `pyproject.toml`
- `uv.lock`

Create the environment under the VidaForge repository and install the official
NeMo-AutoModel checkout plus VidaForge as editable packages:

```bash
REPO_DIR=/path/to/VidaForge
AUTOMODEL_DIR=/path/to/Automodel

cd "${REPO_DIR}"
uv venv .venv-automodel --python 3.12
source .venv-automodel/bin/activate

uv sync --active --frozen --no-install-project \
  --project "${REPO_DIR}/vidaforge_adapters/automodel/uv_patch"

uv pip install --no-deps -e "${AUTOMODEL_DIR}"
uv pip install --no-deps -e "${REPO_DIR}"
```

AutoModel training configs should point to the VidaForge adapter dataloader:

```yaml
_target_: vidaforge_adapters.automodel.VidaForgeVideoDataloaderConfig
```

The typed config keeps the VidaForge Stage 5 dataset, bucket sampler, and
collate behavior while satisfying AutoModel's diffusion dataloader `build()`
contract. The legacy `build_video_multiresolution_dataloader` function remains
available for callers that still use the older function-based API.
