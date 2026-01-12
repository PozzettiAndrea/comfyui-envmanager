# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**comfyui-envmanager** is a Python library for ComfyUI custom nodes that provides:

1. **CUDA Wheel Resolution** - Deterministic wheel URL construction for CUDA packages (nvdiffrast, pytorch3d, etc.)
2. **In-Place Installation** - Install CUDA wheels into current environment without compile
3. **Process Isolation** - Run nodes in separate venvs with different dependencies

This replaces the old `comfyui-isolation` package (version 0.0.1 is a fresh start).

## Architecture

### Type 1 Nodes (Isolated Venv)
Nodes that need separate venv due to conflicting dependencies:
```
ComfyUI Main Process                    Isolated Subprocess
┌─────────────────────────┐            ┌──────────────────────────┐
│  @isolated decorator    │            │  runner.py entrypoint    │
│  intercepts FUNCTION    │   UDS/     │                          │
│  method calls           │  stdin ──► │  Imports node module     │
│                         │            │  (decorator is no-op)    │
│  Tensor IPC via shared  │            │                          │
│  memory / CUDA IPC      │ ◄──────────│  Returns result          │
└─────────────────────────┘            └──────────────────────────┘
```

### Type 2 Nodes (In-Place)
Nodes that just need CUDA wheels resolved:
```
comfyui_env.toml
       │
       ▼
┌──────────────────────────────────────────────┐
│  WheelResolver                               │
│  - Detects CUDA/PyTorch/Python versions      │
│  - Constructs exact wheel URLs               │
│  - pip install --no-deps                     │
└──────────────────────────────────────────────┘
```

## Key Components

| File | Purpose |
|------|---------|
| `src/comfyui_envmanager/install.py` | `install()` function for both modes |
| `src/comfyui_envmanager/resolver.py` | Wheel URL resolution with template expansion |
| `src/comfyui_envmanager/errors.py` | Rich, actionable error messages |
| `src/comfyui_envmanager/cli.py` | `comfy-env` CLI commands |
| `src/comfyui_envmanager/decorator.py` | `@isolated` decorator for process isolation |
| `src/comfyui_envmanager/workers/` | Worker classes (TorchMPWorker, VenvWorker) |
| `src/comfyui_envmanager/env/manager.py` | venv creation with `uv` |
| `src/comfyui_envmanager/env/config_file.py` | TOML config parsing |

## Development Commands

```bash
# Install in development mode
cd /home/shadeform/comfyui-isolation
pip install -e .

# Run CLI
comfy-env info
comfy-env install --dry-run
comfy-env resolve nvdiffrast==0.4.0
```

## Usage Patterns

### In-Place Installation (Type 2)
```python
from comfyui_envmanager import install

# Auto-discover config and install
install()

# Dry run
install(dry_run=True)
```

### Process Isolation (Type 1)
```python
from comfyui_envmanager import isolated

@isolated(env="myenv")
class MyGPUNode:
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)

    def process(self, image):
        # Runs in isolated subprocess
        import heavy_dependency
        return (result,)
```

### Direct Worker Usage
```python
from comfyui_envmanager import TorchMPWorker

worker = TorchMPWorker()
result = worker.call(my_function, image=tensor)
```

## Config File Format

```toml
[env]
name = "my-node"
python = "3.10"
cuda = "auto"

[packages]
requirements = ["transformers>=4.56"]
no_deps = ["nvdiffrast==0.4.0"]

[sources]
wheel_sources = ["https://github.com/.../releases/download/"]
```

## Key Design Decisions

1. **Deterministic Resolution**: Wheel URLs are constructed, not solved. If URL 404s, fail fast with clear message.

2. **No Compilation on User Machines**: If wheel doesn't exist, fail with actionable error showing what combos are available.

3. **Template Variables**: `{cuda_short}`, `{torch_mm}`, `{py_short}`, `{platform}` for URL construction.

4. **Backward Compatibility**: Old config file names (`comfyui_isolation_reqs.toml`) still discovered.

## Related Projects

- **pyisolate** - ComfyUI's official security-focused isolation
- **comfy-cli** - High-level ComfyUI management
- **ComfyUI-SAM3DObjects** - Primary user of this library
