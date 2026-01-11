# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**comfyui-isolation** is a Python library that enables ComfyUI custom nodes to run in isolated subprocess environments with separate Python virtual environments and dependencies. It solves dependency conflicts between nodes by running each node's GPU-heavy code in a separate process with its own venv.

## Architecture

```
ComfyUI Main Process                    Isolated Subprocess
┌─────────────────────────┐            ┌──────────────────────────┐
│  @isolated decorator    │            │  runner.py entrypoint    │
│  intercepts FUNCTION    │            │                          │
│  method calls           │   stdin    │  Imports node module     │
│                         │ ─────────► │  (decorator is no-op)    │
│  Serializes args to     │   JSON     │                          │
│  JSON + base64          │            │  Calls actual method     │
│                         │   stdout   │                          │
│  Deserializes result    │ ◄───────── │  Returns JSON result     │
│  from JSON              │   JSON     │                          │
└─────────────────────────┘            └──────────────────────────┘
```

### Key Components

- **`decorator.py`** - The `@isolated` decorator that intercepts method calls in host process
- **`runner.py`** - Generic subprocess entrypoint that handles JSON-RPC requests
- **`ipc/protocol.py`** - Serialization/deserialization for tensors, images, and complex objects
- **`env/manager.py`** - Virtual environment creation and package installation using `uv`
- **`env/config.py`** - Environment configuration dataclass
- **`stubs/folder_paths.py`** - Stub for ComfyUI's folder_paths module in subprocess

### How It Works

1. **Host Process**: When a node class is decorated with `@isolated`, the decorator:
   - Checks if running in worker mode (`COMFYUI_ISOLATION_WORKER=1`) - if so, becomes a no-op
   - Otherwise, replaces the FUNCTION method with a proxy that forwards calls to subprocess

2. **Subprocess Lifecycle**:
   - First call spawns a subprocess running `runner.py`
   - Subprocess sets up sys.path, imports the node module
   - Node's `@isolated` decorator is a no-op in subprocess (due to env var)
   - Subprocess sends "ready" signal, then processes JSON-RPC requests

3. **IPC Protocol**:
   - Requests/responses are JSON over stdin/stdout
   - Tensors: converted to numpy, pickled, base64 encoded
   - Images (PIL): PNG encoded, base64
   - Complex objects: pickle + base64 fallback

## Development Commands

```bash
# Install in development mode
cd /home/shadeform/comfyui-isolation
pip install -e .

# Run tests
pytest

# The library is used by ComfyUI-SAM3DObjects at:
# /home/shadeform/sam3dobjects/ComfyUI/custom_nodes/ComfyUI-SAM3DObjects/
```

## Key Files

| File | Purpose |
|------|---------|
| `src/comfyui_isolation/decorator.py` | Main `@isolated` decorator and process management |
| `src/comfyui_isolation/runner.py` | Subprocess entrypoint, handles JSON-RPC |
| `src/comfyui_isolation/ipc/protocol.py` | Tensor/image serialization |
| `src/comfyui_isolation/env/manager.py` | venv creation with `uv` |
| `src/comfyui_isolation/env/config.py` | `IsolatedEnv` configuration dataclass |

## Usage Pattern

```python
from comfyui_isolation import isolated

@isolated(env="myenv", import_paths=[".", "../vendor"])
class MyGPUNode:
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)

    def process(self, image):
        # This runs in isolated subprocess
        import torch
        import my_heavy_dependency
        return (result,)
```

## Known Issues & Recent Fixes

1. **C Library stdout pollution**: Libraries like `pymeshfix` print directly to fd 1, corrupting JSON protocol. Fixed by redirecting stdout to stderr at file descriptor level during method execution (see `runner.py` lines 178-193).

2. **Tensor serialization overhead**: Currently copies tensors CPU→numpy→pickle→base64. See CRITICISM.md for planned improvements using CUDA IPC.

## Related Projects

- **pyisolate** (`/home/shadeform/pyisolate`) - ComfyUI's official isolation library, uses multiprocessing.Queue and CUDA IPC for zero-copy tensor sharing. See CRITICISM.md for detailed comparison.

- **ComfyUI-SAM3DObjects** - Primary user of this library for 3D object generation nodes.
