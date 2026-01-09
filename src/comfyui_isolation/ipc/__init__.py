"""IPC (Inter-Process Communication) for comfyui-isolation."""

from .bridge import WorkerBridge
from .worker import BaseWorker, register
from .protocol import Request, Response, encode_object, decode_object

# TorchBridge is optional (requires PyTorch)
try:
    from .torch_bridge import TorchBridge, TorchWorker
    _TORCH_EXPORTS = ["TorchBridge", "TorchWorker"]
except ImportError:
    _TORCH_EXPORTS = []

__all__ = [
    "WorkerBridge",
    "BaseWorker",
    "register",
    "Request",
    "Response",
    "encode_object",
    "decode_object",
] + _TORCH_EXPORTS
