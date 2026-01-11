"""
Generic runner for isolated subprocess execution.

This module is the entry point for subprocess execution. Instead of generating
worker code, the host process spawns this runner with the module/class/method
to execute. The runner imports the original node module and calls the method
directly.

Usage:
    python -m comfyui_isolation.runner \
        --module nodes.depth_estimate \
        --class SAM3D_DepthEstimate \
        --method estimate_depth \
        --node-dir /path/to/ComfyUI-SAM3DObjects \
        --comfyui-base /path/to/ComfyUI \
        --import-paths ".,../vendor"

The runner:
1. Sets COMFYUI_ISOLATION_WORKER=1 (so @isolated decorator becomes no-op)
2. Adds paths to sys.path
3. Imports the module
4. Creates class instance
5. Reads JSON-RPC requests from stdin
6. Calls methods and writes responses to stdout
"""

import os
import sys
import json
import argparse
import traceback
import warnings
import logging
from typing import Any, Dict, Optional

# Suppress warnings that could interfere with JSON IPC
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.disable(logging.WARNING)

# Mark that we're in worker mode - this makes @isolated decorator a no-op
os.environ["COMFYUI_ISOLATION_WORKER"] = "1"


def setup_paths(node_dir: str, comfyui_base: Optional[str], import_paths: Optional[str]):
    """Setup sys.path for imports."""
    from pathlib import Path

    node_path = Path(node_dir)

    # Add ComfyUI base first (for folder_paths, etc.)
    if comfyui_base:
        sys.path.insert(0, comfyui_base)

    # Add import paths
    if import_paths:
        for p in import_paths.split(","):
            p = p.strip()
            if p:
                full_path = node_path / p
                sys.path.insert(0, str(full_path))

    # Add node_dir itself
    sys.path.insert(0, str(node_path))


def import_class(module_name: str, class_name: str):
    """Import a class from a module."""
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def serialize_result(obj: Any) -> Any:
    """Serialize result for JSON transport."""
    # Import protocol's encoder
    from comfyui_isolation.ipc.protocol import encode_object
    return encode_object(obj)


def deserialize_arg(obj: Any) -> Any:
    """Deserialize argument from JSON transport."""
    from comfyui_isolation.ipc.protocol import decode_object
    return decode_object(obj)


def run_worker(module_name: str, class_name: str, node_dir: str,
               comfyui_base: Optional[str], import_paths: Optional[str]):
    """Main worker loop - reads requests from stdin, writes responses to stdout."""

    # Setup paths first
    setup_paths(node_dir, comfyui_base, import_paths)

    # Import the class
    try:
        cls = import_class(module_name, class_name)
        instance = cls()
        print(f"[Runner] Loaded {class_name} from {module_name}", file=sys.stderr)
    except Exception as e:
        # Fatal error - can't even load the class
        error_response = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32000,
                "message": f"Failed to import {module_name}.{class_name}: {e}",
                "data": {"traceback": traceback.format_exc()}
            }
        }
        print(json.dumps(error_response), flush=True)
        sys.exit(1)

    # Send ready signal
    ready_msg = {"status": "ready", "class": class_name, "module": module_name}
    print(json.dumps(ready_msg), flush=True)

    # Main loop - read requests, execute, respond
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        response = {"jsonrpc": "2.0", "id": None}

        try:
            request = json.loads(line)
            response["id"] = request.get("id")

            method_name = request.get("method")
            params = request.get("params", {})

            if method_name == "shutdown":
                # Clean shutdown
                response["result"] = {"status": "shutdown"}
                print(json.dumps(response), flush=True)
                break

            # Get the method
            method = getattr(instance, method_name, None)
            if method is None:
                response["error"] = {
                    "code": -32601,
                    "message": f"Method not found: {method_name}",
                }
                print(json.dumps(response), flush=True)
                continue

            # Deserialize arguments
            deserialized_params = {}
            for key, value in params.items():
                deserialized_params[key] = deserialize_arg(value)

            # Call the method
            print(f"[Runner] Calling {method_name}...", file=sys.stderr)
            result = method(**deserialized_params)

            # Serialize result
            serialized_result = serialize_result(result)
            response["result"] = serialized_result

            print(f"[Runner] {method_name} completed", file=sys.stderr)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Runner] Error: {e}", file=sys.stderr)
            print(tb, file=sys.stderr)
            response["error"] = {
                "code": -32000,
                "message": str(e),
                "data": {"traceback": tb}
            }

        print(json.dumps(response), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Isolated node runner")
    parser.add_argument("--module", required=True, help="Module path (e.g., nodes.depth_estimate)")
    parser.add_argument("--class", dest="class_name", required=True, help="Class name")
    parser.add_argument("--node-dir", required=True, help="Node package directory")
    parser.add_argument("--comfyui-base", help="ComfyUI base directory")
    parser.add_argument("--import-paths", help="Comma-separated import paths")

    args = parser.parse_args()

    run_worker(
        module_name=args.module,
        class_name=args.class_name,
        node_dir=args.node_dir,
        comfyui_base=args.comfyui_base,
        import_paths=args.import_paths,
    )


if __name__ == "__main__":
    main()
