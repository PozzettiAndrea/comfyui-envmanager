"""
Installation API for comfyui-envmanager.

This module provides the main `install()` function that handles both:
- In-place installation (CUDA wheels into current environment)
- Isolated installation (create separate venv with dependencies)

Example:
    from comfyui_envmanager import install

    # In-place install (auto-discovers config)
    install()

    # In-place with explicit config
    install(config="comfyui_env.toml", mode="inplace")

    # Isolated environment
    install(config="comfyui_env.toml", mode="isolated")
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .env.config import IsolatedEnv
from .env.config_file import discover_env_config, load_env_from_file
from .env.manager import IsolatedEnvManager
from .errors import CUDANotFoundError, DependencyError, InstallError, WheelNotFoundError
from .resolver import RuntimeEnv, WheelResolver, parse_wheel_requirement


def install(
    config: Optional[Union[str, Path]] = None,
    mode: str = "inplace",
    node_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
    verify_wheels: bool = False,
) -> bool:
    """
    Install dependencies from a comfyui_env.toml configuration.

    This is the main entry point for installing CUDA dependencies.

    Args:
        config: Path to config file. If None, auto-discovers in node_dir.
        mode: Installation mode - "inplace" (current env) or "isolated" (new venv).
        node_dir: Directory to search for config. Defaults to current directory.
        log_callback: Optional callback for logging. Defaults to print.
        dry_run: If True, show what would be installed without installing.
        verify_wheels: If True, verify wheel URLs exist before installing.

    Returns:
        True if installation succeeded.

    Raises:
        FileNotFoundError: If config file not found.
        WheelNotFoundError: If required wheel cannot be resolved.
        InstallError: If installation fails.

    Example:
        # Simple usage - auto-discover config
        install()

        # Explicit config file
        install(config="comfyui_env.toml")

        # Isolated mode
        install(mode="isolated")

        # Dry run to see what would be installed
        install(dry_run=True)
    """
    log = log_callback or print
    node_dir = Path(node_dir) if node_dir else Path.cwd()

    # Load configuration
    env_config = _load_config(config, node_dir)
    if env_config is None:
        raise FileNotFoundError(
            "No configuration file found. "
            "Create comfyui_env.toml or specify path explicitly."
        )

    log(f"Found configuration: {env_config.name}")

    if mode == "isolated":
        return _install_isolated(env_config, node_dir, log, dry_run)
    else:
        return _install_inplace(env_config, node_dir, log, dry_run, verify_wheels)


def _load_config(
    config: Optional[Union[str, Path]],
    node_dir: Path,
) -> Optional[IsolatedEnv]:
    """Load configuration from file or auto-discover."""
    if config is not None:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = node_dir / config_path
        return load_env_from_file(config_path, node_dir)

    return discover_env_config(node_dir)


def _install_isolated(
    env_config: IsolatedEnv,
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool,
) -> bool:
    """Install in isolated mode using IsolatedEnvManager."""
    log(f"Installing in isolated mode: {env_config.name}")

    if dry_run:
        log("Dry run - would create isolated environment:")
        log(f"  Python: {env_config.python}")
        log(f"  CUDA: {env_config.cuda or 'auto-detect'}")
        if env_config.requirements:
            log(f"  Requirements: {len(env_config.requirements)} packages")
        return True

    manager = IsolatedEnvManager(base_dir=node_dir, log_callback=log)
    env_dir = manager.setup(env_config)
    log(f"Isolated environment ready: {env_dir}")
    return True


def _install_inplace(
    env_config: IsolatedEnv,
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool,
    verify_wheels: bool,
) -> bool:
    """Install in-place into current environment."""
    log("Installing in-place mode")

    # Detect runtime environment
    env = RuntimeEnv.detect()
    log(f"Detected environment: {env}")

    # Check CUDA requirement
    if not env.cuda_version:
        # Check if any requirements need CUDA
        cuda_packages = _get_cuda_packages(env_config)
        if cuda_packages:
            raise CUDANotFoundError(package=", ".join(cuda_packages))

    # Split requirements into CUDA and regular packages
    cuda_packages = _get_cuda_packages(env_config)
    regular_packages = _get_regular_packages(env_config)

    # Resolve CUDA wheel URLs
    resolver = WheelResolver()
    cuda_urls = {}

    for req in cuda_packages:
        package, version = parse_wheel_requirement(req)
        if version is None:
            log(f"Warning: No version specified for {package}, skipping wheel resolution")
            continue

        try:
            url = resolver.resolve(package, version, env, verify=verify_wheels)
            cuda_urls[package] = url
            log(f"Resolved {package}: {url}")
        except WheelNotFoundError as e:
            if dry_run:
                log(f"Warning: Could not resolve {package}=={version}")
            else:
                raise

    if dry_run:
        log("\nDry run - would install:")
        if cuda_urls:
            log("  CUDA packages (--no-deps):")
            for pkg, url in cuda_urls.items():
                log(f"    {pkg}: {url}")
        if regular_packages:
            log("  Regular packages:")
            for pkg in regular_packages:
                log(f"    {pkg}")
        return True

    # Install CUDA packages first (with --no-deps to avoid conflicts)
    if cuda_urls:
        log(f"\nInstalling {len(cuda_urls)} CUDA packages...")
        _pip_install(list(cuda_urls.values()), no_deps=True, log=log)

    # Install regular packages
    if regular_packages:
        log(f"\nInstalling {len(regular_packages)} regular packages...")
        _pip_install(regular_packages, no_deps=False, log=log)

    log("\nInstallation complete!")
    return True


def _get_cuda_packages(env_config: IsolatedEnv) -> List[str]:
    """Extract CUDA packages that need wheel resolution."""
    # For now, treat no_deps_requirements as CUDA packages
    # In future, could parse from [packages.cuda] section
    return env_config.no_deps_requirements or []


def _get_regular_packages(env_config: IsolatedEnv) -> List[str]:
    """Extract regular pip packages."""
    return env_config.requirements or []


def _pip_install(
    packages: List[str],
    no_deps: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """
    Install packages using pip.

    Args:
        packages: List of packages or URLs to install.
        no_deps: If True, use --no-deps flag.
        log: Logging callback.

    Raises:
        InstallError: If pip install fails.
    """
    # Prefer uv if available for speed
    pip_cmd = _get_pip_command()

    args = pip_cmd + ["install"]
    if no_deps:
        args.append("--no-deps")
    args.extend(packages)

    log(f"Running: {' '.join(args[:3])}... ({len(packages)} packages)")

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise InstallError(
            f"Failed to install packages",
            exit_code=result.returncode,
            stderr=result.stderr,
        )


def _get_pip_command() -> List[str]:
    """Get the pip command to use (prefers uv if available)."""
    # Check for uv
    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path, "pip"]

    # Fall back to pip
    return [sys.executable, "-m", "pip"]


def verify_installation(
    packages: List[str],
    log: Callable[[str], None] = print,
) -> bool:
    """
    Verify that packages are importable.

    Args:
        packages: List of package names to verify.
        log: Logging callback.

    Returns:
        True if all packages are importable.
    """
    all_ok = True
    for package in packages:
        # Convert package name to import name
        import_name = package.replace("-", "_").split("[")[0]

        try:
            __import__(import_name)
            log(f"  {package}: OK")
        except ImportError as e:
            log(f"  {package}: FAILED ({e})")
            all_ok = False

    return all_ok
