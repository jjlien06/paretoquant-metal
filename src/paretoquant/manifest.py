"""Integrity-bound execution manifests (not cryptographic signatures).

A seal detects stale/corrupt artifacts relative to the manifest. It does not
authenticate its author or prove that the dispatch was freshly benchmarked.
"""

import copy
import hashlib
import json
import math
import re
from pathlib import Path


class ManifestError(ValueError):
    """A saved dispatch is not safe to activate."""


_BASE_FIELDS = {"schema_version", "dispatch", "profile_device", "mlx", "mlx_lm"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MODULE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.(?:[A-Za-z_][A-Za-z_0-9]*|[0-9]+))*\Z")
_WEIGHT = re.compile(r"model[A-Za-z0-9_.-]*\.safetensors\Z")
_KERNEL = re.compile(r"kernels/[A-Za-z0-9_-]+\.metal\Z")


def _fields(value, required, label, optional=()):
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    if set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ManifestError(f"{label} has missing or unknown fields")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a nonempty string")


def _hash_map(value, label, path_pattern):
    if not isinstance(value, dict) or not value:
        raise ManifestError(f"{label} must be a nonempty hash map")
    for path, digest in value.items():
        if not isinstance(path, str) or not path_pattern(path):
            raise ManifestError(f"{label} has malformed path {path!r}")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ManifestError(f"{label} has malformed SHA-256 for {path}")


def _validate_device(device):
    if not isinstance(device, dict):
        raise ManifestError("profile_device must be an object")
    _text(device.get("device_name"), "profile_device.device_name")
    for key, value in device.items():
        _text(key, "device key")
        if type(value) not in (str, int, float) or (
            type(value) is float and not math.isfinite(value)
        ):
            raise ManifestError(f"invalid profile_device metadata: {key}")


def _validate_environment(environment):
    if not isinstance(environment, dict) or not {"device", "mlx", "mlx_lm"} <= set(environment):
        raise ManifestError("environment requires device, mlx, and mlx_lm metadata")
    _validate_device(environment["device"])
    _text(environment["mlx"], "environment.mlx")
    _text(environment["mlx_lm"], "environment.mlx_lm")


def validate_schema(data, *, allow_legacy=False):
    if not isinstance(data, dict):
        raise ManifestError("execution manifest must be an object")
    version = data.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ManifestError("unsupported execution manifest schema_version; expected integer 2")
    if version == 1 and not allow_legacy:
        raise ManifestError(
            "legacy v1 manifest is unbound; regenerate or seal into a new directory"
        )
    required = _BASE_FIELDS | ({"model_sha256", "kernel_sha256"} if version == 2 else set())
    _fields(data, required, "execution manifest", optional={"seal"} if version == 2 else ())
    if "seal" in data:
        seal = data["seal"]
        _fields(seal, {"source_manifest_sha256", "profiling_performed"}, "seal")
        digest = seal["source_manifest_sha256"]
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ManifestError("seal has malformed source_manifest_sha256")
        if seal["profiling_performed"] is not False:
            raise ManifestError("sealing must not claim profiling_performed")
    _text(data["mlx"], "mlx")
    _text(data["mlx_lm"], "mlx_lm")
    _validate_device(data["profile_device"])
    dispatch = data["dispatch"]
    if not isinstance(dispatch, dict):
        raise ManifestError("dispatch must be an object")
    for name, selection in dispatch.items():
        if not isinstance(name, str) or not _MODULE.fullmatch(name):
            raise ManifestError(f"malformed module path: {name!r}")
        _fields(selection, {"backend", "bits", "rows_per_group"}, f"dispatch {name}")
        if selection["backend"] not in ("stock", "fused"):
            raise ManifestError(f"unknown backend for {name}")
        bits = selection["bits"]
        allowed = (3, 4, 6) if selection["backend"] == "fused" else (2, 3, 4, 6, 8)
        if type(bits) is not int or bits not in allowed:
            raise ManifestError(f"invalid bits for {name}")
        rpg = selection["rows_per_group"]
        if type(rpg) is not int or rpg not in (1, 2, 4, 8):
            raise ManifestError(f"invalid rows_per_group for {name}")
    if version == 2:
        _hash_map(
            data["model_sha256"],
            "model_sha256",
            lambda p: p == "config.json" or _WEIGHT.fullmatch(p),
        )
        if "config.json" not in data["model_sha256"] or len(data["model_sha256"]) < 2:
            raise ManifestError("model_sha256 must include config.json and model weights")
        _hash_map(data["kernel_sha256"], "kernel_sha256", _KERNEL.fullmatch)
    return copy.deepcopy(data)


def _sha256(path):
    if path.is_symlink() or not path.is_file():
        raise ManifestError(f"binding requires a regular non-symlink file: {path}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def model_hashes(source):
    source = Path(source)
    weights = sorted(source.glob("model*.safetensors"))
    if not weights:
        raise ManifestError("model contains no model*.safetensors weights")
    return {path.name: _sha256(path) for path in [source / "config.json", *weights]}


def kernel_hashes():
    root = Path(__file__).parent
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted((root / "kernels").glob("*.metal"))
    }


def create_manifest(model_dir, dispatch, environment_dict):
    """Bind final saved artifacts and shipped kernels to the original profile metadata."""
    source, profile = model_dir, environment_dict
    _validate_environment(profile)
    data = validate_schema(
        {
            "schema_version": 2,
            "dispatch": dispatch,
            "profile_device": profile["device"],
            "mlx": profile["mlx"],
            "mlx_lm": profile["mlx_lm"],
            "model_sha256": model_hashes(source),
            "kernel_sha256": kernel_hashes(),
        }
    )
    _validate_config(source, data["dispatch"])
    return data


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ManifestError(f"nonfinite JSON constant: {value}")


def load_manifest(path, *, allow_legacy=False):
    try:
        data = json.loads(
            Path(path).read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError) as error:
        raise ManifestError(f"invalid execution manifest JSON: {error}") from error
    return validate_schema(data, allow_legacy=allow_legacy)


def validate_manifest(model_dir, manifest_dict, environment_dict, strict_runtime=True):
    """Return validated dispatch; never activate fusion here.

    strict_runtime=False skips only hardware/runtime comparison, not schema,
    artifact, kernel, or saved precision-map checks. Intended for CPU sealing.
    """
    source, current = model_dir, environment_dict
    if type(strict_runtime) is not bool:
        raise ManifestError("strict_runtime must be a boolean")
    if strict_runtime:
        _validate_environment(current)
    data = validate_schema(manifest_dict)
    if data["model_sha256"] != model_hashes(source):
        raise ManifestError("model artifact SHA-256 mismatch")
    if data["kernel_sha256"] != kernel_hashes():
        raise ManifestError("shipped Metal kernel SHA-256 mismatch")
    _validate_config(source, data["dispatch"])
    if strict_runtime and (
        data["profile_device"] != current["device"]
        or data["mlx"] != current["mlx"]
        or data["mlx_lm"] != current["mlx_lm"]
    ):
        raise ManifestError("profile hardware/runtime differs; run a fresh profile")
    return copy.deepcopy(data["dispatch"])


def verify_manifest(source, data, current):
    """Compatibility name for strict validation."""
    return validate_manifest(source, data, current)


def _validate_config(source, dispatch):
    try:
        config = json.loads(
            (Path(source) / "config.json").read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError) as error:
        raise ManifestError(f"invalid model config: {error}") from error
    if not isinstance(config, dict):
        raise ManifestError("model config must be an object")
    quantization = config.get("quantization", config.get("quantization_config"))
    for name, selection in dispatch.items():
        if selection["backend"] == "fused" and config.get("model_type") != "qwen2":
            raise ManifestError("model config: fusion supports only exact Qwen2 MLP semantics")
        if not isinstance(quantization, dict):
            raise ManifestError("model config lacks quantization metadata")
        pairs = [
            quantization.get(f"{name}.{projection}", quantization)
            for projection in ("gate_proj", "up_proj")
        ]
        for pair in pairs:
            if (
                not isinstance(pair, dict)
                or type(pair.get("bits")) is not int
                or pair["bits"] != selection["bits"]
                or pair.get("mode", "affine") != "affine"
                or type(pair.get("group_size")) is not int
                or pair["group_size"] not in (32, 64, 128)
            ):
                raise ManifestError(f"model config quantization disagrees with dispatch: {name}")
        if pairs[0]["group_size"] != pairs[1]["group_size"]:
            raise ManifestError(f"model config gate/up group sizes disagree: {name}")


def validate_model_dispatch(model, dispatch):
    """Ensure saved precision claims describe the loaded projections, before mutation."""
    modules = dict(model.named_modules())
    for name, selection in dispatch.items():
        if name not in modules:
            raise ManifestError(f"Unknown module path in manifest: {name}")
        for projection in ("gate_proj", "up_proj"):
            loaded_bits = getattr(getattr(modules[name], projection, None), "bits", None)
            if type(loaded_bits) is not int or loaded_bits != selection["bits"]:
                raise ManifestError(
                    f"loaded model bits disagree with dispatch: {name}.{projection}"
                )
