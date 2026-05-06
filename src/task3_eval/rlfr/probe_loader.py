"""Load frozen reward-hacking probes for Task 5 RLFR."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from task3_eval.utils.cli import parse_bool


DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_POOLING_METHOD = "completion_mean_pool"
PROBE_EXTENSIONS = (".pt", ".pth", ".bin")
PREFERRED_PROBE_FILENAMES = (
    "probe.pt",
    "probe_model.pt",
    "best_probe.pt",
    "linear_probe.pt",
    "label_best_layer.pt",
    "model.pt",
)


class LinearProbe:
    """Tiny torch linear probe with one scalar logit output."""

    def __new__(cls, input_dim: int):
        import torch.nn as nn

        class _LinearProbe(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = nn.Linear(dim, 1)

            def forward(self, features):  # type: ignore[no-untyped-def]
                return self.linear(features).squeeze(-1)

        return _LinearProbe(input_dim)


class DummyProbe:
    """Smoke-test probe that returns p_hack=0 without model weights."""

    is_dummy = True

    def eval(self) -> "DummyProbe":
        return self

    def requires_grad_(self, requires_grad: bool) -> "DummyProbe":
        return self

    def to(self, device: Any) -> "DummyProbe":
        return self

    def __call__(self, features):  # type: ignore[no-untyped-def]
        import torch

        batch = int(getattr(features, "shape", [1])[0])
        return torch.zeros(batch, dtype=torch.float32, device=getattr(features, "device", None))


@dataclass(slots=True)
class FrozenProbe:
    model: Any
    probe_path: str | None
    architecture: str
    input_dim: int
    layer_idx: int
    pooling_method: str
    feature_normalization: str | None = None
    feature_mean: Any | None = None
    feature_std: Any | None = None
    threshold: float | None = None
    is_dummy: bool = False

    def predict_p_hack(self, features: Any) -> Any:
        """Return p_hack in [0, 1] for each feature row."""

        import torch

        if self.is_dummy:
            return torch.zeros(features.shape[0], dtype=torch.float32, device=features.device)
        normalized = self.normalize(features)
        with torch.no_grad():
            logits = self.model(normalized)
        return torch.sigmoid(logits.float()).clamp(0.0, 1.0)

    def normalize(self, features: Any) -> Any:
        if not self.feature_normalization or self.feature_normalization == "none":
            return features
        if self.feature_normalization == "standard":
            if self.feature_mean is None or self.feature_std is None:
                raise ValueError("feature_normalization=standard requires feature_mean and feature_std.")
            return (features - self.feature_mean.to(features.device)) / self.feature_std.to(features.device).clamp_min(1e-6)
        raise ValueError(f"Unsupported feature_normalization: {self.feature_normalization}")


def _load_sidecar_metadata(probe_path: Path) -> dict[str, Any]:
    candidates = [
        probe_path.with_suffix(".json"),
        probe_path.with_name(f"{probe_path.stem}_metadata.json"),
        probe_path.parent / "probe_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def resolve_probe_checkpoint_path(probe_path: str | Path) -> Path:
    """Resolve either a probe checkpoint file or a directory containing one."""

    path = Path(probe_path)
    if path.is_file():
        return path
    if not path.is_dir():
        return path
    preferred = [path / filename for filename in PREFERRED_PROBE_FILENAMES]
    for candidate in preferred:
        if candidate.exists() and candidate.is_file():
            return candidate
    candidates = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and candidate.suffix.lower() in PROBE_EXTENSIONS
        and "optimizer" not in candidate.name.lower()
        and "scheduler" not in candidate.name.lower()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No probe checkpoint found in {path}. Expected one of {PROBE_EXTENSIONS}."
        )
    names = ", ".join(candidate.name for candidate in candidates)
    raise ValueError(
        f"Multiple probe checkpoint candidates found in {path}: {names}. "
        "Set PROBE_PATH to the exact file."
    )


def _extract_checkpoint_parts(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw, dict):
        return raw, {}
    metadata: dict[str, Any] = {}
    for key in ("metadata", "probe_config", "feature_config"):
        value = raw.get(key)
        if isinstance(value, dict):
            metadata.update(value)
    for state_key in ("state_dict", "model_state_dict", "probe_state_dict"):
        value = raw.get(state_key)
        if isinstance(value, dict):
            return value, metadata
    tensor_like = all(hasattr(value, "shape") for value in raw.values()) if raw else False
    if tensor_like:
        return raw, metadata
    return raw, metadata


def _state_dict_for_linear_probe(state_dict: dict[str, Any]) -> dict[str, Any]:
    if "weight" in state_dict:
        mapped = {"linear.weight": state_dict["weight"]}
        if "bias" in state_dict:
            mapped["linear.bias"] = state_dict["bias"]
        return mapped
    return state_dict


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def load_frozen_probe(
    probe_path: str | None,
    probe_architecture: str = "linear",
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    layer_idx: int | None = None,
    pooling_method: str | None = None,
    feature_normalization: str | None = None,
    threshold: float | None = None,
    allow_dummy: bool = False,
    map_location: str = "cpu",
    verbose: bool = True,
) -> FrozenProbe:
    """Load a frozen probe and its feature definition."""

    import torch

    if probe_path in (None, "", "null"):
        if not allow_dummy:
            raise FileNotFoundError("probe_path is required unless allow_dummy=True.")
        resolved_layer = 0 if layer_idx is None else int(layer_idx)
        resolved_pooling = pooling_method or DEFAULT_POOLING_METHOD
        probe = DummyProbe().eval().requires_grad_(False)
        handle = FrozenProbe(
            model=probe,
            probe_path=None,
            architecture="dummy",
            input_dim=int(hidden_size),
            layer_idx=resolved_layer,
            pooling_method=resolved_pooling,
            feature_normalization=None,
            threshold=threshold,
            is_dummy=True,
        )
        if verbose:
            print_probe_diagnostics(handle)
        return handle

    path = resolve_probe_checkpoint_path(probe_path)
    if not path.exists():
        if allow_dummy:
            return load_frozen_probe(
                None,
                probe_architecture=probe_architecture,
                hidden_size=hidden_size,
                layer_idx=layer_idx,
                pooling_method=pooling_method,
                feature_normalization=feature_normalization,
                threshold=threshold,
                allow_dummy=True,
                map_location=map_location,
                verbose=verbose,
            )
        raise FileNotFoundError(f"Probe checkpoint not found: {path}")

    if probe_architecture != "linear":
        raise ValueError("Only probe_architecture='linear' is implemented for Task5 pilot.")

    raw = torch.load(path, map_location=map_location)
    state_dict, checkpoint_metadata = _extract_checkpoint_parts(raw)
    sidecar_metadata = _load_sidecar_metadata(path)
    metadata = {**sidecar_metadata, **checkpoint_metadata}

    resolved_hidden_size = int(_metadata_value(metadata, "hidden_size", "input_dim") or hidden_size)
    resolved_layer = _metadata_value(metadata, "layer_idx", "probe_layer_idx", "layer")
    resolved_pooling = _metadata_value(metadata, "pooling_method", "probe_pooling_method", "pooling")
    if layer_idx is not None:
        resolved_layer = layer_idx
    if pooling_method is not None:
        resolved_pooling = pooling_method
    if resolved_layer is None or resolved_pooling is None:
        raise ValueError(
            "Probe feature definition is incomplete. Provide layer_idx and pooling_method "
            "or include them in probe metadata."
        )

    normalization = feature_normalization
    if normalization is None:
        normalization = _metadata_value(metadata, "feature_normalization", "normalization")
    mean = _metadata_value(metadata, "feature_mean", "mean")
    std = _metadata_value(metadata, "feature_std", "std")
    feature_mean = torch.as_tensor(mean, dtype=torch.float32) if mean is not None else None
    feature_std = torch.as_tensor(std, dtype=torch.float32) if std is not None else None
    resolved_threshold = threshold if threshold is not None else _metadata_value(metadata, "threshold")

    probe = LinearProbe(resolved_hidden_size)
    probe.load_state_dict(_state_dict_for_linear_probe(state_dict), strict=False)
    probe.eval()
    probe.requires_grad_(False)

    handle = FrozenProbe(
        model=probe,
        probe_path=str(path),
        architecture=probe_architecture,
        input_dim=resolved_hidden_size,
        layer_idx=int(resolved_layer),
        pooling_method=str(resolved_pooling),
        feature_normalization=str(normalization) if normalization else None,
        feature_mean=feature_mean,
        feature_std=feature_std,
        threshold=float(resolved_threshold) if resolved_threshold is not None else None,
        is_dummy=False,
    )
    if verbose:
        print_probe_diagnostics(handle)
    return handle


def print_probe_diagnostics(probe: FrozenProbe) -> None:
    print(f"probe_path={probe.probe_path or 'dummy'}")
    print(f"probe_architecture={probe.architecture}")
    print(f"layer_idx={probe.layer_idx}")
    print(f"pooling_method={probe.pooling_method}")
    print(f"input_dim={probe.input_dim}")
    print(f"normalization_used={bool(probe.feature_normalization and probe.feature_normalization != 'none')}")
    print(f"is_dummy={probe.is_dummy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_path")
    parser.add_argument("--probe_architecture", default="linear")
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--layer_idx", type=int)
    parser.add_argument("--pooling_method", default=DEFAULT_POOLING_METHOD)
    parser.add_argument("--feature_normalization")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--allow_dummy_probe", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--dry_run", "--dry-run", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()
    probe = load_frozen_probe(
        probe_path=args.probe_path,
        probe_architecture=args.probe_architecture,
        hidden_size=args.hidden_size,
        layer_idx=args.layer_idx,
        pooling_method=args.pooling_method,
        feature_normalization=args.feature_normalization,
        threshold=args.threshold,
        allow_dummy=args.allow_dummy_probe,
    )
    if args.dry_run:
        print("dry_run_ok")
    else:
        print("probe_load_ok")
    if probe.is_dummy and not args.allow_dummy_probe:
        raise RuntimeError("Dummy probe was loaded without allow_dummy_probe=True.")


if __name__ == "__main__":
    main()
