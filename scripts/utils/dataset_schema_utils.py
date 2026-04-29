from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

LEGACY_ACTION_SOURCE_NAMES = [
    "delta_ee_pose.x",
    "delta_ee_pose.y",
    "delta_ee_pose.z",
    "delta_ee_pose.rx",
    "delta_ee_pose.ry",
    "delta_ee_pose.rz",
    "gripper_cmd_bin",
]

LEGACY_STATE_SOURCE_NAMES = [
    "ee_pose.x",
    "ee_pose.y",
    "ee_pose.z",
    "ee_pose.rx",
    "ee_pose.ry",
    "ee_pose.rz",
    "gripper_state_norm",
]

LEGACY_IMAGE_SOURCE_MAP = {
    "observation.images.exterior_image": "exterior_image",
    "observation.images.wrist_image": "wrist_image",
}

SUPPORTED_LEGACY_FEATURE_KEYS = {"actions", "state", *LEGACY_IMAGE_SOURCE_MAP.keys()}


def load_dataset_schema_config(schema_ref: str | None, config_dir: Path) -> dict[str, Any] | None:
    if not schema_ref:
        return None

    schema_path = Path(schema_ref)
    if not schema_path.is_absolute():
        schema_path = (config_dir / schema_path).resolve()

    if not schema_path.exists():
        raise FileNotFoundError(f"Dataset schema config not found: {schema_path}")

    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    if not isinstance(schema, dict) or not isinstance(schema.get("features"), dict):
        raise ValueError(f"Invalid dataset schema config: {schema_path}. Expected top-level 'features' mapping.")

    return {
        "path": schema_path,
        "features": schema["features"],
    }


def uses_legacy_dataset_schema(schema_config: dict[str, Any] | None) -> bool:
    return schema_config is not None


def build_legacy_dataset_features(
    schema_config: dict[str, Any],
    action_features: dict[str, Any],
    observation_features: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    dataset_features: dict[str, dict[str, Any]] = {}

    for key, spec in schema_config["features"].items():
        if key not in SUPPORTED_LEGACY_FEATURE_KEYS:
            raise ValueError(
                f"Unsupported legacy dataset feature key: {key}. "
                f"Supported keys: {sorted(SUPPORTED_LEGACY_FEATURE_KEYS)}"
            )

        feature_spec = _normalize_feature_spec(key, spec)

        if key == "actions":
            _validate_vector_feature(
                feature_key=key,
                feature_spec=feature_spec,
                available_names=action_features.keys(),
                required_names=LEGACY_ACTION_SOURCE_NAMES,
            )
        elif key == "state":
            _validate_vector_feature(
                feature_key=key,
                feature_spec=feature_spec,
                available_names=observation_features.keys(),
                required_names=LEGACY_STATE_SOURCE_NAMES,
            )
        else:
            source_key = LEGACY_IMAGE_SOURCE_MAP[key]
            actual_shape = observation_features.get(source_key)
            if not isinstance(actual_shape, tuple):
                raise ValueError(
                    f"Legacy image feature '{key}' requires observation key '{source_key}', but it is not available."
                )

            if tuple(feature_spec["shape"]) != tuple(actual_shape):
                raise ValueError(
                    f"Shape mismatch for '{key}': schema expects {tuple(feature_spec['shape'])}, "
                    f"but runtime camera shape is {tuple(actual_shape)} from '{source_key}'."
                )

            if feature_spec["dtype"] not in {"image", "video"}:
                raise ValueError(f"Legacy image feature '{key}' must use dtype 'image' or 'video'.")

        dataset_features[key] = feature_spec

    return dataset_features


def build_legacy_observation_frame(
    dataset_features: dict[str, dict[str, Any]],
    observation_values: dict[str, Any],
) -> dict[str, Any]:
    frame: dict[str, Any] = {}

    for key in dataset_features:
        if key == "state":
            frame[key] = build_legacy_state_vector(observation_values)
        elif key in LEGACY_IMAGE_SOURCE_MAP:
            source_key = LEGACY_IMAGE_SOURCE_MAP[key]
            frame[key] = observation_values[source_key]

    return frame


def build_legacy_action_frame(
    dataset_features: dict[str, dict[str, Any]],
    action_values: dict[str, Any],
) -> dict[str, np.ndarray]:
    frame: dict[str, np.ndarray] = {}

    if "actions" in dataset_features:
        frame["actions"] = build_legacy_action_vector(action_values)

    return frame


def build_legacy_action_vector(action_values: dict[str, Any]) -> np.ndarray:
    return np.array([action_values[name] for name in LEGACY_ACTION_SOURCE_NAMES], dtype=np.float32)


def build_legacy_state_vector(observation_values: dict[str, Any]) -> np.ndarray:
    return np.array([observation_values[name] for name in LEGACY_STATE_SOURCE_NAMES], dtype=np.float32)


def get_vector_feature_labels(feature_spec: dict[str, Any], vector_length: int) -> list[str]:
    names = feature_spec.get("names") or []
    if len(names) == vector_length:
        return list(names)
    if len(names) == 1:
        return [f"{names[0]}[{idx}]" for idx in range(vector_length)]
    return [f"dim_{idx}" for idx in range(vector_length)]


def _normalize_feature_spec(feature_key: str, spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"Feature spec for '{feature_key}' must be a mapping.")

    required_fields = {"dtype", "shape", "names"}
    missing_fields = required_fields - set(spec)
    if missing_fields:
        raise ValueError(f"Feature spec for '{feature_key}' is missing fields: {sorted(missing_fields)}")

    shape = tuple(spec["shape"])
    if not shape:
        raise ValueError(f"Feature spec for '{feature_key}' must define a non-empty shape.")

    return {
        "dtype": spec["dtype"],
        "shape": shape,
        "names": list(spec["names"]) if spec["names"] is not None else None,
    }


def _validate_vector_feature(
    feature_key: str,
    feature_spec: dict[str, Any],
    available_names,
    required_names: list[str],
) -> None:
    if feature_spec["dtype"] != "float32":
        raise ValueError(f"Legacy vector feature '{feature_key}' must use dtype 'float32'.")

    if len(feature_spec["shape"]) != 1:
        raise ValueError(f"Legacy vector feature '{feature_key}' must be 1D.")

    if feature_spec["shape"][0] != len(required_names):
        raise ValueError(
            f"Legacy vector feature '{feature_key}' must have shape ({len(required_names)},), "
            f"got {feature_spec['shape']}."
        )

    missing_names = [name for name in required_names if name not in available_names]
    if missing_names:
        raise ValueError(
            f"Legacy vector feature '{feature_key}' cannot be built. Missing runtime fields: {missing_names}"
        )
