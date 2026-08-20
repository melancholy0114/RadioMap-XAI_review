"""Backbone construction and checkpoint-compatibility helpers."""

from collections.abc import Mapping
from copy import deepcopy

from model.radio_map_model import Restormer
from model.radiounet import RadioUNetC


_ALIASES = {
    "restormer": "restormer",
    "radiounet_c": "radiounet_c",
    "radiounetc": "radiounet_c",
    "radio_unet_c": "radiounet_c",
}

_RESTORMER_ARGUMENTS = {
    "inp_channels",
    "out_channels",
    "dim",
    "num_blocks",
    "num_refinement_blocks",
    "heads",
    "ffn_expansion_factor",
    "bias",
    "LayerNorm_type",
}

_RADIOUNET_C_ARGUMENTS = {
    "inp_channels",
    "out_channels",
}


def canonical_model_name(name):
    """Return a stable model identifier used in configs and checkpoints."""
    normalized = str(name).strip().lower().replace("-", "_")
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted({"restormer", "radiounet_c"}))
        raise ValueError(f"Unknown model.name {name!r}; choose one of: {choices}") from exc


def model_name_from_config(model_config):
    """Read the configured backbone, preserving legacy Restormer configs."""
    if not isinstance(model_config, Mapping):
        raise TypeError("model configuration must be a mapping")
    return canonical_model_name(model_config.get("name", "restormer"))


def _constructor_arguments(model_config, allowed_arguments, model_name):
    arguments = deepcopy(dict(model_config))
    arguments.pop("name", None)
    unsupported = sorted(set(arguments) - allowed_arguments)
    if unsupported:
        raise ValueError(
            f"Unsupported {model_name} model settings: {', '.join(unsupported)}. "
            "Keep settings for different backbones in their own config files."
        )
    return arguments


def build_model(model_config):
    """Build the backbone selected by ``model.name``.

    ``model.name`` defaults to ``restormer`` so configs and checkpoints created
    before the multi-backbone change remain usable.
    """
    model_name = model_name_from_config(model_config)

    if model_name == "restormer":
        arguments = _constructor_arguments(
            model_config,
            _RESTORMER_ARGUMENTS,
            model_name,
        )
        model = Restormer(**arguments)
    else:
        arguments = _constructor_arguments(
            model_config,
            _RADIOUNET_C_ARGUMENTS,
            model_name,
        )
        model = RadioUNetC(**arguments)

    # Store serializable provenance on the module. DDP checkpoint writers use
    # this metadata without coupling either training pipeline to a backbone.
    model.model_name = model_name
    model.model_config = {"name": model_name, **deepcopy(arguments)}
    return model


def unwrap_model(model):
    """Return the underlying module for plain, DataParallel, or DDP models."""
    return model.module if hasattr(model, "module") else model


def get_model_name(model):
    """Return the canonical identifier for a constructed model."""
    base_model = unwrap_model(model)
    if hasattr(base_model, "model_name"):
        return canonical_model_name(base_model.model_name)
    if isinstance(base_model, Restormer):
        return "restormer"
    if isinstance(base_model, RadioUNetC):
        return "radiounet_c"
    raise ValueError(f"Unsupported model type: {type(base_model).__name__}")


def checkpoint_metadata(model):
    """Return backbone metadata to embed in newly written checkpoints."""
    base_model = unwrap_model(model)
    model_name = get_model_name(base_model)
    model_config = deepcopy(
        getattr(base_model, "model_config", {"name": model_name})
    )
    return {
        "model_name": model_name,
        "model_config": model_config,
    }


def checkpoint_model_name(checkpoint):
    """Read checkpoint provenance; legacy project checkpoints are Restormer."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")

    name = checkpoint.get("model_name")
    if name is None:
        stored_config = checkpoint.get("model_config")
        if isinstance(stored_config, Mapping):
            name = stored_config.get("name")

    # Every project checkpoint predating model metadata used Restormer.
    return canonical_model_name("restormer" if name is None else name)


def validate_checkpoint_model(checkpoint, model):
    """Fail early when a checkpoint belongs to a different backbone."""
    checkpoint_name = checkpoint_model_name(checkpoint)
    configured_name = get_model_name(model)
    if checkpoint_name != configured_name:
        raise ValueError(
            f"Checkpoint backbone is {checkpoint_name!r}, but the config builds "
            f"{configured_name!r}. Use the matching config/checkpoint pair."
        )


def normalize_state_dict(state_dict):
    """Accept weights saved from plain, DataParallel, or DDP models."""
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def get_gradcam_target_layer(model):
    """Return a semantically comparable late feature layer per backbone."""
    base_model = unwrap_model(model)
    target_layer = getattr(base_model, "gradcam_target_layer", None)
    if target_layer is not None:
        return target_layer
    if isinstance(base_model, Restormer):
        return base_model.refinement[-1]
    raise ValueError(
        f"No Grad-CAM target layer is registered for {type(base_model).__name__}"
    )
