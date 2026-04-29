# Configuration classes
from .config_teleop import (
    BaseTeleopConfig,
    DynamixelTeleopConfig,
    SpacemouseTeleopConfig,
    OculusTeleopConfig,
    FrankaTeleopConfig,  # Legacy compatibility
)

# Base class
from .base_teleop import BaseTeleop

# Factory functions
from .teleop_factory import create_teleop, create_teleop_config, get_action_features

__all__ = [
    # Configuration classes
    "BaseTeleopConfig",
    "DynamixelTeleopConfig",
    "SpacemouseTeleopConfig",
    "OculusTeleopConfig",
    "FrankaTeleopConfig",
    # Base class
    "BaseTeleop",
    # Teleoperation implementations
    "DynamixelTeleop",
    "SpacemouseTeleop",
    "OculusTeleop",
    "FrankaTeleop",
    # Factory functions
    "create_teleop",
    "create_teleop_config",
    "get_action_features",
]


def __getattr__(name):
    if name == "DynamixelTeleop":
        from .dynamixel_teleop import DynamixelTeleop

        return DynamixelTeleop
    if name == "SpacemouseTeleop":
        from .spacemouse_teleop import SpacemouseTeleop

        return SpacemouseTeleop
    if name == "OculusTeleop":
        from .oculus_teleop import OculusTeleop

        return OculusTeleop
    if name == "FrankaTeleop":
        from .teleop import FrankaTeleop

        return FrankaTeleop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
