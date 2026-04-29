#!/usr/bin/env python

"""
Factory for creating teleoperation instances.
"""

from .base_teleop import BaseTeleop
from .config_teleop import (
    BaseTeleopConfig,
    SpacemouseTeleopConfig,
    OculusTeleopConfig,
)


def create_teleop(config: BaseTeleopConfig) -> BaseTeleop:
    """
    Create a teleoperation instance based on the configuration.
    
    Args:
        config: Teleoperation configuration (SpacemouseTeleopConfig or OculusTeleopConfig)
    
    Returns:
        A teleoperation instance (SpacemouseTeleop or OculusTeleop)
    
    Raises:
        ValueError: If the control mode is not supported
    """
    if isinstance(config, SpacemouseTeleopConfig) or config.control_mode == "spacemouse":
        from .spacemouse_teleop import SpacemouseTeleop

        return SpacemouseTeleop(config if isinstance(config, SpacemouseTeleopConfig) else SpacemouseTeleopConfig())
    
    elif isinstance(config, OculusTeleopConfig) or config.control_mode == "oculus":
        from .oculus_teleop import OculusTeleop

        return OculusTeleop(config if isinstance(config, OculusTeleopConfig) else OculusTeleopConfig())
    
    else:
        raise ValueError(f"Unsupported control mode: {config.control_mode}. "
                        f"Supported modes: spacemouse, oculus")


def create_teleop_config(control_mode: str, **kwargs) -> BaseTeleopConfig:
    """
    Create a teleoperation configuration based on the control mode.
    
    Args:
        control_mode: The teleoperation mode ("spacemouse" or "oculus")
        **kwargs: Configuration parameters specific to each mode
    
    Returns:
        A teleoperation configuration instance
    
    Raises:
        ValueError: If the control mode is not supported
    """
    if control_mode == "spacemouse":
        return SpacemouseTeleopConfig(**kwargs)
    elif control_mode == "oculus":
        return OculusTeleopConfig(**kwargs)
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}. "
                        f"Supported modes: spacemouse, oculus")


# Convenience function to get action features for a control mode
def get_action_features(control_mode: str, use_gripper: bool = True) -> dict:
    """
    Get the action features for a given control mode.
    
    Args:
        control_mode: The teleoperation mode ("spacemouse" or "oculus")
        use_gripper: Whether gripper is used
    
    Returns:
        Dictionary of action features
    """
    if control_mode in ["spacemouse", "oculus"]:
        features = {}
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"delta_ee_pose.{axis}"] = float
        if use_gripper:
            features["gripper_cmd_bin"] = float
        return features
    
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")
