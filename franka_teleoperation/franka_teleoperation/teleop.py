#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Legacy FrankaTeleop class for backward compatibility.
For new code, use the factory functions or specific teleop classes.
"""

import logging
from pathlib import Path
from typing import Any, Dict
import yaml

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_teleop import BaseTeleopConfig, SpacemouseTeleopConfig
from .spacemouse.spacemouse_robot import SpaceMouseRobot

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class FrankaTeleop(Teleoperator):
    """
    Legacy teleoperation class for backward compatibility.
    Supports spacemouse and oculus modes.
    
    For new code, prefer using:
    - create_teleop(config) from teleop_factory
    - SpacemouseTeleop or OculusTeleop directly
    """

    config_class = BaseTeleopConfig
    name = "FrankaTeleop"
    
    def __init__(self, config: BaseTeleopConfig):
        super().__init__(config)
        self.cfg = config
        self._is_connected = False
        
        if config.control_mode == "spacemouse":
            self.name = "SpacemouseTeleop"
        elif config.control_mode == "oculus":
            self.name = "OculusTeleop"
        else:
            self.name = "unnamed"
            raise ValueError(f"Unknown control mode: {config.control_mode}")
        

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if self.cfg.control_mode == "spacemouse":
            self._check_spacemouse_connection()
            self._is_connected = True
        elif self.cfg.control_mode == "oculus":
            self._check_oculus_connection()
            self._is_connected = True

        logger.info(f"[INFO] {self.name} env initialization completed successfully.\n")
    
    def _check_spacemouse_connection(self) -> None:
        logger.info("\n===== [TELEOP] Connecting to spacemouse =====")
        self.spacemouse_robot = SpaceMouseRobot(
            use_gripper=self.cfg.use_gripper,
            pose_scaler=self.cfg.pose_scaler,
            channel_signs=self.cfg.channel_signs,
            )
        actions = self.spacemouse_robot.get_action()
        formatted_actions = [round(float(j), 4) for j in actions]
        logger.info(f"[TELEOP] Current ee pose actions: {formatted_actions}")
        logger.info("===== [TELEOP] Spacemouse connected successfully. =====\n")
    
    def _check_oculus_connection(self) -> None:
        logger.info("\n===== [TELEOP] Connecting to Oculus Quest =====")
        from .oculus.oculus_robot import OculusRobot

        self.oculus_robot = OculusRobot(
            ip=getattr(self.cfg, 'ip', '192.168.110.62'),
            use_gripper=self.cfg.use_gripper,
            pose_scaler=getattr(self.cfg, 'pose_scaler', [1.0, 1.0]),
            channel_signs=getattr(self.cfg, 'channel_signs', [1, 1, 1, 1, 1, 1]),
            )
        logger.info("===== [TELEOP] Oculus connected successfully. =====\n")


    def calibrate(self) -> None:
        pass

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        if self.cfg.control_mode == "spacemouse":
            return self.spacemouse_robot.get_observations()
        elif self.cfg.control_mode == "oculus":
            return self.oculus_robot.get_observations()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        
        if self.cfg.control_mode == "spacemouse":
            self.spacemouse_robot._expert.close()
        elif self.cfg.control_mode == "oculus":
            pass  # OculusRobot doesn't have explicit disconnect
        logger.info(f"[INFO] ===== All {self.name} connections have been closed =====")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    class RecordConfig:
        def __init__(self, cfg: Dict[str, Any]):
            teleop = cfg["teleop"]
            sm_cfg = teleop["spacemouse_config"]

            if teleop["control_mode"] == "spacemouse":
                # sm teleop config
                self.use_gripper = sm_cfg["use_gripper"]  
                self.pose_scaler = sm_cfg["pose_scaler"]
                self.channel_signs = sm_cfg["channel_signs"]
                self.control_mode = teleop.get("control_mode", "spacemouse")
    
    with open(Path(__file__).parent / "config" / "cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    if record_cfg.control_mode == "spacemouse":
        teleop_config = SpacemouseTeleopConfig(
            use_gripper=record_cfg.use_gripper,
            pose_scaler=record_cfg.pose_scaler,
            channel_signs=record_cfg.channel_signs,
            control_mode=record_cfg.control_mode,    
        )
    teleop = FrankaTeleop(teleop_config)
    teleop.connect()
    for i in range(2):
        teleop.get_action()
