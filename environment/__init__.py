"""Flood Guardian environment package. Importing it registers FloodGuardian-v0 with Gymnasium."""

from gymnasium.envs.registration import register

from environment.custom_env import FloodGuardianEnv

__all__ = ["FloodGuardianEnv", "ENV_ID"]

ENV_ID = "FloodGuardian-v0"

register(id=ENV_ID, entry_point="environment.custom_env:FloodGuardianEnv")
