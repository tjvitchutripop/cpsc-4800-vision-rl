import gymnasium as gym
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class FlattenRGBDSegObservationWrapper(gym.ObservationWrapper):
    """
    Flattens observations including RGB, depth, segmentation, and state data.

    Args:
        rgb (bool): Whether to include RGB images in the observation. Default: True
        depth (bool): Whether to include depth images in the observation. Default: True
        segmentation (bool): Whether to include segmentation masks in the observation. Default: False
        state (bool): Whether to include state data in the observation. Default: True
        separate_channels (bool): Whether to keep RGB, depth, and segmentation as separate keys.
                                   If False, they will be merged into a single "visual" key. Default: True

    Returns observations with keys depending on flags:
        - If separate_channels=True: "rgb", "depth", "segmentation" (depending on which are enabled), "state"
        - If separate_channels=False: "visual" (merged channels), "state"
    """

    def __init__(
        self,
        env,
        rgb=True,
        depth=True,
        segmentation=False,
        state=True,
        separate_channels=True
    ) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_segmentation = segmentation
        self.include_state = state
        self.separate_channels = separate_channels

        # Check if rgb/depth/segmentation data exists in first camera's sensor data
        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "depth" not in first_cam:
            self.include_depth = False
        if "rgb" not in first_cam:
            self.include_rgb = False
        if "segmentation" not in first_cam:
            self.include_segmentation = False

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]

        rgb_images = []
        depth_images = []
        seg_images = []

        for cam_data in sensor_data.values():
            if self.include_rgb:
                rgb_images.append(cam_data["rgb"])
            if self.include_depth:
                depth_images.append(cam_data["depth"])
            if self.include_segmentation:
                seg_images.append(cam_data["segmentation"])

        # Concatenate images from multiple cameras
        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        if len(depth_images) > 0:
            depth_images = torch.concat(depth_images, axis=-1)
        if len(seg_images) > 0:
            seg_images = torch.concat(seg_images, axis=-1)

        # Flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )

        ret = dict()

        # Add state if requested
        if self.include_state:
            ret["state"] = observation

        # Handle visual modalities
        if self.separate_channels:
            # Keep each modality separate
            if self.include_rgb and len(rgb_images) > 0:
                ret["rgb"] = rgb_images
            if self.include_depth and len(depth_images) > 0:
                ret["depth"] = depth_images
            if self.include_segmentation and len(seg_images) > 0:
                ret["segmentation"] = seg_images
        else:
            # Merge all visual modalities into a single tensor
            visual_tensors = []
            if self.include_rgb and len(rgb_images) > 0:
                visual_tensors.append(rgb_images)
            if self.include_depth and len(depth_images) > 0:
                visual_tensors.append(depth_images)
            if self.include_segmentation and len(seg_images) > 0:
                visual_tensors.append(seg_images)

            if len(visual_tensors) > 0:
                ret["visual"] = torch.concat(visual_tensors, axis=-1)

        return ret
