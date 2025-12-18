import glob
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class DistilledTeacherDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        image_size: tuple[int, int] = (128, 128),
        use_frame_stacking: bool = True,
        num_stacked_frames: int = 4,
    ) -> None:
        self.use_frame_stacking = use_frame_stacking
        self.num_stacked_frames = num_stacked_frames
        self.transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(image_size), transforms.ToTensor()]
        )

        self.files = glob.glob(os.path.join(folder_path, "**/*.hdf5"), recursive=True)
        self.indices: list[tuple[str, str, int]] = []
        self.task_prompts: dict[str, str] = {}

        self.action_mean = torch.zeros(7)
        self.action_std = torch.ones(7)
        self.proprio_mean = torch.zeros(7)
        self.proprio_std = torch.ones(7)

        print(f"Indexing {len(self.files)} files in {folder_path} ...")
        for filepath in self.files:
            try:
                with h5py.File(filepath, "r") as contents:
                    prompt = contents.attrs.get("task_description", "do task")
                    if isinstance(prompt, bytes):
                        prompt = prompt.decode("utf-8")
                    self.task_prompts[filepath] = prompt

                    if "data" in contents:
                        for demo in contents["data"].keys():
                            n_frames = contents["data"][demo]["actions"].shape[0]
                            start = (num_stacked_frames - 1) if use_frame_stacking else 0
                            for frame_idx in range(start, n_frames):
                                self.indices.append((filepath, demo, frame_idx))
            except Exception:
                pass
        print(f"Loaded {len(self.indices)} samples.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        fpath, demo, frame_idx = self.indices[idx]
        with h5py.File(fpath, "r") as contents:
            grp = contents["data"][demo]
            obs_group = grp["obs"]
            img_key = "agentview_rgb" if "agentview_rgb" in obs_group else "agentview_image"
            prop_key = "joint_states" if "joint_states" in obs_group else "robot0_joint_pos"

            if self.use_frame_stacking:
                imgs = [
                    self.transform(obs_group[img_key][frame_idx - offset])
                    for offset in range(self.num_stacked_frames - 1, -1, -1)
                ]
                img = torch.cat(imgs, dim=0)
            else:
                img = self.transform(obs_group[img_key][frame_idx])

            prop = torch.from_numpy(obs_group[prop_key][frame_idx][:7]).float()
            act = torch.from_numpy(grp["actions"][frame_idx]).float()

            prop = (prop - self.proprio_mean) / self.proprio_std
            act = (act - self.action_mean) / self.action_std

        return {
            "observations": img,
            "proprio_states": prop,
            "actions": act,
            "prompt": self.task_prompts[fpath],
        }
