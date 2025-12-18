"""Utility to create synthetic HDF5 demos for local CPU smoke tests."""

from __future__ import annotations

import argparse
import os
import numpy as np
import h5py


def create_dummy_dataset(
    output_path: str,
    seq_len: int,
    demos: int,
    image_size: int,
    rng: np.random.Generator,
) -> None:
    with h5py.File(output_path, "w") as handle:
        handle.attrs["task_description"] = "dummy manipulation task"
        data_group = handle.create_group("data")

        for demo_idx in range(demos):
            demo_group = data_group.create_group(f"demo_{demo_idx}")
            actions = rng.standard_normal((seq_len, 7), dtype=np.float32)
            proprio = rng.standard_normal((seq_len, 7), dtype=np.float32)
            images = rng.integers(
                0,
                255,
                size=(seq_len, image_size, image_size, 3),
                dtype=np.uint8,
            )

            demo_group.create_dataset("actions", data=actions)
            obs_group = demo_group.create_group("obs")
            obs_group.create_dataset("agentview_rgb", data=images)
            obs_group.create_dataset("joint_states", data=proprio)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic HDF5 demos.")
    parser.add_argument(
        "--output-dir",
        default="./data/distilled",
        help="Folder where dummy HDF5 files will be written.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=1,
        help="Number of dummy task files to create.",
    )
    parser.add_argument(
        "--demos",
        type=int,
        default=2,
        help="Number of demos per task.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Number of frames per demo.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Spatial resolution for stored RGB frames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for task_idx in range(args.tasks):
        task_name = f"dummy_task_{task_idx}"
        output_path = os.path.join(args.output_dir, f"{task_name}.hdf5")
        create_dummy_dataset(output_path, args.seq_len, args.demos, args.image_size, rng)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
