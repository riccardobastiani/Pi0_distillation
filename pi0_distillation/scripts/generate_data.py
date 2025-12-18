import os
import h5py
import torch
import numpy as np
import cv2
from tqdm import tqdm
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

TEACHER_ID = "lerobot/pi05_libero_finetuned"
SAVE_DIR = "./data/distilled"
TASK_SUITE = "libero_spatial"
EPISODES = 50


def resize_img(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)


def main() -> None:
    print(f"Loading teacher policy {TEACHER_ID} ...")
    policy = PI0Policy.from_pretrained(TEACHER_ID).cuda().eval()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE]()

    os.makedirs(SAVE_DIR, exist_ok=True)

    for task_id in range(task_suite.get_num_tasks()):
        task = task_suite.get_task(task_id)
        print(f"Processing task: {task.name}")

        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(
                os.environ["LIBERO_ASSET_ROOT"],
                "bddl_files",
                task.problem_folder,
                task.bddl_file,
            ),
            render_gpu_device_id=0,
        )
        env.seed(0)

        with h5py.File(os.path.join(SAVE_DIR, f"{task.name}.hdf5"), "w") as handle:
            handle.attrs["task_description"] = task.language
            grp = handle.create_group("data")
            success = 0

            for episode in tqdm(range(EPISODES)):
                obs = env.reset()
                done = False
                steps = 0
                ep_data: dict[str, list[np.ndarray]] = {
                    "obs": [],
                    "prop": [],
                    "act": [],
                }

                while steps < 600 and not done:
                    img_t = (
                        torch.from_numpy(obs["agentview_image"])  # type: ignore[arg-type]
                        .permute(2, 0, 1)
                        .float()
                        .div(255.0)
                        .unsqueeze(0)
                        .cuda()
                    )
                    state_t = torch.from_numpy(obs["robot0_joint_pos"]).float().unsqueeze(0).cuda()

                    with torch.no_grad():
                        action = policy.select_action(  # type: ignore[attr-defined]
                            batch={
                                "observation.images.image": img_t,
                                "observation.state": state_t,
                                "text": [task.language],
                            }
                        )

                    act_np = action.squeeze(0).cpu().numpy()
                    next_obs, _, done, _ = env.step(act_np)

                    ep_data["obs"].append(resize_img(obs["agentview_image"]))
                    ep_data["prop"].append(obs["robot0_joint_pos"])
                    ep_data["act"].append(act_np)

                    obs = next_obs
                    steps += 1

                if done:
                    d_grp = grp.create_group(f"demo_{episode}")
                    d_grp.create_dataset("actions", data=np.array(ep_data["act"]))
                    o_grp = d_grp.create_group("obs")
                    o_grp.create_dataset("agentview_rgb", data=np.array(ep_data["obs"]))
                    o_grp.create_dataset("joint_states", data=np.array(ep_data["prop"]))
                    success += 1
            print(f"Success rate: {success}/{EPISODES}")


if __name__ == "__main__":
    main()
