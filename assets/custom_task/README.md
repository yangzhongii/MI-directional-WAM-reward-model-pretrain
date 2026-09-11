# Generated MuJoCo task assets

Run the following command from the repository root:

```bash
bash requirements/download_mujoco_assets.sh --use-mirrors
```

The downloader fetches only the Franka Emika Panda subtree from Google
DeepMind's MuJoCo Menagerie. The robot model keeps its upstream Apache-2.0
license. The task objects and goals are generated from native MuJoCo primitives
by `mi_reward/scripts/prepare_mujoco_assets.py`.

Generated files are placed in `assets/custom_task/scene/`; the downloaded
upstream checkout is stored in `assets/vendor/mujoco_menagerie/`.
