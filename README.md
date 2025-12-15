# What Visual Features Matter Most for Vision-Based Reinforcement Learning in Robotics?

This is the code for my analysis of how different visual features (e.g., RGB, Segmentation, and Depth) and pretraining affect the training performance of RL policies in robotic manipulation tasks.  This was submitted as a course project for CPSC 5800: Introduction to Computer Vision at Yale University.

## Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended)
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

## Setup

### 1. Install UV

UV is a fast Python package installer and resolver. Install it using:

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, you can install it via pip:
```bash
pip install uv
```

### 2. Clone and Navigate to the Repository

```bash
git clone <repository-url>
cd cpsc-4800-vision-rl
```

### 3. Install Dependencies

UV will automatically create a virtual environment and install all dependencies from `pyproject.toml`:

```bash
uv sync
```

This installs all required packages including:
- ManiSkill (robotics simulation environment)
- PyTorch and TorchVision
- Weights & Biases (for experiment tracking)
- Transformers (for pretrained vision models)
- And other dependencies

### 4. Download Pretrained Models

Download the pretrained evaluation models from Google Drive:

```bash
uv run download_models.py
```

This will:
1. Download a zip file containing pretrained models from Google Drive
2. Extract models into the `models/` directory
3. Clean up the zip file

The models directory structure will look like:
```
models/
├── pickCube/
│   ├── pickCube_rgb.pt
│   ├── pickCube_depth.pt
│   ├── pickCube_rgb_depth.pt
│   └── ...
├── pushCube/
│   └── ...
└── pushT/
    └── ...
```

## Training

### Basic Usage

Use the `train_ppo.py` script to train PPO agents on different tasks with various visual modalities:

```bash
uv run src/train_ppo.py [OPTIONS]
```

### Key Training Arguments

#### Task Selection
- `--env-id`: Environment to train on (default: `"PickCube-v1"`)
  - Options: `PickCube-v1`, `PushCube-v1`, `PushT-v1`

#### Visual Modalities
- `--rgb`: Include RGB camera observations
- `--depth`: Include depth channel observations
- `--segmentation`: Include segmentation channel observations
- `--include-state`: Include robot state information (default: True)
- `--feature-extractor`: Vision encoder architecture (default: `"nature_cnn"`)
  - Options: `nature_cnn`, `theia`, `resnet50`, `densenet121`, `efficientnetb0`

#### Training Hyperparameters
- `--total-timesteps`: Total training timesteps (default: `10000000`)
- `--num-envs`: Number of parallel environments (default: `512`)
- `--num-steps`: Steps per rollout (default: `50`)
- `--learning-rate`: Learning rate (default: `3e-4`)
- `--gamma`: Discount factor (default: `0.8`)
- `--gae-lambda`: GAE lambda (default: `0.9`)
- `--num-minibatches`: Number of minibatches (default: `32`)
- `--update-epochs`: Number of update epochs (default: `4`)

#### Experiment Tracking
- `--track`: Enable Weights & Biases tracking (default: True)
- `--wandb-project-name`: W&B project name (default: `"vision-rl"`)
- `--capture-video`: Save evaluation videos (default: True)
- `--save-model`: Save model checkpoints (default: True)

#### Other Options
- `--seed`: Random seed (default: `1`)
- `--cuda`: Enable CUDA (default: True)
- `--eval-freq`: Evaluation frequency in iterations (default: `25`)

### Training Examples

**Train with RGB only:**
```bash
uv run src/train_ppo.py --env-id PickCube-v1 --rgb
```

**Train with depth only:**
```bash
uv run src/train_ppo.py --env-id PickCube-v1 --depth
```

**Train with RGB + depth:**
```bash
uv run src/train_ppo.py --env-id PickCube-v1 --rgb --depth
```

**Train with all modalities (RGB + depth + segmentation):**
```bash
uv run src/train_ppo.py --env-id PickCube-v1 --rgb --depth --segmentation
```

**Train on PushT task with segmentation and depth:**
```bash
uv run src/train_ppo.py --env-id PushT-v1 --segmentation --depth
```

**Train with pretrained ResNet50 features:**
```bash
uv run src/train_ppo.py --env-id PickCube-v1 --rgb --feature-extractor resnet50
```

**Customize training parameters:**
```bash
uv run src/train_ppo.py \
  --env-id PickCube-v1 \
  --rgb --depth \
  --num-envs 256 \
  --learning-rate 1e-4 \
  --total-timesteps 5000000
```

### Batch Training with Shell Scripts

For systematic evaluation across multiple modality combinations, use the provided shell scripts. These were the scripts used to get the empirical results found within the report.

```bash
# Train all modality combinations for PickCube
bash experiment_run_scripts/pickCube.sh

# Train all modality combinations for PushCube
bash experiment_run_scripts/pushCube.sh

# Train all modality combinations for PushT
bash experiment_run_scripts/pushT.sh
```

### Output

Training outputs are saved to:
- `runs/{run_name}/`: Contains checkpoints, logs, and videos
- `wandb/`: Weights & Biases run data (if tracking enabled)

Model checkpoints are saved as: `runs/{run_name}/model.pt`

## Evaluation

### Basic Usage

Use the `evaluate_model.py` script to evaluate trained models:

```bash
uv run src/evaluate_model.py [OPTIONS]
```

### Key Evaluation Arguments

#### Task and Model Selection
- `--task`: Task to evaluate on (choices: `pickCube`, `pushCube`, `pushT`)
- `--rgb`: Model uses RGB modality
- `--depth`: Model uses depth modality
- `--segmentation`: Model uses segmentation modality
- `--state`: Model uses state information (default: True)
- `--feature-extractor`: Feature extractor used (default: `"nature_cnn"`)
  - Options: `nature_cnn`, `theia`, `resnet50`, `densenet121`, `efficientnetb0`

The script automatically constructs the model path based on task and modalities:
- Nature CNN models: `models/{task}/{task}_{modalities}.pt`
- Pretrained models: `models/{task}/pretrained_features/{task}_{extractor}.pt`

#### Evaluation Parameters
- `--num-episodes`: Number of evaluation episodes (default: `10`)
- `--max-steps`: Maximum steps per episode (default: `200`)
- `--seed`: Random seed (default: `42`)
- `--deterministic`: Use deterministic actions (default: True)
- `--device`: Device to run on (default: auto-detected)

#### Visualization
- `--render`: Render the environment during evaluation
- `--save-video`: Save videos of evaluation episodes
- `--video-dir`: Directory to save videos (default: `"eval_videos"`)

#### Custom Environment (PickCube only)
- `--set_cube_color`: Set custom cube color in RGBA format
  - Example: `"[1.0,0.0,0.0,1.0]"` for red cube

### Evaluation Examples

**Evaluate RGB-only model on PickCube:**
```bash
uv run src/evaluate_model.py --task pickCube --rgb
```

**Evaluate depth-only model:**
```bash
uv run src/evaluate_model.py --task pickCube --depth
```

**Evaluate RGB + depth model:**
```bash
uv run src/evaluate_model.py --task pickCube --rgb --depth
```

**Evaluate with video recording:**
```bash
uv run src/evaluate_model.py \
  --task pickCube \
  --rgb --depth \
  --save-video \
  --video-dir my_evaluation_videos
```

**Evaluate with custom cube color (green cube):**
```bash
uv run src/evaluate_model.py \
  --task pickCube \
  --rgb \
  --set_cube_color "[0.0,1.0,0.0,1.0]"
```

**Evaluate pretrained ResNet50 model:**
```bash
uv run src/evaluate_model.py \
  --task pickCube \
  --rgb \
  --feature-extractor resnet50
```

**Extended evaluation (50 episodes):**
```bash
uv run src/evaluate_model.py \
  --task pushT \
  --segmentation --depth \
  --num-episodes 50
```

### Evaluation Output

The script prints:
- Per-episode rewards, lengths, and success status
- Summary statistics:
  - Success rate
  - Mean/std reward
  - Mean/std episode length
  - Min/max reward

Example output:
```
=============================================================
Evaluating model: models/pickCube/pickCube_rgb_depth.pt
Task: pickCube
Modalities: RGB=True, Depth=True, Seg=False, State=True
Feature Extractor: nature_cnn
Number of episodes: 10
Device: cuda
=============================================================

Episode 1/10: Reward = 2.450, Length = 45, Success = Yes
Episode 2/10: Reward = 2.380, Length = 48, Success = Yes
...

Evaluation Summary:
  Success Rate: 8/10 (80.0%)
  Mean Reward: 2.315 ± 0.124
  Mean Length: 46.2 ± 3.5
  Min Reward: 2.150
  Max Reward: 2.500
=============================================================
```

## Project Structure

```
├── src/
│   ├── train_ppo.py           # PPO training script
│   ├── evaluate_model.py      # Model evaluation script
│   ├── feature_extractor.py   # Vision encoder architectures
│   ├── custom_env.py          # Custom environment modifications
│   └── utils.py               # Utility functions and wrappers
├── models/                     # Pretrained model checkpoints
├── runs/                       # Training outputs and checkpoints
├── experiment_run_scripts/     # Batch training scripts
├── download_models.py          # Model download script
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## Available Tasks

1. **PickCube-v1**: Pick up a cube and place it at a target location
2. **PushCube-v1**: Push a cube to a target location
3. **PushT-v1**: Push a T-shaped object to match a target pose

## Visual Observation Modes

- **RGB**: 128x128x3 color images from camera
- **Depth**: 128x128x1 depth maps
- **Segmentation**: 128x128x1 semantic segmentation masks
- **State**: Robot joint positions, velocities, and task-specific state (we assume these are used by default)

You can combine any subset of these modalities for training and evaluation.

## Troubleshooting

**CUDA Errors**
When training, please make sure that you restrict the number of visible GPU devices to just 1, as ManiSkill3 does not support multi-GPU training yet, which can result in unusual errors when multiple GPUs are available for the script to use.  

## Citation

This project uses ManiSkill and is based on CleanRL's PPO implementation.

```bibtex
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
} 
```
