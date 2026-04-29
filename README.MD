# Franka Teleoperation Data Collection with Isomorphic Joint Mapping and End-Effector Pose

## Introduction 
This project is a teleoperation data collection tool for Franka Research 3 robot (should compatible with Franka Panda), embeded in [LeRobot](https://github.com/huggingface/lerobot.git).

### Support

Now we support the end-effector pose control using [SpaceMouse](https://3dconnexion.com/) devices.

<p align="center">
  <img src="assets/spacemouse.png" alt="Record" width="600">
  <br>
</p>

 The SpaceMouse is a 6-axis input device that can be used to control the end pose of the robot. The left button is used to open the gripper, and the right button is used to close the gripper.

### Support 2

Now we support the end-effector pose control using Oculus Quest 3/3s devices.

<p align="center">
  <img src="assets/quest3s.jpg" alt="Oculus Quest" width="600">
  <br>
</p>

The Oculus Quest controller provides 6-DoF tracking for intuitive end-effector control. The right controller is used for robot manipulation with the following controls:

- **RG (Right Grip)**: Must be pressed to enable robot movement
- **RTr (Right Trigger)**: Controls gripper (press to close, release to open)
- **A Button**: Request robot reset
- **Right controller pose**: Controls end-effector delta pose

## 0. Franka Control via Polymetis

### 0.1 Install [Polymetis](https://polymetis-docs.github.io/) on NUC

1. Requirement of libfranka:
```bash
wget https://github.com/frankarobotics/libfranka/releases/download/<libfranka_version>/libfranka_<libfranka_version>_<ubuntu_codename>_amd64.deb
sudo dpkg -i libfranka_<libfranka_version>_<ubuntu_codename>_amd64.deb
```

Choose a specific version of libfranka according to your machine.

The following example installs **libfranka 0.19.0** on **Ubuntu 22.04 (Jammy)**:
```bash
wget https://github.com/frankarobotics/libfranka/releases/download/0.19.0/libfranka_0.19.0_jammy_amd64.deb
sudo dpkg -i libfranka_0.19.0_jammy_amd64.deb
```

2. Clone repo:
```bash
git clone https://github.com/ryanzhangtianran/polymetis.git
cd polymetis
```

3. Create an environment:
```bash
conda env create -f ./polymetis/environment.yml
conda activate polymetis
```

4. Install
```bash
cd polymetis
pip install -e .

mkdir build && cd build
cmake -DBUILD_FRANKA=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_DOCS=ON ..
make -j32
```

### 0.2. Usage
If you want to use the interface to control the Franka robot, follow the steps below:

```bash
cd polymetis/polymetis/python/scripts

python launch_robot.py robot_client=franka_hardware
python launch_gripper.py gripper=franka_hand
python launch_server.py
```

And run `sudo pkill -9 run_server` after launching the robot.

## 1. Environment Setup
### 1.1 Create and Activate Conda Environment
```bash
conda create -n franka python=3.10
conda activate franka
```

### 1.2 Install lerobot
```bash
# Install specific version 0.3.4
# git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
pip install -e .
```

If you want to use Orbbec series cameras as 3-view and wrist cameras, please use the following specific version of Lerobot:
```bash
git clone https://github.com/ryanzhangtianran/lerobot_0.3.4.git
cd lerobot
pip install -e .
```

### 1.3 Clone and Install Franka Teleoperation Control Source
```bash
git clone https://github.com/ryanzhangtianran/franka_teleop.git
cd franka_teleop
pip install -e .
```

### 1.4 View Available Commands
Run the following command to display all available scrtips:
```bash
franka-help
# ==================================================
#  Franka Teleoperation Utilities - Command Reference
# ==================================================

# Core Commands:
#   franka-record           Record teleoperation dataset
#   franka-replay           Replay a recorded dataset
#   franka-visualize        Visualize recorded dataset
#   franka-reset           Reset the robot to initial state
#   franka-train          Train a policy on the recorded dataset

# Tool Commands:
#   tools-check-dataset   Check local dataset information
#   tools-check-rs        Retrieve connected RealSense camera serial numbers

# Shell Tools:
#   map_gripper.sh        Map Gripper Serial Port

# Test Commands:
#   test-gripper-ctrl     Run gripper control command (operate the gripper)

# --------------------------------------------------
#  Tip: Use 'franka-help' anytime to see this summary.
# ==================================================
```

## 2. Obtain and Configure Necessary Parameters

### 2.1 Get RealSense Camera Serial Number
Please make sure only one camera is connected at a time.
```bash
tools-check-rs
```
Or you can check the serial number in the realsense-viewer.
```
realsense-viewer
```

## 3. Dataset Recording

### 3.1 Upload Data to Hugging Face (Optional)
1. Set the following in `record_cfg.yaml`:
```yaml
push_to_hub: True
```
2. Obtain your Hugging Face account token and login:
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} 
huggingface-cli whoami  # Verify successful login
```

### 3.2 Start Recording
1. Activate **FCI** on the Franka desk interface.  
2. Verify that the parameters in `record_cfg.yaml` are correct.  
3. To ensure standardized data collection, please read `9. Dataset Naming and Recording Details` before running the following command:
```bash
franka-record
```

<p align="center">
  <img src="assets/record.png" alt="Record" width="600">
  <br>
  <b>Figure 1: Record</b>
</p>

## 4. Dataset Replay
```bash
franka-replay # Make sure record_cfg.yaml is properly configured
```

## 5. Dataset Visualization
```bash
franka-visualize # Make sure record_cfg.yaml is properly configured
```
<p align="center">
  <img src="assets/visualize.png" alt="Visualization" width="600">
  <br>
  <b>Figure 2: Visualization</b>
</p>

## 6. Dataset Appending and Resume
If you have already recorded a dataset under the specified `repo_id`, set `resume: True` in `record_cfg.yaml` and enter the dataset name in `resume_dataset` to continue recording new data on the existing dataset. Then, run the following command:

```bash
franka-record
```

## 7. Merge Datasets
If you recorded data in multiple stages, ensure each dataset has a unique `repo_id`. Use the following command to merge them into a single dataset:
```bash
lerobot-edit-dataset 
    --repo_id <merged_repo_id> 
    --operation.type merge 
    --operation.repo_ids "['<repo_id_1>', '<repo_id2>']"
```
- For more dataset processing commands, see [LeRobot](https://huggingface.co/docs/lerobot/using_dataset_tools).

## 8. Recording Control Keys

- **Right Arrow Key**  
   1. Press the right arrow key: end the current episode and save the data. The program will pause.  
   2. Enter the **reset** phase: press enter key to continue, then return the robot to its initial position.  
   3. Press the right arrow key again to start recording the next episode.

- **Left Arrow Key**  
   - Press to restart the current episode recording (overwrite the current data).

- **Esc Key**  
   - Press to exit the recording session and save all recorded episodes.

- **Ctrl+C or Exception**
   - Pressing or on exception: enters exception handling and asks whether to delete the incomplete dataset.
   - If you delete the dataset by accident, you can find it in the trash folder.
  
## 9. Dataset Naming and Recording Details
### 9.1 Dataset Naming
<p align="center">
  <img src="assets/dataset.png" alt="dataset" width="600">
  <br>
  <b>Figure 3: Dataset</b>
</p>

<p align="center">
  <img src="assets/dataset_info.png" alt="dataset_info" width="600">
  <br>
  <b>Figure 4: Dataset Info</b>
</p>

1. The dataset is by default stored in the `~/.cache/huggingface/lerobot` directory and contains three types of items:

   - `dataset_info.txt`: Automatically records local dataset information, including the following fields: `record_id`, `name`, `task`, `date`, `version`, `user_info`, and `type`. The `user_info` field can be annotated via the `user_notes` entry in `record_cfg.yaml`.

   - `dataset_info_backup`: When `tools-check-dataset` is used to manually update `dataset_info.txt`, this folder stores backups of the previous records.

   - Dataset folders: Stores the actual dataset contents.

2. Dataset names follow the format `[description]_[date]_[version]`, where:
   - `description` comes from `record_cfg.yaml` as `repo_id=<user_name>/<description>`;
   - `date` is generated automatically;
   - `version` is incremented automatically if a dataset with the same `repo_id` already exists.
3. Description naming rule: `task.description -> Verb_SourceObject_prep_TargetObject`.  
   For example:  
   `Pick up the green cube and put it into the trash bin -> pick_greencube_into_trashbin`.

### 9.2 Recording Instructions
1. Complete step `2. Get and configure required parameters`.
2. Fill in the task instruction in `record_cfg.yaml` under `task.description` and set `repo_id` according to the dataset naming rules.
3. Check and adjust other parameters in `record_cfg.yaml` to ensure correctness.
4. Run `franka-record`. Then follow `8. Recording Control Keys` to complete the recording.
5. After recording, a `dataset_info.txt` file will be automatically created in the same directory to store local dataset information. If you have manually deleted any datasets, update the `dataset_info.txt` using:
```bash
tools-check-dataset
```

## 10. Training and Evaluation Using Collected Datasets
### 10.1 Training
Config the `train_cfg.yaml` file, including the following fields:
1. dataset_repo_id: The repository ID of the dataset to use for training.
2. policy.type: The policy type to use for training. You can choose from `diffusion` and `act`.
3. output_dir: The directory to save the training outputs.
4. job_name: The name of the training job.
5. batch_size: The batch size for training.
6. steps: The total number of training steps.

Then run the following command to start training:
```bash
franka-train
```

### 10.2 Evaluation
To evaluate the trained policy, config the `record_cfg.yaml` file, including the following fields:
1. run_mode: Set to `run_policy` to enable evaluation mode.
2. policy.type: The policy type to use for evaluation. You can choose from `diffusion` and `act`. Make sure it is the same as the policy type used for training.
3. policy.checkpoint_path: The path to the trained policy checkpoint. The path should include `config.json`.

Then run the following command to start evaluation:
```bash
franka-record
```
