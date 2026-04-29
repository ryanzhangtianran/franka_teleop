#!/usr/bin/env bash
set -euo pipefail

POLYMETIS_SCRIPTS_DIR="/data2/franka/Repos/polymetis/polymetis/python/scripts"
CONDA_SH="/opt/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="polymetis"
ROBOT_STARTUP_WAIT=20
GRIPPER_STARTUP_WAIT=10

cleanup() {
    trap - EXIT INT TERM
    echo "Stopping Polymetis run_server processes..."
    sudo pkill -9 run_server || true
}

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Required file not found: $path" >&2
        exit 1
    fi
}

start_terminal() {
    local title="$1"
    local command="$2"

    gnome-terminal --title "$title" -- bash -lc \
        "source '$CONDA_SH'; conda activate '$CONDA_ENV'; cd '$POLYMETIS_SCRIPTS_DIR'; echo '[START] $command'; exec $command"
}

require_command() {
    local title="$1"
    if ! command -v "$title" >/dev/null 2>&1; then
        echo "Required command not found: $title" >&2
        exit 1
    fi
}

require_command gnome-terminal
require_file "$CONDA_SH"
require_file "$POLYMETIS_SCRIPTS_DIR/launch_robot.py"
require_file "$POLYMETIS_SCRIPTS_DIR/launch_gripper.py"
require_file "$POLYMETIS_SCRIPTS_DIR/launch_server.py"

source "$CONDA_SH"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "Conda environment not found: $CONDA_ENV" >&2
    exit 1
fi

trap cleanup EXIT INT TERM

echo "Starting Polymetis services in separate terminal windows..."
start_terminal "Polymetis Robot" "python launch_robot.py robot_client=franka_hardware"
echo "Waiting ${ROBOT_STARTUP_WAIT} seconds for robot server startup..."
sleep "$ROBOT_STARTUP_WAIT"

start_terminal "Polymetis Gripper" "python launch_gripper.py gripper=franka_hand"
echo "Waiting ${GRIPPER_STARTUP_WAIT} seconds for gripper startup..."
sleep "$GRIPPER_STARTUP_WAIT"

start_terminal "Polymetis Interface Server" "python launch_server.py"

echo "Opened separate terminal windows for Robot / Gripper / Interface Server."
echo "Keep this terminal open while the services are running."
read -r -p "Press Enter to stop Polymetis run_server cleanup..." _
