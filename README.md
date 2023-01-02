# Drone Path Planning

This is the GitHub repository for the drone path planning project. It aims to apply deep reinforcement learning to the problem of path planning for drones.

## Setup

In your working directory, clone this repository.

```bash
git clone https://github.com/jayasting98/drone-path-planning.git
```

This project is intended to run on Python 3.8 (the version of Python in the NUS School of Computing Compute Cluster as of 2 January 2023); ensure you have Python 3.8. If you are using Ubuntu, you can use [this Stack Exchange answer](https://askubuntu.com/questions/682869/how-do-i-install-a-different-python-version-using-apt-get#answer-682875) as a guide to install Python 3.8 especially if you already have other Python versions installed. After that, run `setup.sh`. This will create a Python virtual environment with `venv` and install the required Python packages using pip with reference to `requirements.txt`. Alternatively, you can set up the Python virtual environment yourself (or not, but it is recommended that you do) and you can install the required packages yourself, which can be found in `requirements.txt`.

```bash
bash setup.sh
```
