## Installation

1. Clone robolearn_envs
```
git clone https://github.com/domingoesteban/robolearn_envs_private robolearn_envs
```

2. Install robolearn_envs
```
pip install robolearn_envs
```

3. Clone this repository
```
git clone https://github.com/domingoesteban/hiu_sac
```

4. Install the requirements of this repository
```
cd hiu_sac
pip install -r requirements.txt
```

# Use

- Run HIU-SAC in the reacher environment

```
python train.py -e reacher
```

- Evaluate the results. (Specify the directory that is printed during the learning process )

```
python eval.py path_to_log_directory
```
