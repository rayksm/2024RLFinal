2024RLFinal (Reinforcement learning for logic synthesis.)
--------
This is the source codes for 2024 fall NTU RL course. Our group is "一塊陶土". 

The authors include Ting-Jui Yao, Ren-Hau Shiue, Tzu-Chun Tu and Mu-Yao Chung.

--------
# Prerequsites

# Python environment
Please install the environment through 'requirement.txt'.

The project has other dependencies such as `numpy, six, etc.`
Please installing the dependencies correspondingly.

# abc\_py

The project requires the Python API, [abc\_py](https://github.com/krzhu/abc\_py), for [Berkeley-abc](https://github.com/berkeley-abc/abc).

Please refer to the Github page of abc\_py for installing instruction.

--------

# Benchmarks

Benmarks can be found in [url](https://ddd.fit.cvut.cz/prj/Benchmarks/index.php?page=download).

--------

# Usage

The current version can execute on combinational `.aig` and `.blif` benchmarks.
To run the REINFORCE algorithm, please first edit the `python/rl/testReinforce.py` for the benchmark circuit.
And execute `python3 testReinforce.py`


--------

# Contact

Keren Zhu, UT Austin (keren.zhu AT utexas.edu)
