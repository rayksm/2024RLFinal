#2024RLFinal (Reinforcement learning for logic synthesis.)
--------
This is the source codes for 2024 fall NTU RL course. Our group is "一塊陶土". 

The authors include Ting-Jui Yao, Ren-Hau Shiue, Tzu-Chun Tu and Mu-Yao Chung.

--------
## Prerequsites

### Python environment
Please install the environment through `requirement.txt`.

### abc\_py

The project requires the Python API, [abc\_py](https://github.com/krzhu/abc\_py), for [Berkeley-abc](https://github.com/berkeley-abc/abc).

Please refer to the Github page of abc\_py for installing instruction.

--------

### Benchmarks

Benmarks can be found in [HDL-Benchmark](https://github.com/ispras/hdl-benchmarks).

We use MCNC benchmark for our esperiments.

--------

## Usage

The current version can execute on combinational `.blif` benchmarks.

To run the algorithm, please first edit the `testReinforce.py` for the benchmark circuit.
And execute `python3 testReinforce.py`


--------

## Contact

 Ren-Hau Shiue, Email: r12942046@ntu.edu.tw
