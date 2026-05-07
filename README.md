# Introduction

Hello and welcome to `figs.py`, a static analysis and performance comparison
tool to identify how arithmetic intensity affect the runtime of JAX relative to
PyTorch.

# Installation

Clone this repository, create a virtual environment, and install from
requirements.txt. You can then run `figs.py --help` to see all available
options for analysis.

# Usage

You can use this tool in two ways. First, if you have already run a JAX kernel
and simply want to compare it to its strawman, you can run `python
static_analyzer.py <dir1> <dir2> ... <dirN>` where each directory contains a
compilation in your dumped folder.

Alternatively, you can run `python figs.py ...` as shown below.

```
usage: figs.py [-h]   [-f JAX_FN] [-j JAX_FILE]
                      [-x JAX_IG_FN]
                      [-v] [-s]
                      [-o OUTPUT]

options:
  -h, --help            show this help message and
                        exit
  -j JAX_FILE, --jax_file JAX_FILE
                        Path to JAX program which
                        needs to be profiled.
  -f JAX_FN, --jax_fn JAX_FN
                        Name of function to load
                        from provided JAX file.
  -x JAX_IG_FN, --jax_ig_fn JAX_IG_FN
                        Input generation function
                        name for passed JAX function
                        (assumed as get_inputs).
  -d, --debug           Debug flag for debugging
                        file loading.
  -s, --save            Save dumped XLA after
                        parsing.
  -o OUTPUT, --output OUTPUT
                        Set the output file (should
                        be a csv).
```

Alternatively, you can use the static analyzer by itself to analyze pre-dumped
HLO.

# Credits

This tool was written by William Bradford to contribute to the group project
portion of CS 6501: GPU Architecture in collaboration with Nebil Ozer and
Morteza Baradaran.
