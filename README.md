# Introduction

Hello and welcome to `asdf.py`, a static analysis and performance comparison
tool to identify how arithmetic intensity affect the runtime of JAX relative to
PyTorch.

# Installation

Clone this repository, create a virtual environment, and install from
requirements.txt. You can then run `asdf.py --help` to see all available
options for analysis.

# Usage

You can use this tool in two ways. First, if you have already run a JAX kernel
and simply want to compare it to its strawman, you can run `python
static_analyzer.py <dir1> <dir2> ... <dirN>` where each directory contains a
compilation in your dumped folder.

Alternatively, you can run `python asdf.py ...` as shown below.

```
usage: asdf.py [-h] [-t TORCH_FILE]
                      [-f TORCH_FN] [-j JAX_FILE]
                      [-g JAX_FN] [-x TORCH_IG_FN]
                      [-y JAX_IG_FN] [-d] [-s]
                      [-o OUTPUT]

options:
  -h, --help            show this help message and
                        exit
  -t TORCH_FILE, --torch_file TORCH_FILE
                        Path to PyTorch program
                        which needs to be profiled.
  -f TORCH_FN, --torch_fn TORCH_FN
                        Name of function to load
                        from provided PyTorch file.
  -j JAX_FILE, --jax_file JAX_FILE
                        Path to JAX program which
                        needs to be profiled.
  -g JAX_FN, --jax_fn JAX_FN
                        Name of function to load
                        from provided JAX file.
  -x TORCH_IG_FN, --torch_ig_fn TORCH_IG_FN
                        Input generation function
                        name for passed PyTorch
                        function (assumed as
                        get_inputs).
  -y JAX_IG_FN, --jax_ig_fn JAX_IG_FN
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

# Credits

This tool was written by William Bradford to contribute to the group project
portion of CS 6501: GPU Architecture in collaboration with Nebil Ozer and
Morteza Baradaran.
