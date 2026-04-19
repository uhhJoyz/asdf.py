"""
Use of generative AI disclosure: Generative AI was used in compliance with
course policy to both create and verify regexes which extract the metadata from
files. Note that, before regexes were included in the code, they were
thoroughly reviewed to ensure proper functionality in accordance with our
specifications.

All code in this file is hand-written and formatted with the *black* Python
formatter. Even regexes which were AI-generated were hand-written to ensure
proper understanding of their function.
"""

import os
from pathlib import Path

PID = os.getpid()

# must be an absolute path, so we have to resolve it
dump_dir = Path(f"./temp_{PID}").resolve()

# set the dump directory for JAX to dump MLIR from XLA as text
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"

import argparse
import importlib.util
import sys
import torch
import jax
from torch.profiler import profile, ProfilerActivity
from static_analyzer import gather_stats, get_asdf, serialize_stats, get_csv_header, opt_compare


# a helper function to delete files because
# there is not one in python's os module
# and rmdir will fail if the dir is not empty
def recursive_deletion(file: Path):
    if file.is_dir():
        for f in os.listdir(file):
            recursive_deletion(file / Path(f))
        os.rmdir(file)
    else:
        os.remove(file)


def load_function_from_file(file: Path, function_name: str, debug: bool = False):
    module_name = file.stem
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load file {file} (function {function_name}).")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if debug:
        print("Module name:", module_name)
        print("Module:", str(module))
    spec.loader.exec_module(module)

    try:
        f = getattr(module, function_name)
    except AttributeError:
        raise AttributeError(f"Function {function_name} not found in {file}.")

    if not callable(f):
        raise TypeError(f"The imported function {function_name} is not callable.")

    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--jax_file",
        type=Path,
        help="Path to JAX program which needs to be profiled.",
    )
    parser.add_argument(
        "-f",
        "--jax_fn",
        type=str,
        help="Name of function to load from provided JAX file (assumed as 'kernel' unless otherwise specified).",
    )
    parser.add_argument(
        "-x",
        "--jax_ig_fn",
        type=str,
        help="Input generation function name for passed JAX function (assumed as get_inputs).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Toggle verbose console output.",
    )
    parser.add_argument(
        "-s", "--save", action="store_true", help="Save dumped XLA after parsing."
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Set the output file (should be a csv)."
    )

    args = parser.parse_args()
    debug: bool = args.verbose

    if args.jax_file:
        dump_dir.mkdir(parents=True, exist_ok=True)
        assert dump_dir.is_dir() and os.access(dump_dir, os.W_OK)

        # default function name is kernel
        function_name = args.jax_fn if args.jax_fn else "kernel"
        fn = load_function_from_file(args.jax_file, function_name)

        # default input function name is get_inputs
        input_function_name = args.jax_ig_fn if args.jax_ig_fn else "get_inputs"
        get_inputs = load_function_from_file(args.jax_file, input_function_name)

        if debug:
            print(
                "CUDA available (JAX):", any(d.platform == "gpu" for d in jax.devices())
            )
            print("Available devices (JAX):", jax.devices())

        # compile the function
        jax_fn = jax.jit(fn)

        # run the function to dump XLA compilation at the lowest level
        jax_fn(*(get_inputs())).block_until_ready()
        if debug:
            print(f"Now searching {dump_dir}")
            print(f"Searching for pattern startswith(jit_{function_name})")

        dumps = [
            dump_dir / Path(d)
            for d in os.listdir(dump_dir)
            if d.startswith(f"jit_{function_name}")
        ]

        stats = opt_compare(dumps, debug=True)

        if not args.save:
            recursive_deletion(dump_dir)
        else:
            name = os.path.basename(args.jax_file).split(".")[0]
            os.rename(os.path.basename(dump_dir),
                      f"saved_asdf_{name}_{function_name}_{PID}")
            print(f"Saved MLIR to ./saved_asdf_{name}_{function_name}_{PID}/")
