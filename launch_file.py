"""
Use of generative AI disclosure: Generative AI was used in compliance with course policy to both create and verify regexes which extract the metadata from files. Note that, before regexes were included in the code, they were thoroughly reviewed to ensure proper functionality in accordance with our specifications.

All code in this file is hand-written and formatted with the *black* Python formatter. Even regexes which were AI-generated were hand-written to ensure proper understanding of their function.
"""

import os
from pathlib import Path

PID = os.getpid()

# must be an absolute path, so we have to resolve it
dump_dir = Path(f"./temp_{PID}").resolve()
dump_dir.mkdir(parents=True, exist_ok=True)
assert dump_dir.is_dir() and os.access(dump_dir, os.W_OK)

# set the dump directory for JAX to dump MLIR from XLA as text
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"

import argparse
import importlib.util
import sys
import torch
import jax
from torch.profiler import profile, ProfilerActivity
from static_analyzer import gather_stats, get_asdf, serialize_stats, get_csv_header

def load_function_from_file(file: Path, function_name: str, debug: bool=False):
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
    parser.add_argument("-t", "--torch_file", type=Path, help="Path to PyTorch program which needs to be profiled.")
    parser.add_argument("-f", "--torch_fn", type=str, help="Name of function to load from provided PyTorch file.")
    parser.add_argument("-j", "--jax_file", type=Path, help="Path to JAX program which needs to be profiled.")
    parser.add_argument("-g", "--jax_fn", type=str, help="Name of function to load from provided JAX file.")
    parser.add_argument("-x", "--torch_ig_fn", type=str, help="Input generation function name for passed PyTorch function (assumed as get_inputs).")
    parser.add_argument("-y", "--jax_ig_fn", type=str, help="Input generation function name for passed JAX function (assumed as get_inputs).")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug flag for debugging file loading.")
    parser.add_argument("-s", "--save", action="store_true", help="Save dumped XLA after parsing.")
    parser.add_argument("-o", "--output", type=Path, help="Set the output file (should be a csv).")

    args = parser.parse_args()

    debug = args.debug

    if args.torch_file:
        function_name = args.torch_fn if args.torch_fn else "main"
        fn = load_function_from_file(args.torch_file, function_name)

        input_function_name = args.torch_ig_fn if args.torch_ig_fn else "get_inputs"

        try:
            get_inputs = load_function_from_file(args.torch_file, input_function_name)
        except _:
            raise RuntimeException(
                                   "Must define a function 'get_inputs' which returns the inputs of the profiled function as a tuple."
                                   )
        # safety in case the target file does not do this already
        torch.set_default_device('cuda')
        if debug:
            print("CUDA available (torch):", torch.cuda.is_available())
            print("Current CUDA Device (torch):", torch.cuda.current_device())

        # compile the function
        torch_fn = torch.compile(fn, options={"triton.cudagraphs": True}, fullgraph=True)

        # run the function once before profiling to be safe
        torch_fn(*(get_inputs()))
        torch.cuda.synchronize()

        with profile(
                 activities=[ProfilerActivity.CUDA],
                 record_shapes=False,
                 profile_memory=False,
                 acc_events=True,
                 ) as prof:
            # unpack the inputs into the function input
            torch_fn(*(get_inputs()))
            torch.cuda.synchronize()

        # for whatever reason, prof is not garbage collected after the end
        # of the above with clause and INTENDS for you to use it this way.
        # events are not recorded properly unless you are at a lower
        # indentation level... (mind-boggling)
        num_kernel_launches = sum(e.name == "cudaLaunchKernel" for e in prof.events())
        if debug:
            print(f"PyTorch kernel launches: {num_kernel_launches}")

    if args.jax_file:
        function_name = args.jax_fn if args.jax_fn else "main"
        fn = load_function_from_file(args.jax_file, function_name)

        input_function_name = args.jax_ig_fn if args.jax_ig_fn else "get_inputs"

        try:
            get_inputs = load_function_from_file(args.jax_file, input_function_name)
        except _:
            raise RuntimeException(
                                   "Must define a function 'get_inputs' which returns the inputs of the profiled function as a tuple."
                                   )
        if debug:
            print("CUDA available (JAX):", any(d.platform == "gpu" for d in jax.devices()))
            print("Available devices (JAX):", jax.devices())

        # compile the function
        jax_fn = jax.jit(fn)

        # run the function to dump XLA compilation at the lowest level
        jax_fn(*(get_inputs())).block_until_ready()
        dumps = [dump_dir / Path(d) for d in os.listdir(dump_dir) if d.startswith(f"jit_{args.jax_fn}")]
       
        stats = gather_stats(dumps, debug=True)

        if not args.save:
            os.rmdir(dump_dir)
        else:
            os.rename(os.path.basename(dump_dir), f"saved_asdf_{args.jax_fn}_{PID}")

        if num_kernel_launches:
            for stat_list in stats:
                for stat in stat_list:
                    asdf = get_asdf(stat, num_kernel_launches)
                    stat.torch_asdf = asdf
                    if debug:
                        print("ASDF (vs. PyTorch):", asdf)

        if args.output:
            for stat_list in stats:
                for stat in stat_list:
                    if args.output.is_file():
                        with open(args.output, "a") as f:
                            f.write(serialize_stats(stat, ";") + "\n")
                    else:
                        with open(args.output, "a") as f:
                            f.write(get_csv_header() + "\n")
                            f.write(serialize_stats(stat, ";") + "\n")
