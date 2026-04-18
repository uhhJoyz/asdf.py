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

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

# list found here: https://docs.rs/xla/latest/xla/enum.PrimitiveType.html
#     see also: https://openxla.org/stablehlo/spec
DTYPE_TO_BYTES: dict[str, int] = {
    # some types (shown below) are unsupported because we
    # did not find a reliable source documenting their size
    # "complex<f64>": 16, ?
    # "complex<f32>": 8, ?
    # we also don't have support for tuples as they are relatively uncommon in the performance domains we are interested
    "f64": 8,
    "i64": 8,
    "ui64": 8,
    "f32": 4,
    "i32": 4,
    "ui32": 4,
    "f16": 2,
    "i16": 2,
    "ui16": 2,
    "bf16": 2,
    "i8": 1,
    "ui8": 1,
    "bf8": 1,
    "i1": 1,
}

# a (hopefully) exhaustive list of compute operations in XLA
# this list was sourced from the XLA specification:
#     https://openxla.org/stablehlo/spec
COMPUTE_OPS: frozenset[str] = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "remainder",
        "negate",
        "abs",
        "power",
        "atan2",
        "xor",
        "or",
        "and",
        "not",
        "shift_left",
        "shift_right_logical",
        "shift_right_arithmetic",
        "population_count",
        "count_leading_zeros",
        "compare",
        "select",
        "maximum",
        "minimum",
        "clamp",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "rsqrt",
        "cbrt",
        "logistic",
        "tanh",
        "sine",
        "cosine",
        "is_finite",
        "floor",
        "ceil",
        "round",
        "round_nearest_afz",
        "convert",
        "iota",
    }
)


def get_tensor_size_and_type(type_str: str) -> tuple[int, str]:
    t = type_str.strip()
    # this case matches n-dimensional tensors
    m = re.match(r"^tensor<((?:\d+x)+)(\w+)>$", t)
    if m:
        dims = [int(d) for d in m.group(1).rstrip("x").split("x")]
        n = 1
        # find total number of words used
        for d in dims:
            n *= d
        # right side matches type
        return n, m.group(2)
    # this case matches 0-dimensional tensors (single words)
    m = re.match(r"^tensor<([a-zA-Z]\w*)>$", t)
    if m:
        return 1, m.group(1)
    # base case to ensure consistent behavior
    return 0, ""


def is_scalar_tensor(type_str: str) -> bool:
    return bool(re.match(r"^tensor<[a-zA-Z]\w*>$", type_str.strip()))


@dataclass
class FunctionStats:
    name: str
    input_tensors: list[tuple[int, str]] = field(default_factory=list)
    output_tensors: list[tuple[int, str]] = field(default_factory=list)
    mem_bytes: int = 0
    reg_bytes: int = 0
    compute_ops: int = 0
    external_calls: list[str] = field(default_factory=list)
    unique_ops: int = 0
    arith_int: float = 0
    asdf: float = 0
    torch_asdf: float = 0


def serialize_stats(fs: FunctionStats, sep: str = ";") -> str:
    return (
        f"{fs.name}{sep}"
        f"{fs.input_tensors}{sep}"
        f"{fs.output_tensors}{sep}"
        f"{fs.mem_bytes}{sep}"
        f"{fs.reg_bytes}{sep}"
        f"{sum(b for b, _ in fs.output_tensors)}{sep}"
        f"{fs.compute_ops}{sep}"
        f"{fs.external_calls}{sep}"
        f"{fs.unique_ops}{sep}"
        f"{round(fs.arith_int, 4)}{sep}"
        f"{round(fs.asdf, 4)}{sep}"
        f"{round(fs.torch_asdf, 4)}"
    )


def get_csv_header(sep: str = ";"):
    return (
        f"Name{sep}"
        f"Input Tensors{sep}"
        f"Output Tensors{sep}"
        f"Global Memory Bytes{sep}"
        f"Output Bytes{sep}"
        f"Register Bytes (Scalars){sep}"
        f"Compute Operations{sep}"
        f"External Calls{sep}"
        f"Unique Operations in Kernel{sep}"
        f"Arithmetic Intensity{sep}"
        f"ASDF{sep}"
        f"Torch ASDF"
    )


def find_body(text: str, start: int) -> int:
    # Use this to find the function body's start index within the passed string (text)
    depth_paren = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "{" and depth_paren == 0:
            return i
        i += 1
    return -1


def parse_function(ftext: str) -> FunctionStats:
    name_m = re.search(r"func\.func\s+\w+\s+@(\w+)", ftext)
    fname = name_m.group(1) if name_m else "{could not parse function name}"
    stats = FunctionStats(name=fname)

    pos = find_body(ftext, 0)
    # early exit if no body found
    if pos == -1:
        return stats

    sig_text = ftext[:pos]
    for m in re.finditer(r"%\w+s*:\s*(tensor<[^>]+>)", sig_text):
        t = m.group(1)
        n, dtype = get_tensor_size_and_type(t)
        b = n * DTYPE_TO_BYTES.get(dtype, 0)
        if b > 0:
            stats.input_tensors.append((b, dtype))
            if is_scalar_tensor(t):
                stats.reg_bytes += b
            else:
                stats.mem_bytes += b

    ret_idx = sig_text.rfind("->")
    if ret_idx != -1:
        ret_section = sig_text[ret_idx + 2 :]
        for m in re.finditer(r"tensor<[^>]+>", ret_section):
            t = m.group(0)
            n, dtype = get_tensor_size_and_type(t)
            b = n * DTYPE_TO_BYTES.get(dtype, 0)
            if b > 0:
                stats.output_tensors.append((b, dtype))

    body_text = ftext[pos:]
    for line in body_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in ("{", "}"):
            continue

        if re.search(r"\bcall\s+@", stripped):
            callee_m = re.search(r"call\s+@(\w+)", stripped)
            if callee_m:
                callee = callee_m.group(1)
                if callee not in stats.external_calls:
                    stats.external_calls.append(callee)
            continue

        # ignores all non-stablehlo operations
        op_m = re.search(r"stablehlo\.(\w+)", stripped)
        if not op_m:
            continue
        op = op_m.group(1)
        unique_ops: dict[str, int] = {op: 0 for op in COMPUTE_OPS}
        if op in COMPUTE_OPS:
            all_types = re.findall(r"tensor<[^>]+>", stripped)
            unique_ops[op] += 1
            if all_types:
                n, _ = get_tensor_size_and_type(all_types[-1])
                stats.compute_ops += n
        for v in unique_ops.values():
            stats.unique_ops += v

    return stats


def parse_mlir_file(path: Path) -> list[FunctionStats]:
    content = path.read_text()
    results: list[FunctionStats] = []

    for m in re.finditer(r"func\.func\b", content):
        pos = find_body(content, m.end())
        if pos == -1:
            continue

        depth = 0
        while pos < len(content):
            ch = content[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1

        ftext = content[m.start() : pos + 1]
        results.append(parse_function(ftext))

    return results


def get_stats(dir: Path) -> list[FunctionStats]:
    stats_list = parse_mlir_file(dir / "module.mlir")
    callee_names = {c for s in stats_list for c in s.external_calls}
    return [s for s in stats_list if s.name not in callee_names]


def fmt_bytes(n: int) -> str:
    if n >= 1 << 40:
        return f"{(n >> 40):.6f} TiB, ({n:,} B)"
    if n >= 1 << 30:
        return f"{(n >> 30):.6f} GiB, ({n:,} B)"
    if n >= 1 << 20:
        return f"{(n >> 20):.6f} GiB, ({n:,} B)"
    if n >= 1 << 10:
        return f"{(n >> 10):.6f} GiB, ({n:,} B)"
    return f"{n:,} B"


def fmt_operations(n: int) -> str:
    if n >= 10**9:
        return f"{n / 10 ** 9:.3f} Gops ({n:,} ops)"
    if n >= 10**6:
        return f"{n / 10 ** 6:.3f} Mops ({n:,} ops)"
    if n >= 1000:
        return f"{n / 1000:.3f} Kops ({n:,} ops)"
    return f"{n} ops"


def gather_stats(dirs: list[Path], debug: bool = True):
    stats: list[FunctionStats] = []
    for dir in dirs:
        all_stats: list[FunctionStats] = get_stats(dir)
        stats.append(all_stats)

        if len(all_stats) == 0:
            continue

        callee_names: set[str] = set()
        for fs in all_stats:
            callee_names.update(fs.external_calls)

        for fs in all_stats:
            is_callee = fs.name in callee_names
            if not is_callee:
                out_b = sum(b for b, _ in fs.output_tensors)
                if out_b and debug:
                    print(f"Bytes written to HBM: {fmt_bytes(out_b)}")

                if fs.mem_bytes > 0 and debug:
                    fs.arith_int = fs.compute_ops / fs.mem_bytes
                    # Arithmetic intensity Scaling Due to Fusion, compares to strawman
                    fs.asdf = fs.unique_ops / fs.arith_int
                    if debug:
                        print(f"Arithmetic intensity: {fs.arith_int:.6f}")
                        print(f"ASDF (strawman): {fs.asdf}")

                # NOTE: the below does not properly calculate register pressure
                # it was WIP, but was abandoned because it is not very useful
                # if fs.reg_bytes > 0 and debug:
                #     reg_pressure = fs.reg_bytes / fs.compute_ops
                #     if debug:
                #         print("Register pressure:", reg_pressure)

        if debug:
            print_stats(dir)
    return stats


def get_asdf(stat: FunctionStats, function_calls: int) -> float:
    ext = len(stat.external_calls) + 1
    if stat.arith_int > 0:
        return (stat.arith_int / ext) / (stat.arith_int / function_calls)
    else:
        return 0


def print_stats(dir: Path):
    # I use vim btw
    print("--------------------------------")
    l_fstats: list[FunctionStats] = get_stats(dir)
    for fstat in l_fstats:
        print("Function:", fstat.name)
        print("Global bytes:", fmt_bytes(fstat.mem_bytes))
        print("Shared bytes:", fmt_bytes(fstat.reg_bytes))
        print("Operations:", fmt_operations(fstat.compute_ops))
        print("External Library Calls:", fstat.external_calls)
        print("Unique operations:", fstat.unique_ops)
        print("--------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Static analysis tool for analyzing StableHLO MLIR output at varying levels of lowering from XLA dumps of JAX-compiled functions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example use:\n"
            "python static_analyzer.py <directory 1> <directory 2> ... <directory n>\n"
            "This will compare (pairwise) all of the passed directories' MLIR representations."
        ),
    )
    parser.add_argument(
        "directories", nargs="+", help="Directories containing XLA dumps."
    )

    args = parser.parse_args()
    dirs = [Path(d) for d in args.directories]
    for d in dirs:
        if not d.is_dir() and not (d / "module.mlir").exists():
            raise Exception(
                f"Directory {str(d)} is not an MLIR module (does not contain module.mlir) and therefore cannot be parsed."
            )
        # for s in get_stats(d):
        print_stats(d)

    gather_stats(dirs)
