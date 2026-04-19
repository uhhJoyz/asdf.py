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
    # some types are unsupported because we did not find a reliable source
    # documenting their size we also don't have support for tuples as they
    # are relatively uncommon in the performance domains we are interested in
    "c128": 16,
    "s64": 8,
    "f64": 8,
    "i64": 8,
    "ui64": 8,
    "c64": 8,
    "s32": 4,
    "f32": 4,
    "i32": 4,
    "ui32": 4,
    "f16": 2,
    "i16": 2,
    "s16": 2,
    "u16": 2,
    "bf16": 2,
    "s8": 1,
    "i8": 1,
    "ui8": 1,
    "bf8": 1,
    "i1": 1,
    "pred": 1,
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
        "exponential",
        "dot",
        "dot_general",
        "reduce",
    }
)


HLO_STAGE_PATTERNS: list[tuple[str, str]] = [
    ("before_optimizations", ".before_optimizations.txt"),
    ("after_spmd_partitioner", ".after_spmd_partitioner.txt"),
    ("after_optimizations", ".sm_*_gpu_after_optimizations.txt"),
]

ASSIGNMENT_RE = re.compile(
    r"^\s*(?:ROOT\s+)?%(?P<name>[\w.]+)\s*=\s*(?P<type>\S+)\s+(?P<op>\w+)\((?P<args>[^)]*)\)"
)
FUSION_META_RE = re.compile(r"kind\s*=\s*(?P<kind>\w+).*?calls\s*=\s*%(?P<callee>\w+)")


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


def get_hlo_size_and_type(s: str) -> tuple[int, str]:
    t = s.strip()
    m = re.match(r"^(\w+)\[([^\]]+)\]", t)
    if m:
        dtype = m.group(1)
        try:
            dims = [int(d) for d in m.group(2).split(",")]
        except ValueError:
            return 0, ""
        n = 1
        for d in dims:
            n *= d
        return n, dtype
    # in the scalar case
    m = re.match(r"^(\w+)\[\]", t)
    if m:
        return 1, m.group(1)
    return 0, ""


def type_bytes(type_str: str) -> int:
    n, dtype = get_hlo_size_and_type(type_str)
    return n * DTYPE_TO_BYTES.get(dtype, 0)


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


@dataclass
class KernelStats:
    name: str
    kind: str
    input_bytes: int = 0
    output_bytes: int = 0
    compute_ops: int = 0
    unique_ops: set[str] = field(default_factory=set)

    @property
    def mem_bytes(self) -> int:
        return self.input_bytes + self.output_bytes

    @property
    def arith_int(self) -> float:
        return self.compute_ops / self.mem_bytes if self.mem_bytes > 0 else 0.0


@dataclass
class StageStats:
    stage: str
    path: Path
    kernels: list[KernelStats] = field(default_factory=list)

    @property
    def total_compute_ops(self) -> int:
        return sum(k.compute_ops for k in self.kernels)

    @property
    def total_mem_bytes(self) -> int:
        return sum(k.mem_bytes for k in self.kernels)

    @property
    def overall_arith_int(self) -> float:
        return (
            self.total_compute_ops / self.total_mem_bytes
            if self.total_mem_bytes > 0
            else 0.0
        )

    def serialize(self, name: str, output_file: Path, delim: str = ";"):
        return (
            f"{name}{delim}{os.path.basename(self.path)}{delim}"
            + f"{self.overall_arith_int}{delim}"
        )
        

def serialize_stats(fs: FunctionStats, file: str | Path = "", sep: str = ";") -> str:
    return (
        f"{file}{sep}"
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
        f"File{sep}"
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


def find_block_end(text: str, open_brace: int) -> int:
    depth = 0
    i = open_brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
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
    print("\t\t\tcallee names", callee_names)
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
    if debug:
        print(f"Gathering stats for directories: {dirs}")
    for dir in dirs:
        all_stats: list[FunctionStats] = get_stats(dir)
        stats += all_stats

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


def extract_subcomputations(text: str) -> dict[str, str]:
    subcomps: dict[str, str] = {}
    header_re = re.compile(r"^%(?P<name>\w+)\s+\([^)]*\)\s*->[^{]+\{", re.MULTILINE)

    for m in header_re.finditer(text):
        pos = text.index("{", m.start())
        end = find_block_end(text, pos)
        if end != -1:
            subcomps[m.group("name")] = text[pos + 1 : end]
    return subcomps


def count_body_ops(body: str) -> tuple[int, set[str]]:
    total = 0
    unique_ops: set[str] = set()
    for line in body.splitlines():
        m = ASSIGNMENT_RE.match(line)
        if not m:
            continue
        op = m.group("op")
        if op not in COMPUTE_OPS:
            continue
        n, _ = get_hlo_size_and_type(m.group("type"))
        total += n
        unique_ops.add(op)
    return total, unique_ops


def build_ssa_map(entry_body: str) -> dict[str, str]:
    ssa: dict[str, str] = {}
    for line in entry_body.splitlines():
        m = ASSIGNMENT_RE.match(line)
        if m:
            ssa[m.group("name")] = m.group("type")
    return ssa


def input_bytes_for_args(args_str: str, ssa: dict[str, str]) -> int:
    total = 0
    for arg in re.findall(r"%[\w.]+", args_str):
        name = arg.lstrip("%")
        if name in ssa:
            total += type_bytes(ssa[name])
    return total


def print_entry_stats(annotation: str, ks: KernelStats):
    print(
        f"flat op {ks.name} ({annotation}): "
        f"in={fmt_bytes(ks.input_bytes)}, out={fmt_bytes(ks.output_bytes)}, "
        f"ops={ks.compute_ops}, AI={ks.arith_int:.4f}"
    )


def parse_entry_flat(entry_body: str, debug: bool = False) -> list[KernelStats]:
    ssa = build_ssa_map(entry_body)
    kernels: list[KernelStats] = []

    for line in entry_body.splitlines():
        m = ASSIGNMENT_RE.match(line)
        if not m:
            continue
        op = m.group("op")
        if op not in COMPUTE_OPS:
            continue

        n, _ = get_hlo_size_and_type(m.group("type"))
        out_b = type_bytes(m.group("type"))
        in_b = input_bytes_for_args(m.group("args"), ssa)

        ks = KernelStats(
            name=m.group("name"),
            kind="flat",
            input_bytes=in_b,
            output_bytes=out_b,
            compute_ops=n,
            unique_ops={op},
        )
        kernels.append(ks)

        if debug:
            print_entry_stats(f"{op}", ks)
    return kernels


def parse_entry_fused(
    entry_body: str,
    subcomps: dict[str, str],
    debug: bool = False,
) -> list[KernelStats]:
    ssa = build_ssa_map(entry_body)
    kernels: list[KernelStats] = []
    for line in entry_body.splitlines():
        m = ASSIGNMENT_RE.match(line)
        if not m or m.group("op") != "fusion":
            continue

        fusion_meta = FUSION_META_RE.search(line)
        if fusion_meta:
            kind = fusion_meta.group("kind")
            callee = fusion_meta.group("callee")
        else:
            kind = "unknown"
            callee = ""
        out_b = type_bytes(m.group("type"))
        in_b = input_bytes_for_args(m.group("args"), ssa)

        compute_ops = 0
        unique_ops: set[str] = set()
        if callee in subcomps:
            compute_ops, unique_ops = count_body_ops(subcomps[callee])

        ks = KernelStats(
            name=m.group("name"),
            kind=kind,
            input_bytes=in_b,
            output_bytes=out_b,
            compute_ops=compute_ops,
            unique_ops=unique_ops,
        )

        kernels.append(ks)

        if debug:
            print_entry_stats(f"calls %{callee}", ks)

    return kernels


def parse_hlo_file(path: Path, debug: bool = False) -> list[KernelStats]:
    text = path.read_text()
    subcomps = extract_subcomputations(text)

    entry_m = re.search(r"\bENTRY\b[^{]*\{", text)
    if not entry_m:
        if debug:
            print(f"WARNING: no entry found at {path.name}")
        return []

    pos = text.index("{", entry_m.start())
    end = find_block_end(text, pos)

    if end == -1:
        if debug:
            print(f"WARNING: unmatched brace in entry at {path.name}")
        return []

    entry_body = text[pos + 1 : end]
    has_fusion = bool(re.search(r"\bfusion\b", entry_body))

    if debug:
        mode = "fused" if has_fusion else "flat"
        print(f"parsing in {mode} mode at {path.name}")

    if has_fusion:
        return parse_entry_fused(entry_body, subcomps, debug=debug)
    else:
        return parse_entry_flat(entry_body, debug=debug)


def find_hlo_stages(jit_subdir: Path, function_name: str) -> dict[str, Path]:
    parent = jit_subdir.parent
    stages: dict[str, Path] = {}

    for label, suffix in HLO_STAGE_PATTERNS:
        pattern = f"*.jit_{function_name}{suffix}"
        matches = sorted(parent.glob(pattern))

        if matches:
            stages[label] = matches[0]

    return stages


def compare_stages(
    jit_subdir: Path,
    function_name: str,
    debug: bool = False,
) -> list[StageStats]:
    stage_files = find_hlo_stages(jit_subdir, function_name)
    results: list[StageStats] = []

    for stage_label, path in stage_files.items():
        if debug:
            print(f"Parsing stage '{stage_label}': {path.name}")
        kernels = parse_hlo_file(path, debug=debug)
        results.append(StageStats(stage=stage_label, path=path, kernels=kernels))

    return results


def print_stage_comparison(stage_stats: list[StageStats]):
    sep = "--------------------------------"

    print(f"{sep}\nHLO Optimization Stage Comparisons\n{sep}")

    for ss in stage_stats:
        print(f"\nStage: {ss.stage}")
        print(f"File: {ss.path.name}")
        print(f"Num Kernels: {len(ss.kernels)}")
        print(f"Total Compute Ops: {fmt_operations(ss.total_compute_ops)}")
        print(f"Total Mem Bytes: {fmt_bytes(ss.total_mem_bytes)}")
        print(f"Overall Arithmetic Intensity: {ss.overall_arith_int:.4f} ops/byte")

        for k in ss.kernels:
            print(
                f"\t[{k.kind:8s}] {k.name}: "
                + f"ops={fmt_operations(k.compute_ops)} "
                + f"mem={fmt_bytes(k.mem_bytes)} "
                + f"aint={k.arith_int:.4f}"
            )
            if k.unique_ops:
                print(f"\t\tops in kernel: {sorted(k.unique_ops)}")

    print(f"\n{sep}Fusion Benefit Summary{sep}")

    before = next((s for s in stage_stats if s.stage == "before_optimizations"), None)
    after = next((s for s in stage_stats if s.stage == "after_optimizations"), None)

    if before and after:
        print(
            f"Virtual Kernel Count: {len(before.kernels):4d} -> {len(after.kernels):4d}"
        )
        print(
            f"Overall Arithmetic Intensity (ops/byte): {before.overall_arith_int:.4f} -> {after.overall_arith_int:.4f}"
        )

        if before.overall_arith_int > 0:
            ratio = after.overall_arith_int / before.overall_arith_int
            print(f"ASDF: {ratio:.3f}x")
    else:
        if not before:
            print("before_optimizations stage not found")
        if not after:
            print("after_optimizations stage not found")


def opt_compare(dirs: list[Path], function_name: str = "kernel", debug: bool = False):
    if any(not d.is_dir() for d in dirs):
        raise RuntimeError("One or more of the passed arguments is not a directory.")
    all_stats: list[StageStats] = []
    for d in dirs:
        stage_stats = compare_stages(d, function_name, debug=debug)

        if debug:
            if not stage_stats:
                print(f"No HLO stage files found in directory {d}.")
            else:
                print_stage_comparison(stage_stats)
        all_stats += stage_stats

    return all_stats


def cli_opt_compare(args: argparse.Namespace):
    dirs = [Path(d) for d in args.directories]
    if any(not d.is_dir() for d in dirs):
        raise RuntimeError("One or more of the passed arguments is not a directory.")
    if not args.function_name:
        raise RuntimeError(
            "Cannot perform optimization-level analysis without a passed function name."
        )

    for d in dirs:
        stage_stats = compare_stages(d, args.function_name, debug=args.debug)

        if not stage_stats:
            print(f"No HLO stage files found in directory {d}.")
        else:
            print_stage_comparison(stage_stats)


def functional_analysis(args: argparse.Namespace):
    dirs = [Path(d) for d in args.directories]
    for d in dirs:
        if not d.is_dir() and not (d / "module.mlir").exists():
            raise Exception(
                f"Directory {str(d)} is not an MLIR module (does not contain module.mlir) and therefore cannot be parsed."
            )
        # for s in get_stats(d):
        print_stats(d)

    _ = gather_stats(dirs, True)


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
    _ = parser.add_argument(
        "directories", nargs="+", help="Directories containing XLA dumps."
    )
    _ = parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug print statements."
    )
    _ = parser.add_argument(
        "-o",
        "--opt",
        action="store_true",
        help="Use the opmization stage analysis mode.",
    )
    _ = parser.add_argument(
        "-f", "--function_name", type=str, help="Name of JAX function."
    )

    args = parser.parse_args()
    if args.opt:
        cli_opt_compare(args)
    else:
        functional_analysis(args)
