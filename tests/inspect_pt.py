"""
inspect_pt.py - dissect the .pt files written by FG_SaveConditioningLatent.

Usage (run with the same Python that runs ComfyUI, so that torch and the
ComfyUI classes stored inside the files can be unpickled):

  python inspect_pt.py structure  FILE.pt  [--comfyui-directory X:\\imgAI\\ComfyUI]
  python inspect_pt.py frames     FILE.pt  [--time-axis 2] [--comfyui-directory ...]
  python inspect_pt.py compare    FILE_A.pt FILE_B.pt [--comfyui-directory ...]

structure : prints every key / list entry / tensor with shape, dtype, size in
            kilobytes, minimum, maximum, mean, standard deviation.
frames    : for every tensor with 4 or 5 dimensions (a latent) prints the L2 norm
            and standard deviation of every slice along the time axis, and flags
            slices that are all zeros or that differ sharply from the others.
            For 3-dimensional tensors (text / reference tokens) prints the norm
            of every token and flags the positions where the norm jumps, which is
            where one kind of token stops and the next begins.
compare   : for every tensor path present in both files, reports whether the
            values are bit-identical, the largest absolute difference, and which
            time slices differ.
"""

import argparse
import sys


def load_saved_file(file_path, comfyui_directory):
    if comfyui_directory:
        sys.path.insert(0, comfyui_directory)
    import torch

    try:
        return torch.load(file_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        print(f"Could not unpickle {file_path}: {error}")
        print("The file stores a ComfyUI class. Re-run with "
              "--comfyui-directory pointing at your ComfyUI folder.")
        raise SystemExit(1)


def walk_object(value, path="root"):
    """Yield (path, tensor) for every tensor, including those inside
    dictionaries, lists, tuples and objects such as NestedTensor."""
    import torch

    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, inner_value in value.items():
            yield from walk_object(inner_value, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for index, inner_value in enumerate(value):
            yield from walk_object(inner_value, f"{path}[{index}]")
    elif hasattr(value, "tensors"):  # comfy.nested_tensor.NestedTensor
        print(f"# {path} is a {type(value).__name__} holding "
              f"{len(value.tensors)} tensors (for audio plus video: video first, audio second)")
        for index, inner_value in enumerate(value.tensors):
            yield from walk_object(inner_value, f"{path}.tensors[{index}]")
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for attribute_name, inner_value in vars(value).items():
            yield from walk_object(inner_value, f"{path}.{attribute_name}")


def describe_non_tensors(value, path="root"):
    """Print the plain values (strings, numbers, booleans, None) so that flags
    such as noise masks, indices or reference markers are visible."""
    import torch

    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, dict):
        for key, inner_value in value.items():
            describe_non_tensors(inner_value, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for index, inner_value in enumerate(value):
            describe_non_tensors(inner_value, f"{path}[{index}]")
    elif isinstance(value, (str, int, float, bool)) or value is None:
        print(f"{path} = {value!r}")
    elif hasattr(value, "tensors"):
        describe_non_tensors(list(value.tensors), path + ".tensors")
    else:
        print(f"{path} : {type(value).__name__}")


def structure_report(saved_object):
    print(f"Top-level type: {type(saved_object).__name__}")
    for path, tensor in walk_object(saved_object):
        float_view = tensor.float()
        kilobytes = tensor.numel() * tensor.element_size() / 1024
        print(f"{path}\n    shape={tuple(tensor.shape)} dtype={tensor.dtype} "
              f"size={kilobytes:,.0f}KB min={float_view.min():.4f} "
              f"max={float_view.max():.4f} mean={float_view.mean():.4f} "
              f"std={float_view.std():.4f}")
    print("\n--- non-tensor values ---")
    describe_non_tensors(saved_object)


def flag_outliers(values, label):
    """Flag entries more than 3 median-absolute-deviations from the median."""
    import torch

    median = values.median()
    median_absolute_deviation = (values - median).abs().median().clamp_min(1e-8)
    flagged = [(index, float(values[index]))
               for index in range(len(values))
               if abs(float(values[index] - median)) > 3 * float(median_absolute_deviation)]
    zero_entries = [index for index in range(len(values)) if float(values[index]) == 0.0]
    if zero_entries:
        print(f"    all-zero {label}: {zero_entries}")
    if flagged:
        print(f"    unusual {label} (index, value): {flagged[:20]}")
    if not zero_entries and not flagged:
        print(f"    no unusual {label}")


def frames_report(saved_object, time_axis):
    import torch

    for path, tensor in walk_object(saved_object):
        float_view = tensor.float()
        if tensor.ndim in (4, 5):
            # 5-dimensional video latent: batch, channels, time, height, width.
            # 4-dimensional audio latent: batch, channels, stereo, time (time is last).
            axis = time_axis if tensor.ndim == 5 else tensor.ndim - 1
            moved = float_view.movedim(axis, 0)
            slice_norms = torch.stack([slice_.norm() for slice_ in moved])
            slice_stds = torch.stack([slice_.std() for slice_ in moved])
            print(f"\n{path} shape={tuple(tensor.shape)} treating axis {axis} "
                  f"as time ({moved.shape[0]} slices)")
            print("    norm per slice:", [round(float(v), 1) for v in slice_norms])
            print("    std  per slice:", [round(float(v), 3) for v in slice_stds])
            flag_outliers(slice_norms, "time slices")
        elif tensor.ndim == 3:
            token_norms = float_view[0].norm(dim=-1)
            print(f"\n{path} shape={tuple(tensor.shape)} treating axis 1 as token position")
            jumps = (token_norms[1:] - token_norms[:-1]).abs()
            threshold = jumps.mean() + 4 * jumps.std()
            boundaries = [int(i) + 1 for i in torch.nonzero(jumps > threshold).flatten()]
            print(f"    token norm min={token_norms.min():.2f} "
                  f"max={token_norms.max():.2f} median={token_norms.median():.2f}")
            print(f"    positions where the norm jumps (possible segment boundaries): "
                  f"{boundaries[:40]}")
        elif tensor.ndim == 1 and not tensor.is_floating_point():
            # Integer tag vectors such as minimax_token_tags: print contiguous runs
            # so you can see exactly which token positions carry which tag.
            values = tensor.tolist()
            runs = []
            run_start = 0
            for index in range(1, len(values) + 1):
                if index == len(values) or values[index] != values[run_start]:
                    runs.append((values[run_start], run_start, index - 1))
                    run_start = index
            print(f"\n{path} shape={tuple(tensor.shape)} contiguous runs "
                  f"(tag value, first position, last position): {runs[:60]}")
        else:
            print(f"\n{path} shape={tuple(tensor.shape)} (not profiled)")


def compare_report(saved_object_a, saved_object_b):
    import torch

    tensors_a = dict(walk_object(saved_object_a))
    tensors_b = dict(walk_object(saved_object_b))
    for path in tensors_a:
        if path not in tensors_b:
            print(f"{path}: only in the first file")
            continue
        first, second = tensors_a[path], tensors_b[path]
        if first.shape != second.shape or first.dtype != second.dtype:
            print(f"{path}: different layout "
                  f"{tuple(first.shape)} {first.dtype} versus {tuple(second.shape)} {second.dtype}")
            continue
        identical = torch.equal(first, second)
        difference = (first.float() - second.float()).abs()
        print(f"{path}: bit-identical={identical} max_abs_difference={difference.max():.6g} "
              f"mean_abs_difference={difference.mean():.6g}")
        if not identical and first.ndim in (4, 5):
            axis = 2 if first.ndim == 5 else first.ndim - 1
            per_slice = difference.movedim(axis, 0).flatten(1).max(dim=1).values
            print("    max difference per time slice:",
                  [round(float(v), 5) for v in per_slice])
    for path in tensors_b:
        if path not in tensors_a:
            print(f"{path}: only in the second file")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["structure", "frames", "compare"])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--time-axis", type=int, default=2)
    parser.add_argument("--comfyui-directory", default=None)
    arguments = parser.parse_args()

    loaded = [load_saved_file(path, arguments.comfyui_directory) for path in arguments.files]
    if arguments.command == "structure":
        structure_report(loaded[0])
    elif arguments.command == "frames":
        frames_report(loaded[0], arguments.time_axis)
    else:
        if len(loaded) != 2:
            raise SystemExit("compare needs exactly two files")
        compare_report(loaded[0], loaded[1])


if __name__ == "__main__":
    main()