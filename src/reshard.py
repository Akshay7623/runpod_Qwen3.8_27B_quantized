"""Splits a safetensors checkpoint into a given number of weight files of about
equal size, without changing any tensor.

    python3 reshard.py SRC_DIR DST_DIR NUM_FILES

pekkAi's checkpoint has two weight files, of 10.0 GB and 7.9 GB, and each file
in the image becomes one layer. GHCR, where the GitHub Actions build pushes
the image, rejects layers over 10 GB and uploads that take over 10 minutes.

Each tensor's bytes are copied as they are, and DST_DIR gets a new
model.safetensors.index.json. Each source weight file is deleted once all its
tensors are copied, and so is the old index, so SRC_DIR ends with only the
other files. At the end, every new file is read back with the safetensors
library, and each tensor must have the dtype, shape and SHA-256 it had in the
original files.
"""

import hashlib
import json
import os
import struct
import sys

import torch
from safetensors import safe_open

INDEX = "model.safetensors.index.json"
DTYPE_SIZES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}
CHUNK = 64 * 1024 * 1024


def read_header(path):
    """Return (data_start, {name: info}) for a safetensors file."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return 8 + length, header


def plan(sources, num_files):
    """Assign every tensor, in the order it is stored, to an output file.

    Returns a list of (file_name, [tensor, ...]). A tensor goes to file k when
    it starts in the k-th equal part of all the tensor data, so each file
    exceeds total / num_files by less than one tensor.
    """
    tensors = []
    for src in sources:
        data_start, header = read_header(src)
        end = 0
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"]):
            begin, stop = info["data_offsets"]
            if begin != end:
                sys.exit(f"{src}: tensor data is not contiguous at {name}")
            end = stop
            tensors.append({
                "name": name, "src": src, "dtype": info["dtype"], "shape": info["shape"],
                "offset": data_start + begin, "size": stop - begin,
            })
        if end != os.path.getsize(src) - data_start:
            sys.exit(f"{src}: header does not cover the whole file")

    total = sum(t["size"] for t in tensors)
    files = [(f"model-{k + 1:05d}-of-{num_files:05d}.safetensors", []) for k in range(num_files)]
    position = 0
    for t in tensors:
        files[min(num_files - 1, position * num_files // total)][1].append(t)
        position += t["size"]
    if any(not members for _, members in files):
        sys.exit(f"Cannot split {len(tensors)} tensors into {num_files} files")
    return files, total


def write_file(path, members, open_sources, digests):
    """Write one safetensors file and record each tensor's SHA-256."""
    # Like the safetensors library: tensors with larger elements first, so
    # every tensor starts at a multiple of its element size.
    members = sorted(members, key=lambda t: (-DTYPE_SIZES[t["dtype"]], t["name"]))
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for t in members:
        header[t["name"]] = {
            "dtype": t["dtype"], "shape": t["shape"], "data_offsets": [offset, offset + t["size"]],
        }
        offset += t["size"]
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    # The data starts on an 8-byte boundary; the format pads the header with spaces.
    header_bytes += b" " * (-len(header_bytes) % 8)

    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for t in members:
            src = open_sources.setdefault(t["src"], open(t["src"], "rb"))
            src.seek(t["offset"])
            digest = hashlib.sha256()
            left = t["size"]
            while left:
                chunk = src.read(min(CHUNK, left))
                if not chunk:
                    sys.exit(f"{t['src']} ended inside {t['name']}")
                digest.update(chunk)
                out.write(chunk)
                left -= len(chunk)
            digests[t["name"]] = digest.hexdigest()


def tensor_bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def check_file(path, members, digests):
    """Read a written file back and compare every tensor with the original."""
    with safe_open(path, framework="pt") as f:
        if sorted(f.keys()) != sorted(t["name"] for t in members):
            sys.exit(f"{path}: wrong set of tensors")
        for t in members:
            tensor = f.get_tensor(t["name"])
            if list(tensor.shape) != t["shape"] or tensor.element_size() != DTYPE_SIZES[t["dtype"]]:
                sys.exit(f"{path}: {t['name']} has the wrong shape or dtype")
            if hashlib.sha256(tensor_bytes(tensor)).hexdigest() != digests[t["name"]]:
                sys.exit(f"{path}: {t['name']} differs from the original")


def main():
    src_dir, dst_dir, num_files = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(os.path.join(src_dir, INDEX)) as f:
        sources = sorted({os.path.join(src_dir, name) for name in json.load(f)["weight_map"].values()})
    files, total = plan(sources, num_files)

    os.makedirs(dst_dir, exist_ok=True)
    open_sources, digests, weight_map = {}, {}, {}
    for k, (name, members) in enumerate(files):
        path = os.path.join(dst_dir, name)
        write_file(path, members, open_sources, digests)
        weight_map.update({t["name"]: name for t in members})
        print(f"reshard: {name}: {len(members)} tensors, {os.path.getsize(path) / 1e9:.2f} GB", flush=True)
        # Delete each source file whose tensors are all copied now.
        for src in sources:
            if os.path.exists(src) and all(t["src"] != src for _, later in files[k + 1:] for t in later):
                if src in open_sources:
                    open_sources.pop(src).close()
                os.remove(src)

    with open(os.path.join(dst_dir, INDEX), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
        f.write("\n")
    os.remove(os.path.join(src_dir, INDEX))

    for name, members in files:
        check_file(os.path.join(dst_dir, name), members, digests)
    print(f"reshard: all {len(digests)} tensors read back unchanged", flush=True)


if __name__ == "__main__":
    main()
