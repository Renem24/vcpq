"""
ScanNetv2 ground-truth preprocessing for the S20 label system (ScanNet200-based).

Release copy for the vcpq evaluation code. It produces s20_semantics/ and
s20_instance/ next to color/ (the layout vcpq.protocol.gt reads). Two outputs of
the original script that the evaluation does not read (top-K covisibility ranking,
mask visualisations) were removed; nothing else in the processing was changed.
Unlike the ScanNet++ script, this one has no minimum-segment-size filter.

Reads .sens files with scripts/scannet_sensordata.py (a Python 3 port of the
SensReader of the ScanNet toolkit, https://github.com/ScanNet/ScanNet, MIT
License, scripts/LICENSE-SCANNET). The scene directory holds <scene>.sens,
<scene>_2d-label-filt.zip and <scene>_2d-instance-filt.zip (ScanNet's download
script; ScanNet's terms of use apply).

Attribution: the processing procedure (the instance renumbering of
renumber_instances, the thing/stuff table read by get_thing_semantics, the frame
export and the segmentation_data.pkl format) follows Panoptic Lifting's
dataset/preprocessing/preprocess_scannet.py
(https://github.com/nihalsid/panoptic-lifting, Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International,
https://creativecommons.org/licenses/by-nc-sa/4.0/). The code here is a rewrite
(numpy lookup tables and thread pools) with the label pipeline below; the parts
derived from Panoptic Lifting stay under CC BY-NC-SA 4.0.

Label pipeline:
    raw ScanNet ID → ScanNet200 continuous ID (0-200) via raw_to_s20_200.csv
    Instance renumbering: 198 thing classes → contiguous IDs, 2 stuff → 0

Output structure:
    {output_dir}/{scene_name}/
    ├── color/              # RGB (JPG)
    ├── depth/              # Depth (PNG, uint16, mm)
    ├── pose/               # Camera poses (TXT, 4×4)
    ├── intrinsic/          # Camera intrinsics
    ├── s20_semantics/      # ScanNet200 semantic (PNG, uint8, 0-200)
    ├── s20_instance/       # Thing instance IDs (PNG, uint16, 0=stuff/void)
    ├── segmentation_data.pkl
    └── scene_meta.json

Scene and frame selection (--subset, default eval):
    eval  : the 312 scenes of the packaged evaluation list
            (vcpq/protocol/lists/scannetv2_s20_120f_seed42.json) and only their
            listed 120 frames. The s20_instance numbering is still computed from
            every label frame of the scene, so a written frame is identical to the
            same frame of a full run.
    train : the scenes of --split_file (ScanNet's official scannetv2_train.txt,
            https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark),
            every frame.
    full  : every scene* directory under --scans_root, every frame.
    --sens_root (one scene directory) replaces the scene choice of --subset;
    with --subset eval that scene must be in the evaluation list.

Usage:
    # Evaluation frames of the 312 evaluation scenes
    python scripts/preprocess_scannetv2_s20.py \
        --scans_root <scannet_root>/scans --subset eval \
        --output_dir <data_root>/scannetv2_s20

    # One scene, every frame
    python scripts/preprocess_scannetv2_s20.py \
        --sens_root <scannet_root>/scans/scene0000_00 --subset full \
        --output_dir <data_root>/scannetv2_s20
"""

import sys
import os
import argparse
import io
import json
import pickle
import zipfile
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import imageio
import numpy as np
import png
from PIL import Image
from tqdm import tqdm


# ============================================================================
# LUT Loading
# ============================================================================

# Class-mapping CSVs shipped with the package (vcpq/data/).
RESOURCES_DIR = Path(__file__).resolve().parent.parent / "vcpq" / "data"


@lru_cache(maxsize=1)
def get_raw_to_s200_lut():
    """Raw ScanNet ID → ScanNet200 continuous ID (0-200). Unmapped → 0 (void)."""
    lut = np.zeros(2000, dtype=np.uint8)
    for line in (RESOURCES_DIR / "raw_to_s20_200.csv").read_text().strip().splitlines():
        parts = line.strip().split(',')
        raw_id, s200_id = int(parts[0]), int(parts[2])
        if raw_id < len(lut):
            lut[raw_id] = s200_id
    return lut


@lru_cache(maxsize=1)
def get_s200_thing_mask():
    """Bool array (201,). True = thing class, False = stuff/void."""
    thing_mask = np.zeros(201, dtype=bool)
    for i, line in enumerate((RESOURCES_DIR / "s20_200_things.csv").read_text().strip().splitlines()):
        thing_mask[i + 1] = bool(int(line.strip().split(',')[1]))
    return thing_mask


# ============================================================================
# Step 1: Frame Export
# ============================================================================

def _export_single_frame(args):
    """Export a single frame (color, depth, pose) to disk."""
    idx, frame, color_type, depth_type, depth_h, depth_w, dest = args

    color = frame.decompress_color(color_type)
    imageio.imwrite(str(dest / 'color' / f'{idx}.jpg'), color)

    depth_data = frame.decompress_depth(depth_type)
    depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_h, depth_w)
    with open(dest / 'depth' / f'{idx}.png', 'wb') as f:
        writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
        writer.write(f, depth.reshape(-1, depth.shape[1]).tolist())

    np.savetxt(dest / 'pose' / f'{idx}.txt', frame.camera_to_world, fmt='%f')


def export_all_frames(sd, path_dest, num_workers=8, frames=None):
    """Export all frames (or only the frame indices in ``frames``) and detect
    invalid poses (always over all frames).

    Returns:
        valid_indices: list of frame indices with valid poses
        invalid_indices: list of frame indices with inf/nan poses
    """
    for subdir in ['color', 'depth', 'pose', 'intrinsic']:
        (path_dest / subdir).mkdir(parents=True, exist_ok=True)

    # Intrinsics
    for name, matrix in [
        ('intrinsic_color.txt', sd.intrinsic_color),
        ('extrinsic_color.txt', sd.extrinsic_color),
        ('intrinsic_depth.txt', sd.intrinsic_depth),
        ('extrinsic_depth.txt', sd.extrinsic_depth),
    ]:
        np.savetxt(path_dest / 'intrinsic' / name, matrix, fmt='%f')

    num_frames = len(sd.frames)

    # Detect invalid poses
    valid_indices, invalid_indices = [], []
    for i in range(num_frames):
        pose = sd.frames[i].camera_to_world
        if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
            invalid_indices.append(i)
        else:
            valid_indices.append(i)

    if invalid_indices:
        print(f"  {len(invalid_indices)} frames with invalid poses (inf/nan)")

    # Export all frames (or the requested subset) in parallel
    export_indices = range(num_frames) if frames is None else sorted(frames)
    args_list = [
        (i, sd.frames[i], sd.color_compression_type, sd.depth_compression_type,
         sd.depth_height, sd.depth_width, path_dest)
        for i in export_indices
    ]
    print(f'Exporting {len(args_list)} of {num_frames} frames ({num_workers} workers)')
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(_export_single_frame, args_list),
                  total=len(args_list), desc='Exporting frames'))

    return valid_indices, invalid_indices


# ============================================================================
# Step 2 + 3: Label Mapping + Instance Renumbering
# ============================================================================

def process_labels_s20(path_sens_root, path_dest, num_workers=8, frames=None):
    """Extract labels from zip, map to ScanNet200, renumber instances.

    ``frames`` (label frame stems) limits which PNGs are written; the instance
    renumbering always counts every label frame of the scene.

    Optimized two-pass approach:
      1. Pre-read all zip data into memory (fast sequential read)
      2. Pass 1: parallel semantic mapping + save, collect instance-semantic counts
      3. Pass 2: parallel instance remapping + save (reuse in-memory data)

    Returns:
        List of frame IDs with labels, or None if zips not found.
    """
    raw_to_s200 = get_raw_to_s200_lut()
    thing_mask = get_s200_thing_mask()

    scene_name = path_sens_root.stem
    label_zip = path_sens_root / f"{scene_name}_2d-label-filt.zip"
    instance_zip = path_sens_root / f"{scene_name}_2d-instance-filt.zip"

    for zp in [label_zip, instance_zip]:
        if not zp.exists():
            print(f"WARNING: Not found: {zp}")
            return None

    (path_dest / 's20_semantics').mkdir(parents=True, exist_ok=True)
    (path_dest / 's20_instance').mkdir(parents=True, exist_ok=True)

    # Index zip entries
    with zipfile.ZipFile(label_zip) as lz:
        label_entries = {Path(n).stem: n for n in lz.namelist() if n.endswith('.png')}
    with zipfile.ZipFile(instance_zip) as iz:
        inst_entries = {Path(n).stem: n for n in iz.namelist() if n.endswith('.png')}

    frame_ids = sorted(label_entries.keys() & inst_entries.keys(), key=lambda x: int(x))
    print(f"Found {len(frame_ids)} label frames in zips")
    if frames is None:
        write_ids = frame_ids
    else:
        missing = sorted(set(frames) - set(frame_ids), key=int)
        if missing:
            raise ValueError(f"{scene_name}: requested frames without labels: {missing[:10]}")
        wanted = set(frames)
        write_ids = [fid for fid in frame_ids if fid in wanted]
    write_set = set(write_ids)

    # ---- Pre-read all zip data into memory ----
    print("Loading zip data into memory...")
    with zipfile.ZipFile(label_zip) as lz:
        label_data = {fid: lz.read(label_entries[fid]) for fid in frame_ids}
    with zipfile.ZipFile(instance_zip) as iz:
        inst_data = {fid: iz.read(inst_entries[fid]) for fid in frame_ids}

    # ---- Pass 1: parallel semantic mapping + counting ----
    lut_size = len(raw_to_s200)
    sem_dir = path_dest / 's20_semantics'

    def _pass1(fid):
        sem_raw = np.array(Image.open(io.BytesIO(label_data[fid])))
        sem_s200 = raw_to_s200[np.clip(sem_raw, 0, lut_size - 1)]
        if fid in write_set:
            Image.fromarray(sem_s200).save(sem_dir / f'{fid}.png')

        ins_raw = np.array(Image.open(io.BytesIO(inst_data[fid])))
        combined = ins_raw.flatten().astype(np.int32) * 201 + sem_s200.flatten().astype(np.int32)
        keys, counts = np.unique(combined, return_counts=True)
        frame_counts = {}
        for k, c in zip(keys, counts):
            iid = int(k // 201)
            if iid > 0:
                frame_counts.setdefault(iid, Counter())[int(k % 201)] = int(c)
        return frame_counts

    print(f'Pass 1: semantic mapping + counting ({num_workers} workers)')
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(_pass1, frame_ids),
                            total=len(frame_ids), desc='Pass 1'))

    # Merge counts
    inst_sem_counts = defaultdict(Counter)
    for frame_counts in results:
        for iid, counter in frame_counts.items():
            inst_sem_counts[iid] += counter

    label_data.clear()

    # ---- Build instance remap LUT ----
    inst_to_sem = {iid: c.most_common(1)[0][0] for iid, c in inst_sem_counts.items()}

    remap = {0: 0}
    reverse_remap = {0: 0}
    new_id = 1
    for iid in sorted(inst_to_sem.keys()):
        if thing_mask[inst_to_sem[iid]]:
            remap[iid] = new_id
            reverse_remap[new_id] = iid
            new_id += 1
        else:
            remap[iid] = 0

    print(f"Instance remapping: {len(inst_to_sem)} raw → {new_id - 1} thing instances")

    max_inst = max(remap.keys()) + 1 if remap else 1
    remap_lut = np.zeros(max_inst, dtype=np.uint16)
    for old_id, mapped_id in remap.items():
        remap_lut[old_id] = mapped_id
    lut_len = len(remap_lut)

    # ---- Pass 2: parallel instance remapping ----
    ins_dir = path_dest / 's20_instance'

    def _pass2(fid):
        ins_raw = np.array(Image.open(io.BytesIO(inst_data[fid])))
        ins_remapped = remap_lut[np.clip(ins_raw, 0, lut_len - 1)]
        Image.fromarray(ins_remapped.astype(np.uint16)).save(ins_dir / f'{fid}.png')

    print(f'Pass 2: instance remapping ({num_workers} workers)')
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(_pass2, write_ids),
                  total=len(write_ids), desc='Pass 2'))

    inst_data.clear()

    # ---- segmentation_data.pkl ----
    remapped_to_sem = {0: 0}
    for mid in range(1, new_id):
        remapped_to_sem[mid] = inst_to_sem[reverse_remap[mid]]

    export_dict = {
        'class_system': '200',
        'num_semantic_classes': 201,
        'fg_classes': [i for i in range(201) if thing_mask[i]],
        'bg_classes': [i for i in range(1, 201) if not thing_mask[i]],
        's20_instance_to_semantic': remapped_to_sem,
    }
    with open(path_dest / 'segmentation_data.pkl', 'wb') as f:
        pickle.dump(export_dict, f)

    return frame_ids


# ============================================================================
# Main
# ============================================================================

def process_scene(path_sens_root, path_dest, num_workers=8, frames=None):
    """Process a single ScanNet scene end-to-end.

    ``frames``: frame stems ('4', '7', ...) to write, None = every frame."""
    scene_name = path_sens_root.stem
    path_dest.mkdir(parents=True, exist_ok=True)

    print('#' * 80)
    print(f'Processing {scene_name}')
    print('#' * 80)

    # Step 1: Load .sens + export all frames
    print('\n[1/3] Loading .sens and exporting all frames...')
    sd = SensorData(str(path_sens_root / f'{scene_name}.sens'))
    num_frames = len(sd.frames)
    print(f'Loaded {num_frames} frames')
    if frames is not None:
        bad = [f for f in frames if not f.isdigit() or int(f) >= num_frames]
        if bad:
            raise ValueError(f'{scene_name}: requested frames not in the .sens file: {bad[:10]}')
    valid_indices, invalid_indices = export_all_frames(
        sd, path_dest, num_workers,
        frames=None if frames is None else [int(f) for f in frames])
    del sd

    # Step 2+3: Label mapping + instance renumbering
    print('\n[2/3] Label extraction + instance renumbering...')
    label_frame_ids = process_labels_s20(path_sens_root, path_dest, num_workers, frames=frames)

    # Step 3: Metadata
    print('\n[3/3] Saving metadata...')
    meta = {
        'scene_name': scene_name,
        'total_frames': num_frames,
        'num_valid_frames': len(valid_indices),
        'num_invalid_frames': len(invalid_indices),
        'invalid_pose_indices': invalid_indices,
        'num_label_frames': len(label_frame_ids) if label_frame_ids else 0,
    }
    (path_dest / 'scene_meta.json').write_text(json.dumps(meta, indent=2))

    print(f'\nDone: {path_dest}')
    print(f'  Frames: {num_frames} total, {len(valid_indices)} valid pose, '
          f'{len(label_frame_ids) if label_frame_ids else 0} labels')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ScanNet preprocessing (S20 label system)')
    parser.add_argument('--scans_root', help='ScanNet scans/ directory (one directory per scene)')
    parser.add_argument('--sens_root', help='One raw scene directory (replaces the scene '
                                            'choice of --subset)')
    parser.add_argument('--subset', choices=('eval', 'train', 'full'), default='eval',
                        help='eval (default): the 312 evaluation scenes, listed frames only; '
                             'train: the scenes of --split_file, every frame; '
                             'full: every scene under --scans_root, every frame')
    parser.add_argument('--split_file', help='--subset train: ScanNet scannetv2_train.txt')
    parser.add_argument('--output_dir', required=True, help='Output base directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Parallel workers (default: 8)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='skip scenes whose scene_meta.json (written last) exists')
    args = parser.parse_args()
    if (args.scans_root is None) == (args.sens_root is None):
        parser.error('give exactly one of --scans_root and --sens_root')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    sys.path.insert(1, os.path.dirname(script_dir))  # repository root, for vcpq
    from scannet_sensordata import SensorData
    from preprocess_subsets import eval_frame_list, scannetv2_scenes

    eval_list = eval_frame_list('scannetv2') if args.subset == 'eval' else {}
    if args.sens_root is not None:
        scene_dirs = [Path(args.sens_root)]
    else:
        scene_dirs = [Path(args.scans_root) / s for s in
                      scannetv2_scenes(args.scans_root, args.subset, args.split_file)]

    for i, sens_root in enumerate(scene_dirs):
        scene_name = sens_root.stem
        dest = Path(args.output_dir) / scene_name
        print(f'[{i + 1}/{len(scene_dirs)}] {scene_name}')
        if args.skip_existing and (dest / 'scene_meta.json').is_file():
            print('  Skipping (already exists)')
            continue
        frames = None
        if args.subset == 'eval':
            if scene_name not in eval_list:
                parser.error(f'{scene_name} is not in the evaluation list; '
                             'use --subset train or --subset full for other scenes')
            frames = eval_list[scene_name]
        process_scene(sens_root, dest, num_workers=args.num_workers, frames=frames)
