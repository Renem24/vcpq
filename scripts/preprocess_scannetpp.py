"""
ScanNet++ ground-truth preprocessing for the SPP100 label system (benchmark Top100).

Release copy for the vcpq evaluation code. It writes spp100_semantics/ and
spp100_instance/ next to color/ for the DSLR (920 x 613, frame stems = the
digits of DSC<digits>.JPG) and iPhone (960 x 720) images, the resolutions and
stems of the paper's ground truth (--out_longest_side, default 920 for DSLR and
960 for iPhone for every --subset, 0 = native; the labels are rendered at that
size). Two outputs of the original script that the evaluation does not read
(top-K covisibility ranking, mask visualisations) were removed; the DSLR resize to
--out_longest_side and the digit stems were added for this release (README
"Datasets", reproduction status).

Segment filter: decompose_panoptic() drops every (instance, class) segment with
fewer than MIN_INST_AREA = 50 pixels in a frame (its pixels become void). This is
the filter of PanSt3R's tools/preprocess_scannetpp.py ("Filter noisy segments");
it is in PanSt3R's code, not in the PanSt3R paper.

Attribution: this script is derived from PanSt3R's tools/preprocess_scannetpp.py
(https://github.com/naver/panst3r), which itself is based on DUSt3R's
datasets_preprocess/preprocess_scannetpp.py (https://github.com/naver/dust3r).
The header of both files reads:
    Copyright (C) 2024-present Naver Corporation. All rights reserved.
    Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
The licence is Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International, https://creativecommons.org/licenses/by-nc-sa/4.0/ ; the licensed
material is provided without warranty, as stated in the disclaimer of warranties
of that licence. Two helpers, colmap_to_opencv_intrinsics and
opencv_to_colmap_intrinsics, come from DUSt3R's dust3r/utils/geometry.py
(Copyright (C) 2024-present Naver Corporation, CC BY-NC-SA 4.0, same header).
Shared parts include pose_from_qwxyz_txyz, the segGroups loop that builds
pts_pan_id = inst_id * cls_sep + cls_id, the rgb2id/id2rgb panoptic id encoding
(imported from panst3r.datasets.utils there, written inline here) and the pyrender
scene setup, OPENGL_TO_OPENCV, MIN_INST_AREA = 50, and the "Filter noisy segments" step
(here decompose_panoptic, which also voids ids that are not in the valid id set,
as PanSt3R does). The modifications are ours: the removal of the two
outputs stated above, the SPP100 label mapping, the iPhone source, the
simplifications of the loaders and the command-line interface. The derived parts
stay under CC BY-NC-SA 4.0.
Note on the licence of the PanSt3R file: the header of tools/preprocess_scannetpp.py
states CC BY-NC-SA 4.0, while the LICENSE file at the root of the PanSt3R repository
is the "PanSt3R Non-Commercial License" of NAVER Corporation.

rgb2id and id2rgb follow panopticapi's utils.py (https://github.com/cocodataset/panopticapi),
which PanSt3R adapts in src/panst3r/datasets/utils.py. panopticapi is licensed under
the BSD 2-Clause License; its notice reads:

    Copyright (c) 2018, Alexander Kirillov
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Renders 2D panoptic labels from 3D mesh annotations using pyrender + SEG_VERT.
Requires the ScanNet++ data (official access agreement) and, for rendering, EGL
(PYOPENGL_PLATFORM=egl). ScanNet++ toolkit: https://github.com/scannetpp/scannetpp

Two image sources (--source):
    dslr   : pre-undistorted DSLR images (resized_undistorted_images) + COLMAP poses,
             downscaled (--out_longest_side, default 920)
    iphone : RGB/mask decoded from rgb.mkv/rgb_mask.mkv, undistorted on the fly
             (OPENCV model from iphone/colmap), posed with per-frame aligned_pose
             from pose_intrinsic_imu.json, downscaled (--out_longest_side, default 960).
             The mesh + all label/spp100 logic is shared with dslr,
             so iPhone outputs are format-compatible with the DSLR spp100 set.

Label pipeline:
    raw annotation label → spp100 class ID (1-100) via the mapping that
    scripts/spp100_mapping.py builds from <root_dir>/metadata (semantic_classes.txt,
    semantic_benchmark/top100.txt, top100_instance.txt, map_benchmark.csv)
    Unmapped labels → void (0)
    Instance: 84 thing classes → contiguous IDs, 16 stuff → 0
    panoptic_id = instance_id * 4096 + class_id (CLS_SEP=4096 exceeds every scene-local
    raw label id)

Output structure:
    {output_dir}/{scene_id}/
    ├── color/              # Undistorted RGB (JPG)
    ├── depth/              # Mesh-rendered depth (PNG, uint16, mm)
    ├── pose/               # 4×4 cam2world (TXT)
    ├── intrinsic/          # Per-frame 3×3 K + shared intrinsic_color.txt (4×4)
    ├── semantics/          # Raw scene-local semantic IDs (PNG, uint16, every raw label)
    ├── instance/           # Raw instance IDs (PNG, uint16, all objects)
    ├── spp100_semantics/      # Top100-mapped semantic ID (PNG, uint8, 0-100)
    ├── spp100_instance/       # Top100 thing instance ID (PNG, uint16, 0=stuff/void)
    ├── segmentation_data.pkl
    ├── segments.json       # Segment metadata + local_class_names (for re-mapping)
    ├── categories.json     # SPP100 class definitions
    └── scene_meta.json

Scene and frame selection (--subset, default eval):
    eval  : the scenes of the packaged evaluation list of --source
            (vcpq/protocol/lists/, 50 scenes) and only their listed 120 frames.
            A frame is rendered exactly as in a full run: the mesh labels, the
            instance numbering and the DSLR intrinsics do not depend on which
            frames are rendered, and the iPhone video is still decoded from frame 0.
    train : <root_dir>/splits/nvs_sem_train.txt, every frame.
    full  : nvs_sem_train + nvs_sem_val, every frame.
    --scene replaces the scene choice of --subset; with --subset eval the scene
    must be in the evaluation list. For train and full, two iPhone options save
    disk space: --frame_stride N keeps every Nth video frame and --no_depth skips
    the depth maps.

Usage:
    # Evaluation frames, DSLR
    PYOPENGL_PLATFORM=egl python scripts/preprocess_scannetpp.py \\
        --root_dir <scannetpp_root> --subset eval \\
        --output_dir <data_root>/scannetpp_spp100

    # Evaluation frames, iPhone (aligned_pose, downscaled to 960 longer side)
    PYOPENGL_PLATFORM=egl python scripts/preprocess_scannetpp.py \\
        --root_dir <scannetpp_root> --subset eval --source iphone \\
        --out_longest_side 960 --output_dir <data_root>/scannetpp_spp100_iphone

    # Single scene, every frame
    PYOPENGL_PLATFORM=egl python scripts/preprocess_scannetpp.py \\
        --root_dir <scannetpp_root> --subset full --scene 7b6477cb95 \\
        --output_dir <out_dir>
"""

import sys
import os
import os.path as osp
import argparse
import csv
import json
import pickle
import re
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import PIL.Image as Image
import trimesh
import trimesh.exchange.ply
from scipy.spatial.transform import Rotation
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ for the helper modules, the repository root for vcpq (frame lists)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(1, os.path.dirname(_SCRIPT_DIR))
from preprocess_subsets import SUBSETS, eval_frame_list, scannetpp_scenes, select_frames  # noqa: E402
from spp100_mapping import build_raw_to_spp100, label_maps  # noqa: E402

# Validate pyrender SEG_VERT availability
try:
    import pyrender
    if not hasattr(pyrender.RenderFlags, 'SEG_VERT'):
        raise ImportError(
            "pyrender SEG_VERT flag not found. Install the custom fork:\n"
            '  pip install "git+https://github.com/lojzezust/pyrender.git"')
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


# ============================================================================
# Constants
# ============================================================================

OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])

CLS_SEP = 4096  # Must exceed max scene-local class ID (raw labels can be 2878+)
MIN_INST_AREA = 50

# spp100 class table shipped with the package (vcpq/data/). The raw label ->
# spp100 mapping is built at run time from <root_dir>/metadata (spp100_mapping.py).
_CLASS_MAPPING_DIR = osp.normpath(osp.join(_SCRIPT_DIR, '..', 'vcpq', 'data'))
SPP100_CLASSES_CSV = osp.join(_CLASS_MAPPING_DIR, 'spp100_classes.csv')


# ============================================================================
# Inline Utilities
# ============================================================================

def rgb2id(color):
    """(H, W, 3) uint8 → (H, W) int32 panoptic id."""
    if color.dtype != np.uint8:
        color = color.astype(np.uint8)
    return (color[..., 0].astype(np.int32)
            + 256 * color[..., 1].astype(np.int32)
            + 256 * 256 * color[..., 2].astype(np.int32))


def id2rgb(id_map):
    """(N,) or (H, W) int32 panoptic id → (..., 3) uint8."""
    id_map = id_map.copy()
    rgb = np.zeros((*id_map.shape, 3), dtype=np.uint8)
    for i in range(3):
        rgb[..., i] = id_map % 256
        id_map //= 256
    return rgb


def colmap_to_opencv_intrinsics(K):
    """COLMAP → OpenCV: shift principal point by -0.5."""
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """OpenCV → COLMAP: shift principal point by +0.5."""
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def rescale_intrinsics(K, sx, sy):
    """Scale intrinsics by factors (sx, sy)."""
    K = K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


# ============================================================================
# SfM Loading
# ============================================================================

def pose_from_qwxyz_txyz(elems):
    """COLMAP quaternion + translation → 4×4 cam2world."""
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # world-to-cam → cam-to-world


REGEXPR_DSLR = re.compile(r'^.*DSC(?P<frameid>\d+)\.JPG$')


def get_dslr_frame_id(name):
    """DSLR frame stem = the digits after 'DSC' ('DSC04432.JPG' -> '04432'),
    the stems of the evaluation lists; other names keep their stem."""
    match = REGEXPR_DSLR.match(name)
    return match.group('frameid') if match else get_frame_stem(name)


def output_size(W, H, out_longest_side):
    """(W, H) scaled so that the longer side is out_longest_side (0 = unchanged,
    never upscaled)."""
    if out_longest_side and max(W, H) > out_longest_side:
        s = out_longest_side / float(max(W, H))
        return int(round(W * s)), int(round(H * s))
    return W, H


def get_frame_stem(name):
    """Extract filename stem from image name (e.g. 'DSC05828.JPG' → 'DSC05828')."""
    return os.path.splitext(name)[0]


def load_sfm(sfm_dir):
    """Load COLMAP SfM (cameras + images, DSLR only).

    Returns:
        img_infos: {colmap_idx: {intrinsics, path, frame_id, cam_to_world}}
    """
    with open(osp.join(sfm_dir, 'cameras.txt'), 'r') as f:
        raw = [line for line in f.read().splitlines() if not line.startswith('#')]

    intrinsics = {}
    for camera in raw:
        parts = camera.split(' ')
        intrinsics[int(parts[0])] = [parts[1]] + [float(c) for c in parts[2:]]

    with open(osp.join(sfm_dir, 'images.txt'), 'r') as f:
        raw = [line for line in f.read().splitlines() if not line.startswith('#')]

    img_infos = {}
    for image, _points in zip(raw[0::2], raw[1::2]):
        parts = image.split(' ')
        idx = parts[0]
        img_name = parts[-1]
        img_infos[idx] = dict(
            intrinsics=intrinsics[int(parts[-2])],
            path=img_name,
            frame_id=get_dslr_frame_id(img_name),
            cam_to_world=pose_from_qwxyz_txyz(parts[1:-2]),
        )

    return img_infos


def make_undistorted_intrinsics(raw_intrinsics):
    """Compute undistorted pinhole K from COLMAP fisheye intrinsics.

    Returns (width, height, K_colmap_convention).
    """
    camera_type = raw_intrinsics[0]
    width = int(raw_intrinsics[1])
    height = int(raw_intrinsics[2])
    fx, fy, cx, cy = (raw_intrinsics[3], raw_intrinsics[4],
                       raw_intrinsics[5], raw_intrinsics[6])
    distortion = np.array(raw_intrinsics[7:])

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K = colmap_to_opencv_intrinsics(K)

    if camera_type == "OPENCV_FISHEYE":
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion, (width, height), np.eye(3), balance=0.0)
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (width, height), 1, (width, height), True)

    new_K = opencv_to_colmap_intrinsics(new_K)
    return width, height, new_K


# ============================================================================
# iPhone Loading (video frames + aligned_pose + OPENCV undistortion)
# ============================================================================

def load_iphone_camera(colmap_dir):
    """Load the single OPENCV camera model from iPhone colmap/cameras.txt.

    Layout: CAMERA_ID MODEL W H fx fy cx cy k1 k2 p1 p2  (COLMAP convention).

    Returns dict {model, W, H, K_colmap (3x3), dist (k1,k2,p1,p2[,k3])}.
    """
    with open(osp.join(colmap_dir, 'cameras.txt'), 'r') as f:
        lines = [l for l in f.read().splitlines() if not l.startswith('#')]
    parts = lines[0].split()
    model = parts[1]
    W, H = int(parts[2]), int(parts[3])
    fx, fy, cx, cy = map(float, parts[4:8])
    dist = np.array([float(x) for x in parts[8:]], dtype=np.float64)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return dict(model=model, W=W, H=H, K_colmap=K, dist=dist)


def load_iphone_poses(json_path):
    """Load per-frame mesh-aligned cam2world poses (aligned_pose) from
    pose_intrinsic_imu.json.

    Returns sorted list of (frame_id, cam_to_world 4x4) for every frame that
    carries a valid pose. frame_id is the json key, e.g. 'frame_000300'.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    infos = []
    for fid in sorted(data.keys()):
        pose = data[fid].get('aligned_pose')
        if pose is None:
            continue
        c2w = np.array(pose, dtype=np.float64)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            continue
        infos.append((fid, c2w))
    return infos


def build_iphone_undistort(cam, out_longest_side=0):
    """Precompute OPENCV->pinhole undistortion maps + output intrinsics.

    cv2.undistort/remap operate in OpenCV pixel convention; the rendered/saved
    intrinsics are kept in COLMAP convention to match the DSLR pipeline.

    Returns dict {map1, map2, valid (HxW uint8 255=valid), full_WH, out_WH,
                  K_out (3x3 COLMAP convention, scaled to out_WH)}.
    """
    W, H = cam['W'], cam['H']
    K_cv = colmap_to_opencv_intrinsics(cam['K_colmap'])
    dist = cam['dist']

    # alpha=1 keeps the full FOV (with black borders); the validity mask below
    # marks the undistortion-invalid pixels so they are voided downstream.
    new_K_cv, _roi = cv2.getOptimalNewCameraMatrix(
        K_cv, dist, (W, H), 1, (W, H), True)
    map1, map2 = cv2.initUndistortRectifyMap(
        K_cv, dist, np.eye(3), new_K_cv, (W, H), cv2.CV_32FC1)

    valid = ((map1 >= 0) & (map1 <= W - 1)
             & (map2 >= 0) & (map2 <= H - 1)).astype(np.uint8) * 255

    if out_longest_side and max(W, H) > out_longest_side:
        s = out_longest_side / float(max(W, H))
        out_W, out_H = int(round(W * s)), int(round(H * s))
    else:
        out_W, out_H = W, H

    new_K_colmap = opencv_to_colmap_intrinsics(new_K_cv)
    K_out = rescale_intrinsics(new_K_colmap, out_W / W, out_H / H)
    return dict(map1=map1, map2=map2, valid=valid,
                full_WH=(W, H), out_WH=(out_W, out_H), K_out=K_out)


def read_iphone_frame(rgb_cap, mask_cap, undist, prev_idx, target_idx):
    """Sequentially advance both video readers to target_idx, returning the
    undistorted+resized (rgb, mask) at the iPhone output resolution.

    Sequential grab() is far cheaper/more reliable than per-frame seeking on
    these mkv streams. Returns (rgb uint8 HxWx3, mask uint8 HxW) or (None, None).
    """
    # Advance past skipped frames without decoding them fully.
    for _ in range(target_idx - prev_idx - 1):
        rgb_cap.grab()
        mask_cap.grab()
    ok_r, fr_rgb = rgb_cap.read()
    ok_m, fr_mask = mask_cap.read()
    if not ok_r or not ok_m:
        return None, None

    rgb = fr_rgb[..., ::-1]                    # BGR -> RGB
    mask = fr_mask[..., 0]                      # replicated gray -> single channel

    rgb_u = cv2.remap(np.ascontiguousarray(rgb), undist['map1'], undist['map2'],
                      cv2.INTER_LINEAR)
    mask_u = cv2.remap(mask, undist['map1'], undist['map2'], cv2.INTER_NEAREST)
    mask_u[undist['valid'] == 0] = 0           # fold in undistortion-invalid pixels

    out_WH = undist['out_WH']
    if out_WH != undist['full_WH']:
        rgb_u = cv2.resize(rgb_u, out_WH, interpolation=cv2.INTER_AREA)
        mask_u = cv2.resize(mask_u, out_WH, interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(rgb_u), mask_u


# ============================================================================
# Class System Loading (spp100_: top100 with +1 offset, void=0)
# ============================================================================

def load_spp100_class_definitions(spp100_classes_csv, metadata_dir):
    """Load the ScanNet++ spp100 class definitions.

    spp100_classes.csv (vcpq/data/): spp100_id,name,isthing  (1-100, +1 offset; 0=void)
    raw label -> spp100 mapping: built from <scannetpp_root>/metadata by
    spp100_mapping.build_raw_to_spp100 (rows raw_class,spp100_id,spp100_name,iscrowd)

    Returns:
        label_to_id: {raw_label_name: spp100_class_id (1-100)} (mapped labels only)
        thing_ids: set of spp100 class IDs that are thing classes
        stuff_ids: set of spp100 class IDs that are stuff classes
        crowd_cls: set of raw label names flagged iscrowd=1
        categories: list of {id, name, isthing} dicts
    """
    # spp100_classes.csv -> categories + thing/stuff id sets
    categories = []
    thing_ids = set()
    stuff_ids = set()
    with open(spp100_classes_csv, 'r') as f:
        for row in csv.DictReader(f):
            spp100_id = int(row['spp100_id'])
            isthing = int(row['isthing'])
            categories.append(dict(id=spp100_id, name=row['name'], isthing=isthing))
            (thing_ids if isthing else stuff_ids).add(spp100_id)

    # raw label -> spp100 id (mapped only) + crowd label names
    rows = build_raw_to_spp100(metadata_dir)
    for _, spp100_id, name, _ in rows:
        if spp100_id > 0 and categories[spp100_id - 1]['name'] != name:
            raise ValueError(f'spp100 id {spp100_id} is {name!r} in {metadata_dir} but '
                             f'{categories[spp100_id - 1]["name"]!r} in {spp100_classes_csv}')
    label_to_id, crowd_cls = label_maps(rows)

    return label_to_id, thing_ids, stuff_ids, crowd_cls, categories


# ============================================================================
# Mesh Semantics
# ============================================================================

def read_semantics(segments_path, annotations_path, crowd_classes,
                   num_points, cls_sep):
    """Load mesh annotations and assign panoptic IDs per vertex.

    ALL labels get a scene-local class ID (1-based, no skipping).
    This keeps every raw label, so the output can be re-mapped to another class set.
    Segments of a crowd label are left out (their vertices stay void).

    panoptic_id = instance_id * cls_sep + scene_local_class_id

    Returns:
        segments: list of segment metadata dicts (with orig_class_name)
        pts_pan_id: (N_vertices,) int32 array of panoptic IDs
        local_cls_names: dict {scene_local_class_id: label_name}
    """
    with open(segments_path, 'r') as f:
        segments_data = json.load(f)

    assert segments_data['segIndices'] == list(range(num_points)), \
        "Segment indices mismatch — expected identity mapping"

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Assign scene-local class IDs to ALL labels (1-based, 0=void)
    label_name_to_local = {}
    next_local_id = 1
    for seg_info in annotations['segGroups']:
        name = seg_info['label']
        if name not in label_name_to_local:
            label_name_to_local[name] = next_local_id
            next_local_id += 1

    local_cls_names = {v: k for k, v in label_name_to_local.items()}

    pts_pan_id = np.zeros(num_points, dtype=np.int32)
    segments = []
    inst_id = 1
    for seg_info in annotations['segGroups']:
        vert_idx = np.array(seg_info['segments'])
        label = seg_info['label']
        cls_id = label_name_to_local[label]
        iscrowd = label in crowd_classes
        if iscrowd:
            continue

        segments.append(dict(
            id=inst_id * cls_sep + cls_id,
            instance_id=inst_id,
            class_id=cls_id,
            orig_class_name=label,
            iscrowd=iscrowd,
        ))
        pts_pan_id[vert_idx] = inst_id * cls_sep + cls_id
        inst_id += 1

    return segments, pts_pan_id, local_cls_names


def build_spp100_luts(segments, local_cls_names, label_to_id, thing_ids):
    """Build scene-local -> spp100 lookup tables (shared by DSLR + iPhone paths).

    Returns:
        local_to_spp100_lut: (max_local+1,) uint8, scene-local class -> spp100 id
        inst_remap_lut: (max_inst+1,) uint16, raw instance -> contiguous thing id
        spp100_inst_to_sem: {contiguous_inst_id: spp100_class_id}
        max_local_cls, max_inst_id, new_inst_id (= n_thing_instances + 1)
    """
    # scene-local class ID -> spp100 class ID (top100 mapping)
    max_local_cls = max(local_cls_names.keys(), default=0)
    local_to_spp100_lut = np.zeros(max_local_cls + 1, dtype=np.uint8)
    for local_id, name in local_cls_names.items():
        if name in label_to_id:
            local_to_spp100_lut[local_id] = label_to_id[name]
        # else: stays 0 (void in spp100)

    # Instance remap: spp100 thing -> contiguous IDs, spp100 stuff/void -> 0
    instance_remap = {0: 0}
    spp100_inst_to_sem = {0: 0}
    new_inst_id = 1
    for seg in segments:
        spp100_cls = int(local_to_spp100_lut[seg['class_id']])
        if spp100_cls in thing_ids:
            instance_remap[seg['instance_id']] = new_inst_id
            spp100_inst_to_sem[new_inst_id] = spp100_cls
            new_inst_id += 1
        else:
            instance_remap[seg['instance_id']] = 0

    max_inst_id = max(instance_remap.keys(), default=0)
    inst_remap_lut = np.zeros(max_inst_id + 1, dtype=np.uint16)
    for old_id, new_id in instance_remap.items():
        inst_remap_lut[old_id] = new_id

    return (local_to_spp100_lut, inst_remap_lut, spp100_inst_to_sem,
            max_local_cls, max_inst_id, new_inst_id)


# ============================================================================
# Panoptic Decomposition
# ============================================================================

def decompose_panoptic(pan_mask_ids, cls_sep, min_area=50, valid_ids=None):
    """Split panoptic IDs into semantic and instance maps.

    panoptic_id = instance_id * cls_sep + class_id

    Returns:
        semantic: (H, W) uint16 — scene-local class IDs (0=void)
        instance: (H, W) uint16 — instance IDs (0=void)
    """
    semantic = np.zeros(pan_mask_ids.shape, dtype=np.uint16)
    instance = np.zeros(pan_mask_ids.shape, dtype=np.uint16)

    for pan_id in np.unique(pan_mask_ids):
        if pan_id == 0:
            continue
        mask = pan_mask_ids == pan_id
        if mask.sum() < min_area:
            continue
        if valid_ids is not None and pan_id not in valid_ids:
            continue
        semantic[mask] = pan_id % cls_sep
        instance[mask] = pan_id // cls_sep

    return semantic, instance


# ============================================================================
# Per-Frame I/O (ThreadPoolExecutor worker)
# ============================================================================

def _save_frame(frame_id, rgb, depth, raw_semantic, raw_instance,
                spp100_semantic, spp100_instance, pose, K_3x3, output_dirs):
    """Save all outputs for a single frame. depth=None skips the depth map."""
    Image.fromarray(rgb).save(osp.join(output_dirs['color'], f'{frame_id}.jpg'))
    if depth is not None:
        Image.fromarray(depth).save(osp.join(output_dirs['depth'], f'{frame_id}.png'))
    # Raw (every raw label, scene-local IDs)
    Image.fromarray(raw_semantic).save(
        osp.join(output_dirs['semantics'], f'{frame_id}.png'))
    Image.fromarray(raw_instance).save(
        osp.join(output_dirs['instance'], f'{frame_id}.png'))
    # SPP100 (top100-mapped)
    Image.fromarray(spp100_semantic).save(
        osp.join(output_dirs['spp100_semantics'], f'{frame_id}.png'))
    Image.fromarray(spp100_instance).save(
        osp.join(output_dirs['spp100_instance'], f'{frame_id}.png'))
    np.savetxt(osp.join(output_dirs['pose'], f'{frame_id}.txt'), pose, fmt='%.6f')
    np.savetxt(osp.join(output_dirs['intrinsic'], f'{frame_id}.txt'),
               K_3x3, fmt='%.6f')


# ============================================================================
# Per-Scene Pipeline
# ============================================================================

def process_single_scene(root, scene, label_to_id, thing_ids, stuff_ids,
                         crowd_cls, categories, output_dir,
                         num_workers=8, frames=None, out_longest_side=920):
    """Process a single ScanNet++ scene (DSLR).

    The images are resized so that the longer side is out_longest_side (0 =
    native) and the labels are rendered directly at that size with the
    intrinsics scaled accordingly.

    Returns True on success, False on failure.
    """
    data_dir = osp.join(root, 'data', scene)
    dir_dslr = osp.join(data_dir, 'dslr')
    dir_scans = osp.join(data_dir, 'scans')

    for d in [data_dir, dir_dslr, dir_scans]:
        if not osp.isdir(d):
            print(f"  ERROR: Directory not found: {d}")
            return False

    output_scene = osp.join(output_dir, scene)
    znear, zfar = 0.05, 20.0

    # --- Output directories ---
    output_dirs = {}
    for subdir in ['color', 'depth', 'pose', 'intrinsic',
                   'semantics', 'instance',
                   'spp100_semantics', 'spp100_instance']:
        d = osp.join(output_scene, subdir)
        os.makedirs(d, exist_ok=True)
        output_dirs[subdir] = d

    # --- Step 0a: Load SfM ---
    sfm_dir = osp.join(dir_dslr, 'colmap')
    img_infos = load_sfm(sfm_dir)
    # Sort by frame_id for deterministic ordering
    all_infos = sorted(img_infos.items(), key=lambda x: x[1]['frame_id'])
    print(f"  {len(all_infos)} DSLR frames")

    # --- Step 0b: Undistorted intrinsics (shared, computed once, from the first
    # frame of the whole scene so that a frame subset gets the same K) ---
    first_info = all_infos[0][1]
    orig_W, orig_H, K_undist = make_undistorted_intrinsics(first_info['intrinsics'])

    # Optional frame subset (the --subset eval list): keep frames whose
    # frame_id or its digits is a requested stem ('DSC04540' or '04540').
    if frames:
        kept, missing = select_frames([v['frame_id'] for _, v in all_infos], frames)
        if missing:
            print(f"  ERROR: {len(missing)} requested frames not in colmap: {missing[:10]}")
            return False
        kept = set(kept)
        all_infos = [(k, v) for k, v in all_infos if v['frame_id'] in kept]
        print(f"  frame subset: {len(all_infos)} of the scene's frames kept")

    # --- Step 0c: Load mesh + semantics ---
    ply_path = osp.join(dir_scans, 'mesh_aligned_0.05.ply')
    segments_path = osp.join(dir_scans, 'segments.json')
    annotations_path = osp.join(dir_scans, 'segments_anno.json')

    for p in [ply_path, segments_path, annotations_path]:
        if not osp.isfile(p):
            print(f"  ERROR: File not found: {p}")
            return False

    with open(ply_path, 'rb') as f:
        mesh_kwargs = trimesh.exchange.ply.load_ply(f)
    mesh_trimesh = trimesh.Trimesh(process=False, **mesh_kwargs)

    segments, pts_pan_id, local_cls_names = read_semantics(
        segments_path, annotations_path, crowd_cls,
        len(mesh_trimesh.vertices), CLS_SEP)

    # --- Step 0d: Build LUTs (shared with iPhone path) ---
    (local_to_spp100_lut, inst_remap_lut, spp100_inst_to_sem,
     max_local_cls, max_inst_id, new_inst_id) = build_spp100_luts(
        segments, local_cls_names, label_to_id, thing_ids)

    print(f"  Segments: {len(segments)} annotated ({len(local_cls_names)} classes), "
          f"{new_inst_id - 1} spp100 thing instances")

    # --- Step 0e: Setup pyrender scene ---
    valid_pan_ids = set(np.unique(pts_pan_id)) - {0}
    pts_pan_rgb = id2rgb(pts_pan_id)
    alpha = np.full((pts_pan_rgb.shape[0], 1), 255, dtype='uint8')
    mesh_trimesh.visual.vertex_colors = np.concatenate([pts_pan_rgb, alpha], axis=1)

    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)
    pyrender_scene = pyrender.Scene()
    pyrender_scene.add(mesh)
    renderer = pyrender.OffscreenRenderer(0, 0)

    # --- Step 1: Render + Save ---
    rgb_dir = osp.join(dir_dslr, 'resized_undistorted_images')
    mask_dir = osp.join(dir_dslr, 'resized_undistorted_masks')
    futures = []

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _, info in tqdm(all_infos, desc=f'  Rendering [{scene}]'):
                frame_id = info['frame_id']

                # Load undistorted RGB
                rgb = np.array(Image.open(osp.join(rgb_dir, info['path'])))

                # Load mask (all-valid fallback)
                mask_path = osp.join(mask_dir,
                                     os.path.splitext(info['path'])[0] + '.png')
                if osp.isfile(mask_path):
                    mask = np.array(Image.open(mask_path))
                else:
                    mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)

                # Adjust K if image resolution differs from COLMAP
                H_img, W_img = rgb.shape[:2]
                K = K_undist.copy()
                if W_img != orig_W or H_img != orig_H:
                    K = rescale_intrinsics(K, W_img / orig_W, H_img / orig_H)

                # Resize to the output resolution; render there with scaled K
                out_W, out_H = output_size(W_img, H_img, out_longest_side)
                if (out_W, out_H) != (W_img, H_img):
                    rgb = cv2.resize(rgb, (out_W, out_H), interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, (out_W, out_H),
                                      interpolation=cv2.INTER_NEAREST)
                    K = rescale_intrinsics(K, out_W / W_img, out_H / H_img)
                    H_img, W_img = out_H, out_W

                rgb[mask == 0] = 0  # black out anonymized regions

                # Render
                renderer.viewport_width = W_img
                renderer.viewport_height = H_img
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                camera = pyrender.camera.IntrinsicsCamera(
                    fx, fy, cx, cy, znear=znear, zfar=zfar)
                camera_node = pyrender_scene.add(
                    camera, pose=info['cam_to_world'] @ OPENGL_TO_OPENCV)

                flags = (pyrender.RenderFlags.SKIP_CULL_FACES
                         | pyrender.RenderFlags.SEG_VERT)
                pan_mask_rgb, depth = renderer.render(pyrender_scene, flags)
                pyrender_scene.remove_node(camera_node)

                # Depth → uint16 mm
                depth = (depth * 1000).astype(np.uint16)
                depth[mask < 255] = 0

                # Panoptic → raw semantic + instance (scene-local IDs)
                mask_f = (mask / 255.0)[..., None]
                pan_masked = (pan_mask_rgb * mask_f).astype(np.uint8)
                pan_ids = rgb2id(pan_masked)

                raw_semantic, raw_instance = decompose_panoptic(
                    pan_ids, CLS_SEP, min_area=MIN_INST_AREA,
                    valid_ids=valid_pan_ids)

                # Map to spp100 (top100): scene-local → spp100 class IDs
                spp100_semantic = local_to_spp100_lut[
                    np.minimum(raw_semantic.flatten(), max_local_cls)
                ].reshape(raw_semantic.shape)

                # Remap instance: spp100 thing→contiguous, stuff/void→0
                inst_flat = raw_instance.flatten().astype(np.int64)
                inst_flat = np.minimum(inst_flat, max_inst_id)
                spp100_instance = inst_remap_lut[inst_flat].reshape(
                    raw_instance.shape)

                # Submit parallel save
                fut = executor.submit(
                    _save_frame, frame_id, rgb, depth,
                    raw_semantic, raw_instance,
                    spp100_semantic, spp100_instance,
                    info['cam_to_world'], K,
                    output_dirs)
                futures.append(fut)

            # Wait for all saves and propagate errors
            for fut in futures:
                fut.result()
    finally:
        renderer.delete()

    # --- Step 2: Post-scene metadata ---

    # Shared intrinsic (4×4)
    K_4x4 = np.eye(4)
    K_4x4[:3, :3] = K  # intrinsics of the output resolution (last frame)
    np.savetxt(osp.join(output_scene, 'intrinsic', 'intrinsic_color.txt'),
               K_4x4, fmt='%.6f')

    # segmentation_data.pkl
    seg_data = {
        'class_system': 'spp100',
        'num_semantic_classes': 101,  # 0-100
        'fg_classes': sorted(thing_ids),
        'bg_classes': sorted(stuff_ids),
        'spp100_instance_to_semantic': spp100_inst_to_sem,
    }
    with open(osp.join(output_scene, 'segmentation_data.pkl'), 'wb') as f:
        pickle.dump(seg_data, f)

    # segments.json (includes local_cls_names for re-mapping)
    segments_export = {
        'segments': segments,
        'local_class_names': {str(k): v for k, v in local_cls_names.items()},
        'cls_sep': CLS_SEP,
    }
    with open(osp.join(output_scene, 'segments.json'), 'w') as f:
        json.dump(segments_export, f)

    # categories.json
    with open(osp.join(output_scene, 'categories.json'), 'w') as f:
        json.dump(categories, f)

    # scene_meta.json
    meta = {
        'scene_name': scene,
        'total_frames': len(all_infos),
        'num_thing_instances': new_inst_id - 1,
        'class_system': 'spp100',
        'num_semantic_classes': 101,
    }
    with open(osp.join(output_scene, 'scene_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Done: {len(all_infos)} frames → {output_scene}")
    return True


# ============================================================================
# Per-Scene Pipeline (iPhone: video frames + aligned_pose + undistortion)
# ============================================================================

def process_single_iphone_scene(root, scene, label_to_id, thing_ids, stuff_ids,
                                crowd_cls, categories, output_dir,
                                num_workers=8, frames=None, out_longest_side=960,
                                save_depth=True, frame_stride=1):
    """Process a single ScanNet++ scene from iPhone data.

    Shares all mesh/label/spp100/metadata logic with
    process_single_scene; only the image source differs: RGB+mask are decoded
    from rgb.mkv/rgb_mask.mkv, undistorted (OPENCV model) and resized, and posed
    with per-frame aligned_pose. Mesh is identical (scans/), so labels match.

    Returns True on success, False on failure.
    """
    data_dir = osp.join(root, 'data', scene)
    dir_iphone = osp.join(data_dir, 'iphone')
    dir_scans = osp.join(data_dir, 'scans')

    for d in [data_dir, dir_iphone, dir_scans]:
        if not osp.isdir(d):
            print(f"  ERROR: Directory not found: {d}")
            return False

    rgb_video = osp.join(dir_iphone, 'rgb.mkv')
    mask_video = osp.join(dir_iphone, 'rgb_mask.mkv')
    colmap_dir = osp.join(dir_iphone, 'colmap')
    pose_json = osp.join(dir_iphone, 'pose_intrinsic_imu.json')
    for p in [rgb_video, colmap_dir, pose_json]:
        if not osp.exists(p):
            print(f"  ERROR: iPhone file not found: {p}")
            return False

    output_scene = osp.join(output_dir, scene)
    znear, zfar = 0.05, 20.0

    # Interrupted/partial scene -> wipe and re-render from scratch.
    # main()'s skip_existing already short-circuits fully-completed scenes (those
    # have segmentation_data.pkl), so reaching here with an existing dir means it
    # was left incomplete by a killed/crashed worker. Removing it guarantees an
    # exact rebuild with no stale/truncated frames.
    marker = osp.join(output_scene, 'segmentation_data.pkl')
    if osp.isdir(output_scene) and not osp.isfile(marker):
        print(f"  Partial output found -> wiping for clean restart: {output_scene}")
        shutil.rmtree(output_scene)

    # --- Output directories ---
    subdirs = ['color', 'pose', 'intrinsic', 'semantics', 'instance',
               'spp100_semantics', 'spp100_instance']
    if save_depth:
        subdirs.append('depth')
    output_dirs = {}
    for subdir in subdirs:
        d = osp.join(output_scene, subdir)
        os.makedirs(d, exist_ok=True)
        output_dirs[subdir] = d

    # --- Step 0a: Camera model + per-frame aligned poses ---
    cam = load_iphone_camera(colmap_dir)
    pose_infos = load_iphone_poses(pose_json)
    if not pose_infos:
        print("  ERROR: no aligned poses found")
        return False

    # Optional frame subset (the --subset eval list). The video is still
    # decoded sequentially from frame 0 (grab() for the skipped frames), so a kept
    # frame is decoded exactly as in a full run.
    if frames:
        kept, missing = select_frames([fid for fid, _ in pose_infos], frames)
        if missing:
            print(f"  ERROR: {len(missing)} requested frames have no aligned_pose: {missing[:10]}")
            return False
        kept = set(kept)
        pose_infos = [(fid, c2w) for fid, c2w in pose_infos if fid in kept]
        print(f"  frame subset: {len(pose_infos)} frames kept")

    # frame_id 'frame_000300' -> video frame index 300
    def _vidx(fid):
        return int(''.join(ch for ch in fid if ch.isdigit()))
    pose_infos.sort(key=lambda x: _vidx(x[0]))
    if frame_stride > 1:
        pose_infos = pose_infos[::frame_stride]
    print(f"  {len(pose_infos)} iPhone frames ({cam['model']} "
          f"{cam['W']}x{cam['H']}, stride={frame_stride})")

    # --- Step 0b: Undistortion maps + output intrinsics ---
    undist = build_iphone_undistort(cam, out_longest_side=out_longest_side)
    out_W, out_H = undist['out_WH']
    K_out = undist['K_out']

    # --- Step 0c: Load mesh + semantics (identical to DSLR) ---
    ply_path = osp.join(dir_scans, 'mesh_aligned_0.05.ply')
    segments_path = osp.join(dir_scans, 'segments.json')
    annotations_path = osp.join(dir_scans, 'segments_anno.json')
    for p in [ply_path, segments_path, annotations_path]:
        if not osp.isfile(p):
            print(f"  ERROR: File not found: {p}")
            return False

    with open(ply_path, 'rb') as f:
        mesh_kwargs = trimesh.exchange.ply.load_ply(f)
    mesh_trimesh = trimesh.Trimesh(process=False, **mesh_kwargs)

    segments, pts_pan_id, local_cls_names = read_semantics(
        segments_path, annotations_path, crowd_cls,
        len(mesh_trimesh.vertices), CLS_SEP)

    # --- Step 0d: Build LUTs (shared with DSLR path) ---
    (local_to_spp100_lut, inst_remap_lut, spp100_inst_to_sem,
     max_local_cls, max_inst_id, new_inst_id) = build_spp100_luts(
        segments, local_cls_names, label_to_id, thing_ids)
    print(f"  Segments: {len(segments)} annotated ({len(local_cls_names)} classes), "
          f"{new_inst_id - 1} spp100 thing instances")

    # --- Step 0e: Setup pyrender scene (identical to DSLR) ---
    valid_pan_ids = set(np.unique(pts_pan_id)) - {0}
    pts_pan_rgb = id2rgb(pts_pan_id)
    alpha = np.full((pts_pan_rgb.shape[0], 1), 255, dtype='uint8')
    mesh_trimesh.visual.vertex_colors = np.concatenate([pts_pan_rgb, alpha], axis=1)

    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)
    pyrender_scene = pyrender.Scene()
    pyrender_scene.add(mesh)
    renderer = pyrender.OffscreenRenderer(out_W, out_H)

    fx, fy, cx, cy = K_out[0, 0], K_out[1, 1], K_out[0, 2], K_out[1, 2]
    flags = (pyrender.RenderFlags.SKIP_CULL_FACES
             | pyrender.RenderFlags.SEG_VERT)

    # --- Step 1: Decode video + render + save ---
    rgb_cap = cv2.VideoCapture(rgb_video)
    mask_cap = cv2.VideoCapture(mask_video)
    has_mask = mask_cap.isOpened()
    if not rgb_cap.isOpened():
        print(f"  ERROR: cannot open {rgb_video}")
        return False

    all_frame_ids = []
    futures = []
    prev_idx = -1
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for fid, c2w in tqdm(pose_infos, desc=f'  Rendering [{scene}]'):
                target_idx = _vidx(fid)
                if has_mask:
                    rgb, mask = read_iphone_frame(
                        rgb_cap, mask_cap, undist, prev_idx, target_idx)
                else:
                    # No anonymization mask stream: undistort RGB, mark validity.
                    for _ in range(target_idx - prev_idx - 1):
                        rgb_cap.grab()
                    ok_r, fr_rgb = rgb_cap.read()
                    if not ok_r:
                        rgb, mask = None, None
                    else:
                        rgb_u = cv2.remap(np.ascontiguousarray(fr_rgb[..., ::-1]),
                                          undist['map1'], undist['map2'],
                                          cv2.INTER_LINEAR)
                        mask_u = undist['valid'].copy()
                        if undist['out_WH'] != undist['full_WH']:
                            rgb_u = cv2.resize(rgb_u, undist['out_WH'],
                                               interpolation=cv2.INTER_AREA)
                            mask_u = cv2.resize(mask_u, undist['out_WH'],
                                                interpolation=cv2.INTER_NEAREST)
                        rgb, mask = np.ascontiguousarray(rgb_u), mask_u
                prev_idx = target_idx
                if rgb is None:
                    print(f"  WARNING: failed to read frame {fid}, skipping")
                    continue

                all_frame_ids.append(fid)

                rgb[mask == 0] = 0  # black out anonymized / invalid regions

                # Render at output resolution with output intrinsics
                renderer.viewport_width = out_W
                renderer.viewport_height = out_H
                camera = pyrender.camera.IntrinsicsCamera(
                    fx, fy, cx, cy, znear=znear, zfar=zfar)
                camera_node = pyrender_scene.add(
                    camera, pose=c2w @ OPENGL_TO_OPENCV)
                pan_mask_rgb, depth = renderer.render(pyrender_scene, flags)
                pyrender_scene.remove_node(camera_node)

                # Depth → uint16 mm (optional)
                if save_depth:
                    depth = (depth * 1000).astype(np.uint16)
                    depth[mask < 255] = 0
                else:
                    depth = None

                # Panoptic → raw semantic + instance (scene-local IDs)
                mask_f = (mask / 255.0)[..., None]
                pan_masked = (pan_mask_rgb * mask_f).astype(np.uint8)
                pan_ids = rgb2id(pan_masked)
                raw_semantic, raw_instance = decompose_panoptic(
                    pan_ids, CLS_SEP, min_area=MIN_INST_AREA,
                    valid_ids=valid_pan_ids)

                # Map to spp100 (top100)
                spp100_semantic = local_to_spp100_lut[
                    np.minimum(raw_semantic.flatten(), max_local_cls)
                ].reshape(raw_semantic.shape)
                inst_flat = np.minimum(
                    raw_instance.flatten().astype(np.int64), max_inst_id)
                spp100_instance = inst_remap_lut[inst_flat].reshape(
                    raw_instance.shape)

                fut = executor.submit(
                    _save_frame, fid, rgb, depth,
                    raw_semantic, raw_instance,
                    spp100_semantic, spp100_instance,
                    c2w, K_out, output_dirs)
                futures.append(fut)

            for fut in futures:
                fut.result()
    finally:
        renderer.delete()
        rgb_cap.release()
        if has_mask:
            mask_cap.release()

    # --- Step 2: Post-scene metadata ---
    K_4x4 = np.eye(4)
    K_4x4[:3, :3] = K_out
    np.savetxt(osp.join(output_scene, 'intrinsic', 'intrinsic_color.txt'),
               K_4x4, fmt='%.6f')

    seg_data = {
        'class_system': 'spp100',
        'num_semantic_classes': 101,
        'fg_classes': sorted(thing_ids),
        'bg_classes': sorted(stuff_ids),
        'spp100_instance_to_semantic': spp100_inst_to_sem,
    }
    with open(osp.join(output_scene, 'segmentation_data.pkl'), 'wb') as f:
        pickle.dump(seg_data, f)

    segments_export = {
        'segments': segments,
        'local_class_names': {str(k): v for k, v in local_cls_names.items()},
        'cls_sep': CLS_SEP,
    }
    with open(osp.join(output_scene, 'segments.json'), 'w') as f:
        json.dump(segments_export, f)

    with open(osp.join(output_scene, 'categories.json'), 'w') as f:
        json.dump(categories, f)

    meta = {
        'scene_name': scene,
        'source': 'iphone',
        'total_frames': len(all_frame_ids),
        'num_thing_instances': new_inst_id - 1,
        'class_system': 'spp100',
        'num_semantic_classes': 101,
        'out_resolution': [out_W, out_H],
        'pose_key': 'aligned_pose',
    }
    with open(osp.join(output_scene, 'scene_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Done: {len(all_frame_ids)} frames → {output_scene}")
    return True


# ============================================================================
# Scene List Resolution
# ============================================================================

def get_scene_list(root, args):
    """Resolve the scene list: --scene when given, otherwise the scenes of --subset."""
    if args.scene:
        return [args.scene]
    return scannetpp_scenes(root, args.subset, args.source)


def get_scene_frames(scene, args, eval_list):
    """Frame stems to render for ``scene`` (None = every frame).

    With --subset eval the listed frames of the evaluation list (a scene outside
    that list is an error); train/full render every frame."""
    if args.subset == 'eval':
        if scene not in eval_list:
            raise ValueError(f"{scene} is not in the {args.source} evaluation list; "
                             "use --subset train or --subset full for other scenes")
        return eval_list[scene]
    return None


# ============================================================================
# CLI
# ============================================================================

def get_parser():
    parser = argparse.ArgumentParser(
        description='ScanNet++ preprocessing (SPP100 label system, pyrender SEG_VERT)')
    parser.add_argument('--root_dir', required=True,
                        help='ScanNet++ dataset root')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory')
    parser.add_argument('--subset', choices=SUBSETS, default='eval',
                        help='eval (default): the scenes of the packaged evaluation '
                             'frame list of --source and only their listed frames; '
                             'train: splits/nvs_sem_train.txt, every frame; '
                             'full: nvs_sem_train + nvs_sem_val, every frame. '
                             '--scene replaces the scene choice.')
    parser.add_argument('--scene', type=str, default=None,
                        help='Single scene ID')
    parser.add_argument('--source', choices=['dslr', 'iphone'], default='dslr',
                        help='Image source: dslr (pre-undistorted JPGs) or iphone '
                             '(rgb.mkv + aligned_pose, undistorted on the fly)')
    parser.add_argument('--out_longest_side', type=int, default=None,
                        help='resize the frames so that the longer side equals this '
                             'and render the labels at that size (0 = native: DSLR '
                             '1752x1168 resized_undistorted_images, iPhone 1920x1440). '
                             'Default: 920 for dslr (920x613, the paper\'s DSLR ground '
                             'truth), 960 for iphone (960x720), for every --subset')
    parser.add_argument('--no_depth', action='store_true',
                        help='iPhone only: skip saving mesh-rendered depth '
                             '(about half the bytes per frame; depth is not evaluated)')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='iPhone only, --subset train or full: keep every Nth '
                             'video frame (1 = all; the video is 60 fps)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='parallel frame writers per scene')
    parser.add_argument('--skip_existing', action='store_true',
                        help='skip scenes whose segmentation_data.pkl (written last) exists')
    return parser


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.out_longest_side is None:
        args.out_longest_side = 920 if args.source == 'dslr' else 960
    if args.subset == 'eval' and args.frame_stride != 1:
        parser.error('--frame_stride must be 1 with --subset eval '
                     '(the listed frames are rendered as they are)')

    # Load class definitions (shared across all scenes)
    print("Loading SPP100 class definitions...")
    label_to_id, thing_ids, stuff_ids, crowd_cls, categories = \
        load_spp100_class_definitions(SPP100_CLASSES_CSV,
                                      osp.join(args.root_dir, 'metadata'))
    print(f"  {len(label_to_id)} labels mapped, "
          f"{len(thing_ids)} thing classes, "
          f"{len(stuff_ids)} stuff classes, "
          f"{len(crowd_cls)} crowd labels")

    # Resolve scene list
    scenes = get_scene_list(args.root_dir, args)
    eval_list = eval_frame_list(args.source) if args.subset == 'eval' else {}
    print(f"Processing {len(scenes)} scene(s), subset={args.subset}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save global categories
    with open(osp.join(args.output_dir, 'categories.json'), 'w') as f:
        json.dump(categories, f, indent=2)

    success = 0
    failed = []
    for i, scene in enumerate(scenes):
        print(f"[{i + 1}/{len(scenes)}] {scene}")

        if args.skip_existing:
            marker = osp.join(args.output_dir, scene, 'segmentation_data.pkl')
            if osp.isfile(marker):
                print("  Skipping (already exists)")
                success += 1
                continue

        try:
            frames = get_scene_frames(scene, args, eval_list)
            if args.source == 'iphone':
                ok = process_single_iphone_scene(
                    root=args.root_dir,
                    scene=scene,
                    label_to_id=label_to_id,
                    thing_ids=thing_ids,
                    stuff_ids=stuff_ids,
                    crowd_cls=crowd_cls,
                    categories=categories,
                    output_dir=args.output_dir,
                    num_workers=args.num_workers,
                    frames=frames,
                    out_longest_side=args.out_longest_side,
                    save_depth=not args.no_depth,
                    frame_stride=args.frame_stride,
                )
            else:
                ok = process_single_scene(
                    root=args.root_dir,
                    scene=scene,
                    label_to_id=label_to_id,
                    thing_ids=thing_ids,
                    stuff_ids=stuff_ids,
                    crowd_cls=crowd_cls,
                    categories=categories,
                    output_dir=args.output_dir,
                    num_workers=args.num_workers,
                    frames=frames,
                    out_longest_side=args.out_longest_side,
                )
            if ok:
                success += 1
            else:
                failed.append(scene)
        except Exception:
            traceback.print_exc()
            failed.append(scene)

    print(f"\n{'=' * 60}")
    print(f"Done: {success}/{len(scenes)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed scenes: {failed}")
        failed_path = osp.join(args.output_dir, 'failed_scenes.txt')
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed) + '\n')
        print(f"Written to {failed_path}")
