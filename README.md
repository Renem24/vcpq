# vcpq

Youngmin Lee\*, Byungha Ko\*, Guhnoo Yun & Dong Hwan Kim†
(\*equal contribution, †corresponding author)

Evaluation code of "Every View Counts: View-Consistent Panoptic Quality for
Multi-view Panoptic Segmentation" ([arXiv:2610.05911](https://arxiv.org/abs/2610.05911)). It contains the View-Consistent Panoptic
Quality (VC-PQ) metric, the metric it is compared with (PQ^scene), the
evaluation protocol (the lists of views that form every input set, the
ground-truth label systems, the exclusions) and the open-vocabulary prompt
lists. The settings of the compared methods
are documented in [docs/baselines.md](docs/baselines.md); their adapter code is
not part of this release.

Contents: [Quick start](#quick-start) · [Install](#install) ·
[Score your own predictions](#score-your-own-predictions) ·
[Datasets](#datasets) · [Protocol](#protocol) ·
[Compared methods](#compared-methods) · [Verification](#verification) ·
[License](#license) · [Citation](#citation) · [Ethics](#ethics)

## Quick start

1. Install the package ([Install](#install)).
2. Get access to ScanNet and ScanNet++ and download the files the
   preprocessing reads ([Datasets](#datasets)).
3. Prepare the ground truth of the evaluation scenes (50 ScanNet++ scenes,
   DSLR and iPhone, and 312 ScanNetv2 scenes) with the two scripts in
   `scripts/` ([Preparing the ground truth](#datasets)).
4. Check the pipeline: `vcpq-eval --predictor gt` scores the ground truth
   itself and prints VC-PQ 100.0 ([Run and score](#score-your-own-predictions)).
5. Wrap your method as a `ChunkPredictor` and run `vcpq-eval` once per cell:
   ScanNet++ DSLR, ScanNet++ iPhone and ScanNetv2, each at N = 5, 10, 20 and
   40 ([Score your own predictions](#score-your-own-predictions)).
6. For ScanNet++, pool the DSLR and the iPhone result of each N with
   `vcpq-pool` ([Pool the ScanNet++ cells](#pool-the-scannet-cells)), then
   read VC-PQ and PQ^scene from the result JSON ([Reading the
   result](#score-your-own-predictions)).

## Install

Python 3.10 or newer. The metric code needs only numpy, scipy and pillow.

```bash
git clone https://github.com/Renem24/vcpq.git && cd vcpq
uv sync --extra dev && source .venv/bin/activate   # or: pip install -e ".[dev]"
```

This installs the package `vcpq` and three commands: `vcpq-eval` (runs your
method over a cell and scores it, below), `vcpq-recompute` (scores saved
chunk files again) and `vcpq-pool` (pools the results of several cells, as
the paper's ScanNet++ tables pool DSLR and iPhone).

## Score your own predictions

`vcpq-eval` runs your method over the input sets of a cell (one dataset at one
input-set size N), scores the predictions as they come and writes one result
JSON. The scoring runs on the CPU and keeps one input set in memory at a
time; your predictor may use the GPU.

**Panoptic map format.** One `int32` array `[H, W]` per view, at the
resolution of the ground truth. Each pixel holds `class_id * 1000 +
instance_id`; `0` is void. Stuff classes use instance id 0 (`class_id * 1000`);
a thing pixel needs an instance id from 1 to 999. The instance id of an object
must be the same in every view of the chunk: this cross-view identity is what
VC-PQ measures. Class ids are those of the label system (`vcpq/data/`:
`spp100_classes.csv` for ScanNet++, `s20_200_to_20.csv` and the names in
`vcpq/label_systems.py` for ScanNetv2). `vcpq.panoptic.encode_panoptic`
builds a map from a class map and an instance map, and
`vcpq.panoptic.validate_pred_pan` checks one. `vcpq-eval` checks every
predicted map with it and stops with a `ValueError` that names the scene, the
frame and the id that breaks the format; the scorer itself does not check
what it reads.

**Data layout.** The ground truth is read from
`<data_root>/<dataset>/<scene>/`, with `color/` and the two label directories
of the label system (`spp100_instance/` and `spp100_semantics/`, or
`s20_instance/` and `s20_semantics/`), one PNG per frame stem. `<dataset>`
is one of three directory names: `scannetpp_spp100` (ScanNet++ DSLR),
`scannetpp_spp100_iphone` (ScanNet++ iPhone) and `scannetv2_s20`
(ScanNetv2). Use these names: the dataset name is part of every chunk's
scene tag (`<dataset>__<scene>`). A frame list with no scene under
`<data_root>/<dataset>` raises `ValueError`. How to produce these
directories from ScanNet++ and ScanNet is in [Datasets](#datasets).

**Your predictor.** A `ChunkPredictor` gets the image paths, the frame stems
and the ground-truth size `(H, W)` of one input set (a "chunk") and returns
one map per view. Put it in a module with a function `build(args)` that
returns it. The example predicts "wall" everywhere; replace `predict` with
your model. If your predictions are already saved, `predict` can load them
from the stems instead.

<!-- readme-example:python -->
```python
# my_method.py
import numpy as np

from vcpq.protocol.runner import ChunkPrediction


class MyPredictor:
    """Implements vcpq.protocol.runner.ChunkPredictor."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint   # load your model here

    def predict(self, image_paths, stems, gt_size):
        h, w = gt_size
        # One int32 [h, w] map per view, class_id * 1000 + instance_id, 0 = void.
        # Class 1 of spp100 is the stuff class "wall".
        pred_pans = [np.full((h, w), 1000, dtype=np.int32) for _ in stems]
        return ChunkPrediction(pred_pans=pred_pans)


def add_args(parser):
    """Optional: options of your method, parsed together with those of vcpq-eval."""
    parser.add_argument("--checkpoint", default=None)


def build(args):
    """Called once by vcpq-eval --predictor my_method:build."""
    return MyPredictor(args.checkpoint)
```

**Run and score.** From the directory that holds `my_method.py`:

<!-- readme-example:shell -->
```bash
vcpq-eval --predictor my_method:build \
    --data_root <data_root> --dataset scannetpp_spp100 --label_system spp100 \
    --frame_list scannetpp_spp100_dslr --n 10 \
    --out <out>/results.json
```

The command cuts every scene of the frame list into the paper's input sets
of `--n` views, calls the predictor once per input set, loads the ground
truth, scores the chunk and moves on. It prints a summary (the paper's VC-PQ
with VC-SQ, VC-RQ and the error decomposition, then the paper's PQ^scene) and
writes `results.json`. `--label_system` is `spp100` or `scannetv2_s20`;
`--frame_list` is a packaged list name ([Protocol](#protocol)) or a path to a
JSON of the same form; `--scenes` restricts the run to some scenes. An
existing `--out` file stops the command unless `--overwrite` is given.

`my_method:build` is an importable Python path (the current directory is on
the path) to a function `build(args) -> ChunkPredictor`. If the same module
defines `add_args(parser)`, it is called first, so the module can add its own
options, which then show up in `vcpq-eval --predictor my_method:build --help`.
`--predictor gt` returns the ground truth as the prediction, a check of the
whole pipeline (VC-PQ is then 1.0). A missing module, a missing function or a
`build` that does not return an object with a `predict` method stops the
command with a message naming the `--predictor` value. From Python,
`vcpq.protocol.runner.evaluate_cell(predictor, spec, cfg)` does the same run
and returns the result as a dict (`spec` a `CellSpec`, `cfg` a
`vcpq.recompute.RecomputeConfig`).

**Reading the result.** The two metrics of the paper are read from
`results.json` as follows (for one cell; the paper's ScanNet++ tables pool the
DSLR and the iPhone cell of a method, see [Pool the ScanNet++
cells](#pool-the-scannet-cells)):

| metric | key in `results.json` |
|---|---|
| VC-PQ (pooled class average, robust filtering off) | `vcpq["eta100_gamma0.0"]["class_avg"]["robust_filter_off"]["total"]["vcpq"]` |
| VC-SQ, VC-RQ (same average) | `...["total"]["vcsq"]`, `...["total"]["vcrq"]` (same path as VC-PQ) |
| PQ^scene (per-scene mean, robust filtering on) | `pq_scene["per_scene"]["robust_filter_on"]` |

By default a run computes the metrics as the paper defines them, with
class-wise matching: VC-PQ (with VC-SQ, VC-RQ and the error decomposition)
with robust filtering off, and PQ^scene with robust filtering on. The other
variants are options (below). As in panoptic quality, the class averages of
VC-SQ and VC-RQ do not multiply to the class average of VC-PQ: each class has
`vcpq = vcsq * vcrq`, and the three are averaged over the classes separately.
`things` and `stuff` sit next to `total`; `n_scenes`, `n_chunks` and
`n_void_scenes` count what was scored, and `include_void_scenes` and
`robust_filter_threshold` record the first two options below. Options that
change the result:

- `--include_void_scenes` is off by default. A scene whose ground truth is
  entirely void in every chunk is then left out of the per-scene means and
  counted in `n_void_scenes` of the output. The paper's cells were recomputed
  with the option on, where such a scene enters the per-scene means with value
  0 (as in the paper's `recompute.json` files). Pooled class averages are
  unaffected either way.
- `--robust_filter_threshold` is the robust-filtering threshold of VC-PQ's
  filtered variant (`--both_robust_filter`; default 0.005); `0` turns it off.
  PQ^scene always uses 0.005.
- `--both_robust_filter` also computes the other robust-filtering state of
  each metric: VC-PQ with the filter on (`class_avg["robust_filter_on"]`,
  `per_scene["robust_filter_on"]`, ...) and PQ^scene with it off
  (`per_scene["robust_filter_off"]`, ...).
- `--class_agnostic` adds a `class_agnostic` block to both metrics, for every
  computed state: matching with every segment relabelled to one class (keys
  `robust_filter_off` / `robust_filter_on`) and the same on thing classes
  only (`things_robust_filter_off` / `things_robust_filter_on`).
- `--etas` is the visibility threshold in pixels (paper: 100).
- `--gammas` is the view-weighting exponent (0 is VC-PQ, the default; 1
  weights views by area). Several values give one block per value.

**Error decomposition.** The scorer also splits 1 - VC-PQ into the three
terms of the paper's decomposition (Eq. 10). For a class with
`D = tp + fp/2 + fn/2 > 0` and the matching threshold tau = 0.5:

- `acct_mask` (epsilon_IoU): the mean-IoU shortfall of the matched pairs,
  the sum of `1 - IoU-bar`, plus half of every unmatched ground-truth or
  predicted instance that has no same-class candidate with mean IoU
  (IoU-bar) above tau; divided by `D`.
- `acct_vc` (epsilon_VC): the view-consistency shortfall of the matched
  pairs, the sum of `IoU-bar * (1 - VC)`, plus half of every unmatched
  instance that has such a candidate but none whose VC is above tau; divided
  by `D`.
- `acct_thr` (epsilon_q): half of every unmatched instance that has a
  candidate whose mean IoU and VC both pass tau but whose product q does not;
  divided by `D`.

For every class with `tp + fp + fn > 0`, `vcpq + acct_mask + acct_vc +
acct_thr = 1`; at gamma = 0 (VC-PQ, q = IoU-bar * VC) these are the paper's
three terms. The class averages `total`, `things` and `stuff` are means over
the classes with `tp + fp + fn > 0`, the classes VC-PQ averages over, so they
sum to 1 as well. `vcpq-eval`, `vcpq-recompute` and `vcpq-pool` print the
three terms of `total` (robust filtering off) in their summary. In `results.json` the three keys sit next to `vcpq` in
`vcpq["eta100_gamma0.0"]["class_avg"]["robust_filter_off"]["total"]` (and `things`,
`stuff`, and `robust_filter_on` with `--both_robust_filter`) and, per class, in
`vcpq["eta100_gamma0.0"]["per_class"]["robust_filter_off"]["<class id>"]`;
`vcpq["eta100_gamma0.0"]["accounting"]` repeats the `total` values. In
Python, `VCPQAccumulator.compute_metrics()` returns the same keys in
`per_class[class_id]` and in `total`, `things` and `stuff`.

<!-- readme-example:python -->
```python
import json

block = json.load(open("<out>/results.json"))["vcpq"]["eta100_gamma0.0"]
for group in ("total", "things", "stuff"):
    m = block["class_avg"]["robust_filter_off"][group]
    print(group, *(f"{k} {100 * m[k]:.1f}" for k in ("vcpq", "acct_mask", "acct_vc", "acct_thr")))
```

### Save chunks and rescore later

Add `--dump_dir <out>/raw_chunks` to the `vcpq-eval` command to also save
every chunk: one compressed `.npz` per input set
(`<dataset>__<scene>/c0000.npz`, ...), holding the predicted and the
ground-truth panoptic maps of its views. `vcpq-recompute` scores such a
directory again without the model, which helps when you want other
`--etas`, `--gammas`, `--robust_filter_threshold` or `--include_void_scenes`
values or the `--both_robust_filter` / `--class_agnostic` variants, or want
to score on another machine later. With the same options its numbers equal
those of `vcpq-eval`, and `raw_dir` in both JSON files is the absolute path
of the directory (`null` in `results.json` without `--dump_dir`). Chunk files
of the cell that already exist in `--dump_dir` stop `vcpq-eval` before the
first prediction unless `--overwrite` is given.

<!-- readme-example:shell -->
```bash
vcpq-recompute <out>/raw_chunks --label_system spp100 \
    --gammas 0.0 0.2 0.4 0.6 0.8 1.0 --etas 100 --workers 4 \
    --out <out>/recompute.json
```

`recompute.json` has the keys of `results.json`. `--workers` above 1 scores
scenes in parallel and also writes the per-chunk records
(`recompute_chunks.json.gz`, next to `--out`: the raw counts of every chunk);
with `--workers 1` (the default) no per-chunk records are written.
`--workers` above 1 uses `fork` multiprocessing, which needs Linux; elsewhere
use `--workers 1`. Chunk files can also be written without `vcpq-eval`:
`vcpq.protocol.dump.write_chunk_npz` writes one from maps you already have
(pass `label=` so that the maps are checked), for the scenes and input sets
of `vcpq.protocol.runner.cell_scenes`, `usable_stems` and
`vcpq.protocol.input_sets.make_input_sets`.

### Pool the ScanNet++ cells

The paper's ScanNet++ columns pool the DSLR and the iPhone cell of a method:
99 scene-runs (see Exclusions under [Protocol](#protocol)). Score the two
cells with the same options, then pool their result JSONs:

```bash
vcpq-pool <out>/dslr.json <out>/iphone.json --out <out>/scannetpp.json
```

`scannetpp.json` has the keys of `results.json` except `raw_dir`, the
`accounting` repeat and the class-agnostic blocks, so the two metrics and the
error decomposition are read from the same keys, and `inputs` lists the
pooled files. Class averages (VC-PQ with its three error terms, the pooled
class average of PQ^scene) are computed again from the summed per-class
counts of the inputs; per-scene means are the mean over the scenes of all
inputs. The result equals scoring the chunks of both cells together. For
ScanNet++ the pooled `n_scenes` is 100 and `n_void_scenes` 1: the all-void
iPhone scene is counted but, with the default, left out of the per-scene
means, which therefore average 99 scene-runs. The inputs must share the
label system, `--include_void_scenes`, `--robust_filter_threshold`,
`--both_robust_filter` and the `--etas` and `--gammas` values; a result JSON
without the `include_void_scenes` or `robust_filter_threshold` key (written by
an older version of this code) is refused. From Python,
`vcpq.pool.pool_results([result_dslr, result_iphone])` returns the pooled
dict.

## Datasets

This repository contains no images or labels. The ground truth is prepared
from your own copies of ScanNet and ScanNet++, each under its terms of use,
with the two preprocessing scripts in `scripts/`.

**ScanNetv2.**
- Access: fill out the agreement to the ScanNet Terms of Use with an
  institutional email address and send it to scannet@googlegroups.com, as the
  README of <https://github.com/ScanNet/ScanNet> describes. The download
  script `download-scannet.py` is sent to you after that; it is not in the
  ScanNet GitHub repository.
- Files: for each scene, `<scene>.sens`, `<scene>_2d-label-filt.zip` and
  `<scene>_2d-instance-filt.zip`. The download script puts them under
  `<scannet_root>/scans/<scene>/`; its options include `-o <scannet_root>`,
  `--type` (one file type, for example `--type .sens` or
  `--type _2d-label-filt.zip`) and `--id <scene>` (one scene).
- `scannetv2-labels.combined.tsv` (the download script's `--label_map`) is not read by the
  preprocessing. The raw id to ScanNet200 table it needs ships as
  `vcpq/data/raw_to_s20_200.csv` (sources under [License](#license)).
- `--subset train` also needs the official training list
  `Tasks/Benchmark/scannetv2_train.txt` of the ScanNet GitHub repository,
  passed with `--split_file`. It is not shipped here.

**ScanNet++.**
- Access: create an account on the ScanNet++ website
  (<https://scannetpp.mlsg.cit.tum.de/scannetpp/>), log in and submit an application.
  After it is approved you receive a personal token and the download script
  from your download page. The name and the options of that script are not
  verified here.
- Files read, relative to `<scannetpp_root>/data/<scene>/`:
  DSLR `dslr/resized_undistorted_images/`, `dslr/resized_undistorted_masks/`
  and `dslr/colmap/`; iPhone `iphone/rgb.mkv`, `iphone/rgb_mask.mkv`,
  `iphone/colmap/` and `iphone/pose_intrinsic_imu.json`; for both,
  `scans/mesh_aligned_0.05.ply`, `scans/segments.json` and
  `scans/segments_anno.json`. Also read: `<scannetpp_root>/metadata/semantic_classes.txt`,
  `metadata/semantic_benchmark/` (`top100.txt`, `top100_instance.txt`,
  `map_benchmark.csv`) and, for `--subset train` or `full`,
  `splits/nvs_sem_train.txt` and `splits/nvs_sem_val.txt`.
- The undistorted DSLR images are part of the release: the ScanNet++
  documentation lists `resized_undistorted_images` ("Undistorted DSLR images
  with the same resolution as the resized images") and
  `resized_undistorted_masks` in the DSLR folder, and the dataset page lists
  undistorted DSLR images among its additions. The toolkit's own undistortion
  (`python -m dslr.undistort dslr/configs/undistort.yml` in
  <https://github.com/scannetpp/scannetpp>) writes to `undistorted_images/` by
  default; whether its output equals the released images is not verified.
- The raw label to `spp100` class mapping is built at run time from the
  `metadata/` files above (`scripts/spp100_mapping.py`); only the class table
  `vcpq/data/spp100_classes.csv` ships.

**Preparing the ground truth.** The two scripts need a few more packages,
among them the pyrender fork with `SEG_VERT`; the ScanNet++ script renders
offscreen with EGL, so it needs a GPU driver with EGL. Install them into the
environment of [Install](#install) (`uv pip install` instead of `pip install`
in a `uv sync` environment):

```bash
pip install "opencv-python==4.11.0.86" trimesh scipy imageio pypng tqdm pillow \
    "git+https://github.com/lojzezust/pyrender.git"
```

OpenCV is pinned because a newer OpenCV (5.0) changes the iPhone color images
slightly (labels unchanged). Then run from the repository root:

```bash
# ScanNet++ DSLR -> <data_root>/scannetpp_spp100 (920 x 613 by default)
PYOPENGL_PLATFORM=egl python scripts/preprocess_scannetpp.py \
    --root_dir <scannetpp_root> --subset eval --source dslr \
    --output_dir <data_root>/scannetpp_spp100

# ScanNet++ iPhone -> <data_root>/scannetpp_spp100_iphone
PYOPENGL_PLATFORM=egl python scripts/preprocess_scannetpp.py \
    --root_dir <scannetpp_root> --subset eval --source iphone \
    --out_longest_side 960 --output_dir <data_root>/scannetpp_spp100_iphone

# ScanNetv2 -> <data_root>/scannetv2_s20
python scripts/preprocess_scannetv2_s20.py \
    --scans_root <scannet_root>/scans --subset eval \
    --output_dir <data_root>/scannetv2_s20
```

`--out_longest_side` sets the output resolution of the ScanNet++ script: the
frames are resized so that the longer side has this length and the labels are
rendered at that size. The default is 920 for `--source dslr` (920 x 613) and
960 for `--source iphone` (960 x 720), the resolutions of the paper's ground
truth, for every `--subset`; 0 keeps the native resolution. DSLR frame stems
are the digits of the image name (`04458` for `DSC04458.JPG`), as in the
frame lists.

`--subset` chooses scenes and frames. `eval` (the default) takes the scenes of
the packaged frame lists ([Protocol](#protocol): 50 DSLR, 50 iPhone and 312
ScanNetv2 scenes) and writes only their 120 listed frames each. A listed frame
comes out as it would in a run over every frame: the ScanNet++ labels come
from the mesh, the DSLR intrinsics from the first image of the scene, the
iPhone video is still decoded from its first frame, and the ScanNetv2
instance numbering counts every label frame of the scene. `train` takes the
official training split (`nvs_sem_train`, or `--split_file` for ScanNetv2) and
`full` every scene (ScanNet++ `nvs_sem_train` and `nvs_sem_val`; every
`scene*` directory under `--scans_root`); both write every frame. For
ScanNet++ iPhone with `train` or `full`, `--frame_stride N` keeps every Nth
frame of the 60 fps video and `--no_depth` skips the depth maps, to save disk
space; the evaluation needs neither.
`--scene` (ScanNet++) and `--sens_root` (one ScanNetv2 scene directory)
replace the scene choice; with `--subset eval` the scene must be in the list.
With `--subset eval` the ScanNet++ `scene_meta.json` counts only the written
frames in `total_frames`; the ScanNetv2 one describes the whole scene.

The evaluation reads `<data_root>/<dataset>/<scene>/` with the directory
names `scannetpp_spp100`, `scannetpp_spp100_iphone` and `scannetv2_s20`
([Data layout](#score-your-own-predictions)):

```
<data_root>/scannetpp_spp100/<scene>/          color/ spp100_semantics/ spp100_instance/
<data_root>/scannetpp_spp100_iphone/<scene>/   color/ spp100_semantics/ spp100_instance/
<data_root>/scannetv2_s20/<scene>/             color/ s20_semantics/ s20_instance/
```

Each scene directory also holds `depth/`, `pose/`, `intrinsic/`,
`segmentation_data.pkl` and `scene_meta.json`, and the ScanNet++ ones
`semantics/`, `instance/` (scene-local raw labels), `segments.json` and
`categories.json`. The evaluation reads only `color/` and the two label
directories.

Measured for one scene with `--subset eval` (120 frames) on one machine
with an RTX 3090 Ti (EGL rendering):

| scene | wall time | peak memory (RSS) | output |
|---|---|---|---|
| ScanNet++ DSLR `09c1414f1b` | 30 s | 5.0 GiB | 38 MB |
| ScanNet++ iPhone `09c1414f1b` | 1 min 09 s | 5.2 GiB | 40 MB |
| ScanNetv2 `scene0011_00` | 44 s | 1.3 GiB | 41 MB |

**Reproduction status.** The iPhone and ScanNetv2 ground truth made this way
is byte-identical to the paper's on the scenes checked
([Verification](#verification)). The DSLR one is close but not identical. On
`09c1414f1b` (120 listed frames, rendered at 920 x 613 with digit stems),
`pose/` and the intrinsics are byte-identical to the paper's files, and the
label maps agree except for 327 of 67.7 M semantic pixels (in 72 frames) and
95 instance pixels (in 41 frames). All of these pixels lie on segment
boundaries, and the output is the same across runs. The cause is not
established: the paper's DSLR render was made in an environment that is no
longer available. The color images differ slightly (mean absolute difference
about 1.6 per pixel) because the resize method of that render is unknown. The
paper's numbers were computed on the paper's ground-truth files.

## Protocol

**Input sets.** Every scene contributes a fixed list of 120 frames (stems).
`vcpq/protocol/lists/` packages the three scene to frame-stem lists (sampling
seed 42): `scannetpp_spp100_dslr_120f_seed42.json` (50 scenes), the matching
`..._iphone_120f_seed42.json` (50 scenes) and `scannetv2_s20_120f_seed42.json`
(312 scenes). They are loaded by the names `scannetpp_spp100_dslr`,
`scannetpp_spp100_iphone` and `scannetv2_s20`. A scene's frames are cut into
input sets of N views (N = 5, 10, 20, 40): the frame indices are shuffled
with `random.Random(42)`, sliced into consecutive groups of N, and each group
is sorted back into frame order (`vcpq.protocol.input_sets.make_input_sets`).
The same sets, written out, are in `input_sets_<list>_N{5,10,20,40}.json`, so
the views of every input set can be read without running any code. The
iPhone files leave out the scene `d755b3d9d8` (Exclusions below; 588 sets at
N = 10 instead of 600). A frame of the list without ground truth on disk is skipped with a
warning. For a scene with more frames than N whose frame count N does not
divide (possible with your own list or missing ground truth; never with the
packaged lists), the last input set holds the remainder, as in the paper's
evaluation, and a warning is logged.

The packaged lists are canonical; use them as they are. The two ScanNet++
lists were drawn with `random.Random(42)` from the frames of each scene whose
ground truth is not entirely void (iPhone: every fourth video frame as a
candidate). The ScanNetv2 list has no recorded generator.

**Metrics.** VC-PQ is the pooled class average with robust filtering off
and void-only classes excluded; PQ^scene is the per-scene mean of the scene's
class average with robust filtering on (keys in the result JSON: [Score
your own predictions](#score-your-own-predictions)). The
visibility threshold is eta = 100 pixels at the evaluation resolution (below),
an absolute pixel count and not a fraction of the image. VC-PQ's view
weighting is gamma = 0 (equal weight per view) unless a table says otherwise.
*Robust filtering* is the robust-class filter of Panoptic Lifting: within a
chunk, a semantic class that occupies less than 0.5 % (`robust_filter_threshold =
0.005`) of the non-void ground-truth pixels, in the prediction or the ground
truth, is set to void in both maps, in every view, before scoring
(`vcpq/robust_filter.py`).

**Exclusions.**
- The ScanNet++ iPhone scene `d755b3d9d8` has broken poses from the data
  provider, and its ground truth is entirely void. It stays in the packaged
  iPhone frame list (a faithful record of what was sampled), and `vcpq-eval`
  scores it like any other scene, but it adds nothing to the paper's numbers:
  with no non-void ground truth it has no true positive or false negative,
  its predictions lie entirely on void and are not counted as false
  positives, so the class averages are unchanged; and with the default
  (`--include_void_scenes` off) a scene whose ground truth is entirely void
  is left out of the per-scene means. The paper's ScanNet++ tables pool the
  DSLR and the iPhone run: 50 + 49 = 99 scene-runs (`vcpq-pool`, [Pool the
  ScanNet++ cells](#pool-the-scannet-cells)).
- `vcpq-recompute --include_void_scenes`: see [Score your own
  predictions](#score-your-own-predictions). Off by default; the paper-cell
  comparison in [Verification](#verification) runs with it on.

**Ground truth.**

| dataset | label system name | classes | evaluation resolution (W x H) |
|---|---|---|---|
| ScanNet++ DSLR | `spp100` | official top-100 benchmark classes (84 thing, 16 stuff) | 920 x 613 |
| ScanNet++ iPhone | `spp100` | same | 960 x 720 |
| ScanNetv2 | `scannetv2_s20` | 20 classes (18 thing, 2 stuff), remapped from ScanNet200 ids | 1296 x 968 |

Predictions are compared at these resolutions (`gt_size` is passed to the
predictor). The label system of ScanNetv2 is named `scannetv2_s20`; the
ScanNet++ one `spp100`.

The ground truth is rendered or extracted from the datasets by two scripts
that are copies of the ones used for the paper (preparation and
reproduction status: [Datasets](#datasets)). Neither redistributes data;
you need your own access to ScanNet++
(<https://github.com/scannetpp/scannetpp>) and ScanNet
(<https://github.com/ScanNet/ScanNet>) under their terms of use.

- `scripts/preprocess_scannetpp.py` renders the panoptic labels of the
  ScanNet++ mesh annotations into each DSLR or iPhone frame (pyrender with the
  `SEG_VERT` flag, EGL; it needs the ScanNet++ data layout, `--source dslr` or
  `--source iphone`, and builds the raw label to `spp100` mapping from the
  dataset's `metadata/`).
  It writes `<output_dir>/<scene_id>/`, so point `--output_dir` at
  `<data_root>/scannetpp_spp100` for `--source dslr` and at
  `<data_root>/scannetpp_spp100_iphone` for `--source iphone`.
  The ground truth includes PanSt3R's 50-pixel segment filter: in each frame,
  a segment (an instance of a class) with fewer than 50 pixels is set to void,
  and so is a segment whose id is not in the valid id set of the scene
  (`decompose_panoptic`, `MIN_INST_AREA`), as in PanSt3R. The filter follows
  PanSt3R's `tools/preprocess_scannetpp.py` ("Filter noisy segments"); it is
  in PanSt3R's code and not described in the PanSt3R paper.
- `scripts/preprocess_scannetv2_s20.py` reads a ScanNet scene (`.sens` file
  and the filtered label and instance zips), maps the raw ids to ScanNet200
  and renumbers the thing instances. It writes `<output_dir>/<scene_name>/`;
  point `--output_dir` at `<data_root>/scannetv2_s20`. It has no segment-size filter. The
  `.sens` file is read with `scripts/scannet_sensordata.py`, a Python 3 port
  of the reader of the ScanNet toolkit (the official file is Python 2); the
  20-class remapping happens at load time (`vcpq/data/s20_200_to_20.csv`).

Both scripts are the ones used for the paper's ground truth, with two
outputs the evaluation does not read (top-K covisibility ranking and mask
visualisations) and some command-line options removed;
the ScanNet one also ships the Python 3 `.sens` reader above. They read the
class tables from `vcpq/data/`; the ScanNet++ one builds its raw label
mapping from the dataset's `metadata/`. Both need numpy, pillow and tqdm.
`preprocess_scannetpp.py` also imports opencv (`cv2`), scipy, trimesh and the
pyrender fork with `SEG_VERT` (named in the script).
`preprocess_scannetv2_s20.py` also imports imageio and pypng (`png`); its
`.sens` reader needs only numpy and imageio. The tests run the ScanNetv2 label
pass on synthetic zips (`tests/test_preprocess_subsets.py`, imageio and pypng
stubbed) and the `--subset` selection; they do not render or read `.sens`
files.

## Compared methods

The compared methods are PanSt3R (v1 and v2), SegVGGT, EPS3D and IGGT.
[docs/baselines.md](docs/baselines.md) records what was run for each: the
upstream repository and commit, checkpoint names, input resolution,
post-processing settings and thresholds, the open-vocabulary prompts (built by
`vcpq.vocab.build_vocab`), the structure of the multi-stage EPS3D and IGGT
pipelines, determinism, the largest N that fit in memory, and the one known
discrepancy between the paper's text and the code (the SegVGGT top-600
sentence). The adapter code that produced the dumps is not part of this
release. To run one of these methods, or your own, write a `ChunkPredictor`
and score it with `vcpq-eval --predictor module:factory` as above.

## Verification

The paper's result files use the older key names `rf_off`, `rf_on` and `th_`
(now `robust_filter_off`, `robust_filter_on` and `things_`); the comparisons
below map them.

- VC-PQ and PQ^scene computed by `vcpq-recompute` are bit-identical to
  those of the evaluation code used for the paper's numbers, chunk by chunk,
  on one scene (two input sets) per label system.
- Full cell: PanSt3R v2, ScanNet++ DSLR, N = 10 (50 scenes, 600 input sets,
  gamma 0.0 to 1.0 in steps of 0.2), recomputed with `vcpq-recompute
  --include_void_scenes`: the counts and the VC-PQ and PQ^scene blocks have 0
  differing leaves against the paper's `recompute.json` (ignoring `raw_dir`,
  `label_system`, `chunks_file` and the block of an AP number that this
  release does not compute, see [docs/baselines.md](docs/baselines.md)). The
  per-chunk records agree
  for 600/600 chunks, and their 1200 VC-PQ and PQ^scene blocks are
  identical.
- Cells of the paper's tables, recomputed the same way with a pre-release
  version of this code: 31 of the 34 cells have 0 differing leaves
  (the 24 ScanNet++ cells of PanSt3R v1, PanSt3R v2, EPS3D and IGGT; on
  ScanNetv2, PanSt3R v2 at N = 5, 10, 20 and 40, EPS3D at N = 10 and SegVGGT
  at N = 5 and 10). The other three (SegVGGT on ScanNetv2 at N = 20 and 40,
  EPS3D on ScanNetv2 at N = 5) were not recomputed.
- The released code: PanSt3R v2 at N = 10 on ScanNet++ DSLR and iPhone,
  recomputed with gamma 0.0 to 1.0 in steps of 0.2, once with
  `--both_robust_filter --class_agnostic` and once with the default variants.
  With both options, the VC-PQ blocks (all six gamma values, with the error
  decomposition and the class-agnostic blocks) and the PQ^scene block have 0
  differing leaves against the paper's `recompute.json`, except the iPhone
  per-scene means, which equal the paper's times 50/49: the default leaves
  the all-void scene `d755b3d9d8` out, where the paper's file let it enter as
  0. The default run equals, bit for bit, the same variants of the run with
  both options. `vcpq-pool` on the two default runs gives the paper's
  ScanNet++ numbers for this method and N: PQ^scene 51.3 and VC-PQ 22.2
  (Table 1), and, on thing classes, VC-PQ 22.3 with error terms 67.5, 7.2 and
  3.0 (Table 3).
- Ground-truth preprocessing (`--subset eval`, 120 listed frames per scene,
  compared file by file, then pixel by pixel): ScanNet++ iPhone `09c1414f1b`
  and ScanNetv2 `scene0011_00` are byte-identical to the paper's ground truth
  in the label directories, `color/`, `pose/` and `intrinsic/` (ScanNetv2 also
  `depth/`, `segmentation_data.pkl`, `scene_meta.json`). The iPhone reference
  `depth/` was rewritten later and differs; depth is not evaluated. ScanNet++
  DSLR `09c1414f1b`: `pose/` and `intrinsic/` are
  byte-identical; the label maps differ in 327 semantic and 95 instance pixels
  out of 67.7 M, all on segment boundaries (Reproduction status under
  [Datasets](#datasets)). The raw label to `spp100` mapping built from the
  ScanNet++ metadata equals the table the paper used (2878 rows).

Tests: `python -m pytest -q` runs the tests. They need only this repository
and run on synthetic label maps generated in the test code; the repository
ships no label maps. `tests/test_spp100_mapping.py` also rebuilds the raw
label to `spp100` mapping from your ScanNet++ copy when
`VCPQ_SCANNETPP_ROOT` points at its root, and skips that check otherwise. The
README's Python examples are executed by `tests/test_readme_example.py`.

## License

This repository is licensed under the MIT License (see [`LICENSE`](LICENSE)).

Exceptions: `scripts/preprocess_scannetpp.py` (derived from PanSt3R / DUSt3R,
CC BY-NC-SA 4.0, non-commercial use only, ShareAlike) and
`scripts/preprocess_scannetv2_s20.py` (derived from Panoptic Lifting, CC BY-NC-SA
4.0) stay under CC BY-NC-SA 4.0, as stated in their headers.
For users this means: the metric and scoring code (`vcpq/`) has no such
restriction; only regenerating the ground truth with those two scripts is
limited to non-commercial use.

Third-party notices:
- `scripts/scannet_sensordata.py` is a Python 3 port of `SensorData.py` of the
  ScanNet toolkit (MIT License, Copyright 2017 Angela Dai, Angel X. Chang,
  Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Niessner; text in
  `scripts/LICENSE-SCANNET`). The port lines come from Panoptic Lifting's copy
  of the file (`dataset/preprocessing/sens_reader/SensorData.py`).
- `scripts/preprocess_scannetpp.py` contains code derived from PanSt3R's
  `tools/preprocess_scannetpp.py`, which is based on DUSt3R's
  `datasets_preprocess/preprocess_scannetpp.py` and `dust3r/utils/geometry.py`
  (Copyright (C) 2024-present Naver Corporation, CC BY-NC-SA 4.0,
  non-commercial use only; https://creativecommons.org/licenses/by-nc-sa/4.0/).
  The derived parts stay under CC BY-NC-SA 4.0. The header of the PanSt3R file
  names CC BY-NC-SA 4.0, while the `LICENSE` at the root of the PanSt3R
  repository is the NAVER "PanSt3R Non-Commercial License". Its `rgb2id` /
  `id2rgb`
  follow panopticapi (BSD 2-Clause, Copyright (c) 2018, Alexander Kirillov);
  the notice is reproduced in the file header.
- `scripts/preprocess_scannetv2_s20.py` follows the processing procedure of
  Panoptic Lifting's `dataset/preprocessing/preprocess_scannet.py`
  (https://github.com/nihalsid/panoptic-lifting, CC BY-NC-SA 4.0). The parts
  derived from it stay under CC BY-NC-SA 4.0.

Sources of the class-mapping tables in `vcpq/data/`:
- `raw_to_s20_200.csv` and `s20_200_things.csv`: the ScanNet200 class ids and
  names of ScanNet's `BenchmarkScripts/ScanNet200/scannet200_constants.py`
  (MIT License); the two stuff classes are wall and floor.
- `s20_200_to_20.csv`: each ScanNet200 class mapped to the 20 ScanNet benchmark
  classes through the `nyu40id` column of ScanNet's
  `scannetv2-labels.combined.tsv`.
- `spp100_classes.csv`: the ScanNet++ semantic benchmark metadata
  (`top100.txt`, `top100_instance.txt` under `metadata/semantic_benchmark/` of
  the dataset download). The raw label mapping is not shipped; the ScanNet++
  script builds it from `semantic_classes.txt` and `map_benchmark.csv` of your
  copy (`scripts/spp100_mapping.py`, the mapping loop of PanSt3R's
  `tools/preprocess_scannetpp.py`).

## Citation

Please cite the paper (<https://arxiv.org/abs/2610.05911>):

```bibtex
@misc{vcpq2026,
  title         = {Every View Counts: View-Consistent Panoptic Quality for Multi-view Panoptic Segmentation},
  author        = {Lee, Youngmin and Ko, Byungha and Yun, Guhnoo and Kim, Dong Hwan},
  year          = {2026},
  eprint        = {2610.05911},
  archivePrefix = {arXiv}
}
```

### Data and code we build on

The ground truth, the label systems and the protocol build on the following
work; please cite it as well when you use them. The entries are the ones the
upstream repositories ask for.

```bibtex
@inproceedings{dai2017scannet,
    title={ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes},
    author={Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
    booktitle = {Proc. Computer Vision and Pattern Recognition (CVPR), IEEE},
    year = {2017}
}

@inproceedings{rozenberszki2022language,
    title={Language-Grounded Indoor 3D Semantic Segmentation in the Wild},
    author={Rozenberszki, David and Litany, Or and Dai, Angela},
    booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
    year={2022}
}

@inproceedings{yeshwanth2023scannet++,
  title={Scannet++: A high-fidelity dataset of 3d indoor scenes},
  author={Yeshwanth, Chandan and Liu, Yueh-Cheng and Nie{\ss}ner, Matthias and Dai, Angela},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12--22},
  year={2023}
}

@InProceedings{zust2025panst3r,
  title={PanSt3R: Multi-view Consistent Panoptic Segmentation},
  author={Zust, Lojze and Cabon, Yohann and Marrie, Juliette and Antsfeld, Leonid and Chidlovskii, Boris and Revaud, Jerome and Csurka, Gabriela},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2025}
}

@InProceedings{Siddiqui_2023_CVPR,
    author    = {Siddiqui, Yawar and Porzi, Lorenzo and Bul\`o, Samuel Rota and M\"uller, Norman and Nie{\ss}ner, Matthias and Dai, Angela and Kontschieder, Peter},
    title     = {Panoptic Lifting for 3D Scene Understanding With Neural Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9043-9052}
}

@inproceedings{dust3r_cvpr24,
      title={DUSt3R: Geometric 3D Vision Made Easy},
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      booktitle = {CVPR},
      year = {2024}
}
```

ScanNet is the source of the `.sens` reader, the 20 benchmark classes and the
ScanNet200 constants; ScanNet200 (Rozenberszki et al.) of the 200-class label
set; ScanNet++ of the data and the top-100 benchmark classes; PanSt3R of the
rendering script and the 50-pixel segment filter; Panoptic Lifting of the
robust-class filter, the scene-level protocol and the s20 preprocessing
procedure; DUSt3R of the origin of the ScanNet++ preprocessing script.

## Ethics

This repository contains no dataset images, meshes, scans or annotations of
ScanNet++ or ScanNet, and no model weights. The packaged lists contain scene
ids and frame stems only. You obtain the datasets from their owners under
their terms of use (ScanNet++ and ScanNet each require an access agreement).
The tests do not ship any label maps either: they run on synthetic label maps
generated in the test code. The compared
methods are not redistributed here; their settings are documented in
[docs/baselines.md](docs/baselines.md), and each is used under its own license.
