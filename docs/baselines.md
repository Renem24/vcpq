# Settings of the compared methods

This document records what was run for the compared methods of the paper:
PanSt3R (v1 and v2), SegVGGT, EPS3D and IGGT. It is a description of the
settings, not an installation guide. The adapter code that turned each method
into a `ChunkPredictor` is not part of this release, and this document
contains no code or patches of the methods. To score another method, or to
re-run these ones with your own wrapper, implement a `ChunkPredictor` and
score it with `vcpq-eval --predictor module:factory` (see the README,
"Score your own predictions"); the predictions are then scored exactly as
the paper's dumps.

Contents: [Common to all methods](#common-to-all-methods) ·
[PanSt3R](#panst3r) · [SegVGGT](#segvggt) · [EPS3D](#eps3d) · [IGGT](#iggt) ·
[Determinism](#determinism) · [Known discrepancies](#known-discrepancies)

## Common to all methods

**Cells.** One cell is one method on one dataset at one input-set size N (the
number of views of an input set). The paper's cells are N = 5, 10, 20, 40 for
PanSt3R v2 (ScanNet++ DSLR, ScanNet++ iPhone, ScanNetv2), PanSt3R v1
(ScanNet++ DSLR and iPhone) and SegVGGT (ScanNetv2), and N = 5, 10 for EPS3D
(ScanNet++ DSLR and iPhone, ScanNetv2) and IGGT (ScanNet++ DSLR and iPhone).
EPS3D and IGGT stop at N = 10 for memory (below). The iPhone scene
`d755b3d9d8` is dumped like the others; its ground truth is entirely void, so
it adds nothing to the tables (README, Protocol, Exclusions).

**Output format.** Every method's per-view output was converted to the
protocol's panoptic map, `int32 [H, W]` at ground-truth resolution with
`sem * 1000 + inst`, with the method's own instance identity shared by all
views of the input set. Thing segments are `sem * 1000 + (id + 1)` with `id`
the method's query or cluster index; stuff pixels are `sem * 1000`, and for
PanSt3R, whose queries are all treated as things, the stuff segments of one
class are merged into that class. Pixels that no segment claims, and
segments whose class is outside the label system, are void (0). For PanSt3R,
EPS3D and IGGT the maps of the network resolution are upsampled to
ground-truth size with nearest-neighbour interpolation; SegVGGT instead
upsamples its mask logits bilinearly to ground-truth size before the
per-view competition (below).

**Open-vocabulary prompts.** PanSt3R, EPS3D and IGGT classify with text
prompts. The lists are in `vcpq/vocab.py` (`build_vocab`):

- ScanNet++ (spp100): the 100 class names in id order.
- ScanNetv2 (s20): 41 prompts. The 19 concrete s20 class names, then, in
  place of the catch-all "otherfurniture" (id 20), the 22 ScanNet200 classes
  that `vcpq/data/s20_200_to_20.csv` maps to it; those last 22 prompts all
  score for id 20.

**Hardware and precision.** Python 3.10, torch 2.3.1, torchvision 0.18.1,
CUDA 12.1 (EPS3D's stage: CUDA 11.8). TF32 matrix products
(`torch.backends.cuda.matmul.allow_tf32 = True`) were enabled for PanSt3R and
SegVGGT, and for EPS3D's stage, as in the original evaluation runs. The
paper's dumps also stored per-segment scores for an AP number this release
does not compute; VC-PQ and PQ^scene do not read them.

## PanSt3R

- Upstream: https://github.com/naver/panst3r, commit `0253909`. It builds on
  MUSt3R, https://github.com/naver/must3r, commit `0c0d76b` (its dust3r
  submodule at `bb9f9f5`).
- Checkpoints: `panst3r_v2_512_5ds.pth` for v2 and `panst3r_v1_512_5ds.pth`
  for v1, from the checkpoint links in PanSt3R's README.
- Input: DUSt3R resize with image size 512 (the size of the released
  checkpoints), one forward per input set with every view of the set as a
  keyframe (`num_keyframes` unset), batch size 1, no autocast. Retrieval is
  not used (`use_retrieval=False`); its two dependencies (`asmk`, `faiss`)
  were replaced by empty stub modules because only the retrieval imports
  them. The text encoder was loaded through Hugging Face transformers, with
  `HF_HUB_OFFLINE=1` once cached.
- Post-processing: PanSt3R's own panoptic post-process on the joint
  `[N, S, H, W]` query masks. v2 uses `standard_v2` with class threshold
  0.1; v1 uses `qubo` (simulated annealing, no class threshold). Segment
  scores stored in the paper's dumps (not read by VC-PQ or PQ^scene): the
  query's sigmoid class maximum (`standard_v2`), or class probability times
  mask confidence (`qubo`); things only.
- Vocabulary: as above. ScanNetv2 was run for v2 only (there is no v1
  ScanNetv2 cell).
- Input sets with a single view cannot be run (see Known discrepancies).
- Memory patch (described in words; not included). The paper's runs applied
  a patch to PanSt3R's `src/panst3r/engine/postprocess.py` that changes only
  memory use: a preallocated padding buffer for multi-aspect-ratio inputs,
  freeing the full-resolution mask tensor before the post-process iterations
  when the batch size is 1, and a column-blocked argmax over the queries.
  The output was verified to be identical with and without it; without it,
  N = 40 input sets can run out of memory on a 24 GB GPU.
- Environment: PanSt3R's own `pyproject.toml` requires Python 3.11, while the
  paper's runs used the Python 3.10 environment listed above, with PanSt3R
  and MUSt3R put on the import path rather than installed.
- Determinism: see below (v2 deterministic, v1 not).

## SegVGGT

- Upstream: https://github.com/IDEA-Research/SegVGGT, commit `cd5f156`,
  model config `configs/eval/segvggt_scannetv2.yaml`, checkpoint
  `segvggt_scannetv2.pt` (SegVGGT's Hugging Face release). ScanNetv2 only.
- Input: images resized to width 518 with VGGT's crop mode
  (`load_and_preprocess_images`), bf16 autocast, one forward per input set.
- What the model predicts: 400 queries with mask logits over every view of
  the set and 19 class logits (the 18 s20 thing classes and a no-object
  class). Wall and floor come from a separate semantic head that was not
  used, so the dumped maps never contain wall or floor.
- Post-processing per view, following the panoptic inference of Mask2Former
  plus SegVGGT's own rules: the class logits become 21 columns (`-inf` for
  wall and floor, last column no-object); a query whose mask (sigmoid above
  the mask threshold) covers fewer than `npoint_thr` pixels over all views
  loses every class (SegVGGT's npoint filter, applied before the per-view
  competition); the class score is multiplied by the query's mask quality
  (mean sigmoid over its positive-logit pixels, SegVGGT's
  `obj_normalization`); masks are upsampled bilinearly to ground-truth size
  and every pixel goes to the query with the highest score times sigmoid; a
  query keeps its pixels (also above the mask threshold) only if they make up
  at least the overlap threshold of its thresholded mask and at least
  `npoint_thr` pixels.
- Thresholds: mask threshold 0.4, overlap threshold 0.8, `npoint_thr` 200,
  score threshold 0.0, `obj_normalization` on.
- Segment scores stored in the paper's dumps (not read by VC-PQ or
  PQ^scene): the query's softmax class maximum, without the mask quality
  factor.
- Not applied: SegVGGT's top-600 selection over all (query, class) pairs (see
  Known discrepancies).

## EPS3D

EPS3D predicts an instance embedding and semantic features and has no class
head of its own; the paper's EPS3D row classifies with LSeg. The pipeline has
three stages, and each stage skips the input sets it has already written.

1. **Input sets.** The cell's input sets (key, frame stems, image paths) are
   listed, the same ones the protocol builds for every method, and the prompt
   list of `vcpq/vocab.py` is written for the label system.
2. **EPS3D (GPU, its own environment).** EPS3D runs on every input set with
   all N frames as context views, renders the semantic and instance features
   at EPS3D's own predicted poses, decodes the semantics with LSeg against
   the prompt list, clusters the instance features with HDBSCAN and assigns
   every pixel to its nearest cluster centroid. Per input set the stage
   writes, per frame, the semantic map (prompt index + 1) and the cluster id
   map, and per cluster the mean LSeg probability of the assigned class (the
   cluster's score).
3. **Panoptic maps (no GPU).** Thing pixels `sem * 1000 + (cluster + 1)`,
   stuff pixels `sem * 1000`, nearest resize to ground-truth size; a thing
   segment scores with its cluster's value. A missing or incomplete stage
   output is an error (the original evaluation dumped an all-void map), so an
   unfinished stage cannot lower the scores unnoticed.

- Upstream: https://github.com/Runsong123/EPS3D, commit `147f774`, used
  unmodified. Environment of the paper's runs: Python 3.10, torch 2.3.1 with
  CUDA 11.8, gsplat 1.4.0, torch_scatter 2.1.2, hdbscan 0.8.44.
- Checkpoints (EPS3D's release): the directory with `model.safetensors`,
  `config.json` and LSeg's `demo_e200.ckpt`, and CLIP ViT-B/32 `ViT-B-32.pt`.
- Input: full-frame (short side resized to 448, long side to the nearest
  multiple of 14, no crop). EPS3D's own preprocessing crops the central
  448 x 448 square; the paper did not use that.
- Clustering: HDBSCAN with `min_cluster_size = 400`, `min_samples = 1` on
  50,000 instance features sampled with seed 42.
- Semantic decoding: the LSeg decoding reproduces the decode step of EPS3D's
  LSeg feature extractor, and the per-input-set pipeline follows EPS3D's
  own panoptic test procedure (which is also where the cluster-mean
  probability score comes from).
- Build and runtime: EPS3D's CUDA extensions are compiled at first use, which
  needs `CUDA_HOME` pointing at a CUDA 11.8 toolkit and `ninja` on `PATH`.
  The EPS3D stage ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Memory: N = 5 and 10 only. EPS3D's voxel fusion scatters every pixel of the
  input set, and N = 20 runs out of memory on a 24 GB GPU.

## IGGT

IGGT predicts a per-pixel instance embedding and no class. As in the IGGT
paper (Sec. 3.3), each class-agnostic instance is classified by averaging an
open-vocabulary model's per-pixel features over its mask and matching them to
the class names. The paper's evaluation uses LSeg for this (the same
classifier as the EPS3D row), where the IGGT paper names OpenSeg. ScanNet++
(spp100) only. Four stages, each skipping input sets already written.

1. **Input sets.** As for EPS3D: manifest and the 100 ScanNet++ names.
2. **Stage A, instances (GPU, IGGT environment with cuML).** IGGT forward
   (input 504 x 336, bf16 autocast); world points from the predicted depth
   and cameras; each pixel's instance feature averaged over its 20 nearest 3-D
   neighbours across the views of the set; HDBSCAN (cuML, on the GPU) over all
   pixels of the set with `cluster_selection_epsilon = 0.06`,
   `min_samples = 100`, `min_cluster_size = 500`; every noise pixel is given
   the cluster of its nearest clustered pixel in feature space. The output is
   one cluster id map per frame, ids shared by the views of the set.
3. **Stage B, classes (GPU, EPS3D environment).** LSeg features of each frame
   (short side 480, sides multiples of 32) averaged over each cluster's pixels
   in all views, matched to the 100 names (cosine similarity times LSeg's
   logit scale, softmax). Output: per cluster the class id and its
   probability.
4. **Panoptic maps (no GPU).** Thing pixels `sem * 1000 + (cluster + 1)`,
   stuff pixels `sem * 1000`, clusters without a class are void, nearest
   resize to ground-truth size; a thing segment scores with its cluster's
   probability. A missing stage output is an error, and a cluster id of 999
   or more is an error too, because `sem * 1000 + (id + 1)` would overflow
   into the class field (the paper's largest cluster id is 110).

- Upstream: https://github.com/lifuguan/IGGT_official, commit `bec6604`;
  checkpoint `iggt_checkpoint.pth` (IGGT's release). Stage B uses the EPS3D
  checkout, LSeg's `demo_e200.ckpt` and CLIP ViT-B/32 as in the EPS3D
  section.
- Modifications to the IGGT code (described in words; the patch is not
  included). Four changes to the public code at the commit above:
  1. The cuML import guard in `iggt/utils/misc.py` now records which HDBSCAN
     was imported (a flag, `HDBSCAN_IS_GPU`) and prints why cuML is
     unavailable instead of falling back silently. Stage A of the paper's
     runs refused the CPU `hdbscan` fallback unless it was explicitly allowed
     with an option, and the paper did not use the fallback.
  2. The frame-chunked branch of the DPT head and of the part head, taken
     when an input set has more than 12 views, already existed upstream but
     failed: the DPT head with point features concatenated tuples and
     returned only the features instead of the (predictions, confidence,
     features) tuple of the single-pass branch, and the part head passed frame
     indices that its forward implementation does not accept and unpacked two
     outputs from its single one. The change runs each chunk on its frame slice
     and returns what the single-pass branch returns. This is a bug fix, not
     a memory measure: the paper's IGGT cells are N = 5 and 10, which never
     take that branch.
  3. A module `src/model/norm.py` with a standard RMSNorm was added, because
     `iggt/heads/block.py` falls back to `from src.model.norm import RMSNorm`
     when apex is missing, and `src/model/norm.py` is not in the public IGGT
     repository. The class is never instantiated in these runs (the head's
     attention blocks are built without query/key normalisation).
  4. A module `basicsr/archs/arch_util.py` providing `to_2tuple` and
     `trunc_normal_` was added, because `iggt/heads/window_sa.py` imports them
     from `basicsr`, whose release imports a torchvision module that was
     removed in torchvision 0.17.
  Separately, the exact-path neighbour search and noise fill of stage A are
  our reimplementation of two of IGGT's post-processing steps (scipy cKDTree
  with the tie rule of the `torch_cluster` kernel IGGT calls, neighbours
  summed in a fixed order; exact nearest-neighbour noise fill), and that is
  the path the paper's numbers use.
- Environment of stage A: Python 3.10, torch 2.3.1 (CUDA 12.1), cuML 25.10,
  scipy, scikit-learn and IGGT's requirements; the paper's stage-A outputs
  were computed on several machines (scene shards merged by input-set key).
  cuML allocates GPU memory outside torch's allocator, so stage A should have
  its GPU to itself.
- Memory: N = 5 and 10 only. At N = 20 and 40, cuML HDBSCAN over all pixels of
  the input set fails to allocate on a 24 GB GPU.

## Determinism

- PanSt3R v2 (`standard_v2`) and SegVGGT were deterministic in the paper's
  runs: repeated runs on the same GPU gave identical maps. A different GPU
  model can change a few pixels.
- PanSt3R v1 (`qubo`) is not: its simulated annealing draws from numpy's
  global random generator, which is never seeded. Repeated dumps differ; the
  paper's Appendix A reports a run-to-run difference of at most 0.6 point
  PQ^scene and 0.1 point VC-PQ at N = 10.
- EPS3D is not deterministic either: its encoder gives slightly different
  features for the same input from one forward to the next (most likely the
  atomic additions of torch_scatter in the voxel fusion). Two runs of the
  EPS3D stage differ in some pixels at class boundaries, and their clusters
  can be numbered or drawn differently. The rendering, the LSeg decoding, the
  seeded HDBSCAN subsample, the centroid assignment and the conversion to
  panoptic maps are deterministic: the same stage output always gives the
  same dump.
- IGGT: the neighbour averaging and the noise fill of stage A are
  deterministic given IGGT's features. IGGT's forward and cuML's HDBSCAN run
  on the GPU; the paper did not measure their run-to-run differences.
  HDBSCAN numbers its clusters arbitrarily: the same clusters under other ids
  give other panoptic ids but the same segments and scores. The conversion to
  panoptic maps is deterministic.

## Known discrepancies

- **SegVGGT top-600 selection.** SegVGGT's own inference keeps the 600
  highest-scoring (query, class) pairs. The paper's dumps did not apply this
  step (the panoptic competition needs one class per query); each query takes
  its highest-scoring class instead, as the paper's Appendix A describes.
- **PanSt3R input sets with one view.** PanSt3R needs at least two views. The
  original evaluation skipped such a set without writing a chunk. The
  packaged frame lists (120 frames per scene) only produce one if ground-truth
  frames are missing on disk.

## Third-party terms

The methods above are used under their own licenses; check the license of each
upstream repository before use. PanSt3R's terms are in the `LICENSE` and
`NOTICE` files at the root of its repository. IGGT's README states the MIT
License, but the LICENSE file it refers to is not in the repository at the
commit above.
