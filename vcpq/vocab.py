"""Open-vocabulary prompt lists of the label systems.

Methods that classify with text prompts (PanSt3R, and the LSeg classifier of
EPS3D and IGGT) were prompted with these lists in the paper:

- ScanNet++ (spp100): the 100 class names in id order.
- ScanNetv2 (s20): 41 prompts = the 19 concrete s20 class names, plus, instead
  of the catch-all "otherfurniture" (id 20), the 22 ScanNet200 classes that
  ``vcpq/data/s20_200_to_20.csv`` maps to it; the last 22 prompts all score
  for id 20.

Torch-free.
"""

from __future__ import annotations

import importlib.resources

from vcpq.label_systems import SCANNETV2_S20, SPP100, LabelSystem

S20_OTHERFURNITURE_ID = 20


def build_vocab(label: LabelSystem) -> tuple[list[str], list[int]]:
    """Prompt list and prompt index -> semantic id.

    Returns:
        (prompts, idx_to_semid): ``idx_to_semid[i]`` is the label-system id
        that prompt ``i`` scores for.

    Raises:
        ValueError: ``label`` is neither spp100 nor scannetv2_s20.
    """
    sem_ids = sorted(label.class_names)
    if label.name == SPP100:
        return [label.class_names[k] for k in sem_ids], list(sem_ids)
    if label.name != SCANNETV2_S20:
        raise ValueError(f'no open-vocabulary prompt list for label system {label.name!r}')

    ids = [k for k in sem_ids if k != S20_OTHERFURNITURE_ID]          # 1..19
    prompts = [label.class_names[k] for k in ids]
    idx_to_semid = list(ids)
    csv_text = (importlib.resources.files('vcpq') / 'data' / 's20_200_to_20.csv').read_text()
    for line in csv_text.strip().splitlines():                        # rows: cont_200,name,cont_20
        parts = line.split(',')
        name, cont_20 = ','.join(parts[1:-1]), int(parts[-1])
        if cont_20 == S20_OTHERFURNITURE_ID:
            prompts.append(name)
            idx_to_semid.append(S20_OTHERFURNITURE_ID)
    return prompts, idx_to_semid
