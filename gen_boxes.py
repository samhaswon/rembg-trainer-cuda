import json
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


TILE_SIZE = 512
OVERLAP = 64                # 0 <= OVERLAP < TILE_SIZE

# Selection controls
MIN_POS_FRAC = 0.01         # min foreground fraction to consider a tile at all
BIN_THRESH = 0              # binarize: foreground = mask > BIN_THRESH
REQUIRE_NEGATIVE_OVERLAP = True
MIN_NEG_FRAC = 0.01         # only for mixed tiles if REQUIRE_NEGATIVE_OVERLAP

# Edge-focused sampling + include all-white and all-zero tiles
EDGE_BAND_PX = 12           # pixels from boundary to count as "edge"
EDGE_TARGET_FRAC = 0.8      # target mix across buckets; will renormalize
POS_PURE_FRAC = 0.1         # include all-white tiles
NEG_PURE_FRAC = 0.1         # include all-zero tiles

# Optional grid jitter to reduce checkerboard bias (keep small)
JITTER_PX = 0               # e.g., 4..8 for large images
RNG_SEED = 1234             # set None to make it non-deterministic


def _gen_idxs(length: int, window: int, overlap: int, jitter: int) -> List[int]:
    """
    Build grid start indices covering [0, length), honoring overlap and small jitter.

    :param length: Axis length in pixels.
    :param window: Window size (tile side length).
    :param overlap: Overlap in pixels, 0 <= overlap < window.
    :param jitter: Start offset in pixels, clamped to [0, stride-1].
    :returns: List of starting indices.
    """
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must satisfy 0 <= overlap < window.")
    stride = max(1, window - overlap)
    jit = 0 if stride <= 1 else max(0, min(stride - 1, jitter))
    idxs = list(range(jit, max(1, length - window + 1), stride))
    if not idxs or idxs[-1] != length - window:
        idxs.append(max(0, length - window))
    return idxs


def _select_tiles_edge_mixture(
    mask_gray: np.ndarray,
    tile_size: int,
    overlap: int,
    min_pos_frac: float,
    require_negative_overlap: bool,
    min_neg_frac: float,
    edge_band_px: int,
    edge_target_frac: float,
    pos_pure_frac: float,
    neg_pure_frac: float,
    bin_thresh: int,
    jitter_px: int,
    rng_seed: int | None,
) -> List[Tuple[int, int, int, int]]:
    """
    Select mostly edge tiles, plus a controlled sample of all-white and all-zero tiles.

    Logic detail:
    - Binarize mask to foreground using mask_gray > bin_thresh.
    - Compute distance to boundary via distance transform on both sides, then min.
    - Classify tiles into three buckets:
      mixed (both fg and bg), pos_pure (all fg), neg_pure (all bg).
    - For mixed tiles only, enforce require_negative_overlap/min_neg_frac.
    - Rank mixed tiles by edge density (fraction within edge band), sample to target ratios.
    - Always allow pos_pure and neg_pure candidates into their buckets, then sample by target ratios.

    :returns: List of (x0, y0, x1, y1), x1/y1 are exclusive.
    """
    h, w = mask_gray.shape[:2]
    bin_mask = (mask_gray > bin_thresh).astype(np.uint8)

    dist_fg = cv2.distanceTransform(bin_mask, distanceType=cv2.DIST_L2, maskSize=3)
    dist_bg = cv2.distanceTransform(1 - bin_mask, distanceType=cv2.DIST_L2, maskSize=3)
    dist_to_b = np.minimum(dist_fg, dist_bg)

    rng = np.random.default_rng(rng_seed)
    jx = int(rng.integers(-jitter_px, jitter_px + 1)) if jitter_px > 0 else 0
    jy = int(rng.integers(-jitter_px, jitter_px + 1)) if jitter_px > 0 else 0

    xs = _gen_idxs(w, tile_size, overlap, jx)
    ys = _gen_idxs(h, tile_size, overlap, jy)

    total = tile_size * tile_size
    edge_tiles: list[tuple[Tuple[int, int, int, int], float]] = []
    pos_pure_tiles: list[Tuple[int, int, int, int]] = []
    neg_pure_tiles: list[Tuple[int, int, int, int]] = []

    for y0 in ys:
        for x0 in xs:
            x1, y1 = x0 + tile_size, y0 + tile_size
            tbin = bin_mask[y0:y1, x0:x1]
            if tbin.size != total:
                continue

            pos = int(np.count_nonzero(tbin))
            neg = total - pos
            if pos / total < min_pos_frac:
                continue

            if pos == total:
                # include all-white tiles regardless of negative-overlap requirement
                pos_pure_tiles.append((x0, y0, x1, y1))
                continue
            if neg == total:
                # include all-zero tiles as candidates
                neg_pure_tiles.append((x0, y0, x1, y1))
                continue

            # Mixed tile: optionally require some zero overlap and min_pos_frac
            if pos / total < min_pos_frac:
                continue

            edge_mask = dist_to_b[y0:y1, x0:x1] <= float(edge_band_px)
            edge_frac = float(np.count_nonzero(edge_mask)) / float(total)
            edge_tiles.append(((x0, y0, x1, y1), edge_frac))

    # Rank mixed tiles by edge density
    edge_tiles.sort(key=lambda t: t[1], reverse=True)
    edge_tiles_only = [t[0] for t in edge_tiles]

    # Determine target per-bucket counts relative to availability
    n_pool = len(edge_tiles_only) + len(pos_pure_tiles) + len(neg_pure_tiles)
    if n_pool == 0:
        return []

    weights = np.array([edge_target_frac, pos_pure_frac, neg_pure_frac], dtype=np.float64)
    weights = weights / max(1e-9, weights.sum())

    tgt_edge = min(len(edge_tiles_only), int(round(weights[0] * n_pool)))
    tgt_pos = min(len(pos_pure_tiles), int(round(weights[1] * n_pool)))
    tgt_neg = min(len(neg_pure_tiles), int(round(weights[2] * n_pool)))

    picked: list[Tuple[int, int, int, int]] = []
    picked.extend(edge_tiles_only[:tgt_edge])
    picked.extend(pos_pure_tiles[:tgt_pos])
    picked.extend(neg_pure_tiles[:tgt_neg])

    # Fill shortfall by priority edge -> pos -> neg
    short = n_pool - len(picked)
    if short > 0:
        leftovers = (
            edge_tiles_only[tgt_edge:]
            + pos_pure_tiles[tgt_pos:]
            + neg_pure_tiles[tgt_neg:]
        )
        if leftovers:
            picked.extend(leftovers[:short])

    # Dedup just in case
    seen = set()
    final: List[Tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in picked:
        key = (x0, y0)
        if key not in seen:
            final.append((x0, y0, x1, y1))
            seen.add(key)
    return final


if __name__ == '__main__':
    start = time.perf_counter()
    mask_list = [x.path for x in os.scandir("/home/samuel/da/skindataset/masks")]
    mask_list.sort()
    mask_list = mask_list[:1134]

    box_list: List[Tuple[str, int, int, int, int]] = []
    for mask_path in tqdm(mask_list, desc="Boxing"):
        img_path = mask_path[mask_path.rfind(os.path.sep) + 1:]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        boxes = _select_tiles_edge_mixture(
            mask_gray=mask,
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            min_pos_frac=MIN_POS_FRAC,
            require_negative_overlap=REQUIRE_NEGATIVE_OVERLAP,
            min_neg_frac=MIN_NEG_FRAC,
            edge_band_px=EDGE_BAND_PX,
            edge_target_frac=EDGE_TARGET_FRAC,
            pos_pure_frac=POS_PURE_FRAC,
            neg_pure_frac=NEG_PURE_FRAC,
            bin_thresh=BIN_THRESH,
            jitter_px=JITTER_PX,
            rng_seed=RNG_SEED,
        )
        for box in boxes:
            box_list.append((img_path, *box))
    with open("boxes.json", "w") as f:
        json.dump(box_list, f)

    end = time.perf_counter()
    print(f"Num boxes: {len(box_list)}")
    print(f"Total time: {end - start:.4f}s")
