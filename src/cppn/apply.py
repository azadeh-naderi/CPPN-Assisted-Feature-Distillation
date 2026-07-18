import torch


def apply_pattern(
    images_raw01: torch.Tensor, pattern: torch.Tensor, mode: str = "multiplicative", scale: float = 0.5
) -> torch.Tensor:
    """Combines one evolved/compiled CPPN pattern with a batch of raw-[0,1]
    images to produce the "auxiliary view". `pattern` is [H, W, C] in [0,1]
    (already squashed by compile_genome's outer sigmoid) and is reused
    unchanged across every image in the batch.

    - multiplicative (default): view = image * pattern. Range-safe by
      construction (product of two [0,1] tensors); degenerate cases (all-ones
      = identity, all-zero = blank) are cheap to reason about and are
      suppressed by the gated fitness function in fitness.py.
    - additive: view = clamp(image + scale*(pattern-0.5), 0, 1). Needed
      because multiplicative masking is a no-op on near-zero/background
      pixels; kept as an ablation, not the default.
    - warp: geometric coordinate displacement. Out of scope for the MVP
      (harder to bound than a pixel-space mask) — reserved so the interface
      doesn't need to change if it's added later.
    """
    pattern_bchw = pattern.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    if mode == "multiplicative":
        return images_raw01 * pattern_bchw
    if mode == "additive":
        return torch.clamp(images_raw01 + scale * (pattern_bchw - 0.5), 0.0, 1.0)
    if mode == "warp":
        raise NotImplementedError("view_op='warp' is reserved for future work.")
    raise ValueError(f"Unknown view_op: {mode!r}. Choose 'multiplicative' or 'additive'.")
