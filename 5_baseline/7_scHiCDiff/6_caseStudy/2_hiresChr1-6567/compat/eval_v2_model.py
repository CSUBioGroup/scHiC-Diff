import torch

from scdiff.model import DiffusionModel


def v2_random_masking(x, cell_mask_ratio=0.75):
    if x.ndim != 2:
        raise ValueError(f"v2 masking expects a 2D tensor, got shape {tuple(x.shape)}")
    if not 0.0 <= cell_mask_ratio <= 1.0:
        raise ValueError(
            f"cell_mask_ratio must be between 0 and 1, got {cell_mask_ratio}"
        )

    rows = x.shape[0]
    mask_ratios = torch.rand(rows, 1, device=x.device)
    fully_masked = torch.rand(rows, device=x.device) < cell_mask_ratio
    mask_ratios[fully_masked] = 1.0
    mask = torch.rand_like(x) < mask_ratios
    return x.masked_fill(mask, 0), mask


class EvalV2DiffusionModel(DiffusionModel):
    def __init__(self, eval_v2_cell_mask_ratio=0.75, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= eval_v2_cell_mask_ratio <= 1.0:
            raise ValueError(
                "eval_v2_cell_mask_ratio must be between 0 and 1, "
                f"got {eval_v2_cell_mask_ratio}"
            )
        self.eval_v2_cell_mask_ratio = eval_v2_cell_mask_ratio

    def random_masking(self, x):
        return v2_random_masking(x, self.eval_v2_cell_mask_ratio)
