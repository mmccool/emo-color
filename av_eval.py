# Given: mapper already fit with some seeds
from emotion_color_mapper import rgb_to_lab, deltaE2000, parse_color
import numpy as np

def eval_heldout(mapper, heldout_seed_colors: dict) -> float:
    dists = []
    for emo, col in heldout_seed_colors.items():
        lab_true = rgb_to_lab(parse_color(col))
        lab_pred = mapper._pred_lab_for(emo)  # internal helper
        dists.append(deltaE2000(lab_true, lab_pred))
    return float(np.mean(dists)) if dists else float("nan")

heldout = {"compassion": "pink", "tension": "darkred"}
print("Held-out ΔE2000:", eval_heldout(mapper, heldout))
