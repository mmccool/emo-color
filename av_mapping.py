# av_mapping.py
# Map emotions to and from colors using standard arousal/valence model.
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple, Union, Optional
import colorsys
from difflib import get_close_matches

# -----------------------------------------------------------------------------
# Core idea:
# - Represent emotions in a 2D circumplex: valence ∈ [-1, 1], arousal ∈ [0, 1]
# - Map (valence, arousal) → HSL:
#     hue:      angle around the valence/arousal plane (continuous “wheel”)
#     sat:      increases with arousal
#     light:    increases with valence
# - Convert HSL → sRGB hex for display
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Color:
    r: int  # 0..255
    g: int
    b: int

    @property
    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    @property
    def rgb(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)


@dataclass(frozen=True)
class HSL:
    h: float  # [0, 1]
    s: float  # [0, 1]
    l: float  # [0, 1]


# Curated priors (approximate, but useful) for common emotions.
# valence: -1 (very unpleasant)  →  +1 (very pleasant)
# arousal:  0 (very calm)        →   1 (very activated)
EMOTION_PRIORS: Dict[str, Tuple[float, float]] = {
    # positive
    "joy":          (0.92, 0.63),
    "delight":      (0.90, 0.70),
    "amusement":    (0.78, 0.55),
    "contentment":  (0.70, 0.30),
    "calm":         (0.62, 0.12),
    "relief":       (0.72, 0.22),
    "love":         (0.85, 0.50),
    "gratitude":    (0.80, 0.40),
    "pride":        (0.82, 0.60),
    "hope":         (0.65, 0.45),
    "trust":        (0.55, 0.25),

    # neutral/surprise-ish
    "anticipation": (0.40, 0.62),
    "surprise":     (0.10, 0.85),

    # negative, high arousal
    "anger":        (-0.85, 0.88),
    "rage":         (-0.93, 1.00),
    "fear":         (-0.78, 0.92),
    "anxiety":      (-0.62, 0.74),
    "disgust":      (-0.65, 0.52),
    "shame":        (-0.70, 0.50),
    "guilt":        (-0.72, 0.50),

    # negative, low arousal
    "sadness":      (-0.82, 0.32),
    "grief":        (-0.90, 0.42),
    "boredom":      (-0.40, 0.12),
    "loneliness":   (-0.75, 0.28),
    "regret":       (-0.60, 0.35),
}

# Optional synonyms → canonical keys (extend as you like)
SYNONYMS: Dict[str, str] = {
    "happiness": "joy",
    "joyful": "joy",
    "excited": "delight",
    "chill": "calm",
    "serene": "calm",
    "relaxed": "calm",
    "loveful": "love",
    "angry": "anger",
    "furious": "rage",
    "scared": "fear",
    "worried": "anxiety",
    "grossed out": "disgust",
    "blue": "sadness",
    "depressed": "sadness",
    "lonely": "loneliness",
}

# --------------------------- utility functions -------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())

def _lookup_emotion(label: str) -> Optional[Tuple[float, float]]:
    key = _normalize_label(label)
    key = SYNONYMS.get(key, key)
    if key in EMOTION_PRIORS:
        return EMOTION_PRIORS[key]

    # Fuzzy fallback: nearest known emotion by name (useful for typos)
    candidates = get_close_matches(key, EMOTION_PRIORS.keys(), n=1, cutoff=0.75)
    if candidates:
        return EMOTION_PRIORS[candidates[0]]
    return None

def hsl_to_color(hsl: HSL) -> Color:
    # colorsys uses HLS order and returns floats in [0,1]
    r, g, b = colorsys.hls_to_rgb(hsl.h, hsl.l, hsl.s)
    return Color(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

# ------------------------ the core mapping functions -------------------------

def circumplex_to_hsl(valence: float, arousal: float) -> HSL:
    """
    Map (valence, arousal) to HSL.

    Design:
    - Hue: angle around the plane with center at (0, 0.5). Positive valence points
      map to warmer hues; negative valence to cooler hues. Surprise (high arousal, ~neutral valence)
      lands near cyan/blue; anger (neg. valence, high arousal) near red-magenta; joy (pos. valence, med-high arousal) near yellow.
      Exact angles are conventional but continuous and consistent.
    - Saturation: increases with arousal (more activated → more saturated).
    - Lightness: increases with valence (pleasant → lighter/brighter).
    """
    v = _clamp(valence, -1.0, 1.0)
    a = _clamp(arousal, 0.0, 1.0)

    # Center arousal around 0.5 for angle computation
    x = v
    y = a - 0.5

    angle = math.atan2(y, x)  # [-pi, pi]
    if angle < 0:
        angle += 2 * math.pi
    hue = angle / (2 * math.pi)  # [0,1]

    # Saturation from arousal (soft floor/ceiling)
    #  a=0   → ~0.25   (muted)
    #  a=1   → ~0.95   (vivid)
    sat = 0.25 + 0.70 * (a ** 0.9)
    sat = _clamp(sat, 0.0, 1.0)

    # Lightness from valence:
    #  v=-1 → ~0.30 (dark)
    #  v=+1 → ~0.75 (light)
    light = 0.525 + 0.225 * v
    light = _clamp(light, 0.0, 1.0)

    return HSL(h=hue, s=sat, l=light)

def emotion_to_color(
    emotion: Union[str, Tuple[float, float]],
    *,
    override_va: Optional[Tuple[float, float]] = None,
) -> Color:
    """
    Convert an emotion to a display color.
    - If `emotion` is a string, we use priors (or fuzzy/synonym fallback).
    - If it's a (valence, arousal) tuple, we map directly.
    - You can explicitly pass override_va=(v,a) for a label to bypass priors.

    Returns a Color with .hex and .rgb.
    """
    if isinstance(emotion, tuple):
        v, a = emotion
    else:
        if override_va is not None:
            v, a = override_va
        else:
            va = _lookup_emotion(str(emotion))
            if va is None:
                # Unknown: assume mildly unpleasant and low arousal (a gentle grayish tone).
                v, a = (-0.1, 0.25)
            else:
                v, a = va

    hsl = circumplex_to_hsl(v, a)
    return hsl_to_color(hsl)

def mix_emotions(weights: Mapping[Union[str, Tuple[float, float]], float]) -> Color:
    """
    Blend multiple emotions by weighted average in valence–arousal space.
    Example:
        mix_emotions({"joy": 0.7, "fear": 0.3})
    """
    total = sum(max(0.0, w) for w in weights.values())
    if total == 0:
        return emotion_to_color(("neutral", 0.0))  # fallback (won't be used)
    vx = 0.0
    ax = 0.0
    for key, w in weights.items():
        w = max(0.0, float(w)) / total
        if isinstance(key, tuple):
            v, a = key
        else:
            va = _lookup_emotion(str(key)) or (-0.1, 0.25)
            v, a = va
        vx += w * v
        ax += w * a
    return emotion_to_color((vx, ax))

def make_palette(
    emotions: Iterable[Union[str, Tuple[float, float]]]
) -> Dict[str, Color]:
    """
    Convenience function: build a {label: Color} palette for a list of emotions.
    """
    out: Dict[str, Color] = {}
    for e in emotions:
        if isinstance(e, tuple):
            label = f"({e[0]:+.2f},{e[1]:.2f})"
        else:
            label = _normalize_label(str(e))
        out[label] = emotion_to_color(e)
    return out

# ------------------------------ examples -------------------------------------
if __name__ == "__main__":
    examples = ["joy", "calm", "anticipation", "surprise", "anger", "fear", "sadness", "disgust", "boredom"]
    palette = make_palette(examples)
    for name, color in palette.items():
        print(f"{name:>13}: {color.hex}  rgb{color.rgb}")

    print("\nMix: 70% joy + 30% fear")
    print(mix_emotions({"joy": 0.7, "fear": 0.3}).hex)

    print("\nDirect VA: valence=+0.2, arousal=0.9")
    print(emotion_to_color((0.2, 0.9)).hex)
