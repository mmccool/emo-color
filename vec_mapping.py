# vec_mapping.py
# Generate an emotion-to-color mapping using a model learned from vector embeddings and a set
# of seed mappings.
from __future__ import annotations

import math
import colorsys
import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# Optional deps (strongly recommended):
#   pip install gensim scikit-learn webcolors
try:
    from gensim.models import KeyedVectors
    from gensim.models.word2vec import Word2Vec
    import gensim.downloader as api
except Exception:
    KeyedVectors = None
    Word2Vec = None
    api = None

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neighbors import NearestNeighbors
except Exception:
    raise RuntimeError("Please install scikit-learn: pip install scikit-learn")

try:
    import webcolors  # for CSS color names (closest match)
except Exception:
    webcolors = None  # we fall back to a built-in CSS3 dictionary below

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # matplotlib needs this import

# --------------------------------------------------------------------------------------
# Minimal CSS3 color name fallback (if webcolors isn't available)
# --------------------------------------------------------------------------------------
CSS3_NAMED_COLORS = {
    # name: hex (subset; expand if needed)
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "lime": "#00ff00", "blue": "#0000ff",
    "yellow": "#ffff00", "cyan": "#00ffff", "aqua": "#00ffff", "magenta": "#ff00ff", "fuchsia": "#ff00ff",
    "silver": "#c0c0c0", "gray": "#808080", "maroon": "#800000", "olive": "#808000", "green": "#008000",
    "purple": "#800080", "teal": "#008080", "navy": "#000080",
    "orange": "#ffa500", "gold": "#ffd700", "pink": "#ffc0cb", "brown": "#a52a2a",
    "chocolate": "#d2691e", "coral": "#ff7f50", "crimson": "#dc143c", "darkblue": "#00008b",
    "darkcyan": "#008b8b", "darkgoldenrod": "#b8860b", "darkgray": "#a9a9a9", "darkgreen": "#006400",
    "darkkhaki": "#bdb76b", "darkmagenta": "#8b008b", "darkolivegreen": "#556b2f", "darkorange": "#ff8c00",
    "darkorchid": "#9932cc", "darkred": "#8b0000", "darksalmon": "#e9967a", "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b", "darkslategray": "#2f4f4f", "deeppink": "#ff1493", "deepskyblue": "#00bfff",
    "dimgray": "#696969", "dodgerblue": "#1e90ff", "firebrick": "#b22222", "forestgreen": "#228b22",
    "gainsboro": "#dcdcdc", "ghostwhite": "#f8f8ff", "gray": "#808080", "greenyellow": "#adff2f",
    "hotpink": "#ff69b4", "indianred": "#cd5c5c", "indigo": "#4b0082", "khaki": "#f0e68c",
    "lavender": "#e6e6fa", "lavenderblush": "#fff0f5", "lawngreen": "#7cfc00", "lightblue": "#add8e6",
    "lightcoral": "#f08080", "lightcyan": "#e0ffff", "lightgreen": "#90ee90", "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a", "lightseagreen": "#20b2aa", "lightskyblue": "#87cefa",
    "lightslategray": "#778899", "lightsteelblue": "#b0c4de", "limegreen": "#32cd32",
    "mediumblue": "#0000cd", "mediumorchid": "#ba55d3", "mediumpurple": "#9370db",
    "mediumseagreen": "#3cb371", "mediumslateblue": "#7b68ee", "mediumspringgreen": "#00fa9a",
    "mediumturquoise": "#48d1cc", "midnightblue": "#191970", "moccasin": "#ffe4b5",
    "navajowhite": "#ffdead", "olivedrab": "#6b8e23", "orchid": "#da70d6", "palegreen": "#98fb98",
    "paleturquoise": "#afeeee", "palevioletred": "#db7093", "peachpuff": "#ffdab9",
    "peru": "#cd853f", "plum": "#dda0dd", "powderblue": "#b0e0e6", "rosybrown": "#bc8f8f",
    "royalblue": "#4169e1", "saddlebrown": "#8b4513", "salmon": "#fa8072", "sandybrown": "#f4a460",
    "seagreen": "#2e8b57", "sienna": "#a0522d", "skyblue": "#87ceeb", "slateblue": "#6a5acd",
    "slategray": "#708090", "springgreen": "#00ff7f", "steelblue": "#4682b4", "tan": "#d2b48c",
    "thistle": "#d8bfd8", "tomato": "#ff6347", "turquoise": "#40e0d0", "violet": "#ee82ee",
    "wheat": "#f5deb3", "whitesmoke": "#f5f5f5", "yellowgreen": "#9acd32",
}

# --------------------------------------------------------------------------------------
# Color conversions (sRGB ↔ XYZ ↔ CIELAB), ΔE2000, helpers
# --------------------------------------------------------------------------------------
def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    s = hex_str.strip().lower()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch*2 for ch in s)
    r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
    return r, g, b

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

def rgb_to_xyz(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [x / 255.0 for x in rgb]
    r, g, b = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    # sRGB D65
    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
    return X, Y, Z

def xyz_to_rgb(xyz: Tuple[float, float, float]) -> Tuple[int, int, int]:
    X, Y, Z = xyz
    r =  3.2404542*X - 1.5371385*Y - 0.4985314*Z
    g = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
    b =  0.0556434*X - 0.2040259*Y + 1.0572252*Z
    r, g, b = _linear_to_srgb(r), _linear_to_srgb(g), _linear_to_srgb(b)
    r, g, b = [int(round(max(0, min(1, v)) * 255)) for v in (r, g, b)]
    return r, g, b

# Lab with D65/2° via XYZ using D65 white
# We convert via D65; for Lab reference, often D50 is used—this is sufficient here.
REF_X, REF_Y, REF_Z = 0.95047, 1.00000, 1.08883  # D65 scaled to Y=1

def _f(t: float) -> float:
    d = 6/29
    return t**(1/3) if t > d**3 else (t/(3*d*d) + 4/29)

def xyz_to_lab(xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    X, Y, Z = xyz
    x, y, z = X/REF_X, Y/REF_Y, Z/REF_Z
    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return L, a, b

def lab_to_xyz(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    L, a, b = lab
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    def finv(f):
        d = 6/29
        return f**3 if f > d else 3*d*d*(f - 4/29)
    x, y, z = finv(fx), finv(fy), finv(fz)
    return x*REF_X, y*REF_Y, z*REF_Z

def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return xyz_to_lab(rgb_to_xyz(rgb))

def lab_to_rgb(lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return xyz_to_rgb(lab_to_xyz(lab))

def rgb_to_hsl(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [v/255.0 for v in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # note: HLS order
    return (h, s, l)  # return as H, S, L to match common naming

# CIEDE2000 ΔE (implementation based on the published formula)
def deltaE2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    avg_L = (L1 + L2) / 2.0
    C1 = math.sqrt(a1*a1 + b1*b1)
    C2 = math.sqrt(a2*a2 + b2*b2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.sqrt(a1p*a1p + b1*b1)
    C2p = math.sqrt(a2p*a2p + b2*b2)
    avg_Cp = (C1p + C2p) / 2.0
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dhp = (h2p - h1p)
    if dhp > 180: dhp -= 360
    if dhp < -180: dhp += 360
    dHp = 2 * math.sqrt(C1p*C2p) * math.sin(math.radians(dhp) / 2.0)
    dLp = L2 - L1
    dCp = C2p - C1p
    avg_hp = h1p + dhp/2.0
    if abs(h1p - h2p) > 180:
        avg_hp += 180
    avg_hp = (avg_hp + 360) % 360
    T = (1
         - 0.17 * math.cos(math.radians(avg_hp - 30))
         + 0.24 * math.cos(math.radians(2 * avg_hp))
         + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
         - 0.20 * math.cos(math.radians(4 * avg_hp - 63)))
    Sl = 1 + (0.015 * (avg_L - 50)**2) / math.sqrt(20 + (avg_L - 50)**2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    delta_ro = 30 * math.exp(-((avg_hp - 275) / 25)**2)
    Rc = 2 * math.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
    Rt = - Rc * math.sin(math.radians(2 * delta_ro))
    kl = kc = kh = 1
    dE = math.sqrt(
        (dLp / (kl * Sl))**2 +
        (dCp / (kc * Sc))**2 +
        (dHp / (kh * Sh))**2 +
        Rt * (dCp / (kc * Sc)) * (dHp / (kh * Sh))
    )
    return dE

# --------------------------------------------------------------------------------------
# Color parsing & nearest color names
# --------------------------------------------------------------------------------------
def parse_color(color: Union[str, Tuple[int,int,int]]) -> Tuple[int, int, int]:
    """
    Accept '#RRGGBB' or CSS name (if webcolors or fallback dict) or (R,G,B)
    """
    if isinstance(color, tuple) and len(color) == 3:
        return tuple(int(v) for v in color)
    if isinstance(color, str):
        s = color.strip().lower()
        # Try hex
        if s.startswith("#") or all(ch in "0123456789abcdef" for ch in s.replace("#","")) and len(s) in (3, 6, 7):
            return hex_to_rgb(s)
        # Try css name
        if webcolors:
            try:
                return webcolors.name_to_rgb(s)
            except Exception:
                pass
        if s in CSS3_NAMED_COLORS:
            return hex_to_rgb(CSS3_NAMED_COLORS[s])
    raise ValueError(f"Unrecognized color: {color}")

def nearest_css_color_name(rgb: Tuple[int,int,int]) -> Tuple[str, float]:
    """
    Return (name, ΔE2000) for nearest CSS color (webcolors if available, else fallback dict).
    """
    lab = rgb_to_lab(rgb)
    best = (None, float("inf"))
    if webcolors:
        # Use CSS3 named colors from webcolors
        for name, hexval in webcolors.CSS3_NAMES_TO_HEX.items():
            c = hex_to_rgb(hexval)
            d = deltaE2000(lab, rgb_to_lab(c))
            if d < best[1]:
                best = (name, d)
    else:
        for name, hexval in CSS3_NAMED_COLORS.items():
            c = hex_to_rgb(hexval)
            d = deltaE2000(lab, rgb_to_lab(c))
            if d < best[1]:
                best = (name, d)
    return best[0], best[1]

# --------------------------------------------------------------------------------------
# Main model
# --------------------------------------------------------------------------------------
@dataclass
class ColorPred:
    hex: str
    rgb: Tuple[int,int,int]
    hsl: Tuple[float,float,float]
    nearest_name: str
    nearest_name_deltaE: float
    lab: Tuple[float,float,float]

class EmotionColorMapper:
    """
    Pipeline:
      1) Build/get embeddings for emotion words (word2vec KeyedVectors or train your own).
      2) PCA → 3D.
      3) Regularized regression (Ridge or Kernel Ridge) from 3D → CIELAB.
      4) Predict colors for all emotions; search nearest neighbors in color or space.

    Notes:
      - Provide seed mapping: {emotion: color}.
      - Colors predicted in Lab; converted to sRGB with gamut clipping.
      - Nearest comparisons use ΔE2000 in Lab.
    """

    def __init__(
        self,
        embeddings: Optional["KeyedVectors"]=None,
        *,
        preload: Optional[str]="glove-wiki-gigaword-300",
        lowercase: bool=True,
        oov_policy: str="skip",   # 'skip' or 'avg_subtokens' (split on -_/ )
        pca_components: int=3,
        model_type: str="ridge",  # 'ridge' (linear) or 'kernel_rbf'
        alpha: float=1.0,         # L2 strength (or KernelRidge alpha)
        rbf_gamma: Optional[float]=None,  # None → median heuristic
        random_state: int=42
    ):
        self.embeddings = embeddings
        self.preload = preload
        self.lowercase = lowercase
        self.oov_policy = oov_policy
        self.pca_components = pca_components
        self.model_type = model_type
        self.alpha = alpha
        self.rbf_gamma = rbf_gamma
        self.random_state = random_state

        self._pca = None
        self._X3 = None              # (n,3) PCA coords for emotions
        self._emotions: List[str] = []
        self._Y_lab_seeds = None     # (m,3) LAB targets for seeds
        self._seed_idx = None        # indices into self._emotions for seeds
        self._reg_models = None      # list of 3 regressors (L,a,b)
        self._pred_lab_all = None    # (n,3) predicted lab for all emotions

    # ---------- embeddings ----------
    def _load_pretrained(self) -> "KeyedVectors":
        if self.embeddings is not None:
            return self.embeddings
        if api is None:
            raise RuntimeError("gensim not available. Pass your own KeyedVectors via embeddings=...")
        print(f"[Info] Loading embeddings: {self.preload}")
        kv = api.load(self.preload)  # e.g., 'glove-wiki-gigaword-300'
        self.embeddings = kv
        return kv

    def _get_vector(self, word: str) -> Optional[np.ndarray]:
        if self.lowercase:
            word = word.lower()
        kv = self._load_pretrained()
        if word in kv:
            return kv[word]
        if self.oov_policy == "avg_subtokens":
            # try splitting on common separators
            parts = [p for p in re_split_tokens(word) if p in kv]
            if parts:
                return np.mean([kv[p] for p in parts], axis=0)
        return None

    def build_space(self, emotions: Iterable[str]) -> Tuple[np.ndarray, List[str]]:
        vecs = []
        kept = []
        for w in emotions:
            v = self._get_vector(w)
            if v is not None:
                vecs.append(v)
                kept.append(w)
        if not vecs:
            raise ValueError("No emotions remained after OOV filtering.")
        X = np.vstack(vecs)
        return X, kept

    # ---------- PCA ----------
    def reduce_to_3d(self, X: np.ndarray) -> np.ndarray:
        if self.pca_components != 3:
            raise ValueError("This pipeline targets 3 principal components.")
        self._pca = PCA(n_components=3, random_state=self.random_state)
        X3 = self._pca.fit_transform(X)
        return X3

    # ---------- regression ----------
    def _build_reg(self):
        if self.model_type == "ridge":
            return [Ridge(alpha=self.alpha, random_state=self.random_state) for _ in range(3)]
        elif self.model_type == "kernel_rbf":
            gamma = self.rbf_gamma
            if gamma is None and self._X3 is not None and len(self._X3) >= 2:
                # median distance heuristic
                nbrs = NearestNeighbors(n_neighbors=min(10, len(self._X3))).fit(self._X3)
                dists, _ = nbrs.kneighbors(self._X3)
                med = np.median(dists[:, 1:])  # skip self-dist
                if med <= 0:
                    med = 1.0
                gamma = 1.0 / (2 * med * med)
            return [KernelRidge(alpha=self.alpha, kernel="rbf", gamma=gamma) for _ in range(3)]
        else:
            raise ValueError("model_type must be 'ridge' or 'kernel_rbf'")

    def fit(
        self,
        emotions: Sequence[str],
        seed_colors: Mapping[str, Union[str, Tuple[int,int,int]]],
    ):
        """
        emotions: universe of emotion words to embed & predict for
        seed_colors: subset mapping {emotion: color} for supervised fit
        """
        # 1) Embeddings
        X_hi, kept = self.build_space(emotions)
        self._emotions = list(kept)

        # 2) PCA to 3D
        self._X3 = self.reduce_to_3d(X_hi)

        # 3) Prepare seed supervision in Lab
        seed_idx = []
        seed_lab = []
        name_to_idx = {w: i for i, w in enumerate(self._emotions)}
        for name, col in seed_colors.items():
            key = name.lower() if self.lowercase else name
            if key not in name_to_idx:
                # If OOV or filtered out, skip but warn
                print(f"[Warn] Seed '{name}' not in embedded emotion list; skipping.")
                continue
            seed_idx.append(name_to_idx[key])
            rgb = parse_color(col)
            seed_lab.append(rgb_to_lab(rgb))
        if not seed_idx:
            raise ValueError("No seed labels matched embedded emotions.")
        self._seed_idx = np.array(seed_idx, dtype=int)
        self._Y_lab_seeds = np.vstack(seed_lab)

        # 4) Fit 3 regressors (for L,a,b)
        self._reg_models = self._build_reg()
        Xs = self._X3[self._seed_idx]
        for j in range(3):
            y = self._Y_lab_seeds[:, j]
            self._reg_models[j].fit(Xs, y)

        # 5) Predict Lab for all emotions
        self._pred_lab_all = np.column_stack([m.predict(self._X3) for m in self._reg_models])

    # ---------- prediction & queries ----------
    def _pred_lab_for(self, emotion: str) -> Tuple[float,float,float]:
        if self._pred_lab_all is None:
            raise RuntimeError("Call fit() first.")
        name_to_idx = {w: i for i, w in enumerate(self._emotions)}
        key = emotion.lower() if self.lowercase else emotion
        if key not in name_to_idx:
            raise KeyError(f"Emotion '{emotion}' not in model vocabulary.")
        return tuple(float(v) for v in self._pred_lab_all[name_to_idx[key], :])

    def predict_color(self, emotion: str) -> ColorPred:
        lab = self._pred_lab_for(emotion)
        rgb = lab_to_rgb(lab)
        hsl = rgb_to_hsl(rgb)
        hexv = rgb_to_hex(rgb)
        name, de = nearest_css_color_name(rgb)
        return ColorPred(hex=hexv, rgb=rgb, hsl=hsl, nearest_name=name, nearest_name_deltaE=de, lab=lab)

    def predict_all(self) -> Dict[str, ColorPred]:
        if self._pred_lab_all is None:
            raise RuntimeError("Call fit() first.")
        out = {}
        for i, name in enumerate(self._emotions):
            lab = tuple(float(v) for v in self._pred_lab_all[i, :])
            rgb = lab_to_rgb(lab)
            hsl = rgb_to_hsl(rgb)
            hexv = rgb_to_hex(rgb)
            cname, de = nearest_css_color_name(rgb)
            out[name] = ColorPred(hex=hexv, rgb=rgb, hsl=hsl, nearest_name=cname, nearest_name_deltaE=de, lab=lab)
        return out

    def nearest_emotions_for_color(
        self,
        color: Union[str, Tuple[int,int,int]],
        k: int=5
    ) -> List[Tuple[str, float]]:
        """
        For a given color, return top-k (emotion, ΔE2000) whose predicted colors are nearest.
        """
        if self._pred_lab_all is None:
            raise RuntimeError("Call fit() first.")
        rgb = parse_color(color)
        target_lab = rgb_to_lab(rgb)
        dists = [deltaE2000(target_lab, tuple(self._pred_lab_all[i, :])) for i in range(len(self._emotions))]
        idx = np.argsort(dists)[:k]
        return [(self._emotions[i], float(dists[i])) for i in idx]

    def nearest_color_names_for_emotion(
        self,
        emotion: str,
        k: int=5
    ) -> List[Tuple[str, float]]:
        """
        Nearest named colors (CSS3) to the emotion's predicted color, by ΔE2000.
        """
        lab = self._pred_lab_for(emotion)
        # Build palette
        if webcolors:
            items = webcolors.CSS3_NAMES_TO_HEX.items()
        else:
            items = CSS3_NAMED_COLORS.items()
        pairs = []
        for name, hexval in items:
            rgb = hex_to_rgb(hexval) if isinstance(hexval, str) else hex_to_rgb(hexval)
            d = deltaE2000(lab, rgb_to_lab(rgb))
            pairs.append((name, float(d)))
        pairs.sort(key=lambda x: x[1])
        return pairs[:k]

    # ---------- visualization ----------
    def plot_3d(
        self,
        figsize: Tuple[int,int]=(8,7),
        annotate: bool=False,
        show_seeds: bool=True,
        elev: int=20,
        azim: int=-60,
        title: Optional[str]=None
    ):
        """
        3D scatter in PCA space, colored by predicted sRGB.
        """
        if self._pred_lab_all is None:
            raise RuntimeError("Call fit() first.")
        colors = [lab_to_rgb(tuple(self._pred_lab_all[i,:])) for i in range(len(self._emotions))]
        cols = [tuple(c_i/255 for c_i in c) for c in colors]
        X = self._X3

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:,0], X[:,1], X[:,2], c=cols, s=30, depthshade=True, edgecolor="none")

        if show_seeds and self._seed_idx is not None and len(self._seed_idx) > 0:
            ax.scatter(
                X[self._seed_idx,0], X[self._seed_idx,1], X[self._seed_idx,2],
                facecolors="none", edgecolors="k", s=80, linewidths=1.5, label="seeds"
            )
            ax.legend(loc="best")

        if annotate:
            for i, name in enumerate(self._emotions):
                ax.text(X[i,0], X[i,1], X[i,2], name, fontsize=7)

        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        if title:
            ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        return fig, ax

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
import re
def re_split_tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[-_/ ]+", text) if t]

# --------------------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Your emotion vocabulary (can be big)
    emotions = [
        "joy","delight","amusement","contentment","calm","relief","love","gratitude","pride","hope","trust",
        "anticipation","surprise","anger","rage","fear","anxiety","disgust","shame","guilt",
        "sadness","grief","boredom","loneliness","regret","serenity","excitement","tension","envy","compassion",
    ]

    # 2) Seed color supervision (pick a handful; adjust as you like)
    #    Accepts hex, CSS3 names, or (R,G,B)
    seed_colors = {
        "joy": "#ffd34d",          # warm golden
        "calm": "lightseagreen",   # soft teal
        "love": "crimson",         # deep pink/red
        "fear": "midnightblue",    # dark blue
        "anger": "tomato",         # hot red/orange
        "sadness": "royalblue",    # medium blue
        "surprise": "deepskyblue", # bright cyan-blue
        "disgust": "olive",        # olive green
        "trust": "seagreen",       # green
    }

    # 3) Build & fit model
    mapper = EmotionColorMapper(
        embeddings=None,             # or pass your own KeyedVectors
        preload="glove-wiki-gigaword-300",  # or 'word2vec-google-news-300' if you have it
        model_type="ridge",          # try 'kernel_rbf' for smoother nonlinear mapping
        alpha=3.0,                   # regularization strength
    )
    mapper.fit(emotions, seed_colors)

    # 4) Query: emotion → color
    pred = mapper.predict_color("anticipation")
    print("anticipation →", pred.hex, pred.rgb, "HSL", tuple(round(x,3) for x in pred.hsl), "≈", pred.nearest_name)

    # 5) Query: color → nearest emotions
    print("Nearest emotions to color 'gold':", mapper.nearest_emotions_for_color("gold", k=5))

    # 6) Named colors nearest to an emotion’s predicted color
    print("Nearest CSS color names to 'envy':", mapper.nearest_color_names_for_emotion("envy", k=5))

    # 7) Visualize
    mapper.plot_3d(title="Emotion → Color (PCA space)"); plt.show()
