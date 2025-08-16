# poem_colorizer.py
from __future__ import annotations

import re
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

# External deps
# pip install gensim scikit-learn matplotlib webcolors
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api

# Your earlier module
from emotion_color_mapper import (
    EmotionColorMapper,
    rgb_to_lab, lab_to_rgb, rgb_to_hsl, rgb_to_hex, hex_to_rgb,
    deltaE2000
)

# ----------------------------- Lexicon & Seeds --------------------------------

PRIMARY_EMOTIONS = [
    "joy","trust","fear","surprise","sadness","disgust","anger","anticipation"
]

INTENSITY_VARIANTS = {
    "joy":        ["serenity", "ecstasy"],
    "trust":      ["acceptance", "admiration"],
    "fear":       ["apprehension", "terror"],
    "surprise":   ["distraction", "amazement"],
    "sadness":    ["pensiveness", "grief"],
    "disgust":    ["boredom", "loathing"],
    "anger":      ["annoyance", "rage"],
    "anticipation":["interest", "vigilance"],
}

# Extended "love" subtypes (secondary/complex blends)
LOVE_SUBTYPES = [
    "love", "romantic_love", "parental_love", "friendship", "aesthetic_love"
]

EMOTION_LEXICON: List[str] = (
    PRIMARY_EMOTIONS
    + sum(INTENSITY_VARIANTS.values(), [])
    + LOVE_SUBTYPES
)

# Plutchik primaries + extended love colors (as shared earlier)
PALETTE_SEEDS = {
    "joy": "#FFFF00", "trust": "#00FF00", "fear": "#00FFFF", "surprise": "#0000FF",
    "sadness": "#800080", "disgust": "#008000", "anger": "#FF0000", "anticipation": "#FFA500",
    "love": "#ADFF2F", "romantic_love": "#FFD580", "parental_love": "#40E0D0",
    "friendship": "#E6FFB3", "aesthetic_love": "#87CEFA",
}

# Intensity variants inherit their primary’s seed (the regression will smooth)
for k, variants in INTENSITY_VARIANTS.items():
    for v in variants:
        if v not in PALETTE_SEEDS:
            PALETTE_SEEDS[v] = PALETTE_SEEDS[k]

# ----------------------------- Utilities --------------------------------------

STOPWORDS = set("""
a an and are as at be but by for from had has have he her hers him his i in is it its
of on or our out she so that the their them they this to was were will with you your
me my mine we us our ours not nor if into over under up down again once off own same
""".split())

TOKEN_RE = re.compile(r"[A-Za-z']+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

def wcag_text_color_for_bg(rgb: Tuple[int,int,int]) -> str:
    # Return '#000000' or '#FFFFFF' based on relative luminance (WCAG)
    r, g, b = [x/255.0 for x in rgb]
    def lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
    R, G, B = lin(r), lin(g), lin(b)
    L = 0.2126*R + 0.7152*G + 0.0722*B
    return "#000000" if L > 0.5 else "#FFFFFF"

def safe_mean(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    return np.mean(np.vstack(vectors), axis=0)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

# ----------------------------- Emotion Scorer ---------------------------------

@dataclass
class EmotionScore:
    emotion: str
    score: float

class EmbeddingEmotionScorer:
    """
    Scores text spans (line, sentence, or token windows) against an emotion lexicon
    using cosine similarity in embedding space.

    Strategy:
      - Precompute the vector for each emotion word (from KeyedVectors).
      - For a span, average the token vectors -> span vector.
      - Score = max cosine(span_vec, emotion_vec) or cosine(span_vec, emotion_centroid).
    """

    def __init__(self, kv: KeyedVectors, emotions: Iterable[str]):
        self.kv = kv
        self.emotions = [e.lower() for e in emotions]
        self.evecs: Dict[str, np.ndarray] = {
            e: kv[e] for e in self.emotions if e in kv
        }
        if not self.evecs:
            raise ValueError("None of the emotion words were in the embedding vocab.")

    def span_vector(self, tokens: List[str]) -> Optional[np.ndarray]:
        vecs = [self.kv[t] for t in tokens if t in self.kv and t not in STOPWORDS]
        return safe_mean(vecs)

    def score_span(self, tokens: List[str], topk: int = 3) -> List[EmotionScore]:
        sv = self.span_vector(tokens)
        if sv is None:
            return []
        pairs = []
