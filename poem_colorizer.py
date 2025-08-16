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
        for emo, ev in self.evecs.items():
            pairs.append(EmotionScore(emo, cosine(sv, ev)))
        pairs.sort(key=lambda x: x.score, reverse=True)
        return pairs[:topk]

# ----------------------------- Colorizer --------------------------------------

class PoemColorizer:
    """
    Combines:
      - EmbeddingEmotionScorer for detection
      - EmotionColorMapper for color prediction (Lab regression from 3D PCA)
    Produces:
      - Line-level colored HTML (top emotion or blended top-k)
      - Word-level highlights (optional)
      - Legend
    """

    def __init__(
        self,
        kv: Optional[KeyedVectors] = None,
        preload: str = "glove-wiki-gigaword-300",
        mapper: Optional[EmotionColorMapper] = None,
        model_type: str = "ridge",
        alpha: float = 3.0
    ):
        # Embeddings
        if kv is None:
            self.kv = api.load(preload)
        else:
            self.kv = kv

        # Emotion mapper (trained on lexicon with seeds)
        self.mapper = mapper or EmotionColorMapper(
            embeddings=self.kv,
            model_type=model_type,
            alpha=alpha
        )
        self.mapper.fit(EMOTION_LEXICON, PALETTE_SEEDS)

        # Emotion scorer
        self.scorer = EmbeddingEmotionScorer(self.kv, EMOTION_LEXICON)

        # Cache predicted Lab for speed
        self._emo_lab = {e: self.mapper._pred_lab_for(e) for e in self.mapper._emotions}

    def _blend_emotions_lab(self, emotions: List[EmotionScore]) -> Tuple[int,int,int]:
        """
        Weighted average in Lab of the predicted colors for top emotions,
        then convert to sRGB. Weights = softmax over scores.
        """
        if not emotions:
            return (224, 224, 224)
        scores = np.array([max(0.0, es.score) for es in emotions], dtype=float)
        if np.all(scores == 0):
            weights = np.ones_like(scores) / len(scores)
        else:
            # Softmax with temperature to sharpen (T=0.5)
            T = 0.5
            ex = np.exp(scores / max(1e-6, T))
            weights = ex / np.sum(ex)
        labs = []
        for es in emotions:
            if es.emotion not in self._emo_lab:
                continue
            labs.append(np.array(self._emo_lab[es.emotion], dtype=float))
        if not labs:
            return (224, 224, 224)
        L = np.average(np.stack(labs), axis=0, weights=weights)
        rgb = lab_to_rgb(tuple(float(x) for x in L))
        return rgb

    def _top_emotion_color(self, emotions: List[EmotionScore]) -> Tuple[str, Tuple[int,int,int]]:
        if not emotions:
            return ("neutral", (224,224,224))
        top = emotions[0].emotion
        pred = self.mapper.predict_color(top)
        return (top, pred.rgb)

    def colorize_poem_to_html(
        self,
        poem_text: str,
        *,
        mode: str = "line",      # "line" or "word"
        topk: int = 3,
        blend: bool = True,
        show_scores: bool = False,
        title: Optional[str] = None
    ) -> str:
        """
        Returns a full HTML page as a string.
        mode="line": color each line by its top emotion (or blended top-k)
        mode="word": color each word by its own top emotion (or blended)
        """

        css = """
        <style>
        body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 2rem; }
        .poem { white-space: pre-wrap; line-height: 1.8; }
        .line { padding: 0.25rem 0.5rem; border-radius: 8px; margin: 2px 0; display: block; }
        .word { padding: 0.05rem 0.2rem; border-radius: 6px; margin: 0 2px; display: inline-block; }
        .legend { margin-top: 1.5rem; display: grid; grid-template-columns: repeat(auto-fill,minmax(180px,1fr)); gap: 8px; }
        .legend-item { display: flex; align-items: center; gap: 8px; }
        .swatch { width: 20px; height: 20px; border-radius: 4px; border: 1px solid rgba(0,0,0,.25); }
        .title { font-weight: 700; font-size: 1.2rem; margin-bottom: .75rem; }
        .score { opacity: 0.8; font-size: 0.85em; margin-left: .5rem; }
        </style>
        """

        lines = poem_text.strip("\n").split("\n")
        html_lines: List[str] = []

        if mode == "line":
            for line in lines:
                toks = tokenize(line)
                es = self.scorer.score_span(toks, topk=topk)
                if blend and len(es) > 1:
                    rgb = self._blend_emotions_lab(es)
                    label = " + ".join([e.emotion for e in es])
                else:
                    top, rgb = self._top_emotion_color(es)
                    label = top
                fg = wcag_text_color_for_bg(rgb)
                bg_hex = rgb_to_hex(rgb)
                score_str = ""
                if show_scores and es:
                    score_str = " ".join([f"{e.emotion}:{e.score:.2f}" for e in es])
                    score_str = f'<span class="score">({score_str})</span>'
                html_lines.append(
                    f'<span class="line" style="background:{bg_hex};color:{fg}">{escape_html(line)}'
                    f' <span class="score">[{label}]</span>{score_str}</span>'
                )

        elif mode == "word":
            for line in lines:
                word_spans: List[str] = []
                for token in re.findall(r"\s+|[^\s]+", line):
                    if token.isspace():
                        word_spans.append(token)
                        continue
                    toks = tokenize(token)
                    es = self.scorer.score_span(toks, topk=topk)
                    if blend and len(es) > 1:
                        rgb = self._blend_emotions_lab(es)
                        label = es[0].emotion if es else "neutral"
                    else:
                        label, rgb = self._top_emotion_color(es)
                    fg = wcag_text_color_for_bg(rgb)
                    bg_hex = rgb_to_hex(rgb)
                    word_spans.append(
                        f'<span class="word" title="{label}" style="background:{bg_hex};color:{fg}">{escape_html(token)}</span>'
                    )
                html_lines.append('<span class="line">' + "".join(word_spans) + "</span>")
        else:
            raise ValueError("mode must be 'line' or 'word'")

        legend_html = self._legend_html()

        title_html = f'<div class="title">{escape_html(title)}</div>' if title else ""
        html = (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            + css
            + "</head><body>"
            + title_html
            + '<div class="poem">' + "\n".join(html_lines) + "</div>"
            + legend_html
            + "</body></html>"
        )
        return html

    def _legend_html(self) -> str:
        # Show legend for the core palette (predicted colors for each emotion)
        all_preds = self.mapper.predict_all()
        blocks = []
        # Show a compact, curated subset first (you can switch to full)
        show_list = PRIMARY_EMOTIONS + LOVE_SUBTYPES
        for emo in show_list:
            if emo not in all_preds: 
                continue
            cp = all_preds[emo]
            fg = wcag_text_color_for_bg(cp.rgb)
            blocks.append(
                f'<div class="legend-item"><div class="swatch" style="background:{cp.hex}"></div>'
                f'<div style="color:{fg}">{emo} <span style="opacity:.7">({cp.nearest_name})</span></div></div>'
            )
        return '<div class="legend">' + "".join(blocks) + "</div>"

def escape_html(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
              .replace('"',"&quot;").replace("'","&#39;"))
