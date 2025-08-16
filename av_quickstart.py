# av_quickstart.py

# Quickstart (uses a pretrained embedding if available)
# What this example covers:
# * fit(...), predict_color(...), predict_all(...)
# * nearest_emotions_for_color(...), nearest_color_names_for_emotion(...)
# * plot_3d(...)

# Requires: scikit-learn, matplotlib
# Optional but recommended: gensim, webcolors

from emotion_color_mapper import EmotionColorMapper

# A small universe of emotion words to embed & predict
EMOTIONS = [
    "joy","delight","amusement","contentment","calm","relief","love","gratitude","pride",
    "hope","trust","anticipation","surprise","anger","rage","fear","anxiety","disgust",
    "shame","guilt","sadness","grief","boredom","loneliness","regret","envy","excitement",
    "compassion","tension","serenity"
]

# Seed supervision (emotion -> color). You can use CSS names, hex, or (R,G,B)
SEED_COLORS = {
    "joy": "#ffd34d",
    "calm": "lightseagreen",
    "love": "crimson",
    "fear": "midnightblue",
    "anger": "tomato",
    "sadness": "royalblue",
    "surprise": "deepskyblue",
    "disgust": "olive",
    "trust": "seagreen",
}

# Build the mapper (auto-loads a pretrained embedding if gensim is installed)
mapper = EmotionColorMapper(
    embeddings=None,                        # or pass your KeyedVectors
    preload="glove-wiki-gigaword-300",      # change to what you have locally
    model_type="ridge",                     # try "kernel_rbf" later
    alpha=3.0,                              # L2 regularization
)

# --- Train (fit) ---
mapper.fit(EMOTIONS, SEED_COLORS)

# --- Predict a single emotion → color ---
pred = mapper.predict_color("anticipation")
print("anticipation →")
print("  hex:", pred.hex)
print("  rgb:", pred.rgb)
print("  hsl:", tuple(round(x,3) for x in pred.hsl))
print("  nearest CSS name:", pred.nearest_name, f"(ΔE2000≈{pred.nearest_name_deltaE:.2f})")

# --- Predict all emotions → colors (dict of {emotion: ColorPred}) ---
all_preds = mapper.predict_all()
for k in ["joy","anger","sadness","calm"]:
    cp = all_preds[k]
    print(f"{k:>10} → {cp.hex}  {cp.rgb}  ≈ {cp.nearest_name}")

# --- Color → nearest emotions ---
print("\nNearest emotions to 'gold':")
print(mapper.nearest_emotions_for_color("gold", k=5))

# --- Emotion → nearest color names ---
print("\nNearest CSS color names to 'envy':")
print(mapper.nearest_color_names_for_emotion("envy", k=5))

# --- 3D visualization in PCA space (colored by predicted sRGB) ---
# NOTE: This opens a matplotlib window.
mapper.plot_3d(title="Emotion → Color (PCA space)")
import matplotlib.pyplot as plt
plt.show()
