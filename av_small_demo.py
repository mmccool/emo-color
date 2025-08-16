# Offline demo: train a tiny Word2Vec on the fly (no downloads)
# self-contained example that trains a small Word2Vec on synthetic sentences just to show the API.
# Results won’t be great (tiny corpus), but every function will run.
# * Shows training your own Word2Vec vectors and passing embeddings=...
# * Uses model_type="kernel_rbf" for a smoother fit with sparse seeds.

# Requires: gensim, scikit-learn, matplotlib

from gensim.models import Word2Vec
from emotion_color_mapper import EmotionColorMapper
import matplotlib.pyplot as plt

# Tiny synthetic corpus featuring emotion words co-occurring with “colorish” contexts
sentences = [
    "joy delight excitement sunshine bright happy warm glow".split(),
    "calm serenity contentment gentle breeze quiet lake green".split(),
    "anger rage tension hot fire burning red intense".split(),
    "fear anxiety doubt shadow night cold blue deep".split(),
    "sadness grief loneliness rain gray blue cold quiet".split(),
    "love compassion gratitude heart warmth pink red tender".split(),
    "surprise anticipation sudden shock spark flash sky cyan".split(),
    "disgust revulsion gross rotten swamp olive green murky".split(),
    "trust hope confidence steady stable seagreen balance".split(),
]

# Train a tiny Word2Vec (e.g., 50D just for the demo)
w2v = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    seed=42,
    sg=1,
)
kv = w2v.wv  # KeyedVectors

EMOTIONS = [
    "joy","delight","excitement","calm","serenity","contentment","anger","rage","tension",
    "fear","anxiety","sadness","grief","loneliness","love","compassion","gratitude",
    "surprise","anticipation","disgust","revulsion","trust","hope"
]

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

# Build mapper using our in-memory KeyedVectors
mapper = EmotionColorMapper(
    embeddings=kv,              # <— using our tiny model
    model_type="kernel_rbf",    # non-linear for very small datasets
    alpha=1.0,
)

# Train & predict
mapper.fit(EMOTIONS, SEED_COLORS)

print("Color for 'gratitude':", mapper.predict_color("gratitude").hex)
print("Nearest emotions to 'gold':", mapper.nearest_emotions_for_color("gold", k=3))
print("Nearest CSS names to 'rage':", mapper.nearest_color_names_for_emotion("rage", k=3))

# Plot
mapper.plot_3d(title="Tiny Word2Vec demo (colors are approximate!)")
plt.show()
