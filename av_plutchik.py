# Create mapping using Plutchik's model.
# Uses
# Relative positions of all 24 emotions, consistent with embeddings.
# Colors smoothly interpolated from Plutchik’s seeds via regression.
# Ability to:
# * Map emotion to color (RGB, HSL, nearest CSS name)
# * Map color to closest emotions
# * Visualize in 3D PCA space.

from emotion_color_mapper import EmotionColorMapper
import matplotlib.pyplot as plt

# Use lexicon and Plutchik seeds
mapper = EmotionColorMapper(
    preload="glove-wiki-gigaword-300",
    model_type="ridge",
    alpha=3.0,
)

mapper.fit(EMOTION_LEXICON, PLUTCHIK_COLOR_SEEDS)

# Example: predict “ecstasy” (an intense form of joy)
print(mapper.predict_color("ecstasy"))

# Visualize the whole wheel in 3D PCA space
mapper.plot_3d(title="Plutchik Wheel (Emotion → Color)")
plt.show()
