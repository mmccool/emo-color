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
