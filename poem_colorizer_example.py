# demo_colorize_poem.py
from poem_colorizer import PoemColorizer

poem = """
I saw the sun spill gold across the sea,
and thought of you: breath held before the tide.
A hush, a tremor—lightning under skin—
then laughter, easy trust, and home besides.

But when the night leaned close with salt and glass,
I kept you near; the wind could not divide
the fragile thread we spun from simple hours—
a sky-blue ache, amazed and open-eyed.
""".strip()

# Build the colorizer (auto-loads glove-wiki-gigaword-300 if available)
colorizer = PoemColorizer(preload="glove-wiki-gigaword-300", model_type="kernel_rbf", alpha=1.0)

# 1) Line-level, blended top-3 emotions, show scores
html_line = colorizer.colorize_poem_to_html(
    poem_text=poem,
    mode="line",
    topk=3,
    blend=True,
    show_scores=True,
    title="Poem — Line-level blended emotions"
)
with open("poem_colored_lines.html", "w", encoding="utf-8") as f:
    f.write(html_line)

# 2) Word-level highlights (top-2, blended)
html_word = colorizer.colorize_poem_to_html(
    poem_text=poem,
    mode="word",
    topk=2,
    blend=True,
    show_scores=False,
    title="Poem — Word-level emotional highlights"
)
with open("poem_colored_words.html", "w", encoding="utf-8") as f:
    f.write(html_word)

print("Wrote poem_colored_lines.html and poem_colored_words.html")
