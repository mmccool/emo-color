# Emotion/Color Mapping
Python code to generate mappings between emotion and color words.
There are two codes here: one based on the standard arousal/valence model and a fixed mapping to colors,
and one that is more configurable and can learn a model from word2vec embeddings and specified seed mappings.

# Prompt
Initial code developed by ChatGPT-5 with the following prompt:

* Develop a model, expressed in python code, to map emotions to colors.

This gave some initial code that just used a simple valence/arousal model, checked in under `av_mapping.py`.
However, this was not configurable enough for what I wanted,
in particular I wanted particular emotion words to map to particular colors,
with the other mappings being as consistent as possible.

I have checked that that version of the code for comparison and testing (e.g. to generate seed pairs for testing), but also
updated the prompt with the following.  The result is checked in under `vec_mapping.py`.

* I want to do this from first principles. I want to take a database of emotion words, embed them in a high-dimensional vector space (for instance, using word2vec and a 30 to 300-dimensional space), use PCA to find three principal dimensions describing the variability in that vector space, then as additional input I want to provide color mappings for some number of particular emotions, and then build a 3-dimensional regression model that can map other colors consistently.  The mapping should be regularized.  Finally, I want to be able to visualize the mapping in 3D and also have a function that can take a particular colour and give me a set of the closest emotions it maps to, and vice-versa, that can take a particular emotion and give me the set of the closest colors it maps to.  Colors should be expressed not only in RGB/HSL values but also in color word names.

# Notes - Arousal/Valence Model

* Valence controls lightness: pleasant → lighter/brighter; unpleasant → darker.
* Arousal controls saturation: calm → muted; activated → vivid.
* The hue comes from the angle in the circumplex (centered at arousal 0.5):
* High arousal + negative valence tends toward energetic reds/magentas (anger/fear zone).
* High arousal + near-neutral valence tends toward blues/cyans (surprise).
* Positive valence + moderate arousal tends toward warm yellows/oranges (joy/delight).
* Positive valence + low arousal shifts greenish and softer (calm/trust).

Customize it:
* Edit EMOTION_PRIORS to better match your domain (e.g., UX research, therapy notes, marketing).
* Tweak the sat and light formulas to nudge vividness/brightness for your aesthetic.
* Add more synonyms to SYNONYMS so messy inputs still map nicely.
  
# Notes - Word2Vec/PCA Model

## Embeddings (30–300D):
  Pass your own KeyedVectors (trained Word2Vec) via embeddings=…, or let it auto-load (glove-wiki-gigaword-300).
  The pipeline doesn’t care what the original dimensionality is—PCA reduces to 3.

## Regularization:
  Default is Ridge (linear, L2). Set alpha higher for stronger smoothing.
  For a smoother, possibly more faithful fit with sparse seeds, use model_type="kernel_rbf". Leave rbf_gamma=None to use a median-distance heuristic.

## Seeds:
  Use as many labeled pairs as you like; more seeds → better generalization. You can change the target space easily (e.g., predict LCh instead of Lab).

## Color names:
  If you pip install webcolors, you’ll get the full CSS3 name set automatically. Otherwise, a compact fallback dictionary is used.
  You can swap in the XKCD color survey list (≈950 names) for richer naming: just load that JSON and compute nearest by ΔE2000 like we do for CSS.

## OOV handling:
  Words missing from embeddings are skipped. Set oov_policy="avg_subtokens" to average subword tokens split on hyphens/underscores.

# Gotchas
* If gensim isn’t installed or you’re offline, use the `av_small_demo.py` to create a tiny test embedding.
* Color naming uses CSS3 via webcolors if present, otherwise a small built-in CSS3 fallback list.
  For richer names (~950), swap in the XKCD color list and use the same ΔE2000 nearest-name logic.
* Predictions are built in CIELAB, then converted to sRGB with gamut clipping. Distances (emotion to color) use ΔE2000.
* Regularization: raise alpha for smoother, less wiggly fits; try kernel_rbf when seeds are sparse or the mapping is clearly nonlinear.
