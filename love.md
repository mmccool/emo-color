Where could “love” and related states fit?

As blends of primaries
Plutchik emphasized that complex emotions arise as combinations of the primaries.

Love (romantic, companionate, familial, etc.) → Joy + Trust

Joy contributes positive affect, pleasure.

Trust contributes bonding, connection.

This is explicitly in Plutchik’s writings: he placed “love” at the overlap of joy and trust.

Different “flavors” of love
Using Plutchik’s blending framework, you can differentiate love subtypes by adding nuance:

Romantic/Sexual love: Joy + Trust + Anticipation (positive pleasure, bonding, and arousal/anticipation).

Parental/Protective love: Joy + Trust + Fear (bonding and pleasure, but fear of loss/harm).

Brotherly/Companion love: Joy + Trust + Sadness (mild) (pleasure + bonding + vulnerability).

Friendship: Joy + Trust with lower intensity.

Love of art/beauty: Joy + Surprise (pleasure + awe/novelty).

In other words: “love” is not a missing spoke, but a composite emotion cluster.

Extending the wheel
If you want your palette / lexicon to explicitly include these, you can:

Treat Love as a secondary/complex emotion node on the wheel.

Add sub-lexical entries (romantic, parental, aesthetic) and define them as blends of Plutchik primaries.

This also maps well to embeddings (word2vec will naturally place “love” closer to “joy”/“trust” words than “anger” or “disgust”).

🎨 Where do they sit on the color wheel?

If you keep Plutchik’s geometry, Love belongs between Joy (yellow) and Trust (green) — usually shown as a yellow-green wedge.

Variants can shift hue slightly depending on the blend:

Romantic love → warmer (add orange from anticipation).

Parental love → cooler, softer (add blue-green from fear/trust).

Aesthetic love → toward cyan/blue (joy+surprise).

✅ Practical suggestion for your model

Keep Plutchik’s base 8 primaries + intensities.

Add “Love” and its subtypes as extended lexicon entries.

love (general) = joy + trust

romantic love = joy + trust + anticipation

parental love = joy + trust + fear

friendship = joy + trust (lower intensity)

aesthetic love = joy + surprise

Color assignment: interpolate in LAB or HSL between the primaries involved (your regression model already does smooth blending!).

That way, you respect the Plutchik foundation, but extend it to capture the richer humanistic emotions you mentioned.
