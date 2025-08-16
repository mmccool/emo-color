# Primary Plutchik emotions (8)
# Primary, tertiary, and tertiary terms, giving 24 emotion words;
# Can be expanded with synonyms from the NRC emotion lexicon (14k words).
PRIMARY_EMOTIONS = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation"
]

# Intensity variants (stronger/weaker forms)
INTENSITY_VARIANTS = {
    "joy":        ["serenity", "ecstasy"],
    "trust":      ["acceptance", "admiration"],
    "fear":       ["apprehension", "terror"],
    "surprise":   ["distraction", "amazement"],
    "sadness":    ["pensiveness", "grief"],
    "disgust":    ["boredom", "loathing"],
    "anger":      ["annoyance", "rage"],
    "anticipation":["interest", "vigilance"]
}

# Flatten into a lexicon list
EMOTION_LEXICON = PRIMARY_EMOTIONS + sum(INTENSITY_VARIANTS.values(), [])
