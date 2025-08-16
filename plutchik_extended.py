import colorsys, json

PRIMARY_COLORS = {
    "joy": "#FFFF00",         # Yellow
    "trust": "#00FF00",       # Green
    "fear": "#00FFFF",        # Cyan
    "surprise": "#0000FF",    # Blue
    "sadness": "#800080",     # Purple
    "disgust": "#008000",     # Dark Green
    "anger": "#FF0000",       # Red
    "anticipation": "#FFA500" # Orange
}

INTENSITY_VARIANTS = {
    "joy":        {"weak": "serenity", "strong": "ecstasy"},
    "trust":      {"weak": "acceptance", "strong": "admiration"},
    "fear":       {"weak": "apprehension", "strong": "terror"},
    "surprise":   {"weak": "distraction", "strong": "amazement"},
    "sadness":    {"weak": "pensiveness", "strong": "grief"},
    "disgust":    {"weak": "boredom", "strong": "loathing"},
    "anger":      {"weak": "annoyance", "strong": "rage"},
    "anticipation":{"weak": "interest", "strong": "vigilance"}
}

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def adjust_hsl(hex_color, sat_factor=1.0, light_factor=1.0):
    r, g, b = [v/255.0 for v in hex_to_rgb(hex_color)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # note: HLS order in Python
    s = max(0, min(1, s * sat_factor))
    l = max(0, min(1, l * light_factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return rgb_to_hex((int(r*255), int(g*255), int(b*255)))

# Build palette
palette = {}

for emo, base_hex in PRIMARY_COLORS.items():
    # Primary
    palette[emo] = {"hex": base_hex, "variant": "primary"}
    # Weak
    weak = INTENSITY_VARIANTS[emo]["weak"]
    palette[weak] = {"hex": adjust_hsl(base_hex, sat_factor=0.8, light_factor=1.2), "variant": "weak"}
    # Strong
    strong = INTENSITY_VARIANTS[emo]["strong"]
    palette[strong] = {"hex": adjust_hsl(base_hex, sat_factor=1.2, light_factor=0.8), "variant": "strong"}

# Export
with open("plutchik_palette_extended.json", "w") as f:
    json.dump(palette, f, indent=2)

print(json.dumps(palette, indent=2)[:800])  # preview
