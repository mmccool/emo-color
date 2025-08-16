import json

with open("plutchik_palette.json", "r") as f:
    palette = json.load(f)

print(palette["joy"])        # {'hex': '#FFFF00'}
print(palette["anticipation"])  # {'hex': '#FFA500'}
