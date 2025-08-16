# Export a palette (JSON) of predicted colors
import json
preds = mapper.predict_all()
export = {
    emo: {
        "hex": cp.hex,
        "rgb": cp.rgb,
        "hsl": [float(cp.hsl[0]), float(cp.hsl[1]), float(cp.hsl[2])],
        "nearest_css_name": cp.nearest_name
    }
    for emo, cp in preds.items()
}
with open("emotion_palette.json", "w", encoding="utf-8") as f:
    json.dump(export, f, indent=2)
print("Wrote emotion_palette.json")
