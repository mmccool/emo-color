# Map a color to the k closest emotions
print(mapper.nearest_emotions_for_color("#ffd700", k=7))  # 'gold' in hex
print(mapper.nearest_emotions_for_color((255, 64, 0), k=7))  # RGB tuple
print(mapper.nearest_emotions_for_color("royalblue", k=7))   # CSS name
