# color_code (see results_graph.py) is an (18, 3) array of R/G/B floats in
# [0, 1] - a format matplotlib accepts directly as a color tuple, but Plotly
# needs a CSS color string.
def rgb(color_code, i):
    r, g, b = color_code[i]
    return f"rgb({round(r * 255)},{round(g * 255)},{round(b * 255)})"
