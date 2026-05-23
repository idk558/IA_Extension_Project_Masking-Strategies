import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw


OBJECT_PATTERN = re.compile(
    r"Object\s+\d+\s*-\s*Position:\s*x=(?P<x>-?\d+)\s+y=(?P<y>-?\d+)\s+"
    r"Shape:\s*(?P<shape>\w+)\s+Color:\s*(?P<color>\w+)\s+Material:\s*(?P<material>\w+)",
    flags=re.IGNORECASE,
)

COLOR_MAP = {
    "gray": (150, 150, 150),
    "red": (220, 60, 60),
    "blue": (70, 110, 220),
    "green": (70, 180, 90),
    "brown": (150, 100, 60),
    "purple": (145, 90, 195),
    "cyan": (60, 190, 200),
    "yellow": (230, 200, 60),
}

MATERIAL_HIGHLIGHT = {
    "metal": (255, 255, 255),
    "rubber": (30, 30, 30),
}


def parse_scene_desc(text: str) -> List[Dict]:
    objects = []
    for match in OBJECT_PATTERN.finditer(text):
        objects.append(
            {
                "x": int(match.group("x")),
                "y": int(match.group("y")),
                "shape": match.group("shape").lower(),
                "color": match.group("color").lower(),
                "material": match.group("material").lower(),
            }
        )
    return objects


def scene_to_canvas(x: int, y: int, width: int, height: int):
    px = int(round((x / 100.0) * (width - 1)))
    py = int(round((y / 100.0) * (height - 1)))
    return px, py


def draw_sphere(draw: ImageDraw.ImageDraw, center_x: int, center_y: int, radius: int, fill, material: str):
    bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
    draw.ellipse(bbox, fill=fill, outline=(20, 20, 20), width=2)
    highlight = MATERIAL_HIGHLIGHT.get(material, (255, 255, 255))
    draw.ellipse(
        [center_x - radius // 2, center_y - radius // 2, center_x - radius // 6, center_y - radius // 6],
        fill=highlight,
    )


def draw_cube(draw: ImageDraw.ImageDraw, center_x: int, center_y: int, size: int, fill, material: str):
    half = size // 2
    front = [
        (center_x - half, center_y - half),
        (center_x + half, center_y - half),
        (center_x + half, center_y + half),
        (center_x - half, center_y + half),
    ]
    offset = max(6, size // 5)
    back = [(x + offset, y - offset) for x, y in front]
    draw.polygon(back, fill=tuple(max(0, c - 30) for c in fill), outline=(20, 20, 20))
    for f, b in zip(front, back):
        draw.line([f, b], fill=(20, 20, 20), width=2)
    draw.polygon(front, fill=fill, outline=(20, 20, 20))
    if material == "metal":
        draw.line([front[0], front[1]], fill=(255, 255, 255), width=2)


def draw_cylinder(draw: ImageDraw.ImageDraw, center_x: int, center_y: int, width: int, height: int, fill, material: str):
    half_w = width // 2
    half_h = height // 2
    top_bbox = [center_x - half_w, center_y - half_h - 6, center_x + half_w, center_y - half_h + 6]
    bot_bbox = [center_x - half_w, center_y + half_h - 6, center_x + half_w, center_y + half_h + 6]
    body_bbox = [center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h]
    draw.rectangle(body_bbox, fill=fill, outline=(20, 20, 20), width=2)
    draw.ellipse(top_bbox, fill=tuple(min(255, c + 20) for c in fill), outline=(20, 20, 20), width=2)
    draw.ellipse(bot_bbox, fill=tuple(max(0, c - 20) for c in fill), outline=(20, 20, 20), width=2)
    if material == "metal":
        draw.arc(top_bbox, start=180, end=360, fill=(255, 255, 255), width=2)


def render_scene_desc(text: str, output_path: Path, width: int = 512, height: int = 512):
    objects = parse_scene_desc(text)
    image = Image.new("RGB", (width, height), (245, 245, 248))
    draw = ImageDraw.Draw(image)

    # subtle floor line
    draw.rectangle([0, int(height * 0.78), width, height], fill=(232, 232, 236))

    objects = sorted(objects, key=lambda obj: obj["y"])
    for obj in objects:
        fill = COLOR_MAP.get(obj["color"], (180, 180, 180))
        px, py = scene_to_canvas(obj["x"], obj["y"], width, height)
        py = int(height * 0.85 - (py / height) * (height * 0.6))

        if obj["shape"] == "sphere":
            draw_sphere(draw, px, py, radius=22, fill=fill, material=obj["material"])
        elif obj["shape"] == "cube":
            draw_cube(draw, px, py, size=42, fill=fill, material=obj["material"])
        elif obj["shape"] == "cylinder":
            draw_cylinder(draw, px, py, width=38, height=52, fill=fill, material=obj["material"])
        else:
            draw_sphere(draw, px, py, radius=20, fill=fill, material=obj["material"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render a simple CLEVR-like PNG from a scene description.")
    parser.add_argument("--scene-desc", help="Scene description text.")
    parser.add_argument("--scene-desc-file", help="Optional text file containing the scene description.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    args = parser.parse_args()

    text: Optional[str] = args.scene_desc
    if args.scene_desc_file:
        text = Path(args.scene_desc_file).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Provide either --scene-desc or --scene-desc-file.")

    render_scene_desc(text, Path(args.output))
    print(f"Saved rendered scene to {args.output}")


if __name__ == "__main__":
    main()
