from pathlib import Path
import csv

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FONT_PATH = Path(r"C:\Windows\Fonts\YuGothM.ttc")


def font(size: int):
    return ImageFont.truetype(str(FONT_PATH), size)


def centered(draw, xy, text, text_font, fill="#222222"):
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, font=text_font, fill=fill)


categories = [
    "競技FPS", "ゲーム＋動画編集", "生成AI", "WQHDゲーム", "4Kゲーム",
    "ゲーム配信", "3D制作", "学生向け", "シミュレーション", "VRゲーム",
]
similarities = [0.580, 0.875, 0.753, 0.910, 0.852, 0.863, 0.687, 0.859, 0.761, 0.931]

image = Image.new("RGB", (1800, 1050), "white")
draw = ImageDraw.Draw(image)
centered(draw, (900, 35), "言い換え入力10件に対するTop-1類似度", font(42))
left, right, top, bottom = 430, 1640, 120, 900
for tick in range(0, 11, 2):
    x = left + (right - left) * tick / 10
    draw.line((x, top, x, bottom), fill="#D8DDE5", width=2)
    centered(draw, (x, bottom + 15), f"{tick / 10:.1f}", font(27), "#444444")
row_height = (bottom - top) / len(categories)
for index, (category, value) in enumerate(zip(categories, similarities)):
    y0 = top + index * row_height + 12
    y1 = top + (index + 1) * row_height - 12
    draw.text((35, y0 + 7), category, font=font(29), fill="#222222")
    x1 = left + (right - left) * value
    draw.rectangle((left, y0, x1, y1), fill="#4E79A7")
    draw.text((x1 + 15, y0 + 6), f"{value:.3f}", font=font(28), fill="#222222")
draw.line((left, top, left, bottom), fill="#333333", width=3)
draw.line((left, bottom, right, bottom), fill="#333333", width=3)
centered(draw, (1035, 970), "1位候補のコサイン類似度", font(30))
image.save(OUTPUT_DIR / "evaluation_similarity.png")


with (ROOT / "pc_builds.csv").open(encoding="utf-8-sig", newline="") as file:
    prices = [int(row["price_yen"]) // 10_000 for row in csv.DictReader(file)]

ranges = [(10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69)]
counts = [sum(low <= price <= high for price in prices) for low, high in ranges]
labels = [f"{low}～{high}" for low, high in ranges]

image = Image.new("RGB", (1700, 900), "white")
draw = ImageDraw.Draw(image)
centered(draw, (850, 35), "推薦用コーパス26件の価格帯分布", font(42))
left, right, top, bottom = 180, 1580, 130, 720
maximum = max(counts) + 2
for tick in range(maximum + 1):
    y = bottom - (bottom - top) * tick / maximum
    draw.line((left, y, right, y), fill="#D8DDE5", width=2)
    draw.text((125, y - 17), str(tick), font=font(26), fill="#444444")
slot = (right - left) / len(counts)
for index, (label, value) in enumerate(zip(labels, counts)):
    x0 = left + index * slot + 35
    x1 = left + (index + 1) * slot - 35
    y0 = bottom - (bottom - top) * value / maximum
    draw.rectangle((x0, y0, x1, bottom), fill="#59A14F")
    centered(draw, ((x0 + x1) / 2, y0 - 43), str(value), font(30))
    centered(draw, ((x0 + x1) / 2, bottom + 18), label, font(27))
draw.line((left, top, left, bottom), fill="#333333", width=3)
draw.line((left, bottom, right, bottom), fill="#333333", width=3)
centered(draw, (880, 810), "概算価格帯（万円）", font(30))
draw.text((35, 390), "登録\n構成\n数", font=font(28), fill="#222222", spacing=8)
image.save(OUTPUT_DIR / "corpus_price_distribution.png")
