#!/usr/bin/env python3
"""Render the repository README and one-page annotation guide as readable PDFs."""
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote
from xml.sax.saxutils import escape, quoteattr

from PIL import Image
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image as PDFImage, KeepTogether, LongTable,
    NextPageTemplate, PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE_W, PAGE_H = A4
MARGIN = 40
WIDTH, HEIGHT = PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN
PDF_NAME = "Step3_3주차_실습과제_21114073_차경호.pdf"
GUIDE_NAME = "Step3_3주차_주석기준서_21114073_차경호.pdf"
TOKEN = re.compile(r"(`[^`]+`|\*\*.+?\*\*|\*[^*]+\*|\$[^$\n]+\$|\[[^\]]+\]\([^)]+\)|<br\s*/?>)")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_file(value, base=ROOT):
    path = (base / unquote(str(value))).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise FileNotFoundError(f"Missing repository input: {value}")
    return path


def normalize(text):
    return str(text).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")


def table_cells(line):
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def parse_markdown(text):
    """Read the README's GFM subset; reject unsupported block HTML instead of dropping it."""
    lines = normalize(text).splitlines()
    blocks, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line in ("<details>", "</details>") or re.fullmatch(r"[-*_]{3,}", line):
            i += 1
            continue
        if line.startswith("<!--"):
            while "-->" not in lines[i]:
                i += 1
                if i == len(lines):
                    raise ValueError("Unclosed Markdown comment")
            i += 1
            continue
        if line.startswith("```"):
            content, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                content.append(lines[i]); i += 1
            if i == len(lines):
                raise ValueError("Unclosed fenced code block")
            blocks.append(("code", "\n".join(content))); i += 1
            continue
        if line.startswith("$$"):
            content = line[2:]
            if content.endswith("$$"):
                blocks.append(("math", content[:-2].strip())); i += 1
                continue
            content, i = [content], i + 1
            while i < len(lines) and "$$" not in lines[i]:
                content.append(lines[i]); i += 1
            if i == len(lines):
                raise ValueError("Unclosed display equation")
            content.append(lines[i].split("$$", 1)[0])
            blocks.append(("math", " ".join(content).strip())); i += 1
            continue
        heading = re.fullmatch(r"(#{1,6})\s+(.+?)\s*#*", line)
        if heading:
            blocks.append(("heading", (len(heading[1]), heading[2]))); i += 1
            continue
        summary = re.fullmatch(r"<summary>(.*?)</summary>", line)
        if summary:
            blocks.append(("heading", (3, summary[1]))); i += 1
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image:
            blocks.append(("image", (image[1], image[2]))); i += 1
            continue
        if i + 1 < len(lines) and "|" in line and all(
                re.fullmatch(r":?-{3,}:?", cell) for cell in table_cells(lines[i + 1])):
            rows, i = [table_cells(line)], i + 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                rows.append(table_cells(lines[i])); i += 1
            if any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("Markdown table column counts differ")
            blocks.append(("table", rows))
            continue
        bullet = re.match(r"^(\s*)([-+*]|\d+[.)])\s+(.+)", lines[i])
        if bullet:
            blocks.append(("list", (bullet[2], bullet[3], len(bullet[1])))); i += 1
            continue
        if line.startswith("<") and not line.startswith("<br"):
            raise ValueError(f"Unsupported block HTML: {line[:70]}")
        content, i = [line.removeprefix("> ")], i + 1
        while i < len(lines) and lines[i].strip():
            following = lines[i].strip()
            if re.match(r"^(#{1,6}\s|```|\$\$|!\[|[-+*]\s|\d+[.)]\s|<)", following):
                break
            if i + 1 < len(lines) and "|" in following and re.search(r"\|?\s*:?-{3,}", lines[i + 1]):
                break
            content.append(following.removeprefix("> ")); i += 1
        blocks.append(("paragraph", " ".join(content)))
    return blocks


class Renderer:
    def __init__(self, temporary, font_dir, photo_dpi=180, photo_quality=85):
        self.temporary, self.photo_dpi, self.photo_quality = temporary, photo_dpi, photo_quality
        self.images, self.equations, self.inputs = {}, {}, {}
        for name, filename in (("Nanum", "NanumGothic.ttf"), ("Nanum-Bold", "NanumGothicBold.ttf"),
                               ("NanumCoding", "NanumGothicCoding.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
        pdfmetrics.registerFontFamily("Nanum", normal="Nanum", bold="Nanum-Bold", italic="Nanum", boldItalic="Nanum-Bold")
        def style(name, size, leading, **kwargs):
            return ParagraphStyle(name, fontName="Nanum", fontSize=size, leading=leading,
                                  textColor=colors.HexColor("#20252b"), allowWidows=0, allowOrphans=0,
                                  splitLongWords=True, **kwargs)
        self.styles = {"body": style("body", 10.7, 16.5, spaceAfter=9),
                       "caption": style("caption", 9.2, 13.5, spaceAfter=10),
                       "cell": style("cell", 9.2, 13),
                       "guide": style("guide", 10, 14),
                       "code": style("code", 8.8, 12.5, leftIndent=8, spaceAfter=12)}
        self.styles["code"].fontName = "NanumCoding"
        for level, size in ((1, 18), (2, 14), (3, 12), (4, 11)):
            s = style(f"h{level}", size, size * 1.45, spaceBefore=15 if level > 1 else 0,
                      spaceAfter=9, keepWithNext=True)
            s.fontName = "Nanum-Bold"
            self.styles[f"h{level}"] = s

    def math_image(self, text, size=12):
        key = (text, size)
        if key not in self.equations:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.font_manager import FontProperties
            from matplotlib.mathtext import math_to_image
            expression = re.sub(r"\\text\{([^}]+)\}", r"\\mathrm{\1}", text)
            path = self.temporary / (hashlib.sha256(repr(key).encode()).hexdigest() + ".png")
            math_to_image("$" + expression + "$", str(path), prop=FontProperties(size=size), dpi=240, format="png")
            with Image.open(path) as im:
                w, h = im.size
            self.equations[key] = (path, w * 72 / 240, h * 72 / 240)
        return self.equations[key]

    def inline(self, text):
        result, pos = [], 0
        for match in TOKEN.finditer(normalize(text)):
            result.append(escape(normalize(text)[pos:match.start()]))
            value = match[0]
            if value.startswith("`"):
                result.append('<font name="NanumCoding" size="9">' + escape(value[1:-1]) + "</font>")
            elif value.startswith("**"):
                result.append("<b>" + self.inline(value[2:-2]) + "</b>")
            elif value.startswith("*"):
                result.append("<i>" + self.inline(value[1:-1]) + "</i>")
            elif value.startswith("$"):
                path, w, h = self.math_image(value[1:-1], 11)
                result.append(f'<img src={quoteattr(str(path))} width="{w}" height="{h}" valign="middle"/>')
            elif value.startswith("["):
                label, url = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", value).groups()
                markup = self.inline(label)
                result.append(f'<link href={quoteattr(url)} color="#285b7a">{markup}</link>'
                              if url.startswith(("https://", "http://")) else markup)
            else:
                result.append("<br/>")
            pos = match.end()
        result.append(escape(normalize(text)[pos:]))
        return "".join(result)

    def paragraph(self, text, name="body", **kwargs):
        return Paragraph(self.inline(text).replace("\n", "<br/>"), self.styles[name], **kwargs)

    def image(self, path, max_width=WIDTH, max_height=None):
        path = local_file(path)
        if max_height is None:
            max_height = 320 if path.suffix.lower() in (".jpg", ".jpeg") else HEIGHT - 100
        self.inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        key = (path, max_width)
        if key not in self.images:
            with Image.open(path) as im:
                original = im.size
                if path.suffix.lower() in (".jpg", ".jpeg"):
                    image = im.convert("RGB")
                    width = round(max_width / 72 * self.photo_dpi)
                    if image.width > width:
                        image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
                    buffer = BytesIO()
                    image.save(buffer, "JPEG", quality=self.photo_quality, subsampling=0, optimize=True)
                    payload = buffer.getvalue()
                else:
                    payload = path.read_bytes()
            self.images[key] = (payload, original)
        payload, (w, h) = self.images[key]
        scale = min(max_width / w, max_height / h)
        result = PDFImage(BytesIO(payload), width=w * scale, height=h * scale)
        result.hAlign = "CENTER"
        return result

    def table(self, rows, ratios=None, guide=False):
        count = len(rows[0])
        if ratios is None:
            measures = [min(30, max(7, max(len(re.sub(r"[`*]", "", row[c])) for row in rows)))
                        for c in range(count)]
            ratios = [value ** .65 for value in measures]
        widths = [WIDTH * ratio / sum(ratios) for ratio in ratios]
        data = [[self.paragraph(("**" + cell + "**") if i == 0 else cell, "guide" if guide else "cell")
                 for cell in row] for i, row in enumerate(rows)]
        table = LongTable(data, colWidths=widths, repeatRows=1, splitByRow=1, splitInRow=1,
                          hAlign="LEFT", spaceBefore=5, spaceAfter=12)
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0f2")),
                                  ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#d3d7db")),
                                  ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 5),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                  ("TOPPADDING", (0, 0), (-1, -1), 5),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return table

    def training_curve_sections(self, value):
        """Enlarge the fixed 7x4 overview without changing any plotted data."""
        caption, relative = value
        path = local_file(relative)
        self.inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        parts = []
        with Image.open(path) as original:
            source = original.convert("RGB")
        width, height = source.size
        if abs(width / height - 17.8 / 24.5) > .005:
            raise ValueError("Training overview layout changed; update its PDF enlargement coordinates")
        # These subplot bounds match make_experiment_figures.plot_training.
        axes_height = (.947 - .065) / (7 + 6 * .57)
        bands = []
        for index in range(14):
            row, side = divmod(index, 2)
            top = (1 - .947 + row * axes_height * 1.57) * height
            box = (side * width // 2, round(top - height * .0125),
                   (side + 1) * width // 2, round(top + axes_height * height + height * .0148))
            bands.append(source.crop(box))
        labels = ("기초·A·B·C", "D·E·F·G", "H·I·J", "K·L·M")
        for indices, label in zip((range(0, 4), range(4, 8), range(8, 11), range(11, 14)), labels):
            chosen = [bands[index] for index in indices]
            combined = Image.new("RGB", (chosen[0].width, sum(im.height for im in chosen) + 18 * (len(chosen) - 1)), "white")
            y = 0
            for crop in chosen:
                combined.paste(crop, (0, y)); y += crop.height + 18
            payload = BytesIO(); combined.save(payload, "PNG")
            scale = min(WIDTH / combined.width, (HEIGHT - 92) / combined.height)
            picture = PDFImage(BytesIO(payload.getvalue()), width=combined.width * scale, height=combined.height * scale)
            picture.hAlign = "CENTER"
            legend = self.paragraph("왼쪽: 학습(파랑)·검증(주황) 마스크 손실. 오른쪽: Mask mAP50(보라)·mAP50-95(초록). 검은 점선: 해당 학습의 선택 에폭.", "caption")
            parts.append([self.paragraph(caption + " · " + label, "h3"), legend, picture, Spacer(1, 10)])
        return parts

    def story(self, blocks):
        story = []
        def keep(items):
            # Keep consecutive section titles with the first non-splittable block.
            while story and isinstance(story[-1], Paragraph) and story[-1].style.name.startswith("h"):
                items.insert(0, story.pop())
            story.append(KeepTogether(items))
        for kind, value in blocks:
            if kind == "heading":
                story.append(self.paragraph(value[1], f"h{min(value[0], 4)}"))
            elif kind == "paragraph":
                story.append(self.paragraph(value))
            elif kind == "list":
                prefix, text, indent = value
                paragraph = self.paragraph(text, bulletText="-" if prefix in "-+*" else prefix)
                paragraph.style = ParagraphStyle("list", parent=paragraph.style, leftIndent=14 + min(indent, 8) * 3,
                                                 firstLineIndent=0, bulletIndent=min(indent, 8) * 3)
                story.append(paragraph)
            elif kind == "table":
                table = self.table(value)
                if table.wrap(WIDTH, HEIGHT)[1] <= HEIGHT * .72:
                    keep([table])
                else:
                    story.append(table)
            elif kind == "code":
                lines = []
                for line in value.expandtabs(4).splitlines():
                    remaining = line
                    while pdfmetrics.stringWidth(remaining, "NanumCoding", 8.8) > WIDTH - 16:
                        cut = len(remaining)
                        while pdfmetrics.stringWidth(remaining[:cut], "NanumCoding", 8.8) > WIDTH - 16:
                            cut -= 1
                        lines.append(remaining[:cut]); remaining = "  " + remaining[cut:]
                    lines.append(remaining)
                code = Preformatted("\n".join(lines), self.styles["code"])
                if code.wrap(WIDTH, HEIGHT)[1] < HEIGHT * .8:
                    keep([code])
                else:
                    story.append(code)
            elif kind == "math":
                path, width, height = self.math_image(value, 12)
                scale = min(1, WIDTH / width)
                equation = PDFImage(str(path), width=width * scale, height=height * scale)
                equation.hAlign = "CENTER"
                keep([Spacer(1, 4), equation, Spacer(1, 12)])
            elif kind == "image":
                caption, path = value
                if Path(path).as_posix() == "docs/images/training_curves.png":
                    for section in self.training_curve_sections(value):
                        keep(section)
                    continue
                items = [self.image(path), Spacer(1, 6)]
                if caption:
                    items.append(self.paragraph(caption, "caption"))
                keep(items)
        return story

    def guide_layout(self, guide):
        layout, top = [], MARGIN
        def add(flow, gap=8):
            nonlocal top
            height = flow.wrap(WIDTH, HEIGHT)[1]
            layout.append((flow, MARGIN, top)); top += height + gap
        add(self.paragraph(guide["title"], "h1"), 12)
        add(self.table([guide["headers"], *guide["rows"]], guide.get("widths"), guide=True), 12)
        for rule in guide["rules"]:
            add(self.paragraph("**" + rule["title"] + "**", "guide"), 3)
            add(self.paragraph(rule["text"], "guide"), 9)
        add(self.paragraph("**포함·제외 예시**", "guide"), 7)
        if len(guide["examples"]) != 6:
            raise ValueError("The one-page guideline requires six examples")
        cell = (WIDTH - 20) / 3
        for index, item in enumerate(guide["examples"]):
            image = self.image(item["path"], cell, 82)
            x, y = MARGIN + (index % 3) * (cell + 10), top + (index // 3) * 106
            layout.append((image, x + (cell - image.drawWidth) / 2, y + (82 - image.drawHeight) / 2))
            caption = self.paragraph(item["caption"], "caption")
            if caption.wrap(cell, HEIGHT)[1] > 15:
                raise ValueError("Guideline example caption must fit one line")
            layout.append((caption, x, y + 87))
        top += 212
        if guide.get("source"):
            add(self.paragraph(guide["source"], "caption"), 0)
        if top > PAGE_H - MARGIN:
            raise ValueError(f"Guideline exceeds one page by {top - PAGE_H + MARGIN:.1f} pt")
        return layout


def font_directory(requested):
    candidates = [requested] if requested else []
    if os.environ.get("CKH_ROOT"):
        candidates.append(Path(os.environ["CKH_ROOT"]) / "assets/fonts/nanum")
    candidates.extend((ROOT / "assets/fonts/nanum", Path("/usr/share/fonts/truetype/nanum")))
    for path in candidates:
        if path and all((path / name).is_file() for name in ("NanumGothic.ttf", "NanumGothicBold.ttf", "NanumGothicCoding.ttf")):
            return path
    raise FileNotFoundError("Install Nanum fonts or pass --font-dir containing the three Nanum TTF files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--guideline", type=Path, default=ROOT / "assets/guideline/guideline.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / PDF_NAME)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--photo-dpi", type=int, choices=range(180, 301), default=180)
    parser.add_argument("--photo-quality", type=int, choices=range(85, 93), default=85)
    args = parser.parse_args()
    readme, guide_path = local_file(args.readme), local_file(args.guideline)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "reports") or output.suffix.lower() != ".pdf":
        raise ValueError("PDF outputs must be within the repository reports directory")
    readme_text = readme.read_text(encoding="utf-8")
    readme_sha = digest(readme)
    guide_sha = digest(guide_path)
    blocks = parse_markdown(readme_text)
    if not blocks or blocks[0][0] != "heading":
        raise ValueError("README must begin with its document title")
    guide = json.loads(guide_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache/pdf_review"
    cache.mkdir(parents=True, exist_ok=True)
    rl_config.useA85 = False
    with tempfile.TemporaryDirectory(prefix=".pdf-build-", dir=cache) as directory:
        temporary = Path(directory)
        renderer = Renderer(temporary, font_directory(args.font_dir), args.photo_dpi, args.photo_quality)
        supported = pdfmetrics.getFont("Nanum").face.charToGlyph
        missing = sorted({character for character in normalize(readme_text + json.dumps(guide, ensure_ascii=False))
                          if ord(character) > 31 and ord(character) not in supported})
        if missing:
            raise ValueError(f"Unsupported text glyphs in NanumGothic: {missing}")
        layout = renderer.guide_layout(guide)
        guide_pages = []
        def draw_guide(pdf, doc):
            guide_pages.append(pdf.getPageNumber())
            for flow, x, top in layout:
                width = (WIDTH - 20) / 3 if x != MARGIN else WIDTH
                height = flow.wrap(width, HEIGHT)[1]
                flow.drawOn(pdf, x, PAGE_H - top - height)
        story = renderer.story(blocks)
        story.extend([NextPageTemplate("guide"), PageBreak(), Spacer(1, 1)])
        report_temp, guide_temp = temporary / "report.pdf", temporary / "guideline.pdf"
        doc = BaseDocTemplate(str(report_temp), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                              topMargin=MARGIN, bottomMargin=MARGIN, title=blocks[0][1][1])
        frame = lambda: Frame(MARGIN, MARGIN, WIDTH, HEIGHT, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        doc.addPageTemplates([PageTemplate("body", [frame()]), PageTemplate("guide", [frame()], onPage=draw_guide)])
        doc.build(story)
        if guide_pages != [doc.page]:
            raise ValueError("The guideline must occupy exactly the last report page")
        guide_pdf = canvas.Canvas(str(guide_temp), pagesize=A4)
        guide_pdf.setTitle(guide["title"])
        draw_guide(guide_pdf, None)
        guide_pdf.showPage(); guide_pdf.save()
        if digest(readme) != readme_sha or digest(guide_path) != guide_sha:
            raise ValueError("README or guideline changed during PDF generation; rebuild from the current input")
        if any(digest(ROOT / path) != expected for path, expected in renderer.inputs.items()):
            raise ValueError("A figure changed during PDF generation; rebuild from the current inputs")
        guide_output = output.with_name(GUIDE_NAME)
        report_temp.replace(output); guide_temp.replace(guide_output)
        record = {"readme_sha256": readme_sha, "guideline_sha256": guide_sha, "pages": doc.page,
                  "markdown_blocks": len(blocks), "figures": sum(kind == "image" for kind, _ in blocks),
                  "photo_dpi": args.photo_dpi, "photo_quality": args.photo_quality, "inputs": renderer.inputs,
                  "outputs": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                               "sha256": digest(path)} for path in (output, guide_output)]}
        (cache / "build_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
