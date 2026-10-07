#!/usr/bin/env python3
"""Convert PAPER.md to a professional single-column PDF with embedded figures.

Academic style: Arial (Arial/Helvetica family), 10.5pt body, black headings,
justified-free left alignment, figure images scaled to text width, italic
support, Unicode-safe via registered TrueType fonts.
"""
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

BASE = Path(__file__).resolve().parent
SRC = BASE / "PAPER.md"
OUT = BASE / "PAPER.pdf"

# ---- fonts -----------------------------------------------------------------
FD = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial", FD + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", FD + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", FD + "ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", FD + "arialbi.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold",
                             italic="Arial-Italic", boldItalic="Arial-BoldItalic")

# ---- styles ----------------------------------------------------------------
S = {}
S["title"] = ParagraphStyle("title", fontName="Arial-Bold", fontSize=15.5,
                            leading=19, alignment=TA_CENTER, spaceAfter=10)
S["author"] = ParagraphStyle("author", fontName="Arial", fontSize=11,
                             leading=14, alignment=TA_CENTER)
S["affil"] = ParagraphStyle("affil", fontName="Arial", fontSize=9.5,
                            leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
S["h1"] = ParagraphStyle("h1", fontName="Arial-Bold", fontSize=12.5,
                         leading=15, spaceBefore=14, spaceAfter=5)
S["h2"] = ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11,
                         leading=13.5, spaceBefore=10, spaceAfter=4)
S["body"] = ParagraphStyle("body", fontName="Arial", fontSize=10,
                           leading=14.2, spaceAfter=7)
S["bullet"] = ParagraphStyle("bullet", fontName="Arial", fontSize=10,
                            leading=13.5, spaceAfter=4, leftIndent=16,
                            bulletIndent=4)
S["caption"] = ParagraphStyle("caption", fontName="Arial-Italic", fontSize=8.7,
                              leading=11.5, alignment=TA_LEFT, spaceBefore=3,
                              spaceAfter=12)
S["cell"] = ParagraphStyle("cell", fontName="Arial", fontSize=9,
                           leading=11.5)
S["abstract"] = ParagraphStyle("abstract", fontName="Arial", fontSize=9.6,
                               leading=13.5, spaceAfter=8,
                               leftIndent=14, rightIndent=14)

TXT_W = letter[0] - 1.8 * inch  # usable text width


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(t):
    """markdown inline -> reportlab mini-html (escapes first)."""
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier' size='8.7'>\1</font>", t)
    return t


def para(txt, style):
    return Paragraph(inline(txt), style)


def figure(path, caption, max_h=4.4 * inch):
    img = ImageReader(str(path))
    w, h = img.getSize()
    scale = min(TXT_W / w, max_h / h)
    el = Image(str(path), width=w * scale, height=h * scale)
    return [Spacer(1, 6), el, para(caption, S["caption"])]


def build():
    md = SRC.read_text(encoding="utf-8")
    lines = md.splitlines()
    story = []
    i, n = 0, len(lines)

    # ---- title block -------------------------------------------------------
    # assume structure: # Title / **Author** / Affil — City / email / ## Abstract
    title_lines = []
    while i < n and not lines[i].startswith("## "):
        if lines[i].strip():
            title_lines.append(lines[i].strip())
        i += 1
    story.append(para(title_lines[0].lstrip("# "), S["title"]))
    for tl in title_lines[1:]:
        txt = tl.strip("*")
        if "@" in txt:
            story.append(para(txt, S["affil"]))
        elif txt.startswith("Ternative"):
            story.append(para(txt, S["author"]))
        else:
            story.append(para(txt, S["author"]))
    story.append(Spacer(1, 6))

    while i < n:
        line = lines[i].rstrip()
        s = line.strip()

        if not s:
            i += 1
            continue

        # headings
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            txt = m.group(2)
            style = {1: S["h1"], 2: S["h1"], 3: S["h2"]}[lvl]
            story.append(para(txt, style))
            i += 1
            continue

        # figure: ![caption](path) followed by italic caption line
        if s.startswith("!["):
            m = re.match(r"!\[.*?\]\((.*?)\)", s)
            if m:
                fpath = BASE / m.group(1)
                if fpath.exists():
                    # next nonempty line is the caption
                    j = i + 1
                    cap = ""
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n and lines[j].strip().startswith("*"):
                        cap = lines[j].strip().strip("*")
                    story.extend(figure(fpath, cap))
                    i = j + 1
                    continue
        # horizontal rule
        if re.match(r"^-{3,}$", s):
            i += 1
            continue

        # tables
        if s.startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            data = [[Paragraph(inline(c), S["cell"]) for c in header]]
            for r in rows:
                # pad row if needed
                while len(r) < len(header):
                    r.append("")
                data.append([Paragraph(inline(c), S["cell"]) for c in r])
            ncols = len(header)
            t = Table(data, hAlign="CENTER",
                      colWidths=[TXT_W / ncols] * ncols, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f5f5f5")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([Spacer(1, 4), t, Spacer(1, 10)])
            continue

        # bullets
        if re.match(r"^[-*]\s+", s):
            story.append(para(re.sub(r"^[-*]\s+", "", s), S["bullet"]))
            i += 1
            continue

        # numbered list
        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            story.append(Paragraph(f"<b>{m.group(1)}.</b> " + inline(m.group(2)), S["bullet"]))
            i += 1
            continue

        # abstract styling: paragraph directly after ## Abstract
        if s.startswith("We test whether"):
            story.append(para(s, S["abstract"]))
            i += 1
            continue

        story.append(para(s, S["body"]))
        i += 1

    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.85 * inch, bottomMargin=0.85 * inch,
        title="Hub Concentration Masks the Benefit of Biological Connectome Structure in Small Language Models",
        author="Michelangelo Romero Chisco",
    )
    doc.build(story)
    print(f"PDF written -> {OUT}")


if __name__ == "__main__":
    build()