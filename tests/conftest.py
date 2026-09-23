from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from fieldnote.config import Settings

PARAGRAPH = (
    "High pressure switch trips when condenser airflow is restricted. Check the condenser "
    "coil for debris and verify the condenser fan motor amperage before replacing the switch."
)


def make_manual_pdf(path: Path) -> Path:
    """A two-page stand-in for a service manual: prose + a vector P-T chart, and a raster photo."""
    doc = pymupdf.open()

    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(56, 56, 556, 140), PARAGRAPH, fontsize=11)
    # Vector chart: axes, grid and a curve, the kind of evidence a caption flattens.
    x0, y0, x1, y1 = 100, 200, 500, 500
    shape = page.new_shape()
    shape.draw_line((x0, y1), (x1, y1))
    shape.draw_line((x0, y0), (x0, y1))
    for i in range(1, 8):
        gx = x0 + i * (x1 - x0) / 8
        gy = y0 + i * (y1 - y0) / 8
        shape.draw_line((gx, y0), (gx, y1))
        shape.draw_line((x0, gy), (x1, gy))
    pts = [(x0 + i * 25, y1 - (i**1.4) * 5) for i in range(17)]
    shape.draw_polyline(pts)
    shape.finish(color=(0, 0, 0), width=0.8)
    shape.commit()
    page.insert_text((250, 525), "Outdoor ambient temperature (F)", fontsize=9)
    page.insert_text((60, 190), "Discharge pressure (psig)", fontsize=9)
    page.insert_text((200, 560), "Figure 3. Discharge pressure vs outdoor ambient", fontsize=10)

    page2 = doc.new_page(width=612, height=792)
    img = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(img).rectangle((50, 50, 350, 250), outline="black", width=6)
    img_path = path.parent / "photo_tmp.png"
    img.save(img_path)
    page2.insert_image(pymupdf.Rect(100, 100, 500, 400), filename=str(img_path))
    page2.insert_text((100, 430), "Figure 4. Contactor with pitted contacts", fontsize=10)

    doc.save(path)
    doc.close()
    img_path.unlink()
    return path


@pytest.fixture
def manual_pdf(tmp_path: Path) -> Path:
    manuals = tmp_path / "data" / "manuals"
    manuals.mkdir(parents=True)
    return make_manual_pdf(manuals / "acme_rtu_service.pdf")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        qdrant_path=":memory:",
        reranker="rrf",
        top_n=3,
    )
