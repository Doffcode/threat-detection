"""Finish the Report Kit deck with presentation-sized native results panels."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

here = Path(__file__).resolve().parent
deck = Presentation(here / "out/slides.pptx")
navy = RGBColor.from_string("203040")
muted = RGBColor.from_string("435365")
line = RGBColor.from_string("AEBAC4")
light = RGBColor.from_string("F1F4F6")
white = RGBColor.from_string("FFFFFF")


def fmt(run, size, bold=False, color=navy):
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def txt(slide, value, x, y, width, height, size=22, bold=False, color=navy, center=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for i, row in enumerate(value.split("\n")):
        para = frame.add_paragraph() if i else frame.paragraphs[0]
        para.space_after = Pt(4)
        if center:
            para.alignment = PP_ALIGN.CENTER
        run = para.add_run()
        run.text = row
        fmt(run, size, bold, color)
    return box


def page(number, title):
    slide = deck.slides[number - 1]
    for shape in list(slide.shapes):
        el = shape._element
        el.getparent().remove(el)
    txt(slide, title, .72, .57, 11.9, .85, 31, True)
    txt(slide, str(number), 12.35, 7.06, .3, .2, 10, color=muted)
    return slide


def box(slide, label, x, y, width, height):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = light
    shape.line.color.rgb = line
    frame = shape.text_frame
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = frame.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    run = para.add_run()
    run.text = label
    fmt(run, 22, True)


def arrow(slide, x1, y1, x2, y2):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = muted
    connector.line.width = Pt(1.5)
    tail = OxmlElement("a:tailEnd")
    tail.set("type", "triangle")
    connector._element.spPr.get_or_add_ln().append(tail)


deck.slides[0].notes_slide.notes_text_frame.text = (
    "0:20 · 0:00–0:20\n\n"
    "Good morning. I am Akriti Jaiswal. This is the Phase 3 review of my Surveillance Threat Detection System. "
    "I will describe the dashboard workflow and spend most of the review on the measured results. "
    "The final review is on 4 January."
)

s = page(3, "Summary of methodology — system workflow")
for x in (2.65, 6.66, 10.65):
    arrow(s, 6.66, 2.2, x, 2.78)
    arrow(s, x, 3.58, 6.66, 4.26)
arrow(s, 6.66, 5.05, 6.66, 5.62)
box(s, "Input video", 4.91, 1.55, 3.5, .65)
for x, label in [(1.1, "Weapon detection"), (5.11, "Fight classification"), (9.1, "Movement tracking")]:
    box(s, label, x, 2.78, 3.1, .8)
box(s, "Unified evidence timeline", 4.4, 4.26, 4.52, .79)
box(s, "Playback and download", 4.4, 5.62, 4.52, .79)

s = page(5, "90.4% fight F1 on 300 SCFD clips")
txt(s, "91.0%", .82, 1.78, 4.45, .9, 58, True)
txt(s, "Accuracy", .87, 2.73, 4.35, .48, 25, True)
txt(s, "Whole-clip evaluation\n150 fight · 150 non-fight", .87, 3.37, 4.55, 1.12, 22)
txt(s, "Frozen X3D-M\nThreshold 0.40", .87, 4.8, 4.55, 1.1, 21)
rows = [["Actual / Predicted", "Non-fight", "Fight"],
        ["Non-fight", "146\nTrue negatives", "4\nFalse positives"],
        ["Fight", "23\nFalse negatives", "127\nTrue positives"]]
table_shape = s.shapes.add_table(3, 3, Inches(5.5), Inches(1.9), Inches(6.95), Inches(3.52))
table = table_shape.table
table.first_row = False
table.horz_banding = False
for col, width in zip(table.columns, (2.25, 2.4, 2.3)):
    col.width = Inches(width)
for row, height in zip(table.rows, (.77, 1.37, 1.38)):
    row.height = Inches(height)
for i, row in enumerate(rows):
    for j, value in enumerate(row):
        cell = table.cell(i, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = navy if i == 0 else light if j == 0 or i == j else white
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(.12)
        for k, part in enumerate(value.split("\n")):
            p = cell.text_frame.add_paragraph() if k else cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = part
            fmt(r, 29 if i and j and not k else 17 if k else 18, k == 0, white if i == 0 else navy)
txt(s, "[F] Surveillance Camera Fight Dataset (SCFD) · saved clip predictions", .87, 6.59, 11.5, .39, 14, color=muted)

s = page(6, "Results — weapon detection")
divider = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(6.75), Inches(1.66), Inches(6.75), Inches(6.38))
divider.line.color.rgb = line
txt(s, "Innovatus CCTV [W]", .8, 1.63, 5.5, .5, 27, True)
txt(s, "baseline_m960_v2", .8, 2.19, 5.55, .39, 21)
txt(s, "72 images · 75 weapon annotations", .8, 2.78, 5.65, .39, 19)
txt(s, "17 of 75 matched", .8, 3.37, 5.65, .63, 35, True)
txt(s, "58.6% precision · 22.7% recall", .8, 4.25, 5.7, .5, 23, True)
txt(s, "Confidence 0.30 for precision and recall", .8, 4.83, 5.65, .4, 16, color=muted)
txt(s, "0.252 mAP@0.50", .8, 5.6, 5.65, .55, 27, True)
txt(s, "Tabletop validation [T]", 7.13, 1.63, 5.35, .5, 27, True)
txt(s, "v5_hybrid_m640", 7.13, 2.19, 5.35, .39, 21)
txt(s, "35 images · 40 weapon annotations", 7.13, 2.78, 5.35, .39, 19)
txt(s, "0.919 mAP@0.50", 7.13, 3.37, 5.35, .63, 35, True)
txt(s, "Detector only\nCrop verifier disabled", 7.13, 4.29, 5.35, 1.14, 22)

s = page(7, "Results — movement cue review")
txt(s, "0 of 15", .8, 1.71, 5.35, 1, 56, True)
txt(s, "labeled cue intervals matched", .83, 2.67, 5.55, .98, 26, True)
txt(s, "24 reviewed clips\n6 cue-free · 3 unclear", .83, 3.63, 5.55, 1.12, 23)
txt(s, "0 alerts · 0 processing errors [M]", .83, 5.13, 5.6, .8, 20)
s.shapes.add_picture(str(here.parent / "report/figures/movement_example.png"),
                     Inches(6.23), Inches(1.83), width=Inches(6.23))
txt(s, "Feature demonstration · separate SCFD clip", 6.23, 5.55, 6.23, .6, 16, color=muted, center=True)

s = page(9, "References")
txt(s, "Dataset and models", .8, 1.65, 5.9, .45, 23, True)
txt(s, "Ş. Aktı, G. A. Tataroğlu and H. K. Ekenel.\nVision-based Fight Detection from\nSurveillance Cameras. IPTA, 2019.", .8, 2.29, 5.9, 1.1, 17)
sources = [
    (3.44, "SCFD authors’ repository", "https://github.com/seymanurakti/fight-detection-surv-dataset"),
    (4.17, "VisionLab AI · X3D-M model card and weights", "https://huggingface.co/visionlab-ai/school-violence-detection-models"),
    (4.9, "Ultralytics · YOLO11 documentation", "https://docs.ultralytics.com/models/yolo11/"),
]
for y, label, url in sources:
    ref = txt(s, label, .8, y, 5.9, .4, 17)
    ref.text_frame.paragraphs[0].runs[0].hyperlink.address = url
txt(s, "Local evaluation files", 7.02, 1.65, 5.5, .45, 23, True)
for y, directory, files in [
    (2.29, "[F] outputs/eval/behavior_scfd/", "local_metrics.json · local_predictions.csv"),
    (3.28, "[W] outputs/eval/cctv_test_v2/", "cctv_test_v2_metrics.json · v2.innovatus_rob"),
    (4.27, "[T] work/runs/v5_hybrid_m640/", "metrics_val_tabletop.json"),
    (5.26, "[M] outputs/eval/movement_review/", "summary.json · labels.json · alerts.csv"),
]:
    txt(s, directory, 7.02, y, 5.55, .35, 17, True)
    txt(s, files, 7.02, y + .39, 5.55, .39, 16, color=muted)

deck.save(here / "out/Phase_3_Threat_Detection.pptx")
