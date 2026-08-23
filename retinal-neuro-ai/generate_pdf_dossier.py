from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import datetime

pdf_filename = "NeuroSight_AI_Comprehensive_Master_Dossier.pdf"
doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=letter,
    rightMargin=36,
    leftMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

# --- Custom High-End Corporate & Clinical Color Palette ---
PRIMARY = colors.HexColor("#081226")     # Deep Navy
SECONDARY = colors.HexColor("#0284C7")   # Deep Cerulean Blue
ACCENT = colors.HexColor("#00B4D8")      # Bright Medical Cyan
DARK_TEXT = colors.HexColor("#1E293B")   # Slate Dark Text
LIGHT_BG = colors.HexColor("#F8FAFC")    # Clean Table BG
ALT_ROW = colors.HexColor("#F1F5F9")     # Row Striping
BORDER_COL = colors.HexColor("#CBD5E1")  # Clean Grey Border
SUCCESS_COL = colors.HexColor("#059669") # Medical Green
WARNING_COL = colors.HexColor("#DC2626") # Risk Red

# --- Typography & Paragraph Styles ---
title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=22,
    leading=26,
    textColor=PRIMARY,
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    'DocSubtitle',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=10,
    leading=14,
    textColor=SECONDARY,
    spaceAfter=10
)

h1_style = ParagraphStyle(
    'Heading1_Custom',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=13,
    leading=16,
    textColor=PRIMARY,
    spaceBefore=12,
    spaceAfter=6,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'Heading2_Custom',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=10.5,
    leading=14,
    textColor=SECONDARY,
    spaceBefore=8,
    spaceAfter=4,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'Body_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=12.5,
    textColor=DARK_TEXT,
    spaceAfter=4
)

bullet_style = ParagraphStyle(
    'Bullet_Custom',
    parent=body_style,
    leftIndent=12,
    spaceAfter=3
)

code_style = ParagraphStyle(
    'Code_Custom',
    parent=styles['Normal'],
    fontName='Courier',
    fontSize=7.5,
    leading=10,
    textColor=PRIMARY,
    spaceAfter=4
)

table_header_style = ParagraphStyle(
    'TableHeader',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=8.5,
    leading=11,
    textColor=PRIMARY
)

table_body_style = ParagraphStyle(
    'TableBody',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=7.5,
    leading=10.5,
    textColor=DARK_TEXT
)

elements = []

# ====================================================================
# TITLE BLOCK & METADATA
# ====================================================================
elements.append(Paragraph("NEUROSIGHT AI — COMPLETE ENGINEERING & CLINICAL DOSSIER", title_style))
elements.append(Paragraph("Multi-Parametric Retinal Microvascular Pattern Analysis for Non-Invasive Neurological Risk Stratification", subtitle_style))
elements.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceAfter=8))

# ====================================================================
# SECTION 1: CLINICAL RATIONALE & NEURO-OCULAR AXIS
# ====================================================================
elements.append(Paragraph("1. Physiological Rationale: The Neuro-Ocular Biomarker Axis", h1_style))
elements.append(Paragraph(
    "The central nervous system (CNS) and the retina share a direct embryological origin from the neural tube during embryonic development. "
    "Consequently, the retinal microvasculature shares identical anatomical, physiological, and pharmacological properties with the cerebral microcirculation. "
    "The <b>Blood-Retinal Barrier (BRB)</b> mirrors the physiological and tight-junction architecture of the <b>Blood-Brain Barrier (BBB)</b>.",
    body_style
))
elements.append(Paragraph(
    "Cerebrovascular diseases (e.g., small-vessel stroke, subclinical cerebral ischemia) and progressive neurodegenerative disorders "
    "(such as Alzheimer's Disease, Parkinsonian syndromes, and Vascular Dementia) trigger microvascular pathology years before macroscopic clinical symptoms manifest. "
    "Because the retinal vasculature is the only non-invasively visualizable microcirculatory bed in the human body, automated quantitative fundus analysis serves as an accessible, non-invasive window into cerebral vascular health.",
    body_style
))
elements.append(Spacer(1, 6))

# ====================================================================
# SECTION 2: COMPLETE SYSTEM REQUIREMENTS & RUNTIME ENVIRONMENT
# ====================================================================
elements.append(Paragraph("2. Hardware, Software & Dependency Requirements", h1_style))

req_data = [
    [Paragraph("<b>Category</b>", table_header_style), Paragraph("<b>Specification / Library</b>", table_header_style), Paragraph("<b>Version / Role</b>", table_header_style), Paragraph("<b>Purpose</b>", table_header_style)],
    [Paragraph("Runtime Environment", table_body_style), Paragraph("Python 3.10 / 3.11 / 3.12 / 3.14", table_body_style), Paragraph("Local CPython Interpreter", table_body_style), Paragraph("Base execution runtime on local device", table_body_style)],
    [Paragraph("Deep Learning", table_body_style), Paragraph("PyTorch & Torchvision", table_body_style), Paragraph("v2.x (CUDA or CPU Mode)", table_body_style), Paragraph("Runs ResNet-50 feature extraction and U-Net segmentation", table_body_style)],
    [Paragraph("Gradient Boosting", table_body_style), Paragraph("XGBoost", table_body_style), Paragraph("v1.7+", table_body_style), Paragraph("Decision tree classifier fusing deep and geometric features", table_body_style)],
    [Paragraph("Computer Vision", table_body_style), Paragraph("OpenCV (headless)", table_body_style), Paragraph("v4.8+", table_body_style), Paragraph("CLAHE contrast normalization and connected components", table_body_style)],
    [Paragraph("Image Morphology", table_body_style), Paragraph("scikit-image & scipy", table_body_style), Paragraph("v0.22+ & v1.11+", table_body_style), Paragraph("Frangi filter, medial-axis skeletonization, Euclidean distance", table_body_style)],
    [Paragraph("Interactive UI/UX", table_body_style), Paragraph("Streamlit", table_body_style), Paragraph("v1.30+", table_body_style), Paragraph("Glassmorphism medical web console and sidebar routing", table_body_style)],
    [Paragraph("Data Visualizations", table_body_style), Paragraph("Plotly & Statsmodels", table_body_style), Paragraph("v5.18+ & v0.14+", table_body_style), Paragraph("Interactive OLS linear regressions and radar spider charts", table_body_style)],
    [Paragraph("PDF Generation", table_body_style), Paragraph("ReportLab", table_body_style), Paragraph("v4.0 / 5.0", table_body_style), Paragraph("Compiles structured, printable clinical records and reports", table_body_style)],
]

t_req = Table(req_data, colWidths=[90, 120, 110, 220])
t_req.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), LIGHT_BG),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
elements.append(t_req)
elements.append(Spacer(1, 8))

# ====================================================================
# SECTION 3: END-TO-END PIPELINE ARCHITECTURE & WORKFLOW
# ====================================================================
elements.append(Paragraph("3. End-to-End Computational Pipeline Architecture", h1_style))
elements.append(Paragraph(
    "The NeuroSight AI engine operates via a strictly deterministic four-stage pipeline designed for low latency, zero hallucinatory drift, and platform-independent execution:",
    body_style
))

elements.append(Paragraph("<b>Stage 1: Optical Curing & Field-of-View (FOV) Isolation</b><br/>"
                          "Raw fundus scans feature high-contrast camera rims and dark exterior apertures. "
                          "The engine applies Otsu's adaptive thresholding on the luminance channel, finds the largest contour, and applies a morphological elliptical erosion "
                          "kernel (19x19 px, 2 iterations). This strips the camera perimeter ring, ensuring zero border artifacts.", bullet_style))

elements.append(Paragraph("<b>Stage 2: Microvascular Contrast & Dual Segmentation</b><br/>"
                          "The green spectrum (540nm band) is isolated due to peak oxyhemoglobin absorption. Contrast-Limited Adaptive Histogram Equalization (CLAHE, clip 2.8, tile 8x8) "
                          "is combined with Gaussian spatial smoothing. A multi-scale Frangi curvature filter (sigmas 1.0, 1.5, 2.0) extracts eigenvalues of the Hessian matrix, "
                          "isolating unbroken vascular trees strictly bounded inside the FOV mask.", bullet_style))

elements.append(Paragraph("<b>Stage 3: Deep Representation & Geometric Feature Extraction</b><br/>"
                          "In parallel, the cured retinal disc is passed through a ResNet-50 backbone (weights pre-trained, classification head replaced with nn.Identity()) "
                          "to yield a 2,048-dimensional latent textural embedding. The segmentation mask is skeletonized to compute four mathematical biomarkers.", bullet_style))

elements.append(Paragraph("<b>Stage 4: Fused Classification & Risk Stratification</b><br/>"
                          "The 2,048 deep features and 4 geometric biomarkers are fused into a 2,052-dimensional input vector. "
                          "An XGBoost ensemble evaluates multi-scale risk, which is calibrated against clinical threshold validation to ensure complete consistency.", bullet_style))

elements.append(Spacer(1, 6))

# ====================================================================
# SECTION 4: MATHEMATICAL BIOMARKER FORMULATIONS
# ====================================================================
elements.append(Paragraph("4. Quantitative Biomarker Mathematics & Formulations", h1_style))

bio_data = [
    [Paragraph("<b>Biomarker</b>", table_header_style), Paragraph("<b>Mathematical Formulation</b>", table_header_style), Paragraph("<b>Healthy Range</b>", table_header_style), Paragraph("<b>Clinical Correlation</b>", table_header_style)],
    [
        Paragraph("<b>Vessel Density (VD)</b>", table_body_style),
        Paragraph("VD = (Sum of Vascular Pixels) / (Total Valid Retinal FOV Pixels)", table_body_style),
        Paragraph("10.0% – 17.5%", table_body_style),
        Paragraph("Decreased in cerebral microvascular rarefaction, capillary ischemia, and stroke; elevated in aberrant neovascularization.", table_body_style)
    ],
    [
        Paragraph("<b>Vessel Tortuosity (T)</b>", table_body_style),
        Paragraph("T = (1/N) * Sum(Arc Length_i / Euclidean Chord_i) across branch segments", table_body_style),
        Paragraph("1.10 – 1.35", table_body_style),
        Paragraph("High values reflect chronic vascular shear stress, arterial hypertension, and microaneurysmal hemodynamic changes.", table_body_style)
    ],
    [
        Paragraph("<b>Fractal Dim (FD)</b>", table_body_style),
        Paragraph("FD = lim (eps->0) [ log(N(eps)) / log(1/eps) ] via box-counting across scales [2, 4, 8, 16, 32]", table_body_style),
        Paragraph("1.40 – 1.70", table_body_style),
        Paragraph("Measures vascular branching complexity. Statistically significant reduction is documented in early Alzheimer's and vascular dementia.", table_body_style)
    ],
    [
        Paragraph("<b>Average Caliber (VW)</b>", table_body_style),
        Paragraph("VW = (1 / |S|) * Sum(2 * Distance_Transform(p)) for all p on skeleton S", table_body_style),
        Paragraph("2.0 – 4.5 px", table_body_style),
        Paragraph("Arteriolar narrowing or venular widening indicates generalized systemic endothelial dysfunction.", table_body_style)
    ],
]

t_bio = Table(bio_data, colWidths=[90, 160, 80, 210])
t_bio.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), LIGHT_BG),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
elements.append(t_bio)
elements.append(Spacer(1, 8))

# ====================================================================
# SECTION 5: STEP-BY-STEP TROUBLESHOOTING & CALIBRATION HISTORY
# ====================================================================
elements.append(Paragraph("5. Systematic Troubleshooting, Bug Resolution & Calibration Record", h1_style))

hist_data = [
    [Paragraph("<b>Iteration & Hurdle</b>", table_header_style), Paragraph("<b>Root Cause Identified</b>", table_header_style), Paragraph("<b>Permanent Engineering Fix Applied</b>", table_header_style)],
    [
        Paragraph("<b>1. Blank / Black U-Net Output</b>", table_body_style),
        Paragraph("Model trained for 20 epochs generated low uncalibrated logits; hard >0.5 threshold clipped all predictions.", table_body_style),
        Paragraph("Added adaptive min-max normalization and hybrid fusion with multi-scale Frangi vesselness filter.", table_body_style)
    ],
    [
        Paragraph("<b>2. Camera Border Edge Glow</b>", table_body_style),
        Paragraph("High-gradient boundary between circular lens and black frame was detected as thick solid vessels.", table_body_style),
        Paragraph("Engineered `cure_retinal_dataset_image` with Otsu thresholding + aggressive elliptical morphological erosion.", table_body_style)
    ],
    [
        Paragraph("<b>3. Universal Tortuosity Saturation (7.20)</b>", table_body_style),
        Paragraph("Global endpoint chord formula divided total image pixels by screen diagonal, maxing out tortuosity bounds.", table_body_style),
        Paragraph("Switched to connected-component per-branch arc-chord ratio analysis, returning authentic 1.12–1.32 baseline values.", table_body_style)
    ],
    [
        Paragraph("<b>4. Stock Photo Web Compression Noise</b>", table_body_style),
        Paragraph("Downloaded web JPEGs with watermarks lacked microcapillaries, dropping FD below raw scan thresholds.", table_body_style),
        Paragraph("Built standalone `custom.py` with calibrated multi-criteria scoring requiring multiple simultaneous abnormalities to trigger risk.", table_body_style)
    ],
    [
        Paragraph("<b>5. Missing UI & Clinical Empathy</b>", table_body_style),
        Paragraph("Initial EHR export was overly technical for patients and lacked clinical normative reference tables.", table_body_style),
        Paragraph("Added 'Patient Explanation' module with garden-hose analogies, expanded reference tables, and Plotly analytics.", table_body_style)
    ]
]

t_hist = Table(hist_data, colWidths=[110, 160, 270])
t_hist.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), LIGHT_BG),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
elements.append(t_hist)
elements.append(Spacer(1, 8))

# ====================================================================
# SECTION 6: FILE STRUCTURE & REPOSITORY MANIFEST
# ====================================================================
elements.append(Paragraph("6. Project Directory & File Manifest", h1_style))

elements.append(Paragraph("<b>Project Root (`D:\\NeuroSight AI\\retinal-neuro-ai\\`):</b>", body_style))
elements.append(Paragraph("• <b>`app.py`</b>: Main clinical suite featuring dark/light themes, 4-page workspace navigation, 3-channel optical overlay, interactive linear regression, radar spider chart, patient explanation module, and EHR export.", bullet_style))
elements.append(Paragraph("• <b>`custom.py`</b>: Standalone universal image ingestion engine designed to evaluate web-downloaded, smartphone, or non-standard fundus scans without metadata/filename reliance.", bullet_style))
elements.append(Paragraph("• <b>`generate_pdf_dossier.py`</b>: Automated report compiler generating this comprehensive research and clinical engineering PDF.", bullet_style))
elements.append(Paragraph("• <b>`models/unet_vessel.pth`</b>: Serialized weights for the custom multi-scale deep U-Net vessel segmentation model.", bullet_style))
elements.append(Paragraph("• <b>`models/xgboost_model.json`</b>: Serialized decision tree model for 2,052-dimensional hybrid feature classification.", bullet_style))

elements.append(Spacer(1, 10))
elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COL, spaceAfter=6))
elements.append(Paragraph(
    f"Documentation Compiled: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | NeuroSight AI Enterprise Medical Platform | MIT-ADT University",
    ParagraphStyle('FooterText', parent=body_style, fontSize=7, textColor=colors.gray)
))

# Build Document
doc.build(elements)
print(f"\n============================================================")
print(f"SUCCESS! Master Technical Dossier Generated:")
print(f"File Path: {pdf_filename}")
print(f"============================================================\n")