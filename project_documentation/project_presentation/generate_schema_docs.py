#!/usr/bin/env python3
"""
Generate Database Schema Documentation
- ERD diagram (PNG + SVG) from SQLAlchemy models
- Cover + Table of Contents
- One page per table:
    Purpose (auto/overrides) -> Columns grid -> ORM methods (name + signature)
- Project logger with request IDs
"""

import os
import inspect as pyinspect
from typing import List, Optional, Tuple, Any

# SQLAlchemy / Schema
from sqlalchemy_schemadisplay import create_schema_graph
from sqlalchemy import inspect as sqla_inspect

# Your project configuration & Base
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.base import Base
import modules.emtacdb.emtacdb_fts  # ensure models are imported and attached to Base

# Logger
from modules.configuration.log_config import (
    info_id, error_id, get_request_id, with_request_id
)

# ReportLab for PDF
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

# ---------------------------------------------------------------------
# Environment: ensure Graphviz "dot.exe" is found WITHOUT admin rights.
# Update this path to where your Graphviz was unzipped if needed.
# ---------------------------------------------------------------------
os.environ["PATH"] += os.pathsep + r"C:\Users\10169062\PycharmProjects\windows_10_cmake_Release_Graphviz-14.0.0-win64\Graphviz-14.0.0-win64\bin"

OUTPUT_DIR = "schema_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DB Conn
db_config = DatabaseConfig()
engine = db_config.get_engine()

# ---------------------------------------------------------------------
# Optional hand-written summaries (edit these to fine-tune wording)
# Keys must match actual table names (.__tablename__)
# ---------------------------------------------------------------------
SUMMARY_OVERRIDES = {
    # --- Core domain entities ---
    "part": "Catalog of parts/spares with identifying info (number, make/model, classification) and notes/documentation.",
    "Part": "Catalog of parts/spares with identifying info (number, make/model, classification) and notes/documentation.",

    "position": "Represents a physical installation slot or equipment position at a site/area, linking assets, models, or assemblies.",
    "Position": "Represents a physical installation slot or equipment position at a site/area, linking assets, models, or assemblies.",

    "image": "Stores image assets used to illustrate parts, positions, or assemblies.",
    "Image": "Stores image assets used to illustrate parts, positions, or assemblies.",

    "drawing": "Stores engineering drawings and related metadata (number, name, revision, file path).",
    "Drawing": "Stores engineering drawings and related metadata (number, name, revision, file path).",

    "bill_of_material": "BOM line items (quantity + comment) tied to a specific illustrated position/part image.",
    "BillOfMaterial": "BOM line items (quantity + comment) tied to a specific illustrated position/part image.",

    # --- Plant/site structure & catalog masters ---
    "area": "Logical area or process unit within a site; groups positions/assets by area.",
    "Area": "Logical area or process unit within a site; groups positions/assets by area.",

    "equipment_group": "Grouping of equipment/models by function or family for filtering and reporting.",
    "EquipmentGroup": "Grouping of equipment/models by function or family for filtering and reporting.",

    "model": "OEM model master record used to classify positions/parts.",
    "Model": "OEM model master record used to classify positions/parts.",

    "asset_number": "Enterprise asset identifier that links positions/equipment to CMMS/EAM systems.",
    "AssetNumber": "Enterprise asset identifier that links positions/equipment to CMMS/EAM systems.",

    "location": "Physical plant location metadata (building, area, room, coordinates).",
    "Location": "Physical plant location metadata (building, area, room, coordinates).",

    "site_location": "Top-level site/facility location record and related metadata.",
    "SiteLocation": "Top-level site/facility location record and related metadata.",

    "subassembly": "Subassembly definition used within a larger assembly hierarchy.",
    "Subassembly": "Subassembly definition used within a larger assembly hierarchy.",

    "component_assembly": "Component-level assembly grouping of parts and subassemblies.",
    "ComponentAssembly": "Component-level assembly grouping of parts and subassemblies.",

    "assembly_view": "Pre-arranged assembly visualization/layout, typically referenced by images or drawings.",
    "AssemblyView": "Pre-arranged assembly visualization/layout, typically referenced by images or drawings.",

    "version_info": "Schema/application version metadata for controlled migrations and compatibility.",
    "VersionInfo": "Schema/application version metadata for controlled migrations and compatibility.",

    # --- Problems / solutions / documents / tasks / tools / competencies ---
    "problem": "Tracks problems/failure modes encountered in the field.",
    "Problem": "Tracks problems/failure modes encountered in the field.",

    "solution": "Tracks solutions/remediations that address specific problems.",
    "Solution": "Tracks solutions/remediations that address specific problems.",

    "complete_document": "Canonical record of a complete document (manuals, procedures, job aids).",
    "CompleteDocument": "Canonical record of a complete document (manuals, procedures, job aids).",

    "completed_document": "An instance of a completed/finalized document tied to positions/assets or workflows.",
    "CompletedDocument": "An instance of a completed/finalized document tied to positions/assets or workflows.",

    "task": "Standard task/procedure definitions used for maintenance or training execution.",
    "Task": "Standard task/procedure definitions used for maintenance or training execution.",

    "tool": "Tooling references needed by tasks or associated with positions/equipment.",
    "Tool": "Tooling references needed by tasks or associated with positions/equipment.",

    "core_competency": "Competency topics/skills for training and qualification programs.",
    "CoreCompetency": "Competency topics/skills for training and qualification programs.",

    "quiz_document": "References to quiz/assessment documents used for competency validation.",
    "QuizDocument": "References to quiz/assessment documents used for competency validation.",

    # --- Association / bridge tables (many-to-many and linkers) ---
    "parts_position_image_association": "Joins Part ↔ Position ↔ Image to place a specific part in a position as illustrated by an image.",
    "PartsPositionImageAssociation": "Joins Part ↔ Position ↔ Image to place a specific part in a position as illustrated by an image.",

    "image_position_association": "Associates Image ↔ Position to show where an image depicts a position.",
    "ImagePositionAssociation": "Associates Image ↔ Position to show where an image depicts a position.",

    "drawing_position_association": "Associates Drawing ↔ Position to show where a drawing applies.",
    "DrawingPositionAssociation": "Associates Drawing ↔ Position to show where a drawing applies.",

    "drawing_part_association": "Links Drawing ↔ Part to reference parts that appear on drawings.",
    "DrawingPartAssociation": "Links Drawing ↔ Part to reference parts that appear on drawings.",

    "part_problem_association": "Links Part ↔ Problem to record known issues for a part.",
    "PartProblemAssociation": "Links Part ↔ Problem to record known issues for a part.",

    "drawing_problem_association": "Associates Drawing ↔ Problem to capture issues shown/related to the drawing.",
    "DrawingProblemAssociation": "Associates Drawing ↔ Problem to capture issues shown/related to the drawing.",

    "drawing_solution_association": "Associates Drawing ↔ Solution to capture remedies illustrated/referenced by the drawing.",
    "DrawingSolutionAssociation": "Associates Drawing ↔ Solution to capture remedies illustrated/referenced by the drawing.",

    "image_problem_association": "Associates Image ↔ Problem.",
    "ImageProblemAssociation": "Associates Image ↔ Problem.",

    "image_solution_association": "Associates Image ↔ Solution.",
    "ImageSolutionAssociation": "Associates Image ↔ Solution.",

    "problem_position_association": "Associates Problem ↔ Position (where an issue occurs).",
    "ProblemPositionAssociation": "Associates Problem ↔ Position (where an issue occurs).",

    "complete_document_problem_association": "Associates CompleteDocument ↔ Problem.",
    "CompleteDocumentProblemAssociation": "Associates CompleteDocument ↔ Problem.",

    "complete_document_solution_association": "Associates CompleteDocument ↔ Solution.",
    "CompleteDocumentSolutionAssociation": "Associates CompleteDocument ↔ Solution.",

    "image_completed_document_association": "Associates Image ↔ CompletedDocument.",
    "ImageCompletedDocumentAssociation": "Associates Image ↔ CompletedDocument.",

    "completed_document_position_association": "Associates CompletedDocument ↔ Position.",
    "CompletedDocumentPositionAssociation": "Associates CompletedDocument ↔ Position.",

    "task_position_association": "Associates Task ↔ Position.",
    "TaskPositionAssociation": "Associates Task ↔ Position.",

    "tool_position_association": "Associates Tool ↔ Position.",
    "ToolPositionAssociation": "Associates Tool ↔ Position.",
}


# ---------------------------------------------------------------------
# Utilities to map tables <-> ORM classes and extract method signatures
# ---------------------------------------------------------------------

def _iter_model_classes() -> List[type]:
    """Return a list of ORM classes registered on Base."""
    classes: List[type] = []

    # SQLAlchemy 1.x: Base._decl_class_registry
    registry = getattr(Base, "_decl_class_registry", None)
    if isinstance(registry, dict):
        for _, obj in registry.items():
            if isinstance(obj, type) and hasattr(obj, "__tablename__"):
                classes.append(obj)

    # SQLAlchemy 2.x fallback: Base.registry.mappers
    if not classes and hasattr(Base, "registry"):
        try:
            for mapper in Base.registry.mappers:
                cls = mapper.class_
                if hasattr(cls, "__tablename__"):
                    classes.append(cls)
        except Exception:
            pass

    # Deduplicate while preserving order
    seen = set()
    unique: List[type] = []
    for c in classes:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def _class_for_table(table_name: str) -> Optional[type]:
    """Find the ORM class mapped to a given table name."""
    for cls in _iter_model_classes():
        tn = getattr(cls, "__tablename__", None)
        if tn == table_name:
            return cls
        if hasattr(cls, "__table__") and getattr(cls.__table__, "name", None) == table_name:
            return cls
    return None


def _is_user_method(attr_name: str, attr: Any) -> bool:
    """Filter out dunder, SQLAlchemy internals, and non-callables."""
    if attr_name.startswith("_"):
        return False
    if not callable(attr):
        return False
    blocked_prefixes = ("query", "metadata", "registry")
    if any(attr_name.startswith(pfx) for pfx in blocked_prefixes):
        return False
    return True


def _format_method_signature(name: str, func: Any) -> str:
    """Return 'def name(arg1, ...)' with annotations/defaults if available."""
    try:
        sig = pyinspect.signature(func)
        return f"def {name}{sig}"
    except Exception:
        return f"def {name}(...)"

# ---------------------------------------------------------------------
# Purpose inference
# ---------------------------------------------------------------------

def _fk_targets(inspector, table_name: str) -> List[str]:
    """List of referenced table names for FK constraints on table_name."""
    targets = []
    try:
        for fk in inspector.get_foreign_keys(table_name):
            rt = fk.get("referred_table")
            if rt:
                targets.append(rt)
    except Exception:
        pass
    # dedupe
    seen = set()
    out = []
    for t in targets:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _first_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return text.strip().splitlines()[0].strip() or None


def _infer_purpose(table_name: str, inspector, cls: Optional[type]) -> str:
    """
    Purpose priority:
      1) SUMMARY_OVERRIDES
      2) ORM class __doc__ first line
      3) SQLAlchemy Table.comment (if available via reflection)
      4) Heuristics from name + FKs
      5) Generic fallback
    """
    # 1) override dictionary
    override = SUMMARY_OVERRIDES.get(table_name)
    if override:
        return override

    # 2) class docstring
    if cls:
        doc = _first_line(getattr(cls, "__doc__", None))
        if doc:
            return doc

    # 3) table comment (if inspector returns it—depends on dialect/permissions)
    try:
        # Some dialects expose comments via get_table_comment
        tc = inspector.get_table_comment(table_name)
        if isinstance(tc, dict):
            comm = _first_line(tc.get("text"))
            if comm:
                return comm
        elif isinstance(tc, str):
            comm = _first_line(tc)
            if comm:
                return comm
    except Exception:
        pass

    # 4) heuristics
    low = table_name.lower()
    targets = _fk_targets(inspector, table_name)

    if "association" in low or "link" in low or "junction" in low or "map" in low:
        if len(targets) >= 2:
            return f"Association table linking {' ↔ '.join(targets)}."
        if targets:
            return f"Association table linking to {', '.join(targets)}."
        return "Association table linking related entities."

    if "bom" in low or "billofmaterial" in low or "bill_of_material" in low:
        return "Bill of materials entries that enumerate required parts with quantities and notes."

    if "image" in low and "association" not in low:
        return "Stores image metadata and paths for visual references."

    if "draw" in low and "association" not in low:
        return "Engineering drawings with identifying metadata and file path."

    if "part" in low and "association" not in low:
        return "Master list of parts/spares with identifying and descriptive fields."

    if "position" in low and "association" not in low:
        return "Represents a physical equipment position/slot within a site or assembly."

    if "problem" in low and "association" not in low:
        return "Catalog of field issues or failure modes."

    if "solution" in low and "association" not in low:
        return "Catalog of remediations/solutions addressing problems."

    # 5) generic
    return f"Stores records for {table_name} with primary/foreign key relationships."

# ---------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------

@with_request_id
def generate_erd_graphs(request_id=None) -> Tuple[Optional[str], Optional[str]]:
    """Generate ER diagram as PNG and SVG using Graphviz."""
    try:
        graph = create_schema_graph(
            engine=engine,
            metadata=Base.metadata,
            show_datatypes=True,
            show_indexes=True,
            rankdir="LR",
            concentrate=False
        )

        png_path = os.path.join(OUTPUT_DIR, "schema.png")
        svg_path = os.path.join(OUTPUT_DIR, "schema.svg")

        graph.write_png(png_path)
        graph.write_svg(svg_path)

        info_id(f"ER diagram saved to {png_path} and {svg_path}", request_id)
        return png_path, svg_path
    except Exception as e:
        error_id(f"Failed to generate ERD: {e}", request_id)
        return None, None


@with_request_id
def generate_markdown_summary(request_id=None) -> Optional[str]:
    """Generate a Markdown file with tables/columns/PK/FK info (optional artifact)."""
    md_path = os.path.join(OUTPUT_DIR, "schema.md")
    insp = sqla_inspect(engine)

    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Database Schema Documentation\n\n")

            for table_name in insp.get_table_names():
                f.write(f"## {table_name}\n\n")
                f.write("| Column | Type | PK | FK |\n")
                f.write("|--------|------|----|----|\n")

                pk_cols = insp.get_pk_constraint(table_name).get("constrained_columns", [])

                for column in insp.get_columns(table_name):
                    col_name = column["name"]
                    col_type = str(column["type"])
                    is_pk = "Yes" if col_name in pk_cols else ""
                    fks = [
                        fk["referred_table"] + "." + fk["referred_columns"][0]
                        for fk in insp.get_foreign_keys(table_name)
                        if col_name in fk.get("constrained_columns", [])
                    ]
                    is_fk = ", ".join(fks)
                    f.write(f"| {col_name} | {col_type} | {is_pk} | {is_fk} |\n")

                f.write("\n")

        info_id(f"Markdown schema saved to {md_path}", request_id)
        return md_path
    except Exception as e:
        error_id(f"Failed to generate Markdown schema: {e}", request_id)
        return None


@with_request_id
def generate_pdf(png_path: Optional[str], svg_path: Optional[str], request_id=None) -> None:
    """
    Generate a PDF with:
      - Cover + Table of Contents (auto-filled from headings)
      - ERD page (PNG to preserve arrows/relationships)
      - One page per table: Purpose -> columns grid -> methods bullets
    """
    try:
        pdf_path = os.path.join(OUTPUT_DIR, "schema.pdf")
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        elements = []

        # --- Cover + TOC ---
        toc = TableOfContents()
        toc.levelStyles = [
            styles["Heading1"],
            styles["Heading2"],
        ]

        elements.append(Paragraph("Database Schema Documentation", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Table of Contents", styles["Heading1"]))
        elements.append(toc)
        elements.append(PageBreak())

        # --- ERD Diagram (always PNG for reliable arrows) ---
        if png_path and os.path.exists(png_path):
            elements.append(Paragraph("Entity Relationship Diagram", styles["Heading1"]))
            max_width = A4[0] - 100
            elements.append(Image(png_path, width=max_width, height=500))
            elements.append(PageBreak())
        else:
            error_id("ERD PNG not found, skipping diagram", request_id)

        # --- Tables: Purpose -> columns grid -> methods (bullet list) ---
        insp = sqla_inspect(engine)
        table_names = insp.get_table_names()

        for table_name in table_names:
            # Heading (goes into TOC as level 2)
            elements.append(Paragraph(table_name, styles["Heading2"]))

            # Purpose
            cls = _class_for_table(table_name)
            purpose = _infer_purpose(table_name, insp, cls)
            elements.append(Paragraph("Purpose", styles["Heading3"]))
            elements.append(Paragraph(purpose, styles["Normal"]))
            elements.append(Spacer(1, 0.15 * inch))

            # Columns grid
            data = [["Column", "Type", "PK", "FK"]]
            pk_cols = insp.get_pk_constraint(table_name).get("constrained_columns", [])

            fkeys = insp.get_foreign_keys(table_name)
            fkey_by_col = {}
            for fk in fkeys:
                for c in fk.get("constrained_columns", []) or []:
                    ref_table = fk.get("referred_table")
                    ref_cols = fk.get("referred_columns") or []
                    ref = f"{ref_table}.{ref_cols[0]}" if ref_table and ref_cols else (ref_table or "")
                    if ref:
                        fkey_by_col.setdefault(c, []).append(ref)

            for column in insp.get_columns(table_name):
                col_name = column["name"]
                col_type = str(column["type"])
                is_pk = "Yes" if col_name in pk_cols else ""
                is_fk = ", ".join(fkey_by_col.get(col_name, []))
                data.append([col_name, col_type, is_pk, is_fk])

            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightcyan]),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 0.15 * inch))

            # Methods (bullet list)
            if cls:
                elements.append(Paragraph("Methods", styles["Heading3"]))
                seen_names = set()
                for name in dir(cls):
                    if name in seen_names:
                        continue
                    try:
                        attr = getattr(cls, name)
                    except Exception:
                        continue
                    if not _is_user_method(name, attr):
                        continue
                    func_obj = attr.__func__ if isinstance(attr, (staticmethod, classmethod)) else attr
                    signature_line = _format_method_signature(name, func_obj)
                    elements.append(Paragraph(f"- {signature_line}", styles["Normal"]))
                    seen_names.add(name)

            # New page per table
            elements.append(PageBreak())

        doc.build(elements)
        info_id(f"PDF schema saved to {pdf_path}", request_id)

    except PermissionError as e:
        error_id(f"Failed to generate PDF schema (permission error). "
                 f"Close any open PDF and re-run. Details: {e}", request_id)
    except Exception as e:
        error_id(f"Failed to generate PDF schema: {e}", request_id)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    req_id = get_request_id()
    info_id("Generating schema docs...", req_id)

    png_file, svg_file = generate_erd_graphs()
    generate_markdown_summary()  # optional artifact

    generate_pdf(png_file, svg_file)

    info_id("Done! Check the schema_docs/ folder.", req_id)
