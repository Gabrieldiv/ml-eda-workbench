"""PDF renderers for structured analysis results."""

from fpdf import FPDF

from eda_workbench.analysis.types import MissingValuesResult


def render_missing_values_pdf(pdf: FPDF, result: MissingValuesResult) -> None:
    """Add a Missing Values section to an existing PDF document.

    Args:
        pdf: The shared FPDF document to add this section to. Must already
            have at least one page (``pdf.add_page()`` called beforehand).
        result: Structured result from missing_values().
    """
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Missing Values", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(
        0,
        8,
        f"Instances with missing values: {result.rows_with_missing} "
        f"out of {result.total_rows}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
