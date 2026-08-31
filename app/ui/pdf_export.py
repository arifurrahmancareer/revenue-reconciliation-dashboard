from __future__ import annotations

import io
from decimal import Decimal
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def _fmt(value, currency: str = "") -> str:
    try:
        amount = Decimal(str(value or "0"))
        prefix = f"{currency} " if currency else ""
        return f"{prefix}{amount:,.2f}"
    except Exception:
        return "0.00"

def generate_reconciliation_pdf(run, summary: dict, priorities: list[dict]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']

    elements = []

    # Title
    elements.append(Paragraph(f"Executive Reconciliation Report", title_style))
    elements.append(Paragraph(f"Run ID: {run.id} | Engine v{run.engine_version} | {run.created_at.strftime('%Y-%m-%d %H:%M')}", normal_style))
    elements.append(Spacer(1, 20))

    # Headline KPIs
    elements.append(Paragraph("Headline Overview", subtitle_style))

    kpi_data = [
        ["Total Orders", f"{summary.get('total_orders', 0):,}", "Payments Settled", _fmt(summary.get("total_payments_settled"))],
        ["Matched Rate", f"{summary.get('match_rate_pct', 0)}%", "Total Order Value", _fmt(summary.get("total_order_value"))],
        ["Reconciled Value", _fmt(summary.get("reconciled_value")), "Disputed Value", _fmt(summary.get("disputed_value"))],
        ["Money at Risk", _fmt(summary.get("money_at_risk")), "", ""]
    ]

    kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 130])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.white)
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 20))

    # Priority Queue
    elements.append(Paragraph("Action Required: Priority Queue", subtitle_style))

    if priorities:
        # Table Header
        table_data = [["Order Ref", "Severity", "Risk Direction", "Amount at Risk"]]
        for p in priorities:
            table_data.append([
                str(p.get("order_key", "")),
                str(p.get("severity", "")),
                str(p.get("risk_direction", "")).replace("_", " ").title(),
                _fmt(p.get("amount_at_risk"), p.get("currency") or "")
            ])

        priority_table = Table(table_data, colWidths=[140, 100, 150, 130])
        priority_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.4, 0.6)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(priority_table)
    else:
        elements.append(Paragraph("No critical discrepancies found.", normal_style))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Note: For full evidence and AI explanations, please view the live dashboard.", normal_style))

    # Build Document
    doc.build(elements)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
