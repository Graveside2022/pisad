"""Report generator service for mission analytics."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.backend.services.performance_analytics import MissionPerformanceMetrics
from src.backend.utils.logging import get_logger

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")

logger = get_logger(__name__)


class ReportConfig(BaseModel):
    """Report generation configuration."""

    include_charts: bool = True
    include_recommendations: bool = True
    include_raw_data: bool = False
    chart_dpi: int = 100
    page_size: str = "letter"


class EmailConfig(BaseModel):
    """Email delivery configuration."""

    recipient: str
    subject: str = "Mission Performance Report"
    body: str = "Please find attached the mission performance report."
    smtp_server: str | None = None
    smtp_port: int = 587
    sender_email: str | None = None
    sender_password: str | None = None


class ReportGenerator:
    """Service for generating mission performance reports."""

    def __init__(self) -> None:
        """Initialize the report generator."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        """Setup custom paragraph styles."""
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=24,
                textColor=colors.HexColor("#1976d2"),
                spaceAfter=30,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CustomHeading",
                parent=self.styles["Heading1"],
                fontSize=16,
                textColor=colors.HexColor("#424242"),
                spaceAfter=12,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CustomBody",
                parent=self.styles["BodyText"],
                fontSize=10,
                alignment=TA_JUSTIFY,
            )
        )

    def generate_mission_summary(self, metrics: MissionPerformanceMetrics) -> dict[str, Any]:
        """
        Generate mission summary with key statistics.

        Args:
            metrics: Mission performance metrics

        Returns:
            Summary dictionary
        """
        summary = {
            "mission_id": str(metrics.mission_id),
            "overall_score": round(metrics.overall_score, 1),
            "key_metrics": {
                "total_detections": metrics.detection_metrics.get("total_detections", 0),
                "detections_per_hour": round(
                    metrics.detection_metrics.get("detections_per_hour", 0), 2
                ),
                "final_distance_m": metrics.approach_metrics.get("final_distance_m"),
                "search_time_minutes": round(
                    metrics.search_metrics.get("search_time_minutes", 0), 1
                ),
                "area_covered_km2": round(metrics.search_metrics.get("area_covered_km2", 0), 2),
                "coverage_percentage": round(
                    metrics.search_metrics.get("coverage_percentage", 0), 1
                ),
            },
            "performance_vs_baseline": {
                "time_improvement": f"{metrics.baseline_comparison.get('time_improvement_percent', 0):.1f}%",
                "accuracy_improvement": f"{metrics.baseline_comparison.get('accuracy_improvement_percent', 0):.1f}%",
                "cost_reduction": f"{metrics.baseline_comparison.get('cost_reduction_percent', 0):.1f}%",
            },
            "top_recommendations": metrics.recommendations[:3] if metrics.recommendations else [],
        }
        return summary

    def create_performance_visualizations(
        self, metrics: MissionPerformanceMetrics, output_dir: Path
    ) -> list[Path]:
        """
        Create performance visualization charts.

        Args:
            metrics: Mission performance metrics
            output_dir: Directory to save charts

        Returns:
            List of chart file paths
        """
        chart_files = []
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Detection metrics bar chart
        detection_chart = self._create_detection_chart(metrics, output_dir)
        if detection_chart:
            chart_files.append(detection_chart)

        # 2. Search efficiency pie chart
        efficiency_chart = self._create_efficiency_chart(metrics, output_dir)
        if efficiency_chart:
            chart_files.append(efficiency_chart)

        # 3. Baseline comparison chart
        comparison_chart = self._create_comparison_chart(metrics, output_dir)
        if comparison_chart:
            chart_files.append(comparison_chart)

        # 4. Overall performance radar chart
        radar_chart = self._create_radar_chart(metrics, output_dir)
        if radar_chart:
            chart_files.append(radar_chart)

        return chart_files

    def _create_detection_chart(
        self, metrics: MissionPerformanceMetrics, output_dir: Path
    ) -> Path | None:
        """Create detection metrics bar chart."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            metrics_data = {
                "Total\nDetections": metrics.detection_metrics.get("total_detections", 0),
                "Detections\nper Hour": metrics.detection_metrics.get("detections_per_hour", 0),
                "Detections\nper km²": metrics.detection_metrics.get("detections_per_km2", 0),
                "Mean\nConfidence": metrics.detection_metrics.get("mean_detection_confidence", 0),
            }

            bars = ax.bar(metrics_data.keys(), metrics_data.values(), color="#1976d2")
            ax.set_ylabel("Value")
            ax.set_title("Detection Performance Metrics")
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            chart_path = output_dir / "detection_metrics.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()
            return chart_path
        except Exception as e:
            logger.error(f"Error creating detection chart: {e}")
            return None

    def _create_efficiency_chart(
        self, metrics: MissionPerformanceMetrics, output_dir: Path
    ) -> Path | None:
        """Create search efficiency pie chart."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            covered = metrics.search_metrics.get("area_covered_km2", 0)
            total = metrics.search_metrics.get("total_area_km2", 1)
            uncovered = max(0, total - covered)

            sizes = [covered, uncovered]
            labels = ["Covered Area", "Uncovered Area"]
            colors_list = ["#4caf50", "#e0e0e0"]

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors_list,
                autopct="%1.1f%%",
                startangle=90,
            )

            ax.set_title("Search Area Coverage")
            plt.tight_layout()
            chart_path = output_dir / "search_efficiency.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()
            return chart_path
        except Exception as e:
            logger.error(f"Error creating efficiency chart: {e}")
            return None

    def _create_comparison_chart(
        self, metrics: MissionPerformanceMetrics, output_dir: Path
    ) -> Path | None:
        """Create baseline comparison bar chart."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            categories = ["Time", "Accuracy", "Cost", "Workload"]
            improvements = [
                metrics.baseline_comparison.get("time_improvement_percent", 0),
                metrics.baseline_comparison.get("accuracy_improvement_percent", 0),
                metrics.baseline_comparison.get("cost_reduction_percent", 0),
                metrics.baseline_comparison.get("operator_workload_reduction", 0),
            ]

            x = np.arange(len(categories))
            width = 0.6

            bars = ax.bar(x, improvements, width, color="#ff9800")
            ax.set_ylabel("Improvement (%)")
            ax.set_title("Performance vs Baseline Manual Methods")
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax.grid(axis="y", alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                )

            plt.tight_layout()
            chart_path = output_dir / "baseline_comparison.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()
            return chart_path
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return None

    def _create_radar_chart(
        self, metrics: MissionPerformanceMetrics, output_dir: Path
    ) -> Path | None:
        """Create overall performance radar chart."""
        try:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="polar")

            # Define metrics
            categories = [
                "Detection\nRate",
                "Approach\nAccuracy",
                "Search\nEfficiency",
                "False Positive\nRate",
                "Environmental\nAdaptation",
            ]
            N = len(categories)

            # Get values (normalized to 0-100)
            values = [
                min(100, metrics.detection_metrics.get("mean_detection_confidence", 0)),
                min(100, metrics.approach_metrics.get("approach_efficiency", 0)),
                min(100, metrics.search_metrics.get("search_pattern_efficiency", 0)),
                min(
                    100,
                    (1 - metrics.false_positive_analysis.get("precision", 0)) * 100,
                ),
                min(100, metrics.environmental_correlation.get("weather_impact_score", 50)),
            ]

            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            values += values[:1]  # Complete the circle
            angles += angles[:1]

            # Plot
            ax.plot(angles, values, "o-", linewidth=2, color="#1976d2")
            ax.fill(angles, values, alpha=0.25, color="#1976d2")
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            ax.set_title("Overall Performance Profile", size=16, y=1.08)
            ax.grid(True)

            plt.tight_layout()
            chart_path = output_dir / "performance_radar.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()
            return chart_path
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
            return None

    def generate_pdf_report(
        self,
        metrics: MissionPerformanceMetrics,
        output_path: Path,
        config: ReportConfig | None = None,
    ) -> bool:
        """
        Generate PDF report with charts and analysis.

        Args:
            metrics: Mission performance metrics
            output_path: Path for output PDF file
            config: Report configuration

        Returns:
            True if report generated successfully
        """
        if config is None:
            config = ReportConfig()

        try:
            # Create document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )

            # Build content
            story = []

            # Title
            story.append(
                Paragraph(
                    "Mission Performance Report",
                    self.styles["CustomTitle"],
                )
            )
            story.append(
                Paragraph(
                    f"Mission ID: {metrics.mission_id}",
                    self.styles["Normal"],
                )
            )
            story.append(
                Paragraph(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    self.styles["Normal"],
                )
            )
            story.append(Spacer(1, 0.5 * inch))

            # Executive Summary
            story.append(Paragraph("Executive Summary", self.styles["CustomHeading"]))
            summary = self.generate_mission_summary(metrics)
            summary_text = f"""
            Overall Performance Score: <b>{summary['overall_score']}/100</b><br/>
            <br/>
            The mission achieved {summary['key_metrics']['total_detections']} detections
            over {summary['key_metrics']['search_time_minutes']:.1f} minutes,
            covering {summary['key_metrics']['area_covered_km2']:.2f} km²
            ({summary['key_metrics']['coverage_percentage']:.1f}% of search area).
            """
            story.append(Paragraph(summary_text, self.styles["CustomBody"]))
            story.append(Spacer(1, 0.3 * inch))

            # Key Metrics Table
            story.append(Paragraph("Key Performance Metrics", self.styles["CustomHeading"]))
            metrics_data = [
                ["Metric", "Value", "Baseline", "Improvement"],
                [
                    "Search Time",
                    f"{summary['key_metrics']['search_time_minutes']:.1f} min",
                    "120 min",
                    summary["performance_vs_baseline"]["time_improvement"],
                ],
                [
                    "Final Distance",
                    f"{summary['key_metrics']['final_distance_m'] or 'N/A'} m",
                    "50 m",
                    summary["performance_vs_baseline"]["accuracy_improvement"],
                ],
                [
                    "Detection Rate",
                    f"{summary['key_metrics']['detections_per_hour']:.2f}/hr",
                    "N/A",
                    "N/A",
                ],
                [
                    "Area Coverage",
                    f"{summary['key_metrics']['coverage_percentage']:.1f}%",
                    "N/A",
                    "N/A",
                ],
            ]

            metrics_table = Table(
                metrics_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]
            )
            metrics_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(metrics_table)
            story.append(Spacer(1, 0.3 * inch))

            # Performance Charts
            if config.include_charts:
                story.append(PageBreak())
                story.append(Paragraph("Performance Visualizations", self.styles["CustomHeading"]))

                # Generate charts
                chart_dir = output_path.parent / "charts" / str(metrics.mission_id)
                chart_files = self.create_performance_visualizations(metrics, chart_dir)

                for chart_file in chart_files:
                    if chart_file.exists():
                        img = Image(str(chart_file), width=5 * inch, height=3.75 * inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2 * inch))

            # Recommendations
            if config.include_recommendations and metrics.recommendations:
                story.append(PageBreak())
                story.append(Paragraph("Recommendations", self.styles["CustomHeading"]))
                for i, rec in enumerate(metrics.recommendations, 1):
                    story.append(Paragraph(f"{i}. {rec}", self.styles["CustomBody"]))
                story.append(Spacer(1, 0.3 * inch))

            # Build PDF
            doc.build(story)
            logger.info(f"Generated PDF report: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return False

    async def send_report_email(
        self,
        report_path: Path,
        config: EmailConfig,
    ) -> bool:
        """
        Send report via email.

        Args:
            report_path: Path to report file
            config: Email configuration

        Returns:
            True if email sent successfully
        """
        try:
            import smtplib
            from email.mime.application import MIMEApplication
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            if not config.smtp_server or not config.sender_email:
                logger.error("Email configuration incomplete")
                return False

            # Create message
            msg = MIMEMultipart()
            msg["From"] = config.sender_email
            msg["To"] = config.recipient
            msg["Subject"] = config.subject

            # Add body
            msg.attach(MIMEText(config.body, "plain"))

            # Add attachment
            with open(report_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="pdf")
                attach.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=report_path.name,
                )
                msg.attach(attach)

            # Send email
            with smtplib.SMTP(config.smtp_server, config.smtp_port) as server:
                server.starttls()
                if config.sender_password:
                    server.login(config.sender_email, config.sender_password)
                server.send_message(msg)

            logger.info(f"Report emailed to {config.recipient}")
            return True

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def generate_json_report(
        self,
        metrics: MissionPerformanceMetrics,
        output_path: Path,
    ) -> bool:
        """
        Generate JSON format report.

        Args:
            metrics: Mission performance metrics
            output_path: Path for output JSON file

        Returns:
            True if report generated successfully
        """
        try:
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_version": "1.0",
                    "generator": "PISAD Report Generator",
                },
                "mission_summary": self.generate_mission_summary(metrics),
                "detailed_metrics": metrics.model_dump(),
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Generated JSON report: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            return False
