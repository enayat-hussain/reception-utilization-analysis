import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, Dict
from datetime import timedelta
from matplotlib.backends.backend_pdf import PdfPages
import re

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReceptionAnalyzer:
    def __init__(self, under_threshold=0.5, optimal_threshold=1.0, extreme_threshold=2.0):
        """
        Initialize the analyzer with utilization thresholds.

        Args:
            under_threshold (float): Ratio below which slots are underutilized.
            optimal_threshold (float): Ratio considered optimal.
            extreme_threshold (float): Ratio above which slots are overloaded.
        """
        self.calendar_data = None
        self.shift_data = None
        self.master_data = None

        self.under_threshold = under_threshold
        self.optimal_threshold = optimal_threshold
        self.extreme_threshold = extreme_threshold

    # ---------------------- DATA LOADING ----------------------
    def load_data(self, calendar_path: str, shift_path: str, shift_sheet: str = 0):
        """
        Load calendar and shift data from CSV/Excel files.

        Args:
            calendar_path (str): Path to calendar CSV.
            shift_path (str): Path to shift Excel file.
            shift_sheet (str|int): Sheet name or index in Excel.
        """
        try:
            self.calendar_data = pd.read_csv(calendar_path, dtype=str)
            self.shift_data = pd.read_excel(shift_path, sheet_name=shift_sheet, skiprows=[0], dtype=str)
            self.shift_data.rename(columns={self.shift_data.columns[0]: "Time"}, inplace=True)
            self.shift_data.dropna(axis=1, how="all", inplace=True)
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    # ---------------------- TIME NORMALIZATION ----------------------
    def normalize_time_format(self, t) -> Optional[str]:
        """
        Convert messy time strings to standard HH:MM format.

        Args:
            t: Input time as string or number.

        Returns:
            str or None: Normalized time string or None if invalid.
        """
        if pd.isna(t):
            return None
        s = (
            str(t)
            .upper()
            .replace("AM", "")
            .replace("PM", "")
            .replace(".", ":")
            .replace("-", ":")
            .replace("_", ":")
            .replace(" ", ":")
        )
        s = re.sub(r"[^0-9:]", "", s)
        if not s:
            return None
        try:
            if ":" in s:
                h, m = map(int, s.split(":")[:2])
            else:
                h, m = (int(s[:2]), int(s[2:4])) if len(s) >= 4 else (int(s[0]), int(s[1:3]))
            if 0 <= h <= 23 and 0 <= m <= 59:
                return f"{h:02d}:{m:02d}"
        except Exception:
            return None
        return None

    # ---------------------- CALENDAR CLEANING ----------------------
    def _expand_durations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split long appointments into multiple 20-minute slots.

        Args:
            df (pd.DataFrame): Calendar data with 'duration' or 'dauer' column.

        Returns:
            pd.DataFrame: Expanded calendar with one row per 20-minute slot.
        """
        dur_col = next(
            (c for c in df.columns if "dauer" in c.lower() or "duration" in c.lower()),
            None,
        )
        if not dur_col:
            return df
        df["duration_minutes"] = pd.to_numeric(df[dur_col], errors="coerce").fillna(20)
        expanded = []
        for _, row in df.iterrows():
            if pd.isna(row["slot_datetime"]) or row["duration_minutes"] <= 0:
                expanded.append(row)
                continue
            slots = max(1, int(np.ceil(row["duration_minutes"] / 20)))
            for i in range(slots):
                r = row.copy()
                r["slot_datetime"] = row["slot_datetime"] + timedelta(minutes=20 * i)
                r["slot_20min"] = r["slot_datetime"].floor("20min")
                r["slot_number"], r["total_slots"] = i + 1, slots
                expanded.append(r)
        return pd.DataFrame(expanded)

    def clean_calendar_data(self):
        """
        Clean and preprocess calendar data:
        - Normalize dates and times
        - Mark booked slots
        - Expand durations into 20-minute intervals
        """
        df = self.calendar_data.copy()

        date_col = next(
            (c for c in df.columns if df[c].astype(str).str.contains(r"\d{1,2}/\d{1,2}/\d{2,4}").any()),
            df.columns[0],
        )
        time_col = next(
            (c for c in df.columns if df[c].astype(str).str.contains(r"\d{1,2}:\d{2}").any()),
            df.columns[1],
        )
        book_col = next(
            (c for c in df.columns if df[c].astype(str).str.upper().isin(["J", "N", "Y", "YES", "1", "TRUE"]).any()),
            df.columns[-1],
        )

        df["date_cleaned"] = pd.to_datetime(df[date_col], errors="coerce")
        df["time_normalized"] = df[time_col].apply(self.normalize_time_format)
        df["slot_datetime"] = pd.to_datetime(
            df["date_cleaned"].dt.strftime("%Y-%m-%d") + " " + df["time_normalized"].fillna("00:00"),
            errors="coerce",
        )
        df.dropna(subset=["slot_datetime"], inplace=True)

        df = self._expand_durations(df)
        df["slot_20min"] = df["slot_datetime"].dt.floor("20min")
        df["is_booked"] = df[book_col].astype(str).str.upper().isin(["J", "Y", "YES", "1", "TRUE"])

        admin_words = "|".join(["freizeit", "pause", "orga", "fallbesprechung", "break", "admin"])
        if len(df.columns) > 1:
            df.loc[
                df[df.columns[1]].astype(str).str.lower().str.contains(admin_words, na=False),
                "is_booked",
            ] = False

        self.calendar_data = df.drop_duplicates()

    # ---------------------- SHIFT CLEANING ----------------------
    def clean_shift_data(self):
        """
        Clean and reshape shift data:
        - Normalize times
        - Convert to long format
        - Expand shifts across all dates in calendar
        """
        df = self.shift_data.copy()
        df["time_normalized"] = df["Time"].apply(self.normalize_time_format)
        staff_cols = [c for c in df.columns[1:] if not df[c].isna().all()]

        df_long = df.melt(
            id_vars=["Time", "time_normalized"],
            value_vars=staff_cols,
            var_name="staff_member",
            value_name="present",
        )
        df_long["is_present"] = df_long["present"].notna() & (df_long["present"].astype(str).str.strip() != "")
        df_long = df_long[df_long["time_normalized"].notna()]

        dates = pd.date_range(
            self.calendar_data["slot_datetime"].min().date(),
            self.calendar_data["slot_datetime"].max().date(),
            freq="D",
        )

        expanded = [
            df_long.assign(
                shift_datetime=pd.to_datetime(d.strftime("%Y-%m-%d") + " " + df_long["time_normalized"]).dt.floor(
                    "20min"
                )
            )[["shift_datetime", "staff_member", "is_present"]].rename(columns={"shift_datetime": "slot_20min"})
            for d in dates
        ]

        self.shift_data = pd.concat(expanded, ignore_index=True)

    # ---------------------- MASTER DATA ----------------------
    def create_master_dataset(self):
        """
        Combine calendar and shift data into a master dataset:
        - Calculate patient and staff counts per slot
        - Compute utilization ratios and categories
        """
        patients = (
            self.calendar_data[self.calendar_data["is_booked"]]
            .groupby("slot_20min")
            .size()
            .reset_index(name="patient_count")
        )
        staff = (
            self.shift_data[self.shift_data["is_present"]].groupby("slot_20min").size().reset_index(name="staff_count")
        )

        master = pd.merge(patients, staff, on="slot_20min", how="outer").fillna(0)
        master[["patient_count", "staff_count"]] = master[["patient_count", "staff_count"]].astype(int)

        master["utilization_ratio"] = np.where(
            master["staff_count"] > 0,
            master["patient_count"] / master["staff_count"],
            np.where(master["patient_count"] > 0, np.inf, np.nan),
        )

        bins = [
            -np.inf,
            self.under_threshold,
            self.optimal_threshold,
            self.extreme_threshold,
            np.inf,
        ]
        labels = ["Underutilized", "Optimal", "High", "Overloaded"]
        master["utilization_category"] = pd.cut(
            master["utilization_ratio"].replace(np.inf, np.nan),
            bins=bins,
            labels=labels,
        )
        master["utilization_category"] = master["utilization_category"].cat.add_categories(["Critical"])
        master.loc[master["utilization_ratio"] == np.inf, "utilization_category"] = "Critical"

        master["date"], master["time"], master["weekday"] = (
            master["slot_20min"].dt.date,
            master["slot_20min"].dt.time,
            master["slot_20min"].dt.day_name(),
        )
        master["hour"] = master["slot_20min"].dt.hour
        self.master_data = master

    # ---------------------- REPORTS ----------------------
    def generate_analysis_report(self) -> Dict:
        """
        Generate a summary report of reception utilization.

        Returns:
            dict: Summary including total slots, average patients/staff,
                  average and max utilization, overloaded/underutilized/no-staff slots,
                  and counts of utilization categories.
        """
        df = self.master_data
        return {
            "summary_statistics": {
                "total_slots": len(df),
                "avg_patients": df["patient_count"].mean(),
                "avg_staff": df["staff_count"].mean(),
                "avg_utilization": df["utilization_ratio"].mean(),
                "max_utilization": df["utilization_ratio"].max(),
            },
            "problem_identification": {
                "overloaded_slots": (df["utilization_ratio"] > self.extreme_threshold).sum(),
                "underutilized_slots": (df["utilization_ratio"] < self.under_threshold).sum(),
                "no_staff_slots": (df["staff_count"] == 0).sum(),
            },
            "utilization_categories": df["utilization_category"].value_counts().to_dict(),
        }

    def create_visualizations(self):
        """
        Create and save visualizations of reception utilization:
        - Heatmap of hourly utilization by weekday
        - Average utilization bar chart
        - Utilization ratio histogram
        - Pie chart of utilization categories
        Saves figure as 'reception_analysis.png'.
        """
        df = self.master_data
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Reception Utilization Analysis", fontsize=16, fontweight="bold")

        # Heatmap
        pivot_data = df.pivot_table(values="utilization_ratio", index="hour", columns="weekday", aggfunc="mean")
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        pivot_data = pivot_data.reindex(columns=[d for d in weekday_order if d in pivot_data.columns])
        sns.heatmap(
            pivot_data,
            ax=axes[0, 0],
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=1.0,
        )
        axes[0, 0].set_title("Utilization Heatmap")

        # Bar chart
        daily_avg = df.groupby("weekday")["utilization_ratio"].mean().reindex(weekday_order)
        daily_avg.plot(kind="bar", ax=axes[0, 1], color="skyblue", alpha=0.7)
        axes[0, 1].axhline(
            y=self.optimal_threshold,
            color="green",
            linestyle="-",
            alpha=0.7,
            label="Optimal",
        )
        axes[0, 1].axhline(
            y=self.under_threshold,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Under",
        )
        axes[0, 1].axhline(
            y=self.extreme_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Extreme",
        )
        axes[0, 1].legend()
        axes[0, 1].set_title("Average Utilization by Day")

        # Histogram
        df["utilization_ratio"].hist(bins=30, ax=axes[1, 0], color="lightgreen", alpha=0.7)
        axes[1, 0].axvline(
            x=self.optimal_threshold,
            color="green",
            linestyle="-",
            alpha=0.7,
            label="Optimal",
        )
        axes[1, 0].axvline(
            x=self.under_threshold,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Under",
        )
        axes[1, 0].axvline(
            x=self.extreme_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Extreme",
        )
        axes[1, 0].legend()
        axes[1, 0].set_title("Utilization Ratio Distribution")

        # Pie chart (ignore zeros)
        counts = df["utilization_category"].value_counts()
        counts = counts[counts > 0]
        axes[1, 1].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        axes[1, 1].set_title("Utilization Categories")

        # Adjust layout spacing
        plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
        plt.savefig("reception_analysis.png", dpi=300, bbox_inches="tight")

    def generate_pdf_report(self, filename="reception_report.pdf"):
        """
        Generate a PDF report of reception utilization:
        - Includes summary statistics and a heatmap visualization.

        Args:
            filename (str): Path to save PDF report.
        """
        report = self.generate_analysis_report()
        with PdfPages(filename) as pdf:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [1, 3]})
            ax1.axis("off")
            ax1.text(
                0.05,
                0.95,
                f"""
RECEPTION UTILIZATION REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
• Total Slots: {report['summary_statistics']['total_slots']:,}
• Avg Utilization: {report['summary_statistics']['avg_utilization']:.2f}
• Overloaded: {report['problem_identification']['overloaded_slots']:,}
• Underutilized: {report['problem_identification']['underutilized_slots']:,}
            """,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
            )

            pivot_data = self.master_data.pivot_table(
                values="utilization_ratio",
                index="hour",
                columns="weekday",
                aggfunc="mean",
            )
            sns.heatmap(pivot_data, ax=ax2, annot=True, fmt=".2f", cmap="RdYlBu_r", center=1.0)
            ax2.set_title("Utilization Heatmap")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    def export_enhanced_results(self):
        """
        Export enhanced master dataset to Excel:
        - Adds boolean columns for each utilization category
        - Saves file as 'enhanced_reception_results.xlsx'
        """
        df = self.master_data.copy()
        df["Underutilized_Slot"] = df["utilization_ratio"] < self.under_threshold
        df["Optimal_Slot"] = df["utilization_ratio"].between(self.under_threshold, self.optimal_threshold)
        df["High_Slot"] = df["utilization_ratio"].between(self.optimal_threshold, self.extreme_threshold)
        df["Overloaded_Slot"] = df["utilization_ratio"] > self.extreme_threshold
        df.to_excel("enhanced_reception_results.xlsx", index=False)


# ---------------------- MAIN ----------------------
def main():
    analyzer = ReceptionAnalyzer()
    try:
        analyzer.load_data("CalendarData_RAW.csv", "Shiftplan - Admins.xlsx")
        analyzer.clean_calendar_data()
        analyzer.clean_shift_data()
        analyzer.create_master_dataset()
        report = analyzer.generate_analysis_report()

        print("\nRECEPTION UTILIZATION ANALYSIS")
        print(f"Total slots: {report['summary_statistics']['total_slots']:,}")
        print(f"Avg utilization: {report['summary_statistics']['avg_utilization']:.2f}")
        print(f"Peak utilization: {report['summary_statistics']['max_utilization']:.2f}")

        analyzer.create_visualizations()
        analyzer.generate_pdf_report()
        analyzer.export_enhanced_results()
        print("\n✅ Analysis complete! Results exported.")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()
