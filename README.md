# Reception Utilization Analyzer

## Overview
The **Reception Utilization Analyzer** is a Python tool designed to clean, process, and analyze physiotherapy clinic reception data.  
It helps identify underutilized and overloaded time slots, visualize patterns in staff vs. patient workload, and generate reports for operational optimization.

## Features
- Cleans raw calendar and shift plan data
- Normalizes messy time formats
- Expands long appointments into multiple 20-minute slots
- Handles duplicates and administrative events (e.g., breaks, pauses)
- Combines staff shift data with patient bookings
- Calculates utilization ratios and categorizes slots:
  - Underutilized
  - Optimal
  - High
  - Overloaded
  - Critical (patients booked but no staff)
- Generates:
  - 📊 Summary statistics
  - 📈 Visualizations (heatmap, bar chart, histogram, pie chart)
  - 📄 PDF report
  - 📑 Enhanced Excel results

## Input Data
- `CalendarData_RAW.csv` → Raw patient appointments  
- `Shiftplan - Admins.xlsx` → Staff availability (shift plan)

## Outputs
- `enhanced_reception_analysis.png` → Visualization charts  
- `reception_report.pdf` → PDF report  
- `enhanced_reception_results.xlsx` → Enhanced dataset with utilization flags  

Example visualization (heatmap of utilization):  
![Utilization Heatmap](enhanced_reception_analysis.png)

## Requirements
- Python 3.9+
- pandas  
- numpy  
- matplotlib  
- seaborn  
- openpyxl  

Install all dependencies:
```bash
pip install -r requirements.txt
