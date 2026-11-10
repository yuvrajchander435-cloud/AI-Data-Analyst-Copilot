
import io
import re
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet


# 1. DATA PROFILING
def profile_data(df):

    rows = []

    for col in df.columns:
        rows.append({
            "Column": str(col),
            "Data Type": str(df[col].dtype),
            "Non Null": int(df[col].notna().sum()),
            "Missing": int(df[col].isna().sum()),
            "Unique Values": int(df[col].nunique())
        })

    return pd.DataFrame(rows)


# 2. QUESTION SUGGESTIONS
def suggest_questions(df):

    numeric = list(df.select_dtypes(include=np.number).columns)

    categorical = [
        c for c in df.columns if c not in numeric
    ]

    questions = []

    if numeric:
        questions.append(
            f"What is the average {numeric[0]}?"
        )

        questions.append(
            f"What are the highest 5 values of {numeric[0]}?"
        )

    if numeric and categorical:
        questions.append(
            f"Which {categorical[0]} has the highest {numeric[0]}?"
        )

    if numeric:
        questions.append(
            f"Show summary of {numeric[0]}"
        )

    return questions[:4]


# 3. NATURAL LANGUAGE DATA ANALYSIS
def answer_question(df, question):

    q = question.lower().strip()

    numeric = list(
        df.select_dtypes(include=np.number).columns
    )

    if not numeric:
        return (
            "No numeric columns found in this dataset.",
            None
        )

    metric = numeric[0]

    for col in numeric:
        if str(col).lower() in q:
            metric = col
            break

    # AVERAGE
    if any(word in q for word in ["average", "mean", "avg"]):

        value = df[metric].mean()

        return (
            f"**Average {metric}: {value:,.2f}**",
            None
        )

    # TOTAL
    elif any(word in q for word in ["total", "sum"]):

        value = df[metric].sum()

        return (
            f"**Total {metric}: {value:,.2f}**",
            None
        )

    # HIGHEST VALUES
    elif any(word in q for word in ["highest", "maximum", "max", "top"]):

        match = re.search(r"\btop\s+(\d+)", q)

        n = int(match.group(1)) if match else 5

        result = df.nlargest(
            min(n, len(df)),
            metric
        )

        return (
            f"**Top {len(result)} records by {metric}:**",
            result
        )

    # LOWEST VALUES
    elif any(word in q for word in ["lowest", "minimum", "min", "bottom"]):

        result = df.nsmallest(
            min(5, len(df)),
            metric
        )

        return (
            f"**Lowest 5 records by {metric}:**",
            result
        )

    # STATISTICAL SUMMARY
    elif any(word in q for word in ["summary", "describe", "statistics", "stats"]):

        result = (
            df[[metric]]
            .describe()
            .reset_index()
        )

        return (
            f"**Statistical summary of {metric}:**",
            result
        )

    # GROUP ANALYSIS
    elif any(word in q for word in ["which", "group", "region", "category", "by"]):

        categorical = [
            c for c in df.columns
            if c not in numeric
        ]

        group_col = None

        for col in categorical:
            if str(col).lower() in q:
                group_col = col
                break

        if group_col is None and categorical:
            group_col = categorical[0]

        if group_col is not None:

            result = (
                df.groupby(group_col, dropna=False)[metric]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            result.columns = [
                group_col,
                f"Average {metric}"
            ]

            return (
                f"**Average {metric} by {group_col}:**",
                result
            )

    return (
        "I could not understand that question. "
        "Try asking about average, total, highest, "
        "lowest, summary or group analysis.",
        None
    )


# 4. MACHINE LEARNING FORECASTING
def build_forecast(df, date_col, value_col, periods):

    data = df[[date_col, value_col]].copy()

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce"
    )

    data[value_col] = pd.to_numeric(
        data[value_col],
        errors="coerce"
    )

    data = data.dropna()

    data = data.set_index(date_col).sort_index()

    monthly = (
        data[value_col]
        .resample("MS")
        .sum()
        .reset_index()
    )

    if len(monthly) < 3:
        raise ValueError(
            "At least 3 months of valid data are required."
        )

    monthly.columns = ["Date", value_col]

    # Training data
    X = np.arange(len(monthly)).reshape(-1, 1)

    y = monthly[value_col].values

    # Train Linear Regression
    model = LinearRegression()

    model.fit(X, y)

    # Future prediction
    future_X = np.arange(
        len(monthly),
        len(monthly) + periods
    ).reshape(-1, 1)

    future_dates = pd.date_range(
        start=monthly["Date"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS"
    )

    predictions = model.predict(future_X)

    future = pd.DataFrame({
        "Date": future_dates,
        value_col: predictions.round(2)
    })

    return monthly, future


# 5. ANOMALY DETECTION
def detect_anomalies(df, columns, contamination=0.05):

    result = df.copy()

    values = result[columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    values = values.fillna(
        values.median()
    ).fillna(0)

    if len(values) < 8:
        raise ValueError(
            "At least 8 rows are required."
        )

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(values)

    result["Anomaly"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )

    return result


# 6. PDF REPORT GENERATOR
def make_pdf_report(df):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "AI Data Analyst Copilot Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            f"Total Rows: {len(df)}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Total Columns: {len(df.columns)}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Missing Values: {int(df.isna().sum().sum())}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Duplicate Rows: {int(df.duplicated().sum())}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "Column Information",
            styles["Heading2"]
        )
    )

    profile = profile_data(df)

    table_data = [
        list(profile.columns)
    ] + profile.astype(str).values.tolist()

    table = Table(
        table_data,
        repeatRows=1
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#263B5A")),

            ("TEXTCOLOR", (0, 0), (-1, 0),
             colors.white),

            ("GRID", (0, 0), (-1, -1),
             0.5, colors.grey),

            ("FONTSIZE", (0, 0), (-1, -1), 7),

            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ])
    )

    elements.append(table)

    # Numeric summary
    numeric = df.select_dtypes(
        include=np.number
    )

    if not numeric.empty:

        elements.append(Spacer(1, 15))

        elements.append(
            Paragraph(
                "Statistical Summary",
                styles["Heading2"]
            )
        )

        summary = (
            numeric.describe()
            .round(2)
            .T
            .reset_index()
        )

        summary.columns = [
            str(c) for c in summary.columns
        ]

        summary_data = [
            list(summary.columns)
        ] + summary.astype(str).values.tolist()

        summary_table = Table(
            summary_data,
            repeatRows=1
        )

        summary_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#263B5A")),

                ("TEXTCOLOR", (0, 0), (-1, 0),
                 colors.white),

                ("GRID", (0, 0), (-1, -1),
                 0.5, colors.grey),

                ("FONTSIZE", (0, 0), (-1, -1), 7)
            ])
        )

        elements.append(summary_table)

    document.build(elements)

    return buffer.getvalue()
