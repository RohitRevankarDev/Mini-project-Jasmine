import streamlit as st
import pandas as pd
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Jasmine Weather–Price Analysis",
    page_icon="🌼",
    layout="wide"
)


# ============================================================
# FILE NAMES
# ============================================================

FEATURE_FILE = "feature_dataset_v1.csv"
EVENT_FILE = "weather_events_database.csv"
PREDICTION_FILE = "Phase_7_Jasmine_Predictions.csv"


# ============================================================
# TITLE
# ============================================================

st.title("🌼 Jasmine Weather–Price Analysis")

st.markdown(
    """
    **Weather-Induced Price Volatility in Jasmine**

    Explore historical Jasmine prices, weather events,
    Phase 7 predictions, and model evaluation.
    """
)

st.divider()


# ============================================================
# LOAD FEATURE DATA
# ============================================================

if not os.path.exists(FEATURE_FILE):
    st.error(
        f"❌ `{FEATURE_FILE}` not found.\n\n"
        "Keep feature_dataset_v1.csv in the same folder as app.py."
    )
    st.stop()

df = pd.read_csv(FEATURE_FILE)


# ============================================================
# LOAD WEATHER EVENT DATABASE
# ============================================================

if not os.path.exists(EVENT_FILE):
    st.error(
        f"❌ `{EVENT_FILE}` not found.\n\n"
        "Keep weather_events_database.csv in the same folder as app.py."
    )
    st.stop()

events_df = pd.read_csv(EVENT_FILE)


# ============================================================
# LOAD PREDICTION DATA
# ============================================================

if not os.path.exists(PREDICTION_FILE):
    st.error(
        f"❌ `{PREDICTION_FILE}` not found.\n\n"
        "Keep Phase_7_Jasmine_Predictions.csv in the same folder as app.py."
    )
    st.stop()

prediction_df = pd.read_csv(PREDICTION_FILE)


# ============================================================
# DATE CONVERSION
# ============================================================

if "DATE" in df.columns:
    df["DATE"] = pd.to_datetime(
        df["DATE"],
        errors="coerce"
    )

if "DATE" in events_df.columns:
    events_df["DATE"] = pd.to_datetime(
        events_df["DATE"],
        errors="coerce"
    )

if "DATE" in prediction_df.columns:
    prediction_df["DATE"] = pd.to_datetime(
        prediction_df["DATE"],
        errors="coerce"
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "PRICE_INR",
    "RAINFALL_MM",
    "MAX_TEMP_C",
    "MIN_TEMP_C",
    "MEAN_TEMP_C",
    "HUMIDITY_PERCENT",
    "FLAG_HEAVY_RAIN",
    "FLAG_HOT_DAY",
    "FLAG_HUMID_DAY",
    "FLAG_COMBINED_STRESS"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


if "VALUE" in events_df.columns:

    events_df["VALUE"] = pd.to_numeric(
        events_df["VALUE"],
        errors="coerce"
    )


# ============================================================
# HEAVY RAIN THRESHOLD DEFINITION (> 25mm)
# ============================================================

if "RAINFALL_MM" in df.columns:
    df["FLAG_HEAVY_RAIN"] = (df["RAINFALL_MM"] > 49.0).astype(int)

    if "FLAG_HOT_DAY" in df.columns and "FLAG_HUMID_DAY" in df.columns:
        df["FLAG_COMBINED_STRESS"] = (
            (df["FLAG_HEAVY_RAIN"] == 1) |
            (df["FLAG_HOT_DAY"] == 1) |
            (df["FLAG_HUMID_DAY"] == 1)
        ).astype(int)

if "RAINFALL_MM" in df.columns and "DATE" in df.columns and "EVENT_TYPE" in events_df.columns:
    heavy_rain_days = df[df["RAINFALL_MM"] > 25.0][["DATE", "RAINFALL_MM"]].dropna().copy()
    heavy_rain_days.rename(columns={"RAINFALL_MM": "VALUE"}, inplace=True)
    heavy_rain_days["EVENT_TYPE"] = "HEAVY_RAIN"

    events_df = pd.concat(
        [events_df[events_df["EVENT_TYPE"] != "HEAVY_RAIN"], heavy_rain_days[["DATE", "EVENT_TYPE", "VALUE"]]],
        ignore_index=True
    ).sort_values("DATE").reset_index(drop=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🌼 Project Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "📊 Historical Explorer",
        "🌧️ Weather Events",
        "🤖 Phase 7 Prediction",
        "📈 Model Evaluation"
    ]
)


# ============================================================
# PAGE 1 — HISTORICAL EXPLORER
# ============================================================

if page == "📊 Historical Explorer":

    st.header("📊 Historical Explorer")

    st.write(
        "Explore Jasmine prices and corresponding weather "
        "conditions for individual historical dates."
    )

    st.divider()

    valid_dates = (
        df["DATE"]
        .dropna()
        .sort_values()
        .dt.date
        .unique()
    )

    if len(valid_dates) == 0:

        st.warning("No valid dates found.")

        st.stop()

    selected_date = st.selectbox(
        "📅 Select Date",
        valid_dates
    )

    selected_datetime = pd.Timestamp(
        selected_date
    )

    selected_rows = df[
        df["DATE"].dt.normalize()
        == selected_datetime
    ].copy()

    if selected_rows.empty:

        st.warning(
            "No data available for the selected date."
        )

        st.stop()

    row = selected_rows.iloc[0]

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    st.subheader("🌼 Jasmine Price")

    price = row.get("PRICE_INR")

    if pd.notna(price):

        st.metric(
            "Jasmine Price",
            f"₹{price:.2f}"
        )

    else:

        st.metric(
            "Jasmine Price",
            "N/A"
        )

    st.divider()

    # --------------------------------------------------------
    # WEATHER CONDITIONS
    # --------------------------------------------------------

    st.subheader("🌦️ Weather Conditions")

    weather_columns = [
        "RAINFALL_MM",
        "MAX_TEMP_C",
        "MIN_TEMP_C",
        "MEAN_TEMP_C",
        "HUMIDITY_PERCENT"
    ]

    weather_data = {}

    for col in weather_columns:

        if col in df.columns:

            value = row.get(col)

            if pd.notna(value):

                weather_data[col] = value

    if weather_data:

        weather_cols = st.columns(
            len(weather_data)
        )

        for i, (name, value) in enumerate(
            weather_data.items()
        ):

            display_name = {
                "RAINFALL_MM": "Rainfall",
                "MAX_TEMP_C": "Max Temperature",
                "MIN_TEMP_C": "Min Temperature",
                "MEAN_TEMP_C": "Mean Temperature",
                "HUMIDITY_PERCENT": "Humidity"
            }.get(name, name)

            unit = {
                "RAINFALL_MM": " mm",
                "MAX_TEMP_C": " °C",
                "MIN_TEMP_C": " °C",
                "MEAN_TEMP_C": " °C",
                "HUMIDITY_PERCENT": "%"
            }.get(name, "")

            weather_cols[i].metric(
                display_name,
                f"{value:.2f}{unit}"
            )

    else:

        st.info(
            "Weather information is not available "
            "for this date."
        )

    st.divider()

    # --------------------------------------------------------
    # WEATHER EVENT STATUS
    # --------------------------------------------------------

    st.subheader("🌧️ Weather Event Status")

    event_flags = [
        "FLAG_HEAVY_RAIN",
        "FLAG_HOT_DAY",
        "FLAG_HUMID_DAY",
        "FLAG_COMBINED_STRESS"
    ]

    available_flags = [
        col
        for col in event_flags
        if col in df.columns
    ]

    if available_flags:

        flag_cols = st.columns(
            len(available_flags)
        )

        for i, flag in enumerate(
            available_flags
        ):

            value = pd.to_numeric(
                row.get(flag),
                errors="coerce"
            )

            event_name = (
                flag
                .replace("FLAG_", "")
                .replace("_", " ")
                .title()
            )

            if pd.notna(value) and value == 1:

                flag_cols[i].success(
                    f"✓ {event_name}"
                )

            else:

                flag_cols[i].info(
                    f"— {event_name}"
                )

    st.divider()

    # --------------------------------------------------------
    # NEARBY PRICE HISTORY
    # --------------------------------------------------------

    st.subheader("📅 Nearby Price History")

    history = df[
        (
            df["DATE"]
            >= selected_datetime
            - pd.Timedelta(days=3)
        )
        &
        (
            df["DATE"]
            <= selected_datetime
            + pd.Timedelta(days=3)
        )
    ].copy()

    if not history.empty:

        history_columns = [
            col
            for col in [
                "DATE",
                "PRICE_INR"
            ]
            if col in history.columns
        ]

        history_display = history[
            history_columns
        ].copy()

        history_display["DATE"] = (
            history_display["DATE"]
            .dt.strftime("%Y-%m-%d")
        )

        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 2 — WEATHER EVENTS
# ============================================================

elif page == "🌧️ Weather Events":

    st.header("🌧️ Weather Events")

    st.write(
        "Explore the weather events identified during "
        "Phase 4 and used for weather–price signature analysis."
    )

    st.divider()

    # --------------------------------------------------------
    # EVENT COUNT VALIDATION
    # --------------------------------------------------------

    st.subheader("🔍 Event Count Validation")

    event_counts = (
        events_df["EVENT_TYPE"]
        .value_counts()
    )

    event_count_rows = []

    event_display_names = {
        "PROLONGED_RAIN": "Prolonged Rain",
        "EXTREME_HUMIDITY": "Extreme Humidity",
        "EXTREME_HEAT": "Extreme Heat",
        "HEAVY_RAIN": "Heavy Rain"
    }

    for event_type, count in event_counts.items():

        event_count_rows.append(
            {
                "Weather Event":
                    event_display_names.get(
                        event_type,
                        event_type.replace("_", " ").title()
                    ),
                "Number of Events":
                    int(count)
            }
        )

    event_count_table = pd.DataFrame(
        event_count_rows
    )

    st.dataframe(
        event_count_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Counts are taken directly from the validated "
        "weather_events_database.csv (with Heavy Rain defined as rainfall > 25 mm)."
    )

    st.divider()

    # --------------------------------------------------------
    # SELECT EVENT
    # --------------------------------------------------------

    event_types = sorted(
        events_df["EVENT_TYPE"]
        .dropna()
        .unique()
    )

    selected_event = st.selectbox(
        "🌦️ Select Weather Event",
        event_types,
        format_func=lambda x:
            event_display_names.get(
                x,
                x.replace("_", " ").title()
            )
    )

    selected_event_name = event_display_names.get(
        selected_event,
        selected_event.replace("_", " ").title()
    )

    # --------------------------------------------------------
    # FILTER EVENTS
    # --------------------------------------------------------

    selected_events = events_df[
        events_df["EVENT_TYPE"]
        == selected_event
    ].copy()

    event_count = len(
        selected_events
    )

    # --------------------------------------------------------
    # MERGE EVENT DATA WITH JASMINE PRICE DATA
    # --------------------------------------------------------

    if "DATE" in df.columns:

        price_columns = [
            col
            for col in [
                "DATE",
                "PRICE_INR",
                "RAINFALL_MM",
                "MAX_TEMP_C",
                "MIN_TEMP_C",
                "MEAN_TEMP_C",
                "HUMIDITY_PERCENT"
            ]
            if col in df.columns
        ]

        price_weather = df[
            price_columns
        ].copy()

        event_analysis = selected_events.merge(
            price_weather,
            on="DATE",
            how="left"
        )

    else:

        event_analysis = selected_events.copy()

    # --------------------------------------------------------
    # EVENT SUMMARY
    # --------------------------------------------------------

    st.subheader(
        f"🌦️ {selected_event_name}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Number of Events",
        f"{event_count}"
    )

    if "PRICE_INR" in event_analysis.columns:

        event_prices = pd.to_numeric(
            event_analysis["PRICE_INR"],
            errors="coerce"
        ).dropna()

    else:

        event_prices = pd.Series(
            dtype="float64"
        )

    if not event_prices.empty:

        c2.metric(
            "Average Price",
            f"₹{event_prices.mean():.2f}"
        )

        c3.metric(
            "Minimum Price",
            f"₹{event_prices.min():.2f}"
        )

        c4.metric(
            "Maximum Price",
            f"₹{event_prices.max():.2f}"
        )

    else:

        c2.metric(
            "Average Price",
            "N/A"
        )

        c3.metric(
            "Minimum Price",
            "N/A"
        )

        c4.metric(
            "Maximum Price",
            "N/A"
        )

    st.divider()

    # --------------------------------------------------------
    # EVENT VALUE
    # --------------------------------------------------------

    st.subheader(
        f"📌 {selected_event_name} Event Values"
    )

    if "VALUE" in selected_events.columns:

        value_data = selected_events[
            [
                "DATE",
                "EVENT_TYPE",
                "VALUE"
            ]
        ].copy()

        value_data["DATE"] = (
            value_data["DATE"]
            .dt.strftime("%Y-%m-%d")
        )

        value_data["EVENT_TYPE"] = (
            value_data["EVENT_TYPE"]
            .map(
                lambda x:
                event_display_names.get(
                    x,
                    x.replace("_", " ").title()
                )
            )
        )

        st.dataframe(
            value_data,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # EVENT + PRICE TABLE
    # --------------------------------------------------------

    st.subheader(
        f"📋 {selected_event_name} — Price Association"
    )

    display_columns = [
        col
        for col in [
            "DATE",
            "EVENT_TYPE",
            "VALUE",
            "PRICE_INR",
            "RAINFALL_MM",
            "MAX_TEMP_C",
            "MIN_TEMP_C",
            "MEAN_TEMP_C",
            "HUMIDITY_PERCENT"
        ]
        if col in event_analysis.columns
    ]

    event_display = event_analysis[
        display_columns
    ].copy()

    if "DATE" in event_display.columns:

        event_display["DATE"] = (
            event_display["DATE"]
            .dt.strftime("%Y-%m-%d")
        )

    if "EVENT_TYPE" in event_display.columns:

        event_display["EVENT_TYPE"] = (
            event_display["EVENT_TYPE"]
            .map(
                lambda x:
                event_display_names.get(
                    x,
                    x.replace("_", " ").title()
                )
            )
        )

    st.dataframe(
        event_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3 — PHASE 7 PREDICTION
# ============================================================

elif page == "🤖 Phase 7 Prediction":

    st.header("🤖 Phase 7 Prediction")

    st.write(
        "Predictions generated by the Phase 7 Weather-Aware "
        "XGBoost model on the unseen test set."
    )

    st.divider()

    prediction_dates = (
        prediction_df["DATE"]
        .dropna()
        .sort_values()
        .dt.date
        .unique()
    )

    if len(prediction_dates) == 0:

        st.warning(
            "No prediction dates found."
        )

        st.stop()

    selected_prediction_date = st.selectbox(
        "📅 Select Test Date",
        prediction_dates
    )

    selected_prediction_datetime = pd.Timestamp(
        selected_prediction_date
    )

    selected_prediction = prediction_df[
        prediction_df["DATE"].dt.normalize()
        == selected_prediction_datetime
    ].copy()

    if selected_prediction.empty:

        st.warning(
            "No prediction available for this date."
        )

        st.stop()

    prediction_row = selected_prediction.iloc[0]

    actual_price = pd.to_numeric(
        prediction_row.get("ACTUAL_PRICE"),
        errors="coerce"
    )

    predicted_price = pd.to_numeric(
        prediction_row.get("PREDICTED_PRICE"),
        errors="coerce"
    )

    absolute_error = pd.to_numeric(
        prediction_row.get("ABS_ERROR"),
        errors="coerce"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Actual Jasmine Price",
        f"₹{actual_price:.2f}"
        if pd.notna(actual_price)
        else "N/A"
    )

    c2.metric(
        "Predicted Jasmine Price",
        f"₹{predicted_price:.2f}"
        if pd.notna(predicted_price)
        else "N/A"
    )

    c3.metric(
        "Absolute Error",
        f"₹{absolute_error:.2f}"
        if pd.notna(absolute_error)
        else "N/A"
    )

    st.divider()

    st.subheader(
        "📖 What does this prediction mean?"
    )

    st.info(
        """
        The model was trained using earlier historical data
        and then tested on later dates that were not used
        during training.

        **Actual Price** = observed Jasmine market price.

        **Predicted Price** = price estimated by the Phase 7 model.

        **Absolute Error** = absolute difference between
        actual and predicted price.
        """
    )

    st.divider()

    st.subheader(
        "📋 Test-Set Predictions"
    )

    prediction_display = prediction_df.copy()

    prediction_display["DATE"] = (
        prediction_display["DATE"]
        .dt.strftime("%Y-%m-%d")
    )

    st.dataframe(
        prediction_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 4 — MODEL EVALUATION
# ============================================================

elif page == "📈 Model Evaluation":

    st.header("📈 Model Evaluation")

    st.write(
        "Comparison of the Phase 7 models using the unseen test data."
    )

    st.divider()

    # --------------------------------------------------------
    # EXACT PHASE 7 RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "Model": "Model A",
                "Type": "Simple Baseline",
                "Algorithm": "Linear Regression",
                "Features": "Yesterday Price",
                "MAE": 107.48,
                "RMSE": 141.55,
                "R²": 0.6232
            },
            {
                "Model": "Model B",
                "Type": "Price-Only ML",
                "Algorithm": "Random Forest",
                "Features": "Recent Price Features",
                "MAE": 113.53,
                "RMSE": 150.36,
                "R²": 0.5748
            },
            {
                "Model": "Model B",
                "Type": "Price-Only ML",
                "Algorithm": "XGBoost",
                "Features": "Recent Price Features",
                "MAE": 111.17,
                "RMSE": 145.56,
                "R²": 0.6016
            },
            {
                "Model": "Model C",
                "Type": "Weather-Aware ML",
                "Algorithm": "Random Forest",
                "Features": "Price + Weather Features",
                "MAE": 116.10,
                "RMSE": 149.10,
                "R²": 0.5819
            },
            {
                "Model": "Model C",
                "Type": "Weather-Aware ML",
                "Algorithm": "XGBoost",
                "Features": "Price + Weather Features",
                "MAE": 114.67,
                "RMSE": 148.39,
                "R²": 0.5859
            }
        ]
    )

    st.subheader(
        "📊 Phase 7 Model Comparison"
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # METRIC EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "📌 Evaluation Metrics"
    )

    metric_cols = st.columns(3)

    with metric_cols[0]:

        st.markdown("### MAE")

        st.write(
            "Mean Absolute Error represents the average "
            "absolute prediction error in rupees."
        )

        st.caption(
            "Lower MAE generally indicates smaller average errors."
        )

    with metric_cols[1]:

        st.markdown("### RMSE")

        st.write(
            "Root Mean Squared Error gives greater weight "
            "to larger prediction errors."
        )

        st.caption(
            "Lower RMSE generally indicates fewer large errors."
        )

    with metric_cols[2]:

        st.markdown("### R²")

        st.write(
            "R² indicates how much of the variation in Jasmine "
            "price is explained by the model."
        )

        st.caption(
            "Higher R² generally indicates more explained variation."
        )

    st.divider()

    # --------------------------------------------------------
    # WEATHER FEATURE EFFECT
    # --------------------------------------------------------

    st.subheader(
        "🌦️ Effect of Adding Weather Features"
    )

    effect_data = pd.DataFrame(
        [
            {
                "Algorithm": "Random Forest",
                "Price-Only MAE": 113.53,
                "Weather-Aware MAE": 116.10,
                "MAE Change": "+2.26%"
            },
            {
                "Algorithm": "XGBoost",
                "Price-Only MAE": 111.17,
                "Weather-Aware MAE": 114.67,
                "MAE Change": "+3.15%"
            }
        ]
    )

    st.dataframe(
        effect_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        In this experiment, adding the tested weather features
        increased MAE for both Random Forest and XGBoost.

        Therefore, the tested weather features did not improve
        overall prediction accuracy compared with the corresponding
        price-only models.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # KEY FINDING
    # --------------------------------------------------------

    st.subheader(
        "🔍 Key Finding"
    )

    st.success(
        """
        **For this dataset and the tested weather features,
        recent price information was more useful for prediction
        than the added weather features.**

        This does not mean that weather has no relationship
        with Jasmine prices. It means that the weather features
        tested in this Phase 7 experiment did not provide
        additional predictive improvement.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT STATUS
    # --------------------------------------------------------

    st.subheader(
        "✅ Project Status"
    )

    st.markdown(
        """
        **Completed through Phase 7**

        1. Jasmine price and weather data
        2. Data cleaning and validation
        3. Master Jasmine–weather dataset
        4. Weather feature engineering
        5. Weather event detection
        6. Weather–price signature analysis
        7. Machine learning experiment and evaluation

        **Future Work**

        - SHAP explainability
        - Similar historical case matching
        - Interactive intelligence application
        """
    )

    st.caption(
        "Current implementation and results are limited to Phase 1–7."
    )


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "🌼 Jasmine Weather–Price Analysis"
)

st.sidebar.caption(
    "Phase 1–7 Project Demo"
)

