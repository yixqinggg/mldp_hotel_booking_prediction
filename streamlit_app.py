import joblib
import pandas as pd
import streamlit as st
from datetime import date


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)


# =========================================================
# 2. LOAD MODEL AND SUPPORTING FILES
# =========================================================
@st.cache_resource
def load_model_files():
    model = joblib.load(
        "hotel_cancellation_model.pkl"
    )

    encoded_columns = joblib.load(
        "encoded_columns.pkl"
    )

    selected_features = joblib.load(
        "selected_features.pkl"
    )

    return model, encoded_columns, selected_features


@st.cache_data
def load_dropdown_options():
    categorical_columns = [
        "hotel",
        "meal",
        "country",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "assigned_room_type",
        "deposit_type",
        "customer_type"
    ]

    dataset = pd.read_csv(
        "hotel_booking.csv",
        usecols=categorical_columns
    )

    # Apply the same missing-value treatment used in the notebook
    dataset["country"] = dataset["country"].fillna(
        dataset["country"].mode()[0]
    )

    options = {}

    for column in categorical_columns:
        options[column] = sorted(
            dataset[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    return options


try:
    model, encoded_columns, selected_features = (
        load_model_files()
    )

    dropdown_options = load_dropdown_options()

except FileNotFoundError as error:
    st.error(
        "A required project file could not be found. "
        "Ensure that the dataset, model and feature files "
        "are stored in the same folder as streamlit_app.py."
    )

    st.code(str(error))
    st.stop()

except Exception as error:
    st.error(
        "The application files could not be loaded."
    )

    with st.expander("Technical details"):
        st.code(str(error))

    st.stop()


# =========================================================
# 3. DROPDOWN OPTIONS
# =========================================================
hotel_options = dropdown_options["hotel"]
meal_options = dropdown_options["meal"]
country_options = dropdown_options["country"]

market_segment_options = dropdown_options[
    "market_segment"
]

distribution_channel_options = dropdown_options[
    "distribution_channel"
]

reserved_room_options = dropdown_options[
    "reserved_room_type"
]

assigned_room_options = dropdown_options[
    "assigned_room_type"
]

deposit_type_options = dropdown_options[
    "deposit_type"
]

customer_type_options = dropdown_options[
    "customer_type"
]


# =========================================================
# 4. COUNTRY DISPLAY NAMES
# =========================================================
country_names = {
    "PRT": "Portugal",
    "GBR": "United Kingdom",
    "FRA": "France",
    "ESP": "Spain",
    "DEU": "Germany",
    "ITA": "Italy",
    "IRL": "Ireland",
    "BEL": "Belgium",
    "BRA": "Brazil",
    "NLD": "Netherlands",
    "USA": "United States",
    "CHE": "Switzerland",
    "AUT": "Austria",
    "CHN": "China",
    "AUS": "Australia",
    "SWE": "Sweden",
    "POL": "Poland",
    "NOR": "Norway",
    "FIN": "Finland",
    "DNK": "Denmark",
    "CAN": "Canada",
    "JPN": "Japan",
    "SGP": "Singapore",
    "MYS": "Malaysia",
    "IND": "India"
}


def display_country(country_code):
    country_name = country_names.get(
        country_code,
        country_code
    )

    return f"{country_name} ({country_code})"


# =========================================================
# 5. WEBPAGE STYLING
# =========================================================
st.markdown(
    """
    <style>

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    html, body, [class*="css"] {
        font-size: 18px;
    }

    h1 {
        font-size: 2.7rem !important;
    }

    h2 {
        font-size: 2.1rem !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-size: 1.55rem !important;
    }

    label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #f7dfad,
            #fbeacb
        );
        border: 3px solid #d29a59;
        border-radius: 22px;
        padding: 46px 30px;
        text-align: center;
        box-shadow: 0 7px 20px rgba(92, 61, 32, 0.12);
        margin-bottom: 34px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        color: #3d2f24;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        font-size: 20px;
        color: #655143;
        line-height: 1.5;
    }

    .section-note {
        font-size: 17px;
        color: #6b5b4f;
        margin-bottom: 20px;
    }

    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    div[data-testid="stNumberInput"] input {
        font-size: 17px;
        min-height: 48px;
    }

    div[data-testid="stDateInput"] input {
        font-size: 17px;
        min-height: 48px;
    }

    div[data-baseweb="select"] > div {
        font-size: 17px;
        min-height: 48px;
        border-radius: 10px;
    }

    /* Prediction button */
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 64px;
        border-radius: 15px;
        border: 2px solid #315f67;
        background-color: #3f7882;
        color: white;
        font-size: 21px;
        font-weight: 750;
        letter-spacing: 0.2px;
        box-shadow: 0 5px 12px rgba(35, 76, 85, 0.20);
        transition: 0.15s ease;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #315f69;
        border-color: #244b53;
        color: white;
        transform: translateY(-1px);
    }

    div[data-testid="stFormSubmitButton"] button:active {
        transform: translateY(0);
    }

    /* Neutral result cards */
    div[data-testid="stMetric"] {
        background-color: #f8fafb;
        border: 2px solid #9baeb3;
        border-radius: 15px;
        padding: 22px 25px;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 18px;
        font-weight: 650;
        margin-bottom: 6px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 34px;
        line-height: 1.2;
    }

    /* Keep native Streamlit status colours */
    div[data-testid="stAlert"] {
        border-radius: 15px;
        padding: 22px 24px;
    }

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] li {
        font-size: 18px;
        line-height: 1.65;
    }

    div[data-testid="stAlert"] h3 {
        font-size: 23px !important;
        margin-top: 0 !important;
        margin-bottom: 12px !important;
    }

    div[data-testid="stProgress"] > div > div {
        height: 14px;
        border-radius: 8px;
    }

    @media (max-width: 750px) {

        .hero-title {
            font-size: 34px;
        }

        .hero-subtitle {
            font-size: 17px;
        }

        .hero {
            padding: 34px 18px;
        }

        div[data-testid="stFormSubmitButton"] button {
            font-size: 19px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 6. HEADER
# =========================================================
st.markdown(
    (
        '<div class="hero">'
        '<div class="hero-title">'
        '🏨 Hotel Cancellation Predictor'
        '</div>'
        '<div class="hero-subtitle">'
        'Predict booking cancellation risk and receive practical '
        'recommendations for hotel planning.'
        '</div>'
        '</div>'
    ),
    unsafe_allow_html=True
)


# =========================================================
# 7. INPUT FORM
# =========================================================
st.header("Prediction")


with st.form(
    "prediction_form",
    clear_on_submit=False
):

    # -----------------------------------------------------
    # Main booking details
    # -----------------------------------------------------
    st.subheader("Main Booking Details")

    st.markdown(
        (
            '<div class="section-note">'
            "Enter the guest's stay and reservation information."
            '</div>'
        ),
        unsafe_allow_html=True
    )

    main_col1, main_col2, main_col3 = st.columns(
        3,
        gap="large"
    )


    with main_col1:

        hotel_selected = st.selectbox(
            "Hotel Type",
            hotel_options,
            help="Type of hotel included in the booking."
        )

        lead_time_selected = st.number_input(
            "Lead Time (Days)",
            min_value=0,
            max_value=737,
            value=30,
            help=(
                "Number of days between the booking date "
                "and the arrival date."
            )
        )

        arrival_date_selected = st.date_input(
            "Arrival Date",
            value=date(2017, 7, 15),
            min_value=date(2015, 1, 1),
            max_value=date(2017, 12, 31),
            help=(
                "The training dataset contains arrivals "
                "from 2015 to 2017."
            )
        )

        adults_selected = st.number_input(
            "Number of Adults",
            min_value=0,
            max_value=55,
            value=2
        )


    with main_col2:

        weekend_nights_selected = st.number_input(
            "Weekend Nights",
            min_value=0,
            max_value=19,
            value=1
        )

        weekday_nights_selected = st.number_input(
            "Weekday Nights",
            min_value=0,
            max_value=50,
            value=2
        )

        children_selected = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0
        )

        adr_selected = st.number_input(
            "Average Daily Rate",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
            help="Average room price per occupied night."
        )


    with main_col3:

        market_segment_selected = st.selectbox(
            "Market Segment",
            market_segment_options,
            help=(
                "The market group through which "
                "the booking was made."
            )
        )

        deposit_type_selected = st.selectbox(
            "Deposit Type",
            deposit_type_options
        )

        customer_type_selected = st.selectbox(
            "Customer Type",
            customer_type_options
        )

        meal_selected = st.selectbox(
            "Meal Package",
            meal_options,
            help=(
                "BB: Bed and breakfast, HB: Half board, "
                "FB: Full board, SC: Self-catering."
            )
        )


    st.divider()


    # -----------------------------------------------------
    # Booking history
    # -----------------------------------------------------
    st.subheader("Booking History")

    st.markdown(
        (
            '<div class="section-note">'
            "Provide information about the customer's previous "
            "bookings and reservation changes."
            '</div>'
        ),
        unsafe_allow_html=True
    )

    history_col1, history_col2, history_col3 = st.columns(
        3,
        gap="large"
    )


    with history_col1:

        repeated_guest_selected = st.selectbox(
            "Repeated Guest",
            [
                "No",
                "Yes"
            ]
        )

        previous_cancellations_selected = st.number_input(
            "Previous Cancellations",
            min_value=0,
            max_value=30,
            value=0
        )


    with history_col2:

        previous_bookings_selected = st.number_input(
            "Previous Bookings Not Cancelled",
            min_value=0,
            max_value=75,
            value=0
        )

        booking_changes_selected = st.number_input(
            "Number of Booking Changes",
            min_value=0,
            max_value=21,
            value=0
        )


    with history_col3:

        waiting_days_selected = st.number_input(
            "Days in Waiting List",
            min_value=0,
            max_value=400,
            value=0
        )


    st.divider()


    # -----------------------------------------------------
    # Additional booking details
    # -----------------------------------------------------
    st.subheader("Additional Booking Details")

    st.markdown(
        (
            '<div class="section-note">'
            "Complete the remaining reservation information "
            "used by the prediction model."
            '</div>'
        ),
        unsafe_allow_html=True
    )

    additional_col1, additional_col2, additional_col3 = (
        st.columns(
            3,
            gap="large"
        )
    )


    with additional_col1:

        country_selected = st.selectbox(
            "Customer Country",
            country_options,
            format_func=display_country
        )

        distribution_channel_selected = st.selectbox(
            "Distribution Channel",
            distribution_channel_options
        )


    with additional_col2:

        reserved_room_selected = st.selectbox(
            "Reserved Room Type",
            reserved_room_options
        )

        assigned_room_selected = st.selectbox(
            "Assigned Room Type",
            assigned_room_options
        )


    with additional_col3:

        agent_selected = st.number_input(
            "Travel Agent ID",
            min_value=0,
            max_value=535,
            value=0,
            help=(
                "Enter 0 if the booking was not made through "
                "a registered travel agent."
            )
        )

        parking_spaces_selected = st.number_input(
            "Required Parking Spaces",
            min_value=0,
            max_value=8,
            value=0
        )

        special_requests_selected = st.number_input(
            "Number of Special Requests",
            min_value=0,
            max_value=5,
            value=0
        )


    # -----------------------------------------------------
    # Centred prediction button
    # -----------------------------------------------------
    button_col1, button_col2, button_col3 = st.columns(
        [1, 1.4, 1]
    )

    with button_col2:

        submitted = st.form_submit_button(
            "Predict Cancellation Risk",
            type="primary"
        )


# =========================================================
# 8. INPUT VALIDATION
# =========================================================
if submitted:

    total_guests = (
        adults_selected
        + children_selected
    )

    total_nights = (
        weekend_nights_selected
        + weekday_nights_selected
    )

    validation_errors = []


    if total_guests == 0:
        validation_errors.append(
            "The booking must include at least one guest."
        )

    if adults_selected == 0:
        validation_errors.append(
            "The booking must include at least one adult."
        )

    if total_nights == 0:
        validation_errors.append(
            "The booking must include at least one night."
        )

    if adr_selected <= 0:
        validation_errors.append(
            "The average daily rate must be greater than zero."
        )

    if (
        repeated_guest_selected == "No"
        and (
            previous_cancellations_selected > 0
            or previous_bookings_selected > 0
        )
    ):
        validation_errors.append(
            "A customer with previous booking records should "
            "be marked as a repeated guest."
        )


    if validation_errors:

        st.error(
            "Please correct the following information:"
        )

        for message in validation_errors:
            st.warning(message)


    else:

        try:
            # =================================================
            # 9. CREATE UNSEEN BOOKING DATA
            # =================================================
            arrival_year_selected = (
                arrival_date_selected.year
            )

            arrival_month_selected = (
                arrival_date_selected.strftime("%B")
            )

            arrival_week_selected = int(
                arrival_date_selected.isocalendar().week
            )

            arrival_day_selected = (
                arrival_date_selected.day
            )

            repeated_guest_value = (
                1
                if repeated_guest_selected == "Yes"
                else 0
            )


            df_input = pd.DataFrame({
                "hotel": [
                    hotel_selected
                ],
                "lead_time": [
                    lead_time_selected
                ],
                "arrival_date_year": [
                    arrival_year_selected
                ],
                "arrival_date_month": [
                    arrival_month_selected
                ],
                "arrival_date_week_number": [
                    arrival_week_selected
                ],
                "arrival_date_day_of_month": [
                    arrival_day_selected
                ],
                "stays_in_weekend_nights": [
                    weekend_nights_selected
                ],
                "stays_in_week_nights": [
                    weekday_nights_selected
                ],
                "adults": [
                    adults_selected
                ],
                "children": [
                    children_selected
                ],
                "meal": [
                    meal_selected
                ],
                "country": [
                    country_selected
                ],
                "market_segment": [
                    market_segment_selected
                ],
                "distribution_channel": [
                    distribution_channel_selected
                ],
                "is_repeated_guest": [
                    repeated_guest_value
                ],
                "previous_cancellations": [
                    previous_cancellations_selected
                ],
                "previous_bookings_not_canceled": [
                    previous_bookings_selected
                ],
                "reserved_room_type": [
                    reserved_room_selected
                ],
                "assigned_room_type": [
                    assigned_room_selected
                ],
                "booking_changes": [
                    booking_changes_selected
                ],
                "deposit_type": [
                    deposit_type_selected
                ],
                "agent": [
                    agent_selected
                ],
                "days_in_waiting_list": [
                    waiting_days_selected
                ],
                "customer_type": [
                    customer_type_selected
                ],
                "adr": [
                    adr_selected
                ],
                "required_car_parking_spaces": [
                    parking_spaces_selected
                ],
                "total_of_special_requests": [
                    special_requests_selected
                ]
            })


            # =================================================
            # 10. APPLY SAME PREPROCESSING AS NOTEBOOK
            # =================================================
            df_input_encoded = pd.get_dummies(
                df_input
            )

            df_input_encoded = df_input_encoded.reindex(
                columns=encoded_columns,
                fill_value=0
            )

            df_input_selected = df_input_encoded[
                selected_features
            ]


            # Confirm expected feature order
            if list(df_input_selected.columns) != list(
                selected_features
            ):
                raise ValueError(
                    "The application input columns do not match "
                    "the trained model features."
                )


            # =================================================
            # 11. GENERATE PREDICTION
            # =================================================
            prediction = model.predict(
                df_input_selected
            )[0]

            probabilities = model.predict_proba(
                df_input_selected
            )[0]

            model_classes = list(
                model.classes_
            )

            if 0 not in model_classes or 1 not in model_classes:
                raise ValueError(
                    "The model does not contain the expected "
                    "classification labels."
                )

            cancelled_index = model_classes.index(1)
            not_cancelled_index = model_classes.index(0)

            cancellation_probability = float(
                probabilities[cancelled_index]
            )

            not_cancelled_probability = float(
                probabilities[not_cancelled_index]
            )


            # Confirm probabilities are valid
            probability_total = (
                cancellation_probability
                + not_cancelled_probability
            )

            if abs(probability_total - 1.0) > 0.0001:
                raise ValueError(
                    "The model returned invalid probability values."
                )


            # =================================================
            # 12. DETERMINE RISK LEVEL
            # =================================================
            if cancellation_probability >= 0.70:
                risk_level = "High"

            elif cancellation_probability >= 0.40:
                risk_level = "Moderate"

            else:
                risk_level = "Low"


            # =================================================
            # 13. DISPLAY PREDICTION RESULT
            # =================================================
            st.divider()
            st.header("Prediction Result")


            if prediction == 1:

                st.error(
                    "### Likely to be Cancelled"
                )

                st.write(
                    "The model predicts that this booking has "
                    "a higher likelihood of cancellation."
                )

            else:

                st.success(
                    "### Unlikely to be Cancelled"
                )

                st.write(
                    "The model predicts that this booking has "
                    "a lower likelihood of cancellation."
                )


            result_col1, result_col2 = st.columns(
                2,
                gap="large"
            )


            with result_col1:

                st.metric(
                    "Cancellation Probability",
                    f"{cancellation_probability:.1%}"
                )


            with result_col2:

                st.metric(
                    "Cancellation Risk Level",
                    risk_level
                )


            # =================================================
            # 14. CANCELLATION RISK INDICATOR
            # =================================================
            st.subheader(
                "Cancellation Risk Indicator"
            )

            risk_percentage = int(
                round(
                    cancellation_probability
                    * 100
                )
            )

            st.progress(
                risk_percentage
            )

            st.caption(
                f"The model estimates a {risk_percentage}% "
                "probability that this booking will be cancelled."
            )


            # =================================================
            # 15. RECOMMENDED HOTEL ACTIONS
            # =================================================
            st.header(
                "Recommended Hotel Actions"
            )


            if cancellation_probability >= 0.70:

                st.warning(
                    """
                    ### High-Risk Booking Management

                    - Contact the guest to reconfirm the reservation.
                    - Send an additional reminder closer to arrival.
                    - Review the deposit and cancellation conditions.
                    - Monitor the reservation for further changes.
                    - Prepare a waitlist or room resale plan.
                    """
                )


            elif cancellation_probability >= 0.40:

                st.info(
                    """
                    ### Moderate-Risk Booking Monitoring

                    - Send a booking confirmation reminder.
                    - Monitor any changes made to the reservation.
                    - Follow up with the guest before arrival.
                    - Review the booking again closer to arrival.
                    """
                )


            else:

                st.success(
                    """
                    ### Standard Booking Preparation

                    - Continue with the normal confirmation process.
                    - Send the usual pre-arrival information.
                    - Prepare the assigned room for the guest.
                    - Continue monitoring the booking normally.
                    """
                )


            st.info(
                "The recommended actions are based on the model's "
                "estimated cancellation risk. Hotel staff should "
                "also use operational judgement and communicate "
                "with the guest."
            )


        except Exception as error:

            st.error(
                "The prediction could not be generated. "
                "Please review the booking information and try again."
            )

            with st.expander(
                "Technical details"
            ):
                st.code(str(error))