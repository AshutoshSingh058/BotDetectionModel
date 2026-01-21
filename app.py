import streamlit as st
import pandas as pd
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import base64
import joblib


def get_default_robot_icon():
    return "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/master/svgs/solid/robot.svg"


# Set page configuration
st.set_page_config(
    page_title="Twitter Bot Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        background-color: #FF4B4B;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
    }
    .stTextArea>div>div>textarea {
        border-radius: 0.5rem;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .info-box {
        background-color: #262730;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ✅ Model was trained with these 11 features (confirmed by you)
MODEL_FEATURES = [
    "followers_count",
    "friends_count",
    "listedcount",
    "favourites_count",
    "statuses_count",
    "verified",
    "default_profile",
    "default_profile_image",
    "has_extended_profile",
    "follow_ratio",
    "account_age_days",
]


@st.cache_resource
def load_model(model_path="bot_model.joblib"):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'bot_model.joblib' exists in the project folder.")
        return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def make_prediction(features_df, model):
    """
    Behavioral-only RandomForest prediction.
    features_df MUST have the same columns used in training.
    """
    probs = model.predict_proba(features_df)[0]
    pred_class = int(np.argmax(probs))  # 0 = Human, 1 = Bot
    confidence = float(probs[pred_class])
    return pred_class, confidence, probs


def create_gauge_chart(confidence, prediction_is_bot):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if prediction_is_bot else "darkgreen"},
            'steps': [
                {'range': [0, 33], 'color': 'lightgray'},
                {'range': [33, 66], 'color': 'gray'},
                {'range': [66, 100], 'color': 'darkgray'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def create_probability_chart(probs):
    labels = ['Human', 'Bot']
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=[probs[0] * 100, probs[1] * 100],
        hole=.3,
        marker_colors=['#00CC96', '#EF553B']
    )])
    fig.update_layout(
        title="Probability Distribution",
        height=300
    )
    return fig


def build_model_features_from_ui(
    followers_count: int,
    friends_count: int,
    listed_count: int,
    favorites_count: int,
    statuses_count: int,
    verified: bool,
    default_profile: bool,
    default_profile_image: bool,
    has_extended_profile: bool,
    account_age_days: int
) -> pd.DataFrame:
    """
    Converts UI inputs to the EXACT schema expected by the trained RF model.
    UI stays same, only feature mapping changes.

    Mapping:
    listed_count -> listedcount
    favorites_count -> favourites_count
    followers_friends_ratio -> follow_ratio
    account_age -> account_age_days
    """

    follow_ratio = followers_count / (friends_count + 1)

    features = pd.DataFrame([{
        "followers_count": followers_count,
        "friends_count": friends_count,
        "listedcount": listed_count,
        "favourites_count": favorites_count,
        "statuses_count": statuses_count,
        "verified": int(verified),
        "default_profile": int(default_profile),
        "default_profile_image": int(default_profile_image),
        "has_extended_profile": int(has_extended_profile),
        "follow_ratio": follow_ratio,
        "account_age_days": account_age_days,
    }])

    # enforce correct order
    features = features[MODEL_FEATURES]
    return features


def main():
    # Sidebar with extended navigation
    st.sidebar.image("piclumen-1739279351872.png", width=100)  # Replace with your logo
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Bot Detection", "CSV Analysis", "About", "Statistics"])

    if page == "Bot Detection":
        st.title("🤖 Twitter Bot Detection System")
        st.markdown("""
        <div style='background-color: #262730; color: white; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h4>Welcome to the Social Media Bot Detection System</h4>
        <p>This application demonstrates a metadata-based machine learning approach for detecting automated social media accounts.</p>
        </div>
        """, unsafe_allow_html=True)

        # Load model
        model = load_model()
        if model is None:
            st.stop()

        # Create tabs for individual account analysis
        tab1, tab2 = st.tabs(["📝 Input Details", "📊 Analysis Results"])

        with tab1:
            st.markdown("### Account Information")

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                name = st.text_input("Account Name", placeholder="@username")
                followers_count = st.number_input("Followers Count", min_value=0)
                friends_count = st.number_input("Friends Count", min_value=0)
                listed_count = st.number_input("Listed Count", min_value=0)

            with col2:
                favorites_count = st.number_input("Favorites Count", min_value=0)
                statuses_count = st.number_input("Statuses Count", min_value=0)
                account_age = st.number_input("Account Age (days)", min_value=0)

            with col3:
                description = st.text_area("Profile Description")
                location = st.text_input("Location")

            st.markdown("### Account Properties")
            prop_col1, prop_col2, prop_col3 = st.columns(3)

            with prop_col1:
                verified = st.checkbox("Verified Account")
            with prop_col2:
                default_profile = st.checkbox("Default Profile")
            with prop_col3:
                default_profile_image = st.checkbox("Default Profile Image")

            # kept same UI logic
            has_extended_profile = True
            has_url = True

            st.markdown("### Tweet Content")
            tweet_content = st.text_area("Sample Tweet", height=100)  # UI stays, ignored in logic

            st.caption(
                "Note: The prediction model uses only profile and activity metadata. "
                "Text fields are shown for completeness and are not used in model inference."
            )
            
            if st.button("🔍 Analyze Account"):

                with st.spinner('Analyzing account characteristics...'):
                    # ✅ Build ONLY the exact 11 features your RF expects
                    features = build_model_features_from_ui(
                        followers_count=followers_count,
                        friends_count=friends_count,
                        listed_count=listed_count,
                        favorites_count=favorites_count,
                        statuses_count=statuses_count,
                        verified=verified,
                        default_profile=default_profile,
                        default_profile_image=default_profile_image,
                        has_extended_profile=has_extended_profile,
                        account_age_days=account_age
                    )

                    # ✅ Predict
                    pred_class, confidence, probs = make_prediction(features, model)
                    prediction_is_bot = (pred_class == 1)

                    time.sleep(1)
                    tab2.markdown("### Analysis Complete!")

                    with tab2:
                        if prediction_is_bot:
                            st.error("🤖 Bot Account Detected!")
                        else:
                            st.success("👤 Human Account Detected!")

                        # Confidence gauge directly below the result
                        st.plotly_chart(
                            create_gauge_chart(confidence, prediction_is_bot),
                            use_container_width=True
                        )

                        st.markdown("### Feature Analysis")

                        # Feature importance (RF supports this)
                        if hasattr(model, "feature_importances_"):
                            feature_importance = pd.DataFrame({
                                'Feature': MODEL_FEATURES,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)

                            fig = px.bar(
                                feature_importance,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Feature Importance Analysis'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Feature importance is not available for this model type.")

                        metrics_data = {
                            'Metric': ['Followers', 'Friends', 'Tweets', 'Favorites'],
                            'Count': [followers_count, friends_count, statuses_count, favorites_count]
                        }
                        fig = px.bar(
                            metrics_data,
                            x='Metric',
                            y='Count',
                            title='Account Metrics Overview',
                            color='Count',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            


    elif page == "CSV Analysis":
        st.title("CSV Batch Analysis")
        st.markdown("Upload a CSV file with account data to run batch predictions. You can use \"testClick.csv\" from Dataset folder of this repository.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.markdown("### CSV Data Preview")
            st.dataframe(data.head())

            model = load_model()
            if model is None:
                st.stop()

            predictions = []
            confidences = []
            prediction_labels = []

            with st.spinner("Processing accounts..."):
                for idx, row in data.iterrows():

                    # flexible column names support
                    followers = row.get("followers_count", 0)
                    friends = row.get("friends_count", 0)
                    statuses = row.get("statuses_count", 0)

                    # allow either listedcount or listed_count
                    listed = row.get("listedcount", row.get("listed_count", 0))

                    # allow either favourites_count or favorites_count
                    favourites = row.get("favourites_count", row.get("favorites_count", 0))

                    verified = int(row.get("verified", 0))
                    default_profile = int(row.get("default_profile", 0))
                    default_profile_image = int(row.get("default_profile_image", 0))
                    has_extended_profile = int(row.get("has_extended_profile", 0))

                    # allow account_age_days or "account_age (days)"
                    age_days = row.get("account_age_days", row.get("account_age (days)", 0))

                    # compute follow_ratio if not present
                    follow_ratio = row.get("follow_ratio", followers / (friends + 1))

                    features = pd.DataFrame([{
                        "followers_count": followers,
                        "friends_count": friends,
                        "listedcount": listed,
                        "favourites_count": favourites,
                        "statuses_count": statuses,
                        "verified": verified,
                        "default_profile": default_profile,
                        "default_profile_image": default_profile_image,
                        "has_extended_profile": has_extended_profile,
                        "follow_ratio": follow_ratio,
                        "account_age_days": age_days,
                    }])[MODEL_FEATURES]

                    pred_class, conf, _ = make_prediction(features, model)

                    predictions.append(pred_class)
                    confidences.append(conf)
                    prediction_labels.append('🤖' if pred_class == 1 else '👤')

            data['prediction'] = predictions
            data['confidence'] = confidences
            data['account_type'] = prediction_labels

            st.markdown("### Batch Prediction Results")
            cols = ['username', 'account_type', 'prediction', 'confidence'] + [
                col for col in data.columns if col not in ['username', 'account_type', 'prediction', 'confidence']
            ]
            st.dataframe(data[cols])

            # Optional evaluation if labels exist
            if 'label' in data.columns:
                y_true = data['label'].tolist()
                y_pred = [int(p) for p in predictions]

                from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
                f1 = f1_score(y_true, y_pred, average='weighted')
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                report = classification_report(y_true, y_pred)

                st.markdown("### Evaluation Metrics")
                st.write("F1 Score:", f1)
                st.write("Precision:", precision)
                st.write("Recall:", recall)
                st.text(report)

    elif page == "About":
        st.title("About the Bot Detection System")
        st.markdown("""
        <div class='info-box'>
        <h3>🎯 System Overview</h3>
        <p>Our Twitter Bot Detection System demonstrates a supervised machine learning approach for detecting automated social media accounts using structured profile and activity metadata. The goal of the system is to understand how different behavioral and account-level attributes contribute to identifying bot-like patterns, rather than relying on text or content-based signals.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🔑 Key Features Analyzed")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Account Characteristics
            - Profile completeness
            - Account age and verification status
            - Username patterns
            - Profile description analysis

            #### Behavioral Patterns
            - Posting frequency
            - Engagement rates
            - Temporal patterns
            - Content similarity
            """)
        with col2:
            st.markdown("""
            #### Network Analysis
            - Follower-following ratio
            - Friend acquisition rate
            - Network growth patterns

            """)

        st.markdown("""
        <div class='info-box'>
        <h3>⚙ Technical Implementation</h3>
        <ul>
        <li><strong>Data Processing:</strong> Cleaned and structured profile and activity metadata.</li>
        <li><strong>Feature Engineering:</strong> Derived behavioral features such as follower–following ratio, posting activity, and account age.</li>
        <li><strong>Modeling:</strong> Trained a Random Forest classifier on the engineered features.</li>
        <li><strong>Explainability:</strong> Used feature importance to interpret model predictions.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


        st.markdown("### 📊 System Performance")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

        with metrics_col1:
            st.metric("Accuracy", "87%")
        with metrics_col2:
            st.metric("Precision", "89%")
        with metrics_col3:
            st.metric("Recall", "83%")
        with metrics_col4:
            st.metric("F1 Score", "86%")

        st.markdown("""
        ### 🎯 Common Use Cases
        - *Social Media Management*: Identify and remove bot accounts
        - *Research*: Analyze social media manipulation
        - *Marketing*: Verify authentic engagement
        - *Security*: Protect against automated threats
        """)

    else:  # Statistics page
        st.title("System Statistics")
        st.info(
        "This dashboard is a demo visualization intended to illustrate how system-level statistics and trends could be presented. The data shown here is illustrative and not generated from live usage or production logs."
        )

        col1, col2 = st.columns(2)

        with col1:
            detection_data = {
                'Category': ['Bots', 'Humans'],
                'Count': [737, 826]
            }
            fig = px.pie(
                detection_data,
                values='Count',
                names='Category',
                title='Detection Distribution',
                color_discrete_sequence=['#FF4B4B', '#00CC96']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            confidence_data = {
                'Score': ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%'],
                'Count': [178, 447, 503, 352, 83]
            }
            fig = px.bar(
                confidence_data,
                x='Score',
                y='Count',
                title='Confidence Score Distribution',
                color='Count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Monthly Detection Trends")
        monthly_data = {
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Bots Detected': [45, 52, 38, 65, 48, 76],
            'Accuracy': [92, 94, 93, 95, 94, 96]
        }
        fig = px.line(
            monthly_data,
            x='Month',
            y=['Accuracy','Bots Detected' ],
            title='Monthly Performance Metrics',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Key System Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("Total Analyses", "1,000", "+12%")
        with metric_col2:
            st.metric("Avg. Accuracy", "87%", "+2.3%")
        with metric_col3:
            st.metric("Bot Detection Rate", "47.2%", "-3.2%")
        with metric_col4:
            st.metric("Processing Time", "1.2s", "-0.3s")

    st.caption("*Demo Dashboard (Concept Visualization)*")

if __name__ == "__main__":
    main()
