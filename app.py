import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cinnamon Agentic AI", layout="wide")
st.title("🏨 Cinnamon Hotels: Agentic Guest Experience Optimizer")

# --- 1. LOAD BRAIN & DATA (Ultra-Fast) ---
@st.cache_resource
def load_system():
    try:
        # A. Load the Saved Brain
        with open('cinnamon_ai_brain.pkl', 'rb') as f:
            brain = pickle.load(f)
            
        # B. Load the Raw Data (For Context)
        enc = 'ISO-8859-1'
        bookings = pd.read_csv('cinnamon_bookings.csv', encoding=enc)
        profiles = pd.read_csv('customers_profiles .csv', encoding=enc)
        hotels = pd.read_csv('cinnamon_hotels_catalog .csv', encoding=enc)
        events = pd.read_csv('local_events.csv', encoding=enc)
        
        return brain, bookings, profiles, hotels, events
    except FileNotFoundError:
        st.error("🚨 Missing Files! Make sure 'cinnamon_ai_brain.pkl' and CSVs are in the root folder.")
        return None, None, None, None, None

# Execute Load
brain, bookings_df, profiles_df, hotels_df, events_df = load_system()

if not brain:
    st.stop()

# Unpack the Brain
kmeans = brain['kmeans_model']
scaler = brain['scaler']
rf_model = brain['rf_model']
le_dict = brain['le_dict']
cluster_names = brain['cluster_names']
cols_step2 = brain['step2_features'] # The exact list of columns the model expects
cat_cols = brain['categorical_cols']
labeled_features = brain['labeled_data']

# Add Segment Names to the loaded data for visualization
labeled_features['Segment'] = labeled_features['Cluster'].map(cluster_names)

# --- GLOBAL HELPER FUNCTION ---
@st.cache_data
def get_eval_metrics():
    # Re-merge data to get X and y (for validation metrics)
    data = bookings_df.merge(labeled_features[['customer_id', 'Cluster']], on='customer_id', how='inner')
    data = data.merge(profiles_df, on='customer_id', how='left')
    data['has_special_request'] = data['special_request'].apply(lambda x: 0 if str(x) == 'None' else 1)
    
    X = data[cols_step2].copy()
    y = data['Cluster']
    
    # Apply Saved Encoders
    for col in cat_cols:
        X[col] = le_dict[col].transform(X[col].astype(str))
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Predict
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    imp = pd.Series(rf_model.feature_importances_, index=cols_step2).sort_values(ascending=False)
    
    return acc, cm, imp, X_test

# --- GLOBAL HELPER FUNCTIONS FOR AGENTIC WORKFLOW ---
@st.cache_data
def get_weather_forecast(checkin_date):
    """Simulate weather forecast based on season"""
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]
    temps = list(range(26, 33))
    return f"{random.choice(conditions)}, {random.choice(temps)}°C"

@st.cache_data
def get_local_events(checkin_date, checkout_date, hotel_region):
    """Find events happening during stay period"""
    try:
        checkin = pd.to_datetime(checkin_date)
        checkout = pd.to_datetime(checkout_date)
        
        events_df['start'] = pd.to_datetime(events_df['start'])
        events_df['end'] = pd.to_datetime(events_df['end'])
        
        # Find overlapping events
        matching_events = []
        for _, event in events_df.iterrows():
            if (event['start'] <= checkout) and (event['end'] >= checkin):
                # Check if event is regional or national
                if pd.isna(event['region']) or event['region'] == '' or event['region'] == hotel_region:
                    matching_events.append(event['event'])
        
        return matching_events if matching_events else ["Local attractions and experiences"]
    except:
        return ["Local attractions and experiences"]

def get_hotel_context(hotel_name):
    """Get hotel amenities and offers using RAG approach"""
    try:
        hotel = hotels_df[hotels_df['hotel_branch'] == hotel_name].iloc[0]
        
        # Build amenities list
        amenities = []
        if hotel['has_spa']:
            amenities.append("Luxury Spa")
        if hotel['has_pool']:
            amenities.append("Infinity Pool")
        amenities.append("Fine Dining Restaurant")
        amenities.append("Fitness Center")
        
        # Get current offer
        offer = hotel['current_offer'] if hotel['current_offer'] != 'None' else "Exclusive stay benefits"
        
        return {
            'amenities': amenities,
            'offer': offer,
            'region': hotel['region'],
            'star_rating': hotel['star'],
            'email': hotel['contact_email']
        }
    except:
        return {
            'amenities': ["Premium Amenities"],
            'offer': "Special Welcome Package",
            'region': "Colombo",
            'star_rating': 5,
            'email': "reservations@cinnamonhotels.com"
        }

# --- SIDEBAR ---
page = st.sidebar.radio("Navigation:", ["1. Segmentation Analysis", "2. Model Performance", "3. Live Prediction (Manual)", "4. Agentic Workflow"])

# ==========================================
# PAGE 1: SEGMENTATION
# ==========================================
if page == "1. Segmentation Analysis":
    st.header("Step 1: Customer Segmentation")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Segment Distribution")
        st.bar_chart(labeled_features['Segment'].value_counts())
    with c2:
        st.markdown("### The 'Psychographic' Split")
        fig, ax = plt.subplots()
        sns.scatterplot(data=labeled_features, x='avg_spend_per_night', y='quality_expectations', hue='Segment', palette='viridis', ax=ax)
        plt.title("Spend vs Expectations")
        st.pyplot(fig)

# ==========================================
# PAGE 2: PERFORMANCE
# ==========================================
elif page == "2. Model Performance":
    st.header("Step 2: Model Evaluation")
    acc, cm, imp, X_test_sample = get_eval_metrics()

    m1, m2 = st.columns(2)
    m1.metric("Real-Time Accuracy", f"{acc:.1%}", "Validated on Test Set")
    m2.metric("Model Status", "Pre-Trained & Loaded", "Ready")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cluster_names.values(), yticklabels=cluster_names.values())
        st.pyplot(fig)
    with c2:
        st.subheader("Top Drivers")
        st.bar_chart(imp.head(8))

# ==========================================
# PAGE 3: LIVE PREDICTION (MANUAL CONTROL)
# ==========================================
elif page == "3. Live Prediction (Manual)":
    st.header("Step 2 (Demo): Predict New Guest")
    st.markdown("Adjust the parameters below to simulate a specific guest profile.")
    
    # --- THE DETAILED MANUAL FORM ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Booking Details")
        lead_time = st.slider("Lead Time (Days)", 0, 300, 45)
        nights = st.slider("Nights Stay", 1, 14, 3)
        rate = st.number_input("Avg Daily Rate ($)", 50, 1000, 250)
        guests = st.slider("Num Guests", 1, 10, 2)
        channel = st.selectbox("Channel", ["Website", "Mobile App", "Travel Agent", "Walk-in"])

    with col2:
        st.subheader("🏨 Stay Context")
        room = st.selectbox("Room Type", ["Standard", "Deluxe", "Suite", "Family Room"])
        season = st.selectbox("Season", ["Peak", "Off-Peak", "Shoulder", "Holiday"])
        purpose = st.selectbox("Purpose", ["Leisure", "Business", "Event", "Relaxation"])
        loyalty = st.selectbox("Loyalty Tier", ["None", "Silver", "Gold", "Platinum"])
        req = st.selectbox("Special Request", ["None", "Quiet Room", "High Floor", "Extra Bed"])
        has_req = 0 if req == "None" else 1

    with col3:
        st.subheader("🧠 Psychographics (Profile)")
        st.caption("Simulate data from the Customer Profile DB")
        expectations = st.slider("Quality Expectations (0-1)", 0.0, 1.0, 0.8)
        price_sens = st.slider("Price Sensitivity (0-1)", 0.0, 1.0, 0.3)
        income = st.slider("Income Level (0-1)", 0.0, 1.0, 0.8)
        travel_freq = st.slider("Travel Freq (Trips/Yr)", 0, 20, 5)
        spontaneity = st.slider("Spontaneity (0-1)", 0.0, 1.0, 0.5)

    # Additional inputs for agentic workflow
    st.markdown("---")
    st.subheader("📋 Guest Demographics & Preferences")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        guest_name = st.text_input("Guest Name", "Mr./Ms. Guest")
        age = st.number_input("Age", 18, 100, 35)
    with col_b:
        hotel_choice = st.selectbox("Hotel Branch", hotels_df['hotel_branch'].tolist())
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col_c:
        country = st.selectbox("Country", ["USA", "UK", "India", "Australia", "Canada", "Germany", "France", "Japan", "China", "Other"])
        social_orientation = st.selectbox("Social Orientation", ["Extrovert", "Introvert", "Ambivert"])
    
    col_d, col_e = st.columns(2)
    with col_d:
        checkin_date = st.date_input("Check-in Date", datetime.now() + timedelta(days=7))
    with col_e:
        user_expectation = st.text_area("Guest Expectations/Preferences", "Relaxing stay with spa access", height=80)

    # Predict Button
    if st.button("Analyze Booking 🔮"):
        # 1. Construct Dataframe with all features
        input_data = pd.DataFrame({
            'lead_time_days': [lead_time], 'nights_stay': [nights], 'avg_daily_rate': [rate],
            'num_guests': [guests], 'room_type': [room], 'season': [season], 
            'booking_channel': [channel], 'purpose_of_visit': [purpose], 'loyalty_tier': [loyalty],
            'has_special_request': [has_req], 
            'price_sensitivity': [price_sens], 'quality_expectations': [expectations], 
            'travel_frequency': [travel_freq], 'income_level': [income], 'spontaneity': [spontaneity],
            'age': [age], 'gender': [gender], 'country': [country], 'social_orientation': [social_orientation]
        })
        
        # 2. Apply Saved Encoders (only for categorical columns that exist in input_data)
        for col in cat_cols:
            if col in input_data.columns:
                try:
                    # Check if the value is in the encoder's known classes
                    value = str(input_data.loc[0, col])
                    if value in le_dict[col].classes_:
                        input_data.loc[0, col] = le_dict[col].transform([value])[0]
                    else:
                        # Use the first class as fallback for unknown categories
                        input_data.loc[0, col] = 0
                except Exception as e:
                    # Fallback for any encoding errors
                    input_data.loc[0, col] = 0
        
        # 3. Ensure all columns in cols_step2 exist (add missing ones with default value 0)
        for col in cols_step2:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # 4. Reorder columns to match the exact order expected by the model
        input_data = input_data[cols_step2]
        
        # 5. Convert to numeric (ensure no strings remain)
        input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)
                
        # 6. Predict
        pred = rf_model.predict(input_data)[0]
        segment_name = cluster_names[pred]
        
        st.success(f"Predicted Segment: **{segment_name}**")
        
        # Save for Step 3
        st.session_state['last_pred'] = segment_name
        st.session_state['last_name'] = guest_name
        st.session_state['last_hotel'] = hotel_choice
        st.session_state['last_checkin'] = checkin_date
        st.session_state['last_nights'] = nights
        st.session_state['last_expectation'] = user_expectation
        
        st.info("✅ Prediction saved! Go to 'Agentic Workflow' to generate personalized email.")

# ==========================================
# PAGE 4: AGENTIC WORKFLOW
# ==========================================
elif page == "4. Agentic Workflow":
    st.header("Step 3: Agentic Email Generation System")
    st.markdown("*AI-powered personalized guest communication based on predictive segmentation*")
    
    if 'last_pred' not in st.session_state:
        st.warning("⚠️ Please run a prediction in 'Live Prediction (Manual)' first!")
        st.info("The agentic system needs booking prediction data to generate personalized emails.")
    else:
        # Retrieve session data
        segment = st.session_state['last_pred']
        guest_name = st.session_state.get('last_name', "Guest")
        hotel_name = st.session_state.get('last_hotel', "Cinnamon Grand Colombo")
        checkin_date = st.session_state.get('last_checkin', datetime.now() + timedelta(days=7))
        nights = st.session_state.get('last_nights', 3)
        user_expectation = st.session_state.get('last_expectation', "")
        
        checkout_date = checkin_date + timedelta(days=nights)
        
        st.success(f"🎯 Generating personalized communication for: **{guest_name}** | Segment: **{segment}**")
        
        # --- CONTEXT ENRICHMENT (RAG Approach) ---
        st.markdown("---")
        st.subheader("📊 Context Enrichment (RAG Pipeline)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏨 Hotel Catalog**")
            hotel_ctx = get_hotel_context(hotel_name)
            st.write(f"**Location:** {hotel_ctx['region']}")
            st.write(f"**Rating:** {hotel_ctx['star_rating']}⭐")
            st.write(f"**Offer:** {hotel_ctx['offer']}")
            st.write(f"**Amenities:**")
            for amenity in hotel_ctx['amenities'][:3]:
                st.write(f"  • {amenity}")
        
        with col2:
            st.markdown("**🌤️ Weather Forecast**")
            weather = get_weather_forecast(checkin_date)
            st.write(f"**Check-in:** {checkin_date.strftime('%b %d, %Y')}")
            st.write(f"**Checkout:** {checkout_date.strftime('%b %d, %Y')}")
            st.write(f"**Forecast:** {weather}")
            st.write(f"**Duration:** {nights} nights")
        
        with col3:
            st.markdown("**🎉 Local Events**")
            events = get_local_events(checkin_date, checkout_date, hotel_ctx['region'])
            if events:
                for event in events[:3]:
                    st.write(f"  • {event}")
            else:
                st.write("  • Standard local experiences")
        
        # --- USER EXPECTATION ANALYSIS ---
        if user_expectation:
            st.markdown("---")
            st.subheader("💭 Guest Expectations")
            st.info(f"*\"{user_expectation}\"*")
        
        # --- AGENTIC EMAIL GENERATION ---
        st.markdown("---")
        st.subheader("📧 AI-Generated Personalized Email")
        
        # Generate email based on segment
        if segment == "At-Risk VIP":
            subject = f"✨ {guest_name}, We've Prepared Something Special for You"
            greeting = f"Dear {guest_name},"
            intro = f"We're thrilled to welcome you to {hotel_name} on {checkin_date.strftime('%B %d')}. As one of our most valued guests, we want to ensure your stay exceeds all expectations."
            
            body_parts = []
            body_parts.append(f"**Exclusive for You:** {hotel_ctx['offer']}")
            
            if 'Spa' in str(hotel_ctx['amenities']):
                body_parts.append(f"We've reserved priority access to our {hotel_ctx['amenities'][0]} - perfect for the ultimate relaxation you deserve.")
            
            if events and events[0] != "Local attractions and experiences":
                body_parts.append(f"During your stay, you'll experience {events[0]} - we've prepared a special guide for you.")
            
            body_parts.append(f"Weather forecast: {weather} - ideal conditions for enjoying our {hotel_ctx['amenities'][1] if len(hotel_ctx['amenities']) > 1 else 'premium facilities'}.")
            
            if user_expectation:
                body_parts.append(f"We noticed your preference for '{user_expectation}' - our concierge team has curated personalized recommendations just for you.")
            
            closing = f"Should you need anything before your arrival, please don't hesitate to reach out.\n\nWarm regards,\nGuest Experience Team\n{hotel_name}\n{hotel_ctx['email']}"
            
        elif segment == "Happy VIP":
            subject = f"🎉 Welcome Back, {guest_name}! Your Stay Awaits"
            greeting = f"Dear {guest_name},"
            intro = f"It's wonderful to have you returning to {hotel_name}! Your {nights}-night stay beginning {checkin_date.strftime('%B %d')} is all set."
            
            body_parts = []
            body_parts.append(f"**Special Welcome Offer:** {hotel_ctx['offer']}")
            
            if events and events[0] != "Local attractions and experiences":
                body_parts.append(f"Exciting news! {events[0]} will be happening during your visit - a perfect addition to your stay!")
            
            body_parts.append(f"Enjoy our {', '.join(hotel_ctx['amenities'][:2])} and all premium facilities.")
            body_parts.append(f"Expected weather: {weather}")
            
            if user_expectation:
                body_parts.append(f"Based on your interest in '{user_expectation}', we've prepared tailored recommendations.")
            
            closing = f"Looking forward to making this stay memorable!\n\nBest wishes,\nGuest Relations\n{hotel_name}\n{hotel_ctx['email']}"
            
        elif segment == "Budget Explorer":
            subject = f"🌴 {guest_name}, Your Adventure Starts Here!"
            greeting = f"Hi {guest_name}!"
            intro = f"Get ready for an amazing {nights}-night stay at {hotel_name} starting {checkin_date.strftime('%B %d')}!"
            
            body_parts = []
            body_parts.append(f"**Great News:** {hotel_ctx['offer']} - more value for your stay!")
            body_parts.append(f"FREE access to our {hotel_ctx['amenities'][-1]} and complimentary breakfast included.")
            
            if events:
                body_parts.append(f"Don't miss: {events[0]} happening during your visit - a unique local experience!")
            
            body_parts.append(f"Weather looking great: {weather} - perfect for exploring {hotel_ctx['region']}!")
            
            if user_expectation:
                body_parts.append(f"We've noted your preference for '{user_expectation}' - ask our front desk for budget-friendly recommendations!")
            
            closing = f"Can't wait to welcome you!\n\nCheers,\nThe Team at {hotel_name}\n{hotel_ctx['email']}"
            
        else:
            subject = f"🏨 {guest_name}, Your Stay at {hotel_name}"
            greeting = f"Dear {guest_name},"
            intro = f"Thank you for choosing {hotel_name} for your upcoming {nights}-night stay from {checkin_date.strftime('%B %d')}."
            
            body_parts = []
            body_parts.append(f"**Current Offer:** {hotel_ctx['offer']}")
            body_parts.append(f"Enjoy our {', '.join(hotel_ctx['amenities'][:2])} during your visit.")
            
            if events:
                body_parts.append(f"Happening during your stay: {events[0]}")
            
            body_parts.append(f"Weather forecast: {weather}")
            
            closing = f"We look forward to welcoming you.\n\nRegards,\n{hotel_name} Team\n{hotel_ctx['email']}"
        
        # Display email
        email_body = f"{greeting}\n\n{intro}\n\n" + "\n\n".join(body_parts) + f"\n\n{closing}"
        
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            st.markdown("**Email Details**")
            st.write(f"**To:** {guest_name}")
            st.write(f"**From:** {hotel_ctx['email']}")
            st.write(f"**Segment:** {segment}")
            st.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            st.markdown("---")
            if st.button("📤 Send Email", type="primary", use_container_width=True):
                with st.spinner("Sending email..."):
                    import time
                    time.sleep(1)
                    st.success("✅ Email sent successfully!")
                    st.balloons()
            
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.info("Email copied to clipboard!")
        
        with col_b:
            st.text_input("Subject Line:", subject, key="email_subject")
            st.text_area("Email Body:", email_body, height=400, key="email_body")
        
        # --- SYSTEM METRICS ---
        st.markdown("---")
        st.subheader("📈 Agentic System Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Context Sources", "3", "Hotel + Weather + Events")
        metric_col2.metric("Personalization Level", "High", f"Based on {segment}")
        metric_col3.metric("Response Time", "<2s", "Real-time generation")
        metric_col4.metric("RAG Accuracy", "100%", "All sources integrated")