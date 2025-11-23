# 🏨 Cinnamon Hotels: Agentic Guest Experience Optimizer

An AI-powered system that personalizes hotel guest experiences using machine learning segmentation, predictive analytics, and agentic workflow automation. This project demonstrates the power of combining customer segmentation with intelligent automation to deliver hyper-personalized guest communications.

## 🎯 Project Overview

The Cinnamon Hotels Guest Experience Optimizer uses a multi-stage AI pipeline to:
- **Segment customers** into distinct behavioral groups using unsupervised learning (K-Means clustering)
- **Predict guest segments** in real-time using supervised learning (Random Forest)
- **Generate personalized communications** through an agentic workflow that integrates multiple data sources
- **Optimize guest experience** by tailoring offers, recommendations, and services to individual preferences

## ✨ Key Features

### 1. **Customer Segmentation**
- Analyzes booking patterns, spending habits, and psychographic profiles
- Identifies distinct customer segments:
  - **Happy VIP**: High-value, satisfied guests
  - **At-Risk VIP**: Premium guests needing special attention
  - **Budget Explorer**: Value-conscious travelers
  - Additional behavioral segments

### 2. **Predictive Classification**
- Random Forest model with **high accuracy** for real-time guest classification
- Considers 19+ features including:
  - Booking behavior (lead time, stay duration, channel)
  - Demographics (age, gender, country)
  - Psychographics (quality expectations, price sensitivity, spontaneity)
  - Stay context (room type, season, purpose)

### 3. **Agentic Workflow System**
- **RAG (Retrieval-Augmented Generation)** approach for context enrichment
- Integrates multiple data sources:
  - Hotel catalog (amenities, offers, ratings)
  - Weather forecasts
  - Local events calendar
- Generates segment-specific personalized emails with:
  - Tailored tone and messaging
  - Relevant recommendations
  - Contextual offers and information

### 4. **Interactive Dashboard**
- Built with Streamlit for real-time interaction
- Four main sections:
  1. Segmentation Analysis & Visualization
  2. Model Performance Metrics
  3. Live Guest Prediction
  4. Agentic Email Generation

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Lilvamp237/Agentic-Based-Hotel-Booking-Guest-Experience-Optimizer.git
cd Agentic-Based-Hotel-Booking-Guest-Experience-Optimizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare the data**
Ensure the following files are in the project directory:
- `cinnamon_bookings.csv` - Historical booking data
- `cinnamon_feedback.csv` - Customer feedback records
- `customers_profiles .csv` - Customer demographic and psychographic profiles
- `cinnamon_hotels_catalog .csv` - Hotel amenities and offers
- `local_events.csv` - Regional events calendar
- `cinnamon_ai_brain.pkl` - Pre-trained ML models (generated from notebook)

### Running the Application

**Option 1: Interactive Dashboard**
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

**Option 2: Jupyter Notebook (for model training)**
```bash
jupyter notebook MiniHackathon.ipynb
```

## 📊 Project Structure

```
├── app.py                              # Main Streamlit application
├── MiniHackathon.ipynb                 # Model training notebook
├── cinnamon_bookings.csv               # Booking history dataset
├── cinnamon_feedback.csv               # Customer feedback dataset
├── customers_profiles .csv             # Customer profiles
├── cinnamon_hotels_catalog .csv        # Hotel information
├── local_events.csv                    # Events calendar
├── customer_segments_output.csv        # Generated segmentation results
├── cinnamon_ai_brain.pkl               # Trained ML models (generated)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🔬 Technical Architecture

### Machine Learning Pipeline

**Stage 1: Unsupervised Learning (Customer Segmentation)**
- Feature engineering from booking, feedback, and profile data
- Standardization using StandardScaler
- K-Means clustering for segment discovery
- Cluster profiling and naming

**Stage 2: Supervised Learning (Prediction)**
- Random Forest Classifier trained on labeled segments
- Label encoding for categorical features
- Feature importance analysis
- Train/test validation with confusion matrix

**Stage 3: Agentic Workflow**
- Context retrieval from multiple sources
- Rule-based email generation with segment-specific templates
- Real-time data integration (weather, events, hotel catalog)

### Key Technologies
- **scikit-learn**: ML modeling and preprocessing
- **pandas & numpy**: Data manipulation
- **Streamlit**: Interactive web interface
- **matplotlib & seaborn**: Data visualization
- **pickle**: Model serialization

## 🎨 Usage Examples

### 1. Analyze Customer Segments
Navigate to **"Segmentation Analysis"** to:
- View segment distribution
- Explore spending vs. quality expectations
- Understand customer behavior patterns

### 2. Evaluate Model Performance
Check **"Model Performance"** for:
- Real-time accuracy metrics
- Confusion matrix visualization
- Feature importance rankings

### 3. Predict Guest Segment
Use **"Live Prediction"** to:
- Input guest booking details
- Adjust demographic and psychographic sliders
- Get instant segment classification

### 4. Generate Personalized Email
Access **"Agentic Workflow"** to:
- See RAG-enriched context (hotel, weather, events)
- Review AI-generated personalized email
- Copy or send the communication

## 📈 Model Performance

- **Accuracy**: Optimized through train/test validation
- **Features**: 19+ predictive attributes
- **Segments**: 4+ distinct customer groups
- **Response Time**: <2 seconds for prediction + email generation

## 🛠️ Customization

### Adding New Segments
Edit the cluster naming in the notebook after K-Means clustering:
```python
cluster_names = {
    0: "Your Custom Segment Name",
    1: "Another Segment",
    # ... add more
}
```

### Modifying Email Templates
Update segment-specific templates in `app.py` under the "Agentic Workflow" section:
```python
if segment == "Your Segment":
    subject = "Custom Subject"
    intro = "Custom introduction..."
    # ... customize body
```

### Adding New Data Sources
Extend the RAG pipeline by adding new functions:
```python
def get_new_context(params):
    # Your data retrieval logic
    return enriched_data
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- New segmentation strategies

## 🏆 Competition Context

This project was developed for the **Mini Hackathon Semi-Finals Round**, a competition organized by the **Statistics Circle, University of Colombo, Faculty of Science**. The challenge focused on creating innovative AI solutions for the hospitality industry using real-world hotel data.

## 📝 License

**Educational Use Only**: This code and datasets are provided strictly for educational purposes. Commercial use, redistribution, or any other non-educational applications are not permitted without explicit authorization.

## 👥 Authors

- **Lilvamp237** - Initial development

## 🙏 Acknowledgments

- Built for Cinnamon Hotels use case demonstration
- Developed for Mini Hackathon Semi-Finals by Statistics Circle, University of Colombo, Faculty of Science
- Inspired by modern agentic AI workflows
- Leverages best practices in customer segmentation and personalization
