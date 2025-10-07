import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import io

# Set page configuration
st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .potable {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .not-potable {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class WaterQualityPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                             'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        
    def load_data(self):
        """Load and preprocess the water quality data"""
        try:
            df = pd.read_csv('potability.csv')
            
            # Handle missing values using median (same as notebook)
            for column in df.columns:
                if df[column].isnull().sum() > 0:
                    median = df[column].median()
                    df[column].fillna(median, inplace=True)
            
            return df
        except FileNotFoundError:
            st.error("❌ Dataset file 'potability.csv' not found. Please make sure it's in the same directory.")
            return None
    
    def train_model(self, df):
        """Train the Random Forest model"""
        X = df[self.feature_names]
        y = df['Potability']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest (same parameters as notebook)
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return X_train, X_test, y_train, y_test, y_pred, accuracy
    
    def predict_water_quality(self, input_features):
        """Predict water quality for given features"""
        if self.model is None:
            return None
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_features], columns=self.feature_names)
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0]
        
        return prediction, probability
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

def main():
    # Initialize predictor
    predictor = WaterQualityPredictor()
    
    # Header
    st.markdown('<h1 class="main-header">💧 Water Quality Prediction App</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["📊 Data Overview", "🤖 Model Training", "🔮 Make Prediction", "📈 Model Analysis"]
    )
    
    # Load data
    df = predictor.load_data()
    
    if df is None:
        return
    
    if app_mode == "📊 Data Overview":
        show_data_overview(df)
    
    elif app_mode == "🤖 Model Training":
        show_model_training(predictor, df)
    
    elif app_mode == "🔮 Make Prediction":
        show_prediction_interface(predictor, df)
    
    elif app_mode == "📈 Model Analysis":
        show_model_analysis(predictor, df)

def show_data_overview(df):
    st.header("📊 Dataset Overview")
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(df))
    
    with col2:
        st.metric("Features", len(df.columns) - 1)
    
    with col3:
        potable_count = df['Potability'].sum()
        st.metric("Potable Samples", f"{potable_count} ({potable_count/len(df)*100:.1f}%)")
    
    # Dataset preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Data visualization
    st.subheader("Data Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Potability distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        potability_counts = df['Potability'].value_counts()
        colors = ['lightcoral', 'lightgreen']
        plt.pie(potability_counts.values, labels=['Not Potable (0)', 'Potable (1)'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Water Potability Distribution')
        st.pyplot(fig)
    
    with col2:
        # Feature distributions
        selected_feature = st.selectbox("Select feature to visualize:", predictor.feature_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.hist(df[selected_feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel(selected_feature)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {selected_feature}')
        st.pyplot(fig)

def show_model_training(predictor, df):
    st.header("🤖 Model Training")
    
    if st.button("🚀 Train Random Forest Model"):
        with st.spinner("Training model... This may take a few seconds."):
            X_train, X_test, y_train, y_test, y_pred, accuracy = predictor.train_model(df)
            
            st.success(f"✅ Model trained successfully!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Accuracy", f"{accuracy:.3f}")
                st.metric("Training Samples", len(X_train))
                st.metric("Test Samples", len(X_test))
            
            with col2:
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Not Potable', 'Potable'],
                           yticklabels=['Not Potable', 'Potable'])
                plt.title('Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
            
            # Save model
            joblib.dump(predictor.model, 'water_quality_model.pkl')
            st.info("💾 Model saved as 'water_quality_model.pkl'")
            
            # Show classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

def show_prediction_interface(predictor, df):
    st.header("🔮 Predict Water Quality")
    
    # Load model if exists, otherwise train first
    try:
        predictor.model = joblib.load('water_quality_model.pkl')
        st.success("✅ Pre-trained model loaded successfully!")
    except:
        st.warning("⚠️ No pre-trained model found. Please train the model first in the 'Model Training' section.")
        if st.button("Train Model Now"):
            with st.spinner("Training model..."):
                predictor.train_model(df)
                joblib.dump(predictor.model, 'water_quality_model.pkl')
                st.success("✅ Model trained and loaded!")
    
    if predictor.model is None:
        return
    
    # Input form
    st.subheader("Enter Water Quality Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
        hardness = st.slider("Hardness", 0.0, 500.0, 196.0, 1.0)
        solids = st.slider("Solids", 0.0, 60000.0, 22000.0, 100.0)
    
    with col2:
        chloramines = st.slider("Chloramines", 0.0, 15.0, 7.0, 0.1)
        sulfate = st.slider("Sulfate", 0.0, 500.0, 333.0, 1.0)
        conductivity = st.slider("Conductivity", 0.0, 800.0, 426.0, 1.0)
    
    with col3:
        organic_carbon = st.slider("Organic Carbon", 0.0, 30.0, 14.0, 0.1)
        trihalomethanes = st.slider("Trihalomethanes", 0.0, 130.0, 66.0, 0.1)
        turbidity = st.slider("Turbidity", 0.0, 10.0, 4.0, 0.1)
    
    # Prediction button
    if st.button("🔍 Predict Water Quality", type="primary"):
        input_features = [ph, hardness, solids, chloramines, sulfate, 
                         conductivity, organic_carbon, trihalomethanes, turbidity]
        
        prediction, probability = predictor.predict_water_quality(input_features)
        
        # Display result
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.markdown('<div class="prediction-box potable">'
                       '<h3>✅ Potable Water</h3>'
                       '<p>This water is safe for drinking!</p>'
                       f'<p><strong>Confidence:</strong> {probability[1]:.2%}</p>'
                       '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box not-potable">'
                       '<h3>❌ Not Potable Water</h3>'
                       '<p>This water is not safe for drinking.</p>'
                       f'<p><strong>Confidence:</strong> {probability[0]:.2%}</p>'
                       '</div>', unsafe_allow_html=True)
        
        # Show probability breakdown
        st.subheader("Probability Breakdown")
        prob_df = pd.DataFrame({
            'Class': ['Not Potable', 'Potable'],
            'Probability': probability
        })
        st.dataframe(prob_df, use_container_width=True)

def show_model_analysis(predictor, df):
    st.header("📈 Model Analysis")
    
    # Load model if exists
    try:
        predictor.model = joblib.load('water_quality_model.pkl')
    except:
        st.warning("⚠️ Please train the model first in the 'Model Training' section.")
        return
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance_df = predictor.get_feature_importance()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importance in Random Forest Model')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        st.pyplot(fig)
    
    with col2:
        st.dataframe(importance_df, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    st.pyplot(fig)
    
    # Model comparison (placeholder - you can extend this)
    st.subheader("Model Performance")
    st.info("""
    The Random Forest model was chosen for this application due to its:
    - High accuracy in classification tasks
    - Robustness to outliers
    - Feature importance analysis capability
    - Good performance on tabular data
    """)

if __name__ == "__main__":
    main()