import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class ChurnAnalyzer:
    """
    Main class for customer churn analysis and prediction
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, uploaded_file):
        """Load and perform initial data inspection"""
        try:
            self.data = pd.read_csv(uploaded_file)
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.data is None:
            return False
            
        # Create a copy for processing
        self.processed_data = self.data.copy()
        
        # Handle missing values
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.processed_data.select_dtypes(include=['object']).columns
        
        # Fill missing values
        for col in numeric_columns:
            self.processed_data[col].fillna(self.processed_data[col].median(), inplace=True)
        
        for col in categorical_columns:
            self.processed_data[col].fillna(self.processed_data[col].mode()[0], inplace=True)
        
        # Convert TotalCharges to numeric if it exists (common issue in Telco dataset)
        if 'TotalCharges' in self.processed_data.columns:
            self.processed_data['TotalCharges'] = pd.to_numeric(
                self.processed_data['TotalCharges'], errors='coerce'
            )
            self.processed_data['TotalCharges'].fillna(
                self.processed_data['TotalCharges'].median(), inplace=True
            )
        
        # Encode categorical variables
        for col in categorical_columns:
            if col != 'Churn':  # Don't encode target variable yet
                le = LabelEncoder()
                self.processed_data[col] = le.fit_transform(self.processed_data[col])
                self.label_encoders[col] = le
        
        # Handle target variable
        if 'Churn' in self.processed_data.columns:
            self.processed_data['Churn'] = self.processed_data['Churn'].map({'Yes': 1, 'No': 0})
        
        # Remove outliers using IQR method
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'Churn']
        
        for col in numeric_columns:
            Q1 = self.processed_data[col].quantile(0.25)
            Q3 = self.processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.processed_data = self.processed_data[
                (self.processed_data[col] >= lower_bound) & 
                (self.processed_data[col] <= upper_bound)
            ]
        
        return True
    
    def perform_eda(self):
        """Perform exploratory data analysis"""
        if self.data is None:
            return None
            
        eda_results = {}
        
        # Basic statistics
        eda_results['basic_stats'] = self.data.describe()
        eda_results['missing_values'] = self.data.isnull().sum()
        eda_results['data_types'] = self.data.dtypes
        
        # Churn distribution
        if 'Churn' in self.data.columns:
            churn_counts = self.data['Churn'].value_counts()
            eda_results['churn_distribution'] = churn_counts
            eda_results['churn_rate'] = churn_counts.get('Yes', 0) / len(self.data)
        
        return eda_results
    
    def train_model(self, model_type='logistic_regression'):
        """Train machine learning model"""
        if self.processed_data is None:
            return False
            
        # Prepare features and target
        X = self.processed_data.drop(['Churn'], axis=1)
        y = self.processed_data['Churn']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train the model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42)
            self.model.fit(self.X_train_scaled, self.y_train)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42, max_depth=10)
            self.model.fit(self.X_train, self.y_train)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
        
        return True
    
    def evaluate_model(self, model_type='logistic_regression'):
        """Evaluate the trained model"""
        if self.model is None:
            return None
            
        # Make predictions
        if model_type == 'logistic_regression':
            y_pred = self.model.predict(self.X_test_scaled)
            y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        else:
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': classification_report(self.y_test, y_pred),
            'roc_curve': (fpr, tpr),
            'roc_auc': roc_auc,
            'y_pred_proba': y_pred_proba
        }

def create_visualizations(analyzer):
    """Create various visualizations for EDA"""
    
    # Churn distribution pie chart
    fig_churn = px.pie(
        values=analyzer.data['Churn'].value_counts().values,
        names=analyzer.data['Churn'].value_counts().index,
        title="Customer Churn Distribution",
        color_discrete_map={'No': '#2E86AB', 'Yes': '#A23B72'}
    )
    
    return fig_churn

def create_feature_analysis(analyzer):
    """Create feature analysis visualizations"""
    
    categorical_cols = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineBackup']
    categorical_cols = [col for col in categorical_cols if col in analyzer.data.columns]
    
    figures = {}
    
    for col in categorical_cols:
        # Create crosstab for churn analysis
        crosstab = pd.crosstab(analyzer.data[col], analyzer.data['Churn'])
        crosstab_pct = pd.crosstab(analyzer.data[col], analyzer.data['Churn'], normalize='index') * 100
        
        fig = px.bar(
            x=crosstab_pct.index,
            y=crosstab_pct['Yes'],
            title=f'Churn Rate by {col}',
            labels={'x': col, 'y': 'Churn Rate (%)'},
            color=crosstab_pct['Yes'],
            color_continuous_scale='Reds'
        )
        
        figures[col] = fig
    
    return figures

def create_correlation_heatmap(analyzer):
    """Create correlation heatmap"""
    if analyzer.processed_data is None:
        return None
        
    # Calculate correlation matrix
    corr_matrix = analyzer.processed_data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu'
    )
    
    return fig

def create_model_evaluation_plots(evaluation_results):
    """Create model evaluation visualizations"""
    
    # Confusion Matrix
    cm = evaluation_results['confusion_matrix']
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        x=['No Churn', 'Churn'],
        y=['No Churn', 'Churn']
    )
    
    # ROC Curve
    fpr, tpr = evaluation_results['roc_curve']
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {evaluation_results["roc_auc"]:.3f})',
        line=dict(color='#2E86AB', width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    fig_roc.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    
    return fig_cm, fig_roc

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Analysis & Prediction</h1>', 
                unsafe_allow_html=True)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ChurnAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📤 Data Upload", "🔍 EDA", "🤖 Model Training", "📈 Predictions", "📊 Dashboard"]
    )
    
    if page == "📤 Data Upload":
        st.header("Data Upload and Preprocessing")
        
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file containing customer data with a 'Churn' column"
        )
        
        if uploaded_file is not None:
            if analyzer.load_data(uploaded_file):
                st.success("✅ Data loaded successfully!")
                
                # Display basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Customers", len(analyzer.data))
                with col2:
                    st.metric("Features", len(analyzer.data.columns))
                with col3:
                    if 'Churn' in analyzer.data.columns:
                        churn_rate = (analyzer.data['Churn'] == 'Yes').mean()
                        st.metric("Churn Rate", f"{churn_rate:.2%}")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(analyzer.data.head())
                
                # Data cleaning
                if st.button("🧹 Clean Data"):
                    with st.spinner("Cleaning data..."):
                        if analyzer.clean_data():
                            st.success("✅ Data cleaned successfully!")
                            
                            # Show cleaned data info
                            st.subheader("Cleaned Data Info")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Rows After Cleaning", len(analyzer.processed_data))
                            with col2:
                                st.metric("Missing Values", analyzer.processed_data.isnull().sum().sum())
                        else:
                            st.error("❌ Error cleaning data")
    
    elif page == "🔍 EDA":
        st.header("Exploratory Data Analysis")
        
        if analyzer.data is None:
            st.warning("⚠️ Please upload data first!")
            return
        
        # Perform EDA
        eda_results = analyzer.perform_eda()
        
        if eda_results:
            # Basic statistics
            st.subheader("Basic Statistics")
            st.dataframe(eda_results['basic_stats'])
            
            # Churn distribution
            st.subheader("Churn Distribution")
            fig_churn = create_visualizations(analyzer)
            st.plotly_chart(fig_churn, use_container_width=True)
            
            # Feature analysis
            st.subheader("Feature Analysis")
            feature_figs = create_feature_analysis(analyzer)
            
            for feature, fig in feature_figs.items():
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            if analyzer.processed_data is not None:
                st.subheader("Feature Correlations")
                fig_corr = create_correlation_heatmap(analyzer)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
    
    elif page == "🤖 Model Training":
        st.header("Model Training & Evaluation")
        
        if analyzer.processed_data is None:
            st.warning("⚠️ Please upload and clean data first!")
            return
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["logistic_regression", "decision_tree", "random_forest"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                if analyzer.train_model(model_type):
                    st.success("✅ Model trained successfully!")
                    
                    # Evaluate model
                    evaluation_results = analyzer.evaluate_model(model_type)
                    
                    if evaluation_results:
                        # Display metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{evaluation_results['accuracy']:.3f}")
                        with col2:
                            st.metric("AUC Score", f"{evaluation_results['roc_auc']:.3f}")
                        
                        # Model evaluation plots
                        st.subheader("Model Evaluation")
                        fig_cm, fig_roc = create_model_evaluation_plots(evaluation_results)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_cm, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_roc, use_container_width=True)
                        
                        # Classification report
                        st.subheader("Classification Report")
                        st.text(evaluation_results['classification_report'])
                else:
                    st.error("❌ Error training model")
    
    elif page == "📈 Predictions":
        st.header("Churn Prediction")
        
        if analyzer.model is None:
            st.warning("⚠️ Please train a model first!")
            return
        
        st.subheader("Predict Individual Customer Churn")
        
        # Create input form based on original data columns
        if analyzer.data is not None:
            # Get categorical columns for input
            categorical_cols = analyzer.data.select_dtypes(include=['object']).columns
            categorical_cols = [col for col in categorical_cols if col != 'Churn']
            
            # Get numerical columns for input
            numerical_cols = analyzer.data.select_dtypes(include=[np.number]).columns
            
            # Create input form
            input_data = {}
            
            col1, col2 = st.columns(2)
            
            with col1:
                for col in categorical_cols[:len(categorical_cols)//2]:
                    unique_values = analyzer.data[col].unique()
                    input_data[col] = st.selectbox(f"Select {col}", unique_values)
            
            with col2:
                for col in categorical_cols[len(categorical_cols)//2:]:
                    unique_values = analyzer.data[col].unique()
                    input_data[col] = st.selectbox(f"Select {col}", unique_values)
            
            # Numerical inputs
            for col in numerical_cols:
                min_val = float(analyzer.data[col].min())
                max_val = float(analyzer.data[col].max())
                mean_val = float(analyzer.data[col].mean())
                input_data[col] = st.slider(f"{col}", min_val, max_val, mean_val)
            
            if st.button("🔮 Predict Churn"):
                # Process input data
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical variables
                for col in categorical_cols:
                    if col in analyzer.label_encoders:
                        try:
                            input_df[col] = analyzer.label_encoders[col].transform(input_df[col])
                        except ValueError:
                            st.error(f"Unknown value for {col}")
                            return
                
                # Make prediction
                try:
                    if hasattr(analyzer.model, 'predict_proba'):
                        if 'logistic_regression' in str(type(analyzer.model)):
                            input_scaled = analyzer.scaler.transform(input_df)
                            prediction = analyzer.model.predict(input_scaled)[0]
                            probability = analyzer.model.predict_proba(input_scaled)[0]
                        else:
                            prediction = analyzer.model.predict(input_df)[0]
                            probability = analyzer.model.predict_proba(input_df)[0]
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            churn_status = "Will Churn" if prediction == 1 else "Will Not Churn"
                            st.metric("Prediction", churn_status)
                        with col2:
                            churn_prob = probability[1]
                            st.metric("Churn Probability", f"{churn_prob:.2%}")
                        
                        # Probability gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = churn_prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Probability (%)"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    elif page == "📊 Dashboard":
        st.header("Customer Churn Dashboard")
        
        if analyzer.data is None:
            st.warning("⚠️ Please upload data first!")
            return
        
        # KPIs
        st.subheader("Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(analyzer.data)
            st.metric("Total Customers", total_customers)
        
        with col2:
            if 'Churn' in analyzer.data.columns:
                churned_customers = (analyzer.data['Churn'] == 'Yes').sum()
                st.metric("Churned Customers", churned_customers)
        
        with col3:
            if 'Churn' in analyzer.data.columns:
                churn_rate = (analyzer.data['Churn'] == 'Yes').mean()
                st.metric("Churn Rate", f"{churn_rate:.2%}")
        
        with col4:
            if 'Churn' in analyzer.data.columns:
                retention_rate = 1 - churn_rate
                st.metric("Retention Rate", f"{retention_rate:.2%}")
        
        # Filters
        st.subheader("Filters")
        
        col1, col2 = st.columns(2)
        
        # Add filters based on available columns
        filters = {}
        categorical_cols = analyzer.data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Churn']
        
        with col1:
            if 'Contract' in categorical_cols:
                filters['Contract'] = st.multiselect(
                    "Contract Type", 
                    analyzer.data['Contract'].unique(),
                    default=analyzer.data['Contract'].unique()
                )
        
        with col2:
            if 'InternetService' in categorical_cols:
                filters['InternetService'] = st.multiselect(
                    "Internet Service", 
                    analyzer.data['InternetService'].unique(),
                    default=analyzer.data['InternetService'].unique()
                )
        
        # Apply filters
        filtered_data = analyzer.data.copy()
        for filter_col, filter_values in filters.items():
            if filter_values:
                filtered_data = filtered_data[filtered_data[filter_col].isin(filter_values)]
        
        # Update KPIs based on filtered data
        if len(filtered_data) > 0:
            st.subheader("Filtered Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Filtered Customers", len(filtered_data))
            
            with col2:
                if 'Churn' in filtered_data.columns:
                    filtered_churn_rate = (filtered_data['Churn'] == 'Yes').mean()
                    st.metric("Filtered Churn Rate", f"{filtered_churn_rate:.2%}")
            
            # Visualizations for filtered data
            st.subheader("Visualizations")
            
            # Churn distribution for filtered data
            if 'Churn' in filtered_data.columns:
                fig_filtered_churn = px.pie(
                    values=filtered_data['Churn'].value_counts().values,
                    names=filtered_data['Churn'].value_counts().index,
                    title="Churn Distribution (Filtered)",
                    color_discrete_map={'No': '#2E86AB', 'Yes': '#A23B72'}
                )
                st.plotly_chart(fig_filtered_churn, use_container_width=True)
        
        # Show filtered data
        if st.checkbox("Show Filtered Data"):
            st.dataframe(filtered_data)

if __name__ == "__main__":
    main()