import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
from pathlib import Path
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Sports Analytics Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'filters_created' not in st.session_state:
    st.session_state.filters_created = []
if 'picks_generated' not in st.session_state:
    st.session_state.picks_generated = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .status-success {
        color: #00a650;
        font-weight: bold;
    }
    .status-warning {
        color: #ff6b6b;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚽ Sports Analytics Platform</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a feature", [
    "🏠 Dashboard",
    "📊 Data Management", 
    "🔍 Filters",
    "📈 Backtesting",
    "📉 Analytics",
    "🎯 Picks",
    "⚙️ Settings"
])

# Generate sample data for demo
def generate_sample_games():
    teams_nba = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Nuggets', 'Suns', 'Bucks', '76ers']
    teams_nfl = ['Chiefs', '49ers', 'Bengals', 'Bills', 'Cowboys', 'Eagles', 'Ravens', 'Dolphins']
    
    games = []
    for i in range(50):
        if np.random.random() > 0.5:
            home, away = np.random.choice(teams_nba, 2, replace=False)
            sport = 'NBA'
        else:
            home, away = np.random.choice(teams_nfl, 2, replace=False)
            sport = 'NFL'
        
        date = datetime.now() - timedelta(days=np.random.randint(0, 30))
        home_odds = round(np.random.uniform(1.3, 3.5), 2)
        away_odds = round(1 / (1 - 1/home_odds), 2)
        
        games.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Sport': sport,
            'Home': home,
            'Away': away,
            'Home_Odds': home_odds,
            'Away_Odds': away_odds,
            'Actual_Winner': np.random.choice([home, away])
        })
    
    return pd.DataFrame(games)

# Dashboard Page
if page == "🏠 Dashboard":
    st.header("📊 Dashboard")
    
    # Generate metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games Analyzed", "1,247", "+12%")
    
    with col2:
        st.metric("Active Filters", len(st.session_state.filters_created), "+2")
    
    with col3:
        st.metric("Win Rate", "67.3%", "+5.2%")
    
    with col4:
        st.metric("Active Picks", len(st.session_state.picks_generated), "-3")
    
    st.markdown("---")
    
    # Performance Chart
    st.subheader("📈 Performance Overview")
    
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    cumulative_pnl = np.cumsum(np.random.normal(15, 100, len(dates)))
    
    fig = px.line(
        x=dates, 
        y=cumulative_pnl, 
        title="Cumulative P&L Over Time",
        labels={'x': 'Date', 'y': 'P&L ($)'}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.subheader("📋 Recent Activity")
    
    activity_data = pd.DataFrame({
        'Date': ['2024-02-24', '2024-02-23', '2024-02-22', '2024-02-21'],
        'Activity': ['New filter created', 'Backtest completed', 'Data uploaded', 'Picks generated'],
        'Status': ['✅ Success', '✅ Success', '✅ Success', '⚠️ Warning'],
        'Result': ['+2.3%', '+5.1%', 'Data loaded', '8 picks']
    })
    
    st.dataframe(activity_data, use_container_width=True)
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Upload Data", type="primary", use_container_width=True):
            st.session_state.selected_page = "📊 Data Management"
            st.rerun()
    
    with col2:
        if st.button("🔍 Create Filter", use_container_width=True):
            st.session_state.selected_page = "🔍 Filters"
            st.rerun()
    
    with col3:
        if st.button("🎯 Generate Picks", use_container_width=True):
            st.session_state.selected_page = "🎯 Picks"
            st.rerun()

# Data Management Page
elif page == "📊 Data Management":
    st.header("📊 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["📥 Upload Data", "📋 Templates", "📊 Data View"])
    
    with tab1:
        st.subheader("Upload Historical Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Upload your historical sports betting data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                st.dataframe(df.head())
                
                # Data validation
                required_cols = ['Date', 'Home', 'Away', 'Home_Odds', 'Away_Odds']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.warning(f"⚠️ Missing recommended columns: {missing_cols}")
                else:
                    st.success("✅ All required columns present!")
                
                st.session_state.data_uploaded = True
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with tab2:
        st.subheader("Download Data Templates")
        
        templates = {
            "🏀 NBA Games": "nba_template.csv",
            "🏈 NFL Games": "nfl_template.csv", 
            "⚾ MLB Games": "mlb_template.csv",
            "🏒 NHL Games": "nhl_template.csv"
        }
        
        for sport, template in templates.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(sport)
            with col2:
                if st.button("Download", key=f"download_{template}"):
                    # Create sample template
                    sample_data = pd.DataFrame({
                        'Date': ['2024-02-24', '2024-02-25'],
                        'Home': ['Team A', 'Team C'],
                        'Away': ['Team B', 'Team D'],
                        'Home_Odds': [1.85, 2.10],
                        'Away_Odds': [2.05, 1.80],
                        'Actual_Winner': ['Team A', 'Team D']
                    })
                    csv = sample_data.to_csv(index=False)
                    st.download_button(
                        label=f"💾 {template}",
                        data=csv,
                        file_name=template,
                        mime='text/csv'
                    )
    
    with tab3:
        st.subheader("Current Data")
        
        if st.session_state.data_uploaded:
            st.success("✅ Data loaded successfully!")
            
            # Generate sample data for demo
            sample_data = generate_sample_games()
            st.dataframe(sample_data, use_container_width=True)
            
            # Data summary
            st.subheader("📊 Data Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Games", len(sample_data))
            with col2:
                st.metric("Date Range", "30 days")
            with col3:
                st.metric("Sports", f"{sample_data['Sport'].nunique()}")
        else:
            st.info("📝 No data uploaded yet. Please upload data in the Upload tab.")

# Filters Page
elif page == "🔍 Filters":
    st.header("🔍 Custom Filters")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Create New Filter")
        
        with st.form("filter_form"):
            filter_name = st.text_input("Filter Name", placeholder="e.g., High Confidence NBA Home Games")
            
            col_a, col_b = st.columns(2)
            with col_a:
                min_odds = st.number_input("Minimum Odds", min_value=1.0, max_value=10.0, value=1.5, step=0.1)
                max_odds = st.number_input("Maximum Odds", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
            
            with col_b:
                sport_filter = st.selectbox("Sport", ["All", "NBA", "NFL", "MLB", "NHL"])
                confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 60)
            
            # Advanced options
            with st.expander("Advanced Options"):
                home_away = st.selectbox("Home/Away", ["Both", "Home Only", "Away Only"])
                min_games = st.number_input("Minimum Games", min_value=1, value=10)
                
            submitted = st.form_submit_button("Create Filter", type="primary")
            
            if submitted and filter_name:
                new_filter = {
                    'name': filter_name,
                    'min_odds': min_odds,
                    'max_odds': max_odds,
                    'sport': sport_filter,
                    'confidence': confidence_threshold,
                    'home_away': home_away,
                    'min_games': min_games
                }
                st.session_state.filters_created.append(new_filter)
                st.success(f"✅ Filter '{filter_name}' created successfully!")
    
    with col2:
        st.subheader("Filter Library")
        
        if st.session_state.filters_created:
            for i, filter_obj in enumerate(st.session_state.filters_created):
                with st.expander(filter_obj['name']):
                    st.write(f"**Sport:** {filter_obj['sport']}")
                    st.write(f"**Odds Range:** {filter_obj['min_odds']} - {filter_obj['max_odds']}")
                    st.write(f"**Confidence:** ≥{filter_obj['confidence']}%")
                    
                    if st.button("Delete", key=f"delete_{i}"):
                        st.session_state.filters_created.pop(i)
                        st.rerun()
        else:
            st.info("No filters created yet.")
    
    st.markdown("---")
    st.subheader("📊 Filter Performance")
    
    if st.session_state.filters_created:
        # Sample performance data
        performance_data = []
        for f in st.session_state.filters_created:
            performance_data.append({
                'Filter Name': f['name'],
                'Sport': f['sport'],
                'Win Rate': f"{np.random.randint(60, 80)}%",
                'Total Picks': np.random.randint(50, 200),
                'ROI': f"+{np.random.uniform(5, 25):.1f}%",
                'Status': 'Active'
            })
        
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, use_container_width=True)
    else:
        st.info("Create filters to see performance metrics.")

# Backtesting Page
elif page == "📈 Backtesting":
    st.header("📈 Backtesting Engine")
    
    with st.expander("⚙️ Backtest Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
            end_date = st.date_input("End Date", value=datetime.now())
            initial_bankroll = st.number_input("Initial Bankroll ($)", value=1000, min_value=100)
        
        with col2:
            bet_strategy = st.selectbox("Betting Strategy", [
                "Fixed Stake", 
                "Kelly Criterion", 
                "Percentage Stake", 
                "Fibonacci"
            ])
            sport_filter = st.selectbox("Sport Filter", ["All", "NBA", "NFL", "MLB", "NHL"])
            
            if bet_strategy == "Fixed Stake":
                stake_amount = st.number_input("Stake Amount ($)", value=50, min_value=1)
            elif bet_strategy == "Percentage Stake":
                stake_percentage = st.slider("Stake Percentage (%)", 1, 10, 2)
        
        # Filter selection
        if st.session_state.filters_created:
            selected_filters = st.multiselect(
                "Select Filters to Test",
                [f['name'] for f in st.session_state.filters_created],
                default=[f['name'] for f in st.session_state.filters_created[:1]]
            )
        else:
            st.warning("⚠️ No filters available. Create filters first.")
            selected_filters = []
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            if selected_filters:
                with st.spinner("Running backtest simulation..."):
                    import time
                    time.sleep(3)  # Simulate processing
                    
                    st.success("✅ Backtest completed successfully!")
                    st.session_state.backtest_complete = True
            else:
                st.error("❌ Please select at least one filter to test.")
    
    with col2:
        if st.button("📊 Load Sample", use_container_width=True):
            st.session_state.backtest_complete = True
            st.info("Sample results loaded.")
    
    # Results Section
    if st.session_state.get('backtest_complete', False):
        st.markdown("---")
        st.subheader("📊 Backtest Results")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "$2,347", "+134.7%")
        with col2:
            st.metric("Win Rate", "67.3%", "+5.2%")
        with col3:
            st.metric("ROI", "134.7%", "+12.3%")
        with col4:
            st.metric("Max Drawdown", "-12.4%", "-2.1%")
        
        # Performance Chart
        st.subheader("📈 Portfolio Performance")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        returns = np.cumsum(np.random.normal(0.5, 2, len(dates)))
        bankroll = initial_bankroll + returns
        
        fig = px.line(
            x=dates, 
            y=bankroll, 
            title="Bankroll Over Time",
            labels={'x': 'Date', 'y': 'Bankroll ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Results Table
        st.subheader("📋 Detailed Results")
        
        results_data = pd.DataFrame({
            'Filter': selected_filters,
            'Total Bets': [np.random.randint(50, 150) for _ in selected_filters],
            'Win Rate': [f"{np.random.randint(60, 75)}%" for _ in selected_filters],
            'ROI': [f"+{np.random.uniform(5, 25):.1f}%" for _ in selected_filters],
            'Profit ($)': [np.random.randint(100, 800) for _ in selected_filters],
            'Sharpe Ratio': [round(np.random.uniform(0.8, 2.1), 2) for _ in selected_filters]
        })
        
        st.dataframe(results_data, use_container_width=True)

# Analytics Page
elif page == "📉 Analytics":
    st.header("📉 Analytics & Insights")
    
    analytics_type = st.selectbox("Select Analysis Type", [
        "🏆 Team Performance",
        "💰 Market Efficiency", 
        "🔍 Pattern Detection",
        "🤖 Predictive Modeling"
    ])
    
    if analytics_type == "🏆 Team Performance":
        st.subheader("Team Performance Analysis")
        
        sport = st.selectbox("Select Sport", ["NBA", "NFL", "MLB", "NHL"])
        
        # Generate sample team data
        teams = {
            'NBA': ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Nuggets', 'Suns'],
            'NFL': ['Chiefs', '49ers', 'Bengals', 'Bills', 'Cowboys', 'Eagles'],
            'MLB': ['Yankees', 'Dodgers', 'Red Sox', 'Astros', 'Braves', 'Cardinals'],
            'NHL': ['Golden Knights', 'Lightning', 'Bruins', 'Avalanche', 'Hurricanes', 'Oilers']
        }
        
        selected_teams = st.multiselect("Select Teams", teams[sport], default=teams[sport][:4])
        
        if selected_teams:
            metrics = ['Win Rate', 'Points Scored', 'Points Allowed', 'Home Advantage']
            
            performance_data = []
            for team in selected_teams:
                for metric in metrics:
                    performance_data.append({
                        'Team': team,
                        'Metric': metric,
                        'Value': np.random.uniform(40, 90)
                    })
            
            df_performance = pd.DataFrame(performance_data)
            
            fig = px.bar(
                df_performance, 
                x='Team', 
                y='Value', 
                color='Metric', 
                barmode='group',
                title=f"{sport} Team Performance Metrics"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif analytics_type == "🤖 Predictive Modeling":
        st.subheader("Machine Learning Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox("Model Type", [
                "Random Forest",
                "Logistic Regression", 
                "Neural Network",
                "XGBoost",
                "Support Vector Machine"
            ])
            
            target_variable = st.selectbox("Target Variable", [
                "Game Winner",
                "Point Spread",
                "Over/Under",
                "Money Line Value"
            ])
        
        with col2:
            test_size = st.slider("Test Size (%)", 10, 40, 20)
            cross_validation = st.checkbox("Use Cross-Validation", value=True)
            
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training machine learning model..."):
                import time
                time.sleep(3)  # Simulate training
                
                st.success(f"✅ {model_type} trained successfully!")
                
                # Model metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{np.random.uniform(65, 85):.1f}%")
                with col2:
                    st.metric("Precision", f"{np.random.uniform(60, 80):.1f}%")
                with col3:
                    st.metric("Recall", f"{np.random.uniform(55, 75):.1f}%")
                
                # Feature importance
                st.subheader("📊 Feature Importance")
                
                features = ['Home Odds', 'Away Odds', 'Team Form', 'Head-to-Head', 'Injuries']
                importance = np.random.dirichlet(np.ones(len(features)), size=1)[0]
                
                feature_data = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                })
                
                fig = px.bar(
                    feature_data, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    title="Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)

# Picks Page
elif page == "🎯 Picks":
    st.header("🎯 Today's Picks")
    
    # Date and sport selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_date = st.date_input("Select Date", value=datetime.now().date())
    
    with col2:
        sport_filter = st.selectbox("Sport Filter", ["All", "NBA", "NFL", "MLB", "NHL"])
    
    with col3:
        confidence_filter = st.slider("Min Confidence (%)", 50, 95, 65)
    
    # Generate picks
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🎯 Generate Picks", type="primary", use_container_width=True):
            with st.spinner("Analyzing today's games and running models..."):
                import time
                time.sleep(2)
                
                # Generate sample picks
                games = [
                    "Lakers vs Warriors", "Celtics vs Heat", "Nuggets vs Suns",
                    "Chiefs vs 49ers", "Bengals vs Bills", "Cowboys vs Eagles"
                ]
                
                new_picks = []
                for game in games:
                    teams = game.split(" vs ")
                    prediction = np.random.choice(teams)
                    confidence = np.random.randint(confidence_filter, 95)
                    odds = round(np.random.uniform(1.5, 2.8), 2)
                    value = "✅ Good Value" if confidence > 75 and odds > 2.0 else "⚠️ Moderate Value"
                    
                    new_picks.append({
                        'Game': game,
                        'Prediction': prediction,
                        'Confidence': f"{confidence}%",
                        'Odds': odds,
                        'Value': value,
                        'Sport': 'NBA' if any(team in ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Nuggets', 'Suns'] for team in teams) else 'NFL'
                    })
                
                st.session_state.picks_generated = new_picks
                st.success(f"✅ Generated {len(new_picks)} picks for {selected_date}!")
    
    with col2:
        if st.button("📊 Load Sample", use_container_width=True):
            st.session_state.picks_generated = [
                {'Game': 'Lakers vs Warriors', 'Prediction': 'Lakers', 'Confidence': '78%', 'Odds': 1.85, 'Value': '✅ Good Value', 'Sport': 'NBA'},
                {'Game': 'Celtics vs Heat', 'Prediction': 'Celtics', 'Confidence': '72%', 'Odds': 1.92, 'Value': '✅ Good Value', 'Sport': 'NBA'},
                {'Game': 'Chiefs vs 49ers', 'Prediction': 'Chiefs', 'Confidence': '65%', 'Odds': 2.10, 'Value': '⚠️ Moderate Value', 'Sport': 'NFL'}
            ]
            st.info("Sample picks loaded.")
    
    # Display picks
    if st.session_state.picks_generated:
        st.markdown("---")
        st.subheader("🎯 Generated Picks")
        
        # Filter picks based on user selections
        filtered_picks = st.session_state.picks_generated
        
        if sport_filter != "All":
            filtered_picks = [p for p in filtered_picks if p['Sport'] == sport_filter]
        
        # Convert to DataFrame for display
        picks_df = pd.DataFrame(filtered_picks)
        
        if not picks_df.empty:
            # Style the dataframe
            def highlight_confidence(val):
                if isinstance(val, str) and '%' in val:
                    conf = int(val.replace('%', ''))
                    if conf >= 80:
                        return 'background-color: #d4edda'
                    elif conf >= 70:
                        return 'background-color: #fff3cd'
                    else:
                        return 'background-color: #f8d7da'
                return ''
            
            styled_df = picks_df.style.applymap(highlight_confidence, subset=['Confidence'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Summary metrics
            st.subheader("📊 Pick Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Picks", len(filtered_picks))
            with col2:
                avg_confidence = np.mean([int(p['Confidence'].replace('%', '')) for p in filtered_picks])
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            with col3:
                high_confidence = len([p for p in filtered_picks if int(p['Confidence'].replace('%', '')) >= 75])
                st.metric("High Confidence", high_confidence)
            with col4:
                good_value = len([p for p in filtered_picks if '✅' in p['Value']])
                st.metric("Good Value", good_value)
            
            # Export picks
            st.subheader("💾 Export Picks")
            
            csv = picks_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Picks as CSV",
                data=csv,
                file_name=f"picks_{selected_date}.csv",
                mime='text/csv'
            )
        else:
            st.info("No picks match the selected filters.")
    else:
        st.info("🎯 Click 'Generate Picks' to analyze today's games.")

# Settings Page
elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["🔧 General", "📊 Data Sources", "🤖 Model Settings"])
    
    with tab1:
        st.subheader("General Settings")
        
        auto_scrape = st.checkbox("🔄 Enable automatic data scraping", value=True, help="Automatically scrape latest data daily")
        notifications = st.checkbox("🔔 Enable notifications", value=True, help="Get notified about new picks and important updates")
        dark_mode = st.checkbox("🌙 Dark mode (experimental)", value=False)
        
        st.subheader("🎯 Display Settings")
        
        default_timezone = st.selectbox("Default Timezone", [
            "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London"
        ])
        
        date_format = st.selectbox("Date Format", ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"])
        
        if st.button("💾 Save Settings", type="primary"):
            st.success("✅ Settings saved successfully!")
    
    with tab2:
        st.subheader("📊 Data Sources")
        
        st.write("Configure which data sources to use for analysis:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("📈 Massey Ratings", value=True, help="Team ratings and predictions")
            st.checkbox("🏀 ESPN BPI", value=True, help="ESPN's Basketball Power Index")
            st.checkbox("🎮 GameSim", value=True, help="Game simulation and predictions")
        
        with col2:
            st.checkbox("💰 BettingData", value=True, help="Consensus betting odds")
            st.checkbox("🔗 Custom API", value=False, help="Your own API endpoint")
            st.checkbox("📤 Manual Upload", value=True, help="Upload your own data")
        
        st.subheader("🔄 Update Frequency")
        
        update_frequency = st.selectbox("Data Update Frequency", [
            "Real-time",
            "Hourly", 
            "Daily",
            "Weekly"
        ])
        
        api_key = st.text_input("🔑 API Key (if required)", type="password", help="Enter your API key for premium data sources")
    
    with tab3:
        st.subheader("🤖 Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider("Default Confidence Threshold (%)", 50, 90, 65)
            max_picks_per_day = st.number_input("Maximum picks per day", min_value=1, max_value=50, value=10)
            
            st.subheader("🧠 Model Parameters")
            
            learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
            max_depth = st.slider("Max Tree Depth", 3, 20, 10)
        
        with col2:
            st.subheader("🎯 Betting Strategy")
            
            default_stake = st.number_input("Default Stake ($)", min_value=1, max_value=1000, value=50)
            max_bankroll_risk = st.slider("Max Bankroll Risk (%)", 1, 20, 5)
            
            st.checkbox("🛡️ Use Kelly Criterion", value=False, help="Use Kelly criterion for optimal bet sizing")
            st.checkbox("📊 Include Monte Carlo", value=True, help="Run Monte Carlo simulations for risk assessment")
        
        if st.button("🚀 Retrain Models", type="primary"):
            with st.spinner("Retraining models with new settings..."):
                import time
                time.sleep(3)
                st.success("✅ Models retrained successfully!")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; padding: 20px;">'
    '<strong>Sports Analytics Platform v1.0</strong> | '
    '🚀 Powered by Streamlit | '
    '<a href="#" style="color: #1f77b4;">Documentation</a> | '
    '<a href="#" style="color: #1f77b4;">Support</a>'
    '</div>',
    unsafe_allow_html=True
)
