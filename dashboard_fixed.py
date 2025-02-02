import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dispatch_analysis_new import DispatchAnalyzer

def display_peak_hours(analyzer):
    """Display the Peak Hours Analysis"""
    st.title("Peak Hours Analysis")
    
    # Get the analysis results
    hourly_df = analyzer.analyze_peak_hours()
    
    if hourly_df is not None:
        fig = go.Figure()
        
        # Add auction price line
        fig.add_trace(go.Scatter(
            x=hourly_df['hour'],
            y=hourly_df['avg_auction_price'],
            name='Auction Price',
            line=dict(color='#00B8D9', width=2)
        ))
        
        # Add revenue line
        fig.add_trace(go.Scatter(
            x=hourly_df['hour'],
            y=hourly_df['avg_revenue'],
            name='Average Revenue',
            line=dict(color='#FF5630', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Average Auction Prices vs Revenue by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Amount (EUR)',
            template='plotly_dark',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        st.subheader("Hourly Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Bookings per Hour", f"{hourly_df['total_bookings'].mean():.1f}")
        
        with col2:
            st.metric("Average Revenue per Hour", f"€{hourly_df['avg_revenue'].mean():.2f}")
        
        with col3:
            st.metric("Average Duration", f"{hourly_df['avg_duration'].mean():.1f} min")
    else:
        st.error("Failed to load peak hours analysis data")

def display_shift_utilization(analyzer):
    """Display the Shift Utilization analysis"""
    st.title("Shift Utilization Analysis")
    
    # Get the analysis results
    analysis = analyzer.analyze_shift_utilization()
    
    if analysis:
        # Display key metrics
        st.subheader("Utilization Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Shifts", f"{analysis['total_shifts']:,}")
            st.metric("Total Working Hours", f"{analysis['total_working_hours']:.1f}")
        
        with col2:
            st.metric("Average Hours per Shift", f"{analysis['avg_hours_per_shift']:.1f}")
            st.metric("Average Hourly Rate", f"€{analysis['avg_hourly_rate']:.2f}")
        
        with col3:
            st.metric("Total Cost", f"€{analysis['total_cost']:,.2f}")
            st.metric("Cost per Hour", f"€{analysis['cost_per_hour']:.2f}")
    else:
        st.error("Failed to load shift utilization data")

def display_supply_demand(analyzer):
    """Display the Supply and Demand analysis"""
    st.title("Supply and Demand Analysis")
    
    # Get the analysis results
    analysis = analyzer.analyze_supply_demand()
    
    if analysis:
        # Create the line chart
        fig = go.Figure()
        
        # Add supply line
        fig.add_trace(go.Scatter(
            x=analysis['date'],
            y=analysis['total_shift_hours'],
            name='Supply (Shift Hours)',
            line=dict(color='#00B8D9', width=2)
        ))
        
        # Add demand line
        fig.add_trace(go.Scatter(
            x=analysis['date'],
            y=analysis['total_booking_hours'],
            name='Demand (Booking Hours)',
            line=dict(color='#FF5630', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Daily Supply vs Demand',
            xaxis_title='Date',
            yaxis_title='Hours',
            template='plotly_dark',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        st.subheader("Daily Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Supply", f"{analysis['total_shift_hours'].mean():.1f} hours")
            st.metric("Total Supply", f"{analysis['total_shift_hours'].sum():.1f} hours")
        
        with col2:
            st.metric("Average Demand", f"{analysis['total_booking_hours'].mean():.1f} hours")
            st.metric("Total Demand", f"{analysis['total_booking_hours'].sum():.1f} hours")
        
        with col3:
            utilization = (analysis['total_booking_hours'].sum() / analysis['total_shift_hours'].sum() * 100
                         if analysis['total_shift_hours'].sum() > 0 else 0)
            st.metric("Overall Utilization", f"{utilization:.1f}%")
    else:
        st.error("Failed to load supply and demand data")

def display_cost_revenue(analyzer):
    """Display the Cost and Revenue analysis"""
    st.title("Cost and Revenue Analysis")
    
    # Get the analysis results
    analysis = analyzer.analyze_cost_revenue()
    
    if analysis:
        st.subheader("Cost and Revenue Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Revenue", f"€{analysis['total_revenue']:,.2f}")
            st.metric("Total Shift Cost", f"€{analysis['total_shift_cost']:,.2f}")
        
        with col2:
            st.metric("Profit/Loss", f"€{analysis['profit_loss']:,.2f}")
            st.metric("Average Revenue per Booking", f"€{analysis['avg_revenue_per_booking']:,.2f}")
        
        with col3:
            st.metric("Average Cost per Shift", f"€{analysis['avg_cost_per_shift']:,.2f}")
    else:
        st.error("Failed to load cost and revenue data")

def display_lost_revenue(analyzer):
    """Display analysis of bookings with zero revenue"""
    st.title("Lost Revenue Analysis")
    
    # Get the analysis results
    analysis = analyzer.analyze_lost_revenue()
    
    if analysis:
        st.subheader("Analysis of Lost Revenue Cases (Revenue = 0)")
        
        # Display key metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Lost Cases", f"{analysis['total_cases']:,}")
            st.metric("Total Hours", f"{analysis['total_hours']:.1f}")
        
        with col2:
            st.metric("Average Distance per Case", f"{analysis['avg_distance_km']:.1f} km")
            st.metric("Average Duration per Case", f"{analysis['avg_duration_minutes']:.1f} min")
        
        with col3:
            st.metric("Estimated Total Lost Revenue", f"€{analysis['estimated_lost_revenue']:,.2f}")
        
        # Display detailed data table
        st.subheader("Detailed Lost Revenue Cases")
        
        # Prepare DataFrame for display
        display_df = analysis['lost_cases_df'].copy()
        display_df['booked_start_at'] = pd.to_datetime(display_df['booked_start_at']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['booked_duration'] = display_df['booked_duration'] / 60  # Convert to minutes
        
        # Select and rename columns for display
        display_df = display_df[[
            'booking_uuid', 'booked_start_at', 'booked_distance', 'booked_duration'
        ]].rename(columns={
            'booking_uuid': 'Booking ID',
            'booked_start_at': 'Start Time',
            'booked_distance': 'Distance (km)',
            'booked_duration': 'Duration (min)'
        })
        
        st.dataframe(
            display_df,
            column_config={
                "Booking ID": st.column_config.TextColumn(width="medium"),
                "Start Time": st.column_config.TextColumn(width="medium"),
                "Distance (km)": st.column_config.NumberColumn(width="small", format="%.1f"),
                "Duration (min)": st.column_config.NumberColumn(width="small", format="%.1f")
            },
            hide_index=True
        )
    else:
        st.error("Failed to load lost revenue analysis data")

def main():
    st.set_page_config(
        page_title="Dispatch Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        # Initialize the analyzer
        analyzer = DispatchAnalyzer()
        
        # Load data
        analyzer.load_data(
            bookings_path='data/raw/disp_incoming_bookings.csv',
            shifts_path='data/raw/disp_pre_purchased_shifts.csv',
            auctions_path='data/raw/disp_historical_auction_data.csv'
        )
        
        # Add tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Peak Hours Analysis",
            "Shift Utilization",
            "Supply & Demand",
            "Cost & Revenue",
            "Lost Revenue Analysis"
        ])
        
        with tab1:
            try:
                display_peak_hours(analyzer)
            except Exception as e:
                st.error(f"Error in Peak Hours tab: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        with tab2:
            try:
                display_shift_utilization(analyzer)
            except Exception as e:
                st.error(f"Error in Shift Utilization tab: {str(e)}")
        
        with tab3:
            try:
                display_supply_demand(analyzer)
            except Exception as e:
                st.error(f"Error in Supply & Demand tab: {str(e)}")
        
        with tab4:
            try:
                display_cost_revenue(analyzer)
            except Exception as e:
                st.error(f"Error in Cost & Revenue tab: {str(e)}")
        
        with tab5:
            try:
                display_lost_revenue(analyzer)
            except Exception as e:
                st.error(f"Error in Lost Revenue tab: {str(e)}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
