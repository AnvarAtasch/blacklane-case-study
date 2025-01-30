import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dispatch_analysis import DispatchAnalyzer

def display_auction_vs_shift_costs(analyzer):
    """Display the Auction Prices vs Shift Costs analysis"""
    st.title("Auction Prices vs Shift Costs")
    
    # Get the analysis results
    analysis = analyzer.analyze_peak_hours()
    
    if analysis:
        # Create the line chart for hourly metrics
        hourly_df = analysis['hourly_metrics']
        
        fig = go.Figure()
        
        # Add auction price line
        fig.add_trace(go.Scatter(
            x=hourly_df['hour'],
            y=hourly_df['avg_auction_price'],
            name='Auction Price',
            line=dict(color='#00B8D9', width=2)
        ))
        
        # Add shift cost line
        fig.add_trace(go.Scatter(
            x=hourly_df['hour'],
            y=hourly_df['avg_shift_cost'],
            name='Shift Cost',
            line=dict(color='#FF5630', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Average Auction Prices vs Shift Costs by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Price (EUR)',
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display peak hours analysis table
        st.subheader("Peak Hours Analysis")
        
        peak_df = analysis['peak_hours_analysis']
        
        # Format the DataFrame for display
        formatted_df = peak_df.copy()
        formatted_df['avg_auction_price'] = formatted_df['avg_auction_price'].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
        formatted_df['avg_shift_cost'] = formatted_df['avg_shift_cost'].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
        formatted_df['price_difference'] = formatted_df['price_difference'].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
        
        # Rename columns for display
        formatted_df.columns = ['Time Range', 'Number of Bookings', 'Avg Auction Price (EUR)', 'Avg Shift Cost (EUR)', 'Price Difference (EUR)']
        
        # Display the table
        st.dataframe(
            formatted_df,
            column_config={
                "Time Range": st.column_config.TextColumn(width="medium"),
                "Number of Bookings": st.column_config.NumberColumn(width="small"),
                "Avg Auction Price (EUR)": st.column_config.TextColumn(width="medium"),
                "Avg Shift Cost (EUR)": st.column_config.TextColumn(width="medium"),
                "Price Difference (EUR)": st.column_config.TextColumn(width="medium")
            },
            hide_index=True
        )
    else:
        st.error("Failed to load auction and shift cost analysis data")

def display_shift_utilization(analyzer):
    """Display the Shift Utilization analysis"""
    st.title("Shift Utilization Analysis")
    
    # Get the analysis results
    analysis = analyzer.analyze_hours_distribution()
    
    if analysis:
        # Display overall metrics
        st.subheader("Overall Utilization Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Scheduled Hours",
                f"{analysis['total_prepurchased']:,.0f}"
            )
        with col2:
            st.metric(
                "Total Active Hours",
                f"{analysis['total_utilized']:,.0f}"
            )
        with col3:
            st.metric(
                "Unused Hours",
                f"{analysis['unused_hours']:,.0f}"
            )
        with col4:
            st.metric(
                "Average Utilization",
                f"{analysis['utilization_rate']:.1f}%"
            )
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hourly Utilization")
            
            # Hours used by time of day
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=analysis['utilized_hours']['hour_of_day'],
                y=analysis['utilized_hours']['utilized_hours'],
                name='Hours Used',
                marker_color='#00B4D8'
            ))
            
            fig.update_layout(
                title='Hours Used by Time of Day',
                xaxis_title='Hour of Day',
                yaxis_title='Hours Used',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cost vs Revenue")
            
            # Get cost vs revenue data
            cost_revenue = analyzer.analyze_cost_revenue()
            
            if cost_revenue:
                # Create bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Cost', 'Revenue'],
                    y=[cost_revenue['total_cost'], cost_revenue['total_revenue']],
                    marker_color=['#FB8500', '#00B4D8']
                ))
                
                fig.update_layout(
                    title='Total Cost vs Revenue',
                    yaxis_title='Amount (EUR)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False,
                    yaxis=dict(tickformat='€,.0f')
                )
                
                st.plotly_chart(fig, use_container_width=True)

def display_supply_demand(analyzer):
    """Display the Supply and Demand analysis"""
    st.title("Supply & Demand Analysis")
    
    # Get the analysis results
    analysis = analyzer.analyze_supply_demand()
    
    if analysis:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Bookings", f"{analysis['demand_metrics']['total_bookings']:,}")
            st.metric("Total Hours", f"{analysis['supply_metrics']['total_hours']:.1f}")
        
        with col2:
            metrics = analysis['efficiency_metrics']
            st.metric("Hours per Booking", f"{metrics['hours_per_booking']:.2f}")
            st.metric("Revenue per Hour", f"€{metrics['revenue_per_hour']:.2f}")
            st.metric("Utilization Rate", f"{metrics['utilization_rate']:.1f}%")
        
        # Display hourly statistics
        st.subheader("Hourly Statistics")
        
        # Create the line chart
        fig = go.Figure()
        
        # Add auction statistics
        hourly_auctions = analysis['hourly_auctions']
        
        # Add mean auction price line
        fig.add_trace(go.Scatter(
            x=hourly_auctions['hour'],
            y=hourly_auctions['auction_winning_price']['mean'],
            name='Mean Auction Price',
            line=dict(color='blue')
        ))
        
        # Add standard deviation range
        upper_bound = hourly_auctions['auction_winning_price']['mean'] + hourly_auctions['auction_winning_price']['std']
        lower_bound = hourly_auctions['auction_winning_price']['mean'] - hourly_auctions['auction_winning_price']['std']
        
        fig.add_trace(go.Scatter(
            x=hourly_auctions['hour'],
            y=upper_bound,
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_auctions['hour'],
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name='Standard Deviation'
        ))
        
        # Add min/max corridor
        fig.add_trace(go.Scatter(
            x=hourly_auctions['hour'],
            y=hourly_auctions['auction_corridor_max_price'],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Max Price Corridor'
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_auctions['hour'],
            y=hourly_auctions['auction_corridor_min_price'],
            mode='lines',
            line=dict(dash='dash', color='green'),
            name='Min Price Corridor'
        ))
        
        # Update layout
        fig.update_layout(
            title='Auction Prices by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Price (EUR)',
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display hourly supply and demand
        st.subheader("Supply vs Demand by Hour")
        
        # Create a new figure for supply vs demand
        fig2 = go.Figure()
        
        # Add supply line
        hourly_supply = analysis['hourly_supply']
        fig2.add_trace(go.Scatter(
            x=hourly_supply['hour'],
            y=hourly_supply['shift_working_hours'],
            name='Supply (Hours)',
            line=dict(color='orange')
        ))
        
        # Add demand line
        hourly_demand = analysis['hourly_demand']
        fig2.add_trace(go.Scatter(
            x=hourly_demand['hour'],
            y=hourly_demand['booking_uuid']['count'],
            name='Demand (Bookings)',
            line=dict(color='blue')
        ))
        
        # Update layout
        fig2.update_layout(
            title='Supply and Demand by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Count',
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.error("Failed to load supply and demand analysis data")

def display_auction_vs_revenue(analyzer):
    """Display the Auction Winning Price vs Gross Revenue analysis"""
    st.title("Auction Winning Price vs Gross Revenue")
    
    # Get the data from analyzer
    bookings_df = analyzer.bookings_df
    auctions_df = analyzer.auctions_df
    
    if not bookings_df.empty and not auctions_df.empty:
        # Merge bookings and auctions data
        merged_df = pd.merge(
            bookings_df[['booking_uuid', 'booked_start_at', 'gross_revenue_eur']],
            auctions_df[['booking_uuid', 'auction_winning_price']],
            on='booking_uuid',
            how='inner'
        )
        
        # Extract hour from timestamp
        merged_df['hour'] = pd.to_datetime(merged_df['booked_start_at']).dt.hour
        
        # Group by hour and calculate means
        hourly_metrics = merged_df.groupby('hour').agg({
            'auction_winning_price': 'mean',
            'gross_revenue_eur': 'mean'
        }).reset_index()
        
        # Create the line chart
        fig = go.Figure()
        
        # Add auction winning price line
        fig.add_trace(go.Scatter(
            x=hourly_metrics['hour'],
            y=hourly_metrics['auction_winning_price'],
            name='Auction Winning Price',
            line=dict(color='orange', width=2)
        ))
        
        # Add gross revenue line
        fig.add_trace(go.Scatter(
            x=hourly_metrics['hour'],
            y=hourly_metrics['gross_revenue_eur'],
            name='Gross Revenue',
            line=dict(color='blue', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Average Auction Winning Price vs Gross Revenue by Hour of Day',
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display hourly metrics table
        st.subheader("Hourly Metrics")
        
        # Format the DataFrame for display
        formatted_df = hourly_metrics.copy()
        formatted_df['auction_winning_price'] = formatted_df['auction_winning_price'].apply(lambda x: f"€{x:.2f}")
        formatted_df['gross_revenue_eur'] = formatted_df['gross_revenue_eur'].apply(lambda x: f"€{x:.2f}")
        
        # Rename columns for display
        formatted_df.columns = ['Hour', 'Avg Auction Winning Price (EUR)', 'Avg Gross Revenue (EUR)']
        
        st.dataframe(
            formatted_df,
            column_config={
                "Hour": st.column_config.NumberColumn(width="small"),
                "Avg Auction Winning Price (EUR)": st.column_config.TextColumn(width="medium"),
                "Avg Gross Revenue (EUR)": st.column_config.TextColumn(width="medium")
            },
            hide_index=True
        )
    else:
        st.error("Failed to load auction and revenue analysis data")

def main():
    st.set_page_config(
        page_title="Shift Utilization Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        # Initialize the analyzer
        analyzer = DispatchAnalyzer()
        
        # Load data
        success = analyzer.load_data(
            bookings_path='data/raw/disp_incoming_bookings.csv',
            shifts_path='data/raw/disp_pre_purchased_shifts.csv',
            auctions_path='data/raw/disp_historical_auction_data.csv'
        )
        
        if not success:
            st.error("Failed to load data. Please check the data files and try again.")
            return
        
        # Add tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance Metrics",
            "Shift Utilization Analysis",
            "Supply & Demand Analysis",
            "Auction Winning Price vs Gross Revenue"
        ])
        
        with tab1:
            try:
                display_auction_vs_shift_costs(analyzer)
            except Exception as e:
                st.error(f"Error in Performance Metrics tab: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        with tab2:
            try:
                display_shift_utilization(analyzer)
            except Exception as e:
                st.error(f"Error in Shift Utilization tab: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        with tab3:
            try:
                display_supply_demand(analyzer)
            except Exception as e:
                st.error(f"Error in Supply & Demand tab: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        with tab4:
            try:
                display_auction_vs_revenue(analyzer)
            except Exception as e:
                st.error(f"Error in Auction Winning Price vs Gross Revenue tab: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error initializing dashboard: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
