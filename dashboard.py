import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Shift Utilization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dispatch_analysis_new import DispatchAnalyzer

def display_peak_hours(analyzer):
    """Display the Peak Hours Analysis"""
    st.title("Peak Hours Analysis")
    
    # Get the analysis results
    analysis = analyzer.analyze_peak_hours()
    
    if analysis is not None:
        hourly_metrics = analysis['hourly_metrics']
        
        fig = go.Figure()
        
        # Add revenue line
        fig.add_trace(go.Scatter(
            x=hourly_metrics['hour'],
            y=hourly_metrics['avg_revenue'],
            name='Average Revenue',
            line=dict(color='#36B37E', width=2)
        ))
        
        # Add shift cost line
        fig.add_trace(go.Scatter(
            x=hourly_metrics['hour'],
            y=hourly_metrics['avg_shift_cost'],
            name='Average Shift Cost',
            line=dict(color='#FF5630', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Average Revenue vs Shift Cost by Hour of Day',
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
        
        # Display bookings volume
        volume_fig = go.Figure()
        
        volume_fig.add_trace(go.Bar(
            x=hourly_metrics['hour'],
            y=hourly_metrics['bookings_count'],
            marker=dict(color='#00B8D9')
        ))
        
        volume_fig.update_layout(
            title='Number of Bookings by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Bookings',
            template='plotly_dark',
            height=300,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(volume_fig, use_container_width=True)
        
        # Display detailed metrics
        st.markdown("### Hourly Metrics")
        st.dataframe(analysis['peak_hours_analysis'])
    else:
        st.error("Failed to load peak hours data")

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
                f"{analysis['total_scheduled_hours']:,.0f}"
            )
        with col2:
            st.metric(
                "Total Active Hours",
                f"{analysis['total_active_hours']:,.0f}"
            )
        with col3:
            st.metric(
                "Unused Hours",
                f"{analysis['unused_hours']:,.0f}"
            )
        with col4:
            st.metric(
                "Utilization Rate",
                f"{analysis['utilization_rate']:.1f}%"
            )
        
        # Create utilization chart
        fig = go.Figure()
        
        # Add utilized hours
        fig.add_trace(go.Bar(
            name='Active Hours',
            y=['Hours'],
            x=[analysis['total_active_hours']],
            orientation='h',
            marker=dict(color='#00B8D9')
        ))
        
        # Add unused hours
        fig.add_trace(go.Bar(
            name='Unused Hours',
            y=['Hours'],
            x=[analysis['unused_hours']],
            orientation='h',
            marker=dict(color='#FF5630')
        ))
        
        # Update layout
        fig.update_layout(
            title='Hours Utilization',
            barmode='stack',
            showlegend=True,
            height=200,
            margin=dict(t=30, b=0, l=0, r=0),
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display hourly distribution
        if 'hours_by_time' in analysis:
            st.subheader("Hourly Distribution")
            
            # Create hourly distribution chart
            hours_fig = go.Figure()
            
            hours_fig.add_trace(go.Bar(
                x=analysis['hours_by_time'].index,
                y=analysis['hours_by_time'].values,
                marker=dict(color='#00B8D9')
            ))
            
            hours_fig.update_layout(
                title='Hours Distribution by Time of Day',
                xaxis_title='Hour of Day',
                yaxis_title='Hours',
                template='plotly_dark',
                height=300,
                margin=dict(t=30, b=30, l=30, r=30)
            )
            
            st.plotly_chart(hours_fig, use_container_width=True)
    else:
        st.error("Failed to load shift utilization data")

def display_supply_demand(analyzer):
    """Display the Supply and Demand analysis"""
    st.title("Supply and Demand Analysis")
    
    # Get the analysis results
    daily_metrics = analyzer.analyze_supply_demand()
    
    if daily_metrics is not None:
        # Display overall metrics
        st.subheader("Daily Metrics Overview")
        
        # Calculate averages
        avg_metrics = {
            'avg_shifts': daily_metrics['total_shifts'].mean(),
            'avg_bookings': daily_metrics['total_bookings'].mean(),
            'avg_utilization': daily_metrics['utilization_rate'].mean() * 100,
            'avg_revenue': daily_metrics['revenue_per_hour'].mean()
        }
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Daily Shifts",
                f"{avg_metrics['avg_shifts']:.1f}"
            )
        
        with col2:
            st.metric(
                "Avg Daily Bookings",
                f"{avg_metrics['avg_bookings']:.1f}"
            )
        
        with col3:
            st.metric(
                "Avg Utilization Rate",
                f"{avg_metrics['avg_utilization']:.1f}%"
            )
        
        with col4:
            st.metric(
                "Avg Revenue/Hour",
                f"€{avg_metrics['avg_revenue']:.2f}"
            )
        
        # Create supply vs demand chart
        fig = go.Figure()
        
        # Add supply line
        fig.add_trace(go.Scatter(
            x=daily_metrics['date'],
            y=daily_metrics['total_shifts'],
            name='Total Shifts',
            line=dict(color='#00B8D9', width=2)
        ))
        
        # Add demand line
        fig.add_trace(go.Scatter(
            x=daily_metrics['date'],
            y=daily_metrics['total_bookings'],
            name='Total Bookings',
            line=dict(color='#FF5630', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Daily Supply vs Demand',
            xaxis_title='Date',
            yaxis_title='Count',
            template='plotly_dark',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display utilization rate over time
        util_fig = go.Figure()
        
        util_fig.add_trace(go.Scatter(
            x=daily_metrics['date'],
            y=daily_metrics['utilization_rate'] * 100,
            line=dict(color='#36B37E', width=2)
        ))
        
        util_fig.update_layout(
            title='Daily Utilization Rate',
            xaxis_title='Date',
            yaxis_title='Utilization Rate (%)',
            template='plotly_dark',
            height=300,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(util_fig, use_container_width=True)
        
        # Display revenue metrics
        rev_fig = go.Figure()
        
        rev_fig.add_trace(go.Scatter(
            x=daily_metrics['date'],
            y=daily_metrics['revenue_per_hour'],
            line=dict(color='#00B8D9', width=2)
        ))
        
        rev_fig.update_layout(
            title='Daily Revenue per Hour',
            xaxis_title='Date',
            yaxis_title='EUR/Hour',
            template='plotly_dark',
            height=300,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(rev_fig, use_container_width=True)
    else:
        st.error("Failed to load supply and demand analysis data")

def display_cost_revenue(analyzer):
    """Display the Cost and Revenue analysis"""
    st.title("Cost and Revenue Analysis")
    
    # Get the analysis results
    analysis = analyzer.analyze_cost_revenue()
    
    if analysis:
        # Display overall metrics
        st.subheader("Overall Financial Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Revenue",
                f"€{analysis['total_revenue']:,.2f}"
            )
        with col2:
            st.metric(
                "Total Cost",
                f"€{analysis['total_cost']:,.2f}"
            )
        with col3:
            profit = analysis['total_revenue'] - analysis['total_cost']
            st.metric(
                "Net Profit",
                f"€{profit:,.2f}",
                delta=f"{(profit/analysis['total_revenue']*100):.1f}% margin"
            )
        
        # Create revenue vs cost chart
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            name='Revenue',
            x=['Amount'],
            y=[analysis['total_revenue']],
            marker=dict(color='#36B37E')
        ))
        
        fig.add_trace(go.Bar(
            name='Cost',
            x=['Amount'],
            y=[analysis['total_cost']],
            marker=dict(color='#FF5630')
        ))
        
        # Update layout
        fig.update_layout(
            title='Total Revenue vs Cost',
            showlegend=True,
            template='plotly_dark',
            height=300,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to load cost and revenue analysis data")

def display_lost_revenue(analyzer):
    """Display analysis of bookings with zero revenue"""
    st.title("Lost Revenue Analysis")
    
    # Get the analysis results
    analysis = analyzer.analyze_lost_revenue()
    
    if analysis:
        # Display overall metrics
        st.subheader("Lost Revenue Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Lost Cases",
                f"{analysis['total_cases']:,}"
            )
        with col2:
            st.metric(
                "Total Lost Hours",
                f"{analysis['total_hours']:.1f}"
            )
        with col3:
            st.metric(
                "Avg Distance (km)",
                f"{analysis['avg_distance_km']:.1f}"
            )
        with col4:
            st.metric(
                "Avg Duration (min)",
                f"{analysis['avg_duration_minutes']:.1f}"
            )
        
        # Display estimated lost revenue
        st.metric(
            "Estimated Lost Revenue",
            f"€{analysis['estimated_lost_revenue']:,.2f}"
        )
        
        # Display lost cases data
        if 'lost_cases_df' in analysis:
            st.subheader("Lost Cases Details")
            st.dataframe(analysis['lost_cases_df'])
    else:
        st.error("Failed to load lost revenue analysis data")

def display_auction_vs_shift_costs(analyzer):
    """Display the Performance Metrics analysis"""
    st.title("Performance Metrics")
    
    # Get the analysis results
    analysis = analyzer.analyze_peak_hours()
    
    if analysis is not None and isinstance(analysis, dict):
        peak_df = analysis.get('peak_hours_analysis')
        hourly_data = analysis.get('hourly_metrics')
        
        if peak_df is not None and hourly_data is not None:
            # Create visualization
            st.subheader("Hourly Revenue vs Cost")
            fig = go.Figure()
            
            # Add revenue line
            fig.add_trace(go.Scatter(
                x=hourly_data['hour'],
                y=hourly_data['avg_revenue'],
                name='Average Revenue',
                line=dict(color='#00B8D9', width=2)
            ))
            
            # Add shift cost line
            fig.add_trace(go.Scatter(
                x=hourly_data['hour'],
                y=hourly_data['avg_shift_cost'],
                name='Average Shift Cost',
                line=dict(color='#FF5630', width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title='Average Revenue vs Shift Cost by Hour of Day',
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
            
            # Display peak hours analysis table
            st.subheader("Peak Hours Analysis")
            
            # Format the DataFrame for display
            formatted_df = peak_df.copy()
            
            # Format monetary values
            numeric_cols = ['avg_revenue', 'avg_shift_cost', 'revenue_difference']
            for col in numeric_cols:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
            
            # Display the table
            st.dataframe(
                formatted_df,
                column_config={
                    "time_range": st.column_config.TextColumn("Time Range", width="medium"),
                    "num_bookings": st.column_config.NumberColumn("Number of Bookings", width="small"),
                    "avg_revenue": st.column_config.TextColumn("Avg Revenue (EUR)", width="medium"),
                    "avg_shift_cost": st.column_config.TextColumn("Avg Shift Cost (EUR)", width="medium"),
                    "revenue_difference": st.column_config.TextColumn("Revenue Difference (EUR)", width="medium")
                },
                hide_index=True
            )
            
            # Add booking volume visualization
            st.subheader("Hourly Booking Volume")
            volume_fig = go.Figure()
            
            volume_fig.add_trace(go.Bar(
                x=hourly_data['hour'],
                y=hourly_data['bookings_count'],
                marker_color='#36B37E'
            ))
            
            volume_fig.update_layout(
                title='Number of Bookings by Hour of Day',
                xaxis_title='Hour of Day',
                yaxis_title='Number of Bookings',
                template='plotly_dark',
                height=400,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(volume_fig, use_container_width=True)
        else:
            st.error("Error: Invalid data structure returned from analysis")
    else:
        st.error("Failed to load performance metrics data")

def display_shift_utilization(analyzer):
    """Display the Shift Utilization analysis"""
    st.title("Shift Utilization Analysis")
    
    # Get utilization analysis
    analysis = analyzer.analyze_shift_utilization()
    if analysis is not None:
        # Display summary metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Total Utilized Hours",
                value=f"{analysis['total_utilized_hours']:.1f}",
                delta=f"{analysis['utilization_rate']:.1f}%"
            )
        
        with col2:
            st.metric(
                label="Total Shift Hours",
                value=f"{analysis['total_shift_hours']:.1f}",
                delta=f"-{analysis['unused_hours']:.1f} unused"
            )
        
        # Create utilization chart
        fig = go.Figure()
        
        # Add utilized hours
        fig.add_trace(go.Bar(
            name='Utilized Hours',
            y=['Hours'],
            x=[analysis['total_utilized_hours']],
            orientation='h',
            marker=dict(color='#00B8D9')
        ))
        
        # Add unused hours
        fig.add_trace(go.Bar(
            name='Unused Hours',
            y=['Hours'],
            x=[analysis['unused_hours']],
            orientation='h',
            marker=dict(color='#FF5630')
        ))
        
        # Update layout
        fig.update_layout(
            title='Hours Utilization',
            barmode='stack',
            showlegend=True,
            height=200,
            margin=dict(t=30, b=0, l=0, r=0),
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown(f"""
        ### Utilization Analysis
        - **Total Shift Hours**: {analysis['total_shift_hours']:.1f} hours available
        - **Utilized Hours**: {analysis['total_utilized_hours']:.1f} hours used ({analysis['utilization_rate']:.1f}%)
        - **Unused Hours**: {analysis['unused_hours']:.1f} hours unused
        """)
        
        # Add revenue section
        st.markdown("### Revenue vs Cost Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Total Revenue",
                value=f"€{analysis['total_revenue']:,.2f}",
                delta=f"€{analysis['revenue_per_hour']:.2f}/hour"
            )
        
        with col2:
            st.metric(
                label="Total Shift Cost",
                value=f"€{analysis['total_shift_cost']:,.2f}",
                delta=f"€{analysis['cost_per_hour']:.2f}/hour"
            )
            
        # Create revenue vs cost chart
        fig2 = go.Figure()
        
        # Add revenue bar
        fig2.add_trace(go.Bar(
            name='Revenue',
            x=['Amount'],
            y=[analysis['total_revenue']],
            marker=dict(color='#36B37E')
        ))
        
        # Add cost bar
        fig2.add_trace(go.Bar(
            name='Shift Cost',
            x=['Amount'],
            y=[analysis['total_shift_cost']],
            marker=dict(color='#FF5630')
        ))
        
        # Update layout
        fig2.update_layout(
            title='Revenue vs Shift Cost',
            showlegend=True,
            height=300,
            margin=dict(t=30, b=0, l=0, r=0),
            template='plotly_dark',
            yaxis_title='EUR'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.error("Failed to load shift utilization data")    # Get the analysis results
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
    try:
        analysis = analyzer.analyze_supply_demand()
        if analysis is None:
            st.error("Failed to load supply and demand analysis data. Please check if the data is loaded correctly.")
            return
            
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
        
        # Add supply line (make it more visible)
        hourly_supply = analysis['hourly_supply']
        fig2.add_trace(go.Scatter(
            x=hourly_supply['hour'],
            y=hourly_supply['working_hours'],
            name='Supply (Hours)',
            line=dict(color='#FB8500', width=3),
            mode='lines+markers'
        ))
        
        # Add demand line
        hourly_demand = analysis['hourly_demand']
        fig2.add_trace(go.Scatter(
            x=hourly_demand['hour'],
            y=hourly_demand['booking_count'],
            name='Demand (Bookings)',
            line=dict(color='#00B4D8', width=2),
            mode='lines+markers'
        ))

        

        
        # Update layout with improved visibility
        fig2.update_layout(
            title='Supply and Demand by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Count',
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                range=[-0.5, 23.5]
            ),
            showlegend=True
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to load supply and demand analysis data: {str(e)}")

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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Peak Hours Analysis",
            "Shift Utilization",
            "Supply & Demand",
            "Cost & Revenue",
            "Lost Revenue Analysis"
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
                
        with tab5:
            try:
                display_lost_revenue(analyzer)
            except Exception as e:
                st.error(f"Error in Lost Revenue Analysis tab: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error initializing dashboard: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
