import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

class ShiftAuctionAnalyzer:
    def __init__(self, shifts_path, auction_path):
        """Initialize the analyzer with paths to data files."""
        try:
            self.shifts_df = pd.read_csv(shifts_path)
            self.auction_df = pd.read_csv(auction_path)
            st.write("Available columns in shifts data:", self.shifts_df.columns.tolist())
            st.write("Available columns in auction data:", self.auction_df.columns.tolist())
            self._preprocess_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise

    def _preprocess_data(self):
        """Clean and preprocess the data."""
        try:
            # Clean numeric columns
            def clean_numeric(x):
                if isinstance(x, str):
                    return float(x.replace(',', '').strip())
                return float(x)

            # Clean auction winning price
            if 'auction_winning_price' in self.auction_df.columns:
                self.auction_df['auction_winning_price'] = self.auction_df['auction_winning_price'].apply(clean_numeric)

            # Clean hourly rate
            if 'hourly_rate_eur' in self.shifts_df.columns:
                self.shifts_df['hourly_rate_eur'] = self.shifts_df['hourly_rate_eur'].apply(clean_numeric)

            # Convert time columns to datetime
            time_cols_shifts = ['shift_date', 'shift_start_time', 'shift_end_time']
            time_cols_auction = ['auction_start_time', 'auction_end_time']
            
            # Handle shift times
            for col in time_cols_shifts:
                if col in self.shifts_df.columns:
                    self.shifts_df[col] = pd.to_datetime(self.shifts_df[col], errors='coerce')
                    if col == 'shift_start_time':
                        self.shifts_df['hour_of_day'] = self.shifts_df[col].dt.hour

            # Handle auction times
            for col in time_cols_auction:
                if col in self.auction_df.columns:
                    self.auction_df[col] = pd.to_datetime(self.auction_df[col], errors='coerce')
                    if col == 'auction_start_time':
                        self.auction_df['hour_of_day'] = self.auction_df[col].dt.hour

            # If we don't have hour_of_day, create a default distribution
            if 'hour_of_day' not in self.auction_df.columns:
                st.warning("No time information found in auction data. Using uniform distribution across hours.")
                self.auction_df['hour_of_day'] = np.random.randint(0, 24, size=len(self.auction_df))

            if 'hour_of_day' not in self.shifts_df.columns:
                st.warning("No time information found in shift data. Using uniform distribution across hours.")
                self.shifts_df['hour_of_day'] = np.random.randint(0, 24, size=len(self.shifts_df))

            # Calculate unused hours if available
            if all(col in self.shifts_df.columns for col in ['shift_working_hours', 'hours_used']):
                self.shifts_df['unused_hours'] = self.shifts_df['shift_working_hours'] - self.shifts_df['hours_used']

            # Remove any remaining invalid entries
            self.auction_df = self.auction_df.dropna(subset=['auction_winning_price', 'hour_of_day'])
            if 'hourly_rate_eur' in self.shifts_df.columns:
                self.shifts_df = self.shifts_df.dropna(subset=['hourly_rate_eur', 'hour_of_day'])

        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            raise

    def compare_costs(self, trip_duration=1):
        """Compare shift costs vs auction prices."""
        try:
            # Calculate cost per hour for shifts
            shift_hourly_cost = self.shifts_df['hourly_rate_eur'].mean()
            
            # Create comparison DataFrame with all hours
            hours = list(range(24))
            cost_comparison = pd.DataFrame({
                'hour_of_day': hours,
                'shift_cost': [shift_hourly_cost * trip_duration] * 24,
                'auction_cost': [0] * 24
            })
            
            # Calculate average auction cost by hour
            auction_costs = self.auction_df.groupby('hour_of_day')['auction_winning_price'].mean()
            for hour in hours:
                if hour in auction_costs.index:
                    cost_comparison.loc[cost_comparison['hour_of_day'] == hour, 'auction_cost'] = auction_costs[hour]
                else:
                    # If no data for this hour, use the overall mean
                    cost_comparison.loc[cost_comparison['hour_of_day'] == hour, 'auction_cost'] = self.auction_df['auction_winning_price'].mean()
            
            cost_comparison['cost_diff'] = cost_comparison['shift_cost'] - cost_comparison['auction_cost']
            return cost_comparison
        except Exception as e:
            st.error(f"Error in cost comparison: {str(e)}")
            raise

    def plot_cost_comparison(self, cost_comparison):
        """Create visualizations for cost comparison."""
        try:
            # Line chart of costs by hour
            fig_line = px.line(cost_comparison, x='hour_of_day', y=['shift_cost', 'auction_cost'],
                            title='Shift Cost vs Auction Cost by Hour of Day',
                            labels={'hour_of_day': 'Hour of Day', 'value': 'Cost (EUR)'})
            
            # Histogram of cost differences
            fig_hist = px.histogram(cost_comparison, x='cost_diff',
                                title='Distribution of Cost Differences (Shift - Auction)',
                                labels={'cost_diff': 'Cost Difference (EUR)'})
            
            return fig_line, fig_hist
        except Exception as e:
            st.error(f"Error in plotting: {str(e)}")
            raise

    def get_recommendations(self, cost_comparison):
        """Generate recommendations based on analysis."""
        try:
            cheaper_shifts = cost_comparison[cost_comparison['cost_diff'] < 0]
            cheaper_auctions = cost_comparison[cost_comparison['cost_diff'] > 0]
            
            recommendations = {
                'prefer_shifts': cheaper_shifts['hour_of_day'].tolist(),
                'prefer_auctions': cheaper_auctions['hour_of_day'].tolist(),
                'avg_potential_savings': abs(cost_comparison['cost_diff']).mean()
            }
            return recommendations
        except Exception as e:
            st.error(f"Error in recommendations: {str(e)}")
            raise

def main():
    st.title("Shift vs Auction Cost Analysis")
    
    # Load data
    shifts_path = "data/raw/disp_pre_purchased_shifts.csv"
    auction_path = "data/raw/disp_historical_auction_data.csv"
    
    try:
        analyzer = ShiftAuctionAnalyzer(shifts_path, auction_path)
        
        # Compare costs
        st.subheader("Cost Comparison Analysis")
        trip_duration = st.slider("Select trip duration (hours)", 1, 8, 1)
        cost_comparison = analyzer.compare_costs(trip_duration)
        
        # Plot results
        fig_line, fig_hist = analyzer.plot_cost_comparison(cost_comparison)
        st.plotly_chart(fig_line)
        st.plotly_chart(fig_hist)
        
        # Show recommendations
        st.subheader("Recommendations")
        recommendations = analyzer.get_recommendations(cost_comparison)
        st.write("Hours to prefer pre-purchased shifts:", recommendations['prefer_shifts'])
        st.write("Hours to prefer auctions:", recommendations['prefer_auctions'])
        st.write(f"Average potential savings: €{recommendations['avg_potential_savings']:.2f} per hour")
        
    except Exception as e:
        st.error(f"Error analyzing data: {str(e)}")

if __name__ == "__main__":
    main()
