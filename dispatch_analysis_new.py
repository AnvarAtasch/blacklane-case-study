import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DispatchAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        self.bookings_df = None
        self.shifts_df = None
        self.auctions_df = None
        
    def load_data(self, bookings_path, shifts_path, auctions_path):
        """Load data from CSV files and convert data types"""
        try:
            print("\nDebug: Starting data load...")
            
            # Load data with proper parsing and handle missing values
            print(f"\nLoading bookings from: {bookings_path}")
            self.bookings_df = pd.read_csv(bookings_path)
            print(f"Loaded {len(self.bookings_df)} bookings records")
            
            print(f"\nLoading shifts from: {shifts_path}")
            self.shifts_df = pd.read_csv(shifts_path)
            print(f"Loaded {len(self.shifts_df)} shifts records")
            
            print(f"\nLoading auctions from: {auctions_path}")
            self.auctions_df = pd.read_csv(auctions_path)
            print(f"Loaded {len(self.auctions_df)} auctions records")
            
            # Convert datetime columns
            print("\nConverting datetime columns...")
            if 'booked_start_at' in self.bookings_df.columns:
                self.bookings_df['booked_start_at'] = pd.to_datetime(self.bookings_df['booked_start_at'])
            
            if 'shift_date' in self.shifts_df.columns:
                self.shifts_df['shift_date'] = pd.to_datetime(self.shifts_df['shift_date'])
            
            if 'auction_time' in self.auctions_df.columns:
                self.auctions_df['auction_time'] = pd.to_datetime(self.auctions_df['auction_time'])
            
            # Handle numeric columns
            print("\nConverting numeric columns...")
            numeric_cols = {
                'bookings_df': {
                    'booked_duration': 'float64',
                    'gross_revenue_eur': 'float64'
                },
                'shifts_df': {
                    'hourly_rate_eur': 'float64'
                },
                'auctions_df': {
                    'auction_winning_price': 'float64'
                }
            }
            
            for df_name, cols in numeric_cols.items():
                df = getattr(self, df_name)
                for col, dtype in cols.items():
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
            
            print("\nData loading completed successfully!")
            return True
            
            # Convert datetime columns
            print("\nConverting datetime columns...")
            datetime_cols = {
                'bookings_df': ['pickup_time', 'booked_start_at'],  # Try both column names
                'shifts_df': ['start_time', 'shift_date'],  # Try both column names
                'auctions_df': ['auction_time']
            }
            
            for df_name, cols in datetime_cols.items():
                df = getattr(self, df_name)
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        print(f"Converted {df_name}.{col} to datetime")
            
            # Handle missing values and numeric conversions
            print("\nHandling missing values and numeric conversions...")
            
            # For shifts
            if 'cost' in self.shifts_df.columns:
                self.shifts_df['cost'] = pd.to_numeric(self.shifts_df['cost'], errors='coerce')
                median_cost = self.shifts_df['cost'].median()
                self.shifts_df['cost'] = self.shifts_df['cost'].fillna(median_cost)
            
            # For bookings
            numeric_cols_bookings = ['booked_duration', 'gross_revenue_eur']
            for col in numeric_cols_bookings:
                if col in self.bookings_df.columns:
                    self.bookings_df[col] = pd.to_numeric(self.bookings_df[col], errors='coerce')
            
            # For auctions
            if 'auction_winning_price' in self.auctions_df.columns:
                self.auctions_df['auction_winning_price'] = pd.to_numeric(self.auctions_df['auction_winning_price'], errors='coerce')
            
            # Verify required columns exist
            required_cols = {
                'bookings_df': ['pickup_time', 'booked_duration', 'gross_revenue_eur'],
                'shifts_df': ['start_time', 'cost', 'duration'],
                'auctions_df': ['auction_time', 'auction_winning_price']
            }
            
            for df_name, cols in required_cols.items():
                df = getattr(self, df_name)
                missing_cols = [col for col in cols if col not in df.columns]
                if missing_cols:
                    print(f"Error: Missing required columns in {df_name}: {missing_cols}")
                    return False
            
            print("\nData loading completed successfully!")
            return True
            
            for df_name, cols in numeric_cols.items():
                df = getattr(self, df_name)
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"Converted {df_name}.{col} to numeric. Null values: {df[col].isnull().sum()}")
            
            # Fill any remaining missing values
            self.shifts_df['shift_working_hours'] = self.shifts_df['shift_working_hours'].fillna(8.0)
            
            print("\nData types after conversion:")
            for df_name in ['bookings_df', 'shifts_df', 'auctions_df']:
                print(f"\n{df_name}:")
                print(getattr(self, df_name).dtypes)
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_lost_revenue(self):
        """Analyze bookings with zero revenue"""
        try:
            print("\nDEBUG: Starting lost revenue analysis")
            print(f"Total bookings before filtering: {len(self.bookings_df)}")
            
            # Ensure numeric types
            self.bookings_df['gross_revenue_eur'] = pd.to_numeric(self.bookings_df['gross_revenue_eur'], errors='coerce')
            self.bookings_df['booked_duration'] = pd.to_numeric(self.bookings_df['booked_duration'], errors='coerce')
            self.bookings_df['booked_distance'] = pd.to_numeric(self.bookings_df['booked_distance'], errors='coerce')
            
            # Filter bookings with zero revenue
            lost_revenue_cases = self.bookings_df[self.bookings_df['gross_revenue_eur'] == 0].copy()
            print(f"\nFound {len(lost_revenue_cases)} cases with zero revenue")
            
            if len(lost_revenue_cases) == 0:
                print("No cases with zero revenue found")
                return None
            
            # Calculate metrics
            total_cases = len(lost_revenue_cases)
            total_hours = lost_revenue_cases['booked_duration'].sum() / 3600  # Convert seconds to hours
            avg_distance = lost_revenue_cases['booked_distance'].mean()
            avg_duration = lost_revenue_cases['booked_duration'].mean() / 60  # Convert seconds to minutes
            
            print(f"\nMetrics calculated:")
            print(f"Total hours: {total_hours:.2f}")
            print(f"Average distance: {avg_distance:.2f} km")
            print(f"Average duration: {avg_duration:.2f} min")
            
            # Calculate average revenue from successful bookings
            successful_bookings = self.bookings_df[self.bookings_df['gross_revenue_eur'] > 0]
            print(f"\nCalculating average revenue from {len(successful_bookings)} successful bookings")
            
            avg_booking_revenue = successful_bookings['gross_revenue_eur'].mean()
            avg_booking_duration = successful_bookings['booked_duration'].mean() / 60  # Convert to minutes
            
            avg_revenue_per_minute = avg_booking_revenue / avg_booking_duration if avg_booking_duration > 0 else 0
            print(f"Average revenue per minute: €{avg_revenue_per_minute:.2f}")
            
            # Calculate estimated lost revenue
            estimated_lost_revenue = avg_revenue_per_minute * (lost_revenue_cases['booked_duration'].sum() / 60)
            print(f"Estimated total lost revenue: €{estimated_lost_revenue:.2f}")
            
            return {
                'total_cases': total_cases,
                'total_hours': total_hours,
                'avg_distance_km': avg_distance,
                'avg_duration_minutes': avg_duration,
                'estimated_lost_revenue': estimated_lost_revenue,
                'lost_cases_df': lost_revenue_cases
            }
            
        except Exception as e:
            print(f"Error in analyze_lost_revenue: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_supply_demand(self):
        """Analyze supply and demand patterns with detailed metrics"""
        try:
            print("\nDebug: Starting supply and demand analysis...")
            if self.shifts_df is None or self.bookings_df is None:
                print("Error: Required data not loaded")
                return None
                
            print("\nShifts DataFrame columns:", self.shifts_df.columns.tolist())
            print("Bookings DataFrame columns:", self.bookings_df.columns.tolist())
            
            # Analyze supply (shifts)
            daily_shifts = self.shifts_df.groupby('date').agg({
                'shift_uuid': 'count',  # Count unique shifts
                'duration': 'sum'  # Sum of shift durations
            }).reset_index()
            daily_shifts.columns = ['date', 'total_shifts', 'total_shift_hours']
            
            # Analyze demand (bookings)
            daily_bookings = self.bookings_df.groupby(pd.to_datetime(self.bookings_df['booked_start_at']).dt.date).agg({
                'booking_uuid': 'count',  # Count unique bookings
                'booked_duration': 'sum',  # Sum of booking durations
                'gross_revenue_eur': 'sum'  # Sum of revenue
            }).reset_index()
            daily_bookings.columns = ['date', 'total_bookings', 'total_booking_duration', 'total_revenue']
            
            print("\nProcessed daily metrics:")
            print("Shifts shape:", daily_shifts.shape)
            print("Bookings shape:", daily_bookings.shape)
            
            # Convert booking duration to hours (if it's in seconds)
            daily_bookings['total_booking_hours'] = daily_bookings['total_booking_duration'] / 3600
            
            # Merge supply and demand data
            daily_metrics = pd.merge(daily_shifts, daily_bookings, on='date', how='outer').fillna(0)
            
            # Calculate utilization rate
            daily_metrics['utilization_rate'] = (daily_metrics['total_booking_hours'] / 
                                               daily_metrics['total_shift_hours']).clip(0, 1)
            
            # Calculate revenue per hour
            daily_metrics['revenue_per_hour'] = (daily_metrics['total_revenue'] / 
                                               daily_metrics['total_booking_hours']).replace([np.inf, -np.inf], 0)
            
            print("\nFinal metrics shape:", daily_metrics.shape)
            return daily_metrics
            
        except Exception as e:
            print(f"Error in analyze_supply_demand: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
        except Exception as e:
            print(f"Error in analyze_supply_demand: {str(e)}")
            return None

    def analyze_hours_distribution(self):
        """Analyze pre-purchased hours vs utilized hours by hour of day"""
        if self.shifts_df is None or self.bookings_df is None:
            return None

        try:
            # Calculate total metrics
            total_scheduled_hours = self.shifts_df['duration'].sum()
            total_active_hours = self.bookings_df['booked_duration'].sum()
            unused_hours = total_scheduled_hours - total_active_hours
            utilization_rate = (total_active_hours / total_scheduled_hours * 100) if total_scheduled_hours > 0 else 0

            # Calculate hourly utilization
            hours_by_time = self.shifts_df.groupby(self.shifts_df['start_time'].dt.hour)['duration'].sum()

            # Calculate cost vs revenue
            total_cost = self.shifts_df['cost'].sum()
            total_revenue = self.bookings_df['gross_revenue_eur'].sum()

            return {
                'total_scheduled_hours': total_scheduled_hours,
                'total_active_hours': total_active_hours,
                'unused_hours': unused_hours,
                'utilization_rate': utilization_rate,
                'hours_by_time': hours_by_time,
                'total_cost': total_cost,
                'total_revenue': total_revenue
            }

        except Exception as e:
            print(f"Error in analyze_hours_distribution: {str(e)}")
            return None

    def analyze_shift_utilization(self):
        """Analyze shift utilization by comparing scheduled vs actual hours"""
        try:
            # Calculate total utilized hours from bookings
            total_utilized_seconds = self.bookings_df['booked_duration'].sum()
            total_utilized_hours = total_utilized_seconds / 3600
            
            # Calculate total pre-purchased hours from shifts
            total_shift_hours = self.shifts_df['shift_working_hours'].sum()
            
            # Calculate unused hours and utilization rate
            unused_hours = total_shift_hours - total_utilized_hours
            utilization_rate = (total_utilized_hours / total_shift_hours) * 100 if total_shift_hours > 0 else 0
            
            # Calculate revenue metrics
            total_revenue = self.bookings_df['gross_revenue_eur'].sum()
            
            # Calculate total shift cost (hourly rate * hours for each shift)
            total_shift_cost = (self.shifts_df['hourly_rate_eur'] * self.shifts_df['shift_working_hours']).sum()
            
            revenue_per_hour = total_revenue / total_utilized_hours if total_utilized_hours > 0 else 0
            cost_per_hour = total_shift_cost / total_shift_hours if total_shift_hours > 0 else 0
            
            print("\nDebug Information:")
            print(f"Total utilized hours (seconds): {total_utilized_seconds}")
            print(f"Total utilized hours (hours): {total_utilized_hours:.2f}")
            print(f"Total pre-purchased hours: {total_shift_hours:.2f}")
            print(f"Unused hours: {unused_hours:.2f}")
            print(f"Utilization rate: {utilization_rate:.2f}%")
            print(f"Total revenue: €{total_revenue:.2f}")
            print(f"Total shift cost: €{total_shift_cost:.2f}")
            
            return {
                'total_utilized_hours': total_utilized_hours,
                'total_shift_hours': total_shift_hours,
                'unused_hours': unused_hours,
                'utilization_rate': utilization_rate,
                'total_revenue': total_revenue,
                'total_shift_cost': total_shift_cost,
                'revenue_per_hour': revenue_per_hour,
                'cost_per_hour': cost_per_hour
            }
            
        except Exception as e:
            print(f"Error in analyze_shift_utilization: {str(e)}")
            return None

    def analyze_peak_hours(self):
        """Analyze peak hours with auction prices and revenue"""
        if self.auctions_df is None or self.shifts_df is None:
            print("Error: Required data not loaded")
            return None

        try:
            # Convert booked_start_at to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self.bookings_df['booked_start_at']):
                self.bookings_df['booked_start_at'] = pd.to_datetime(self.bookings_df['booked_start_at'])
            
            # Group bookings by hour
            bookings_hourly = self.bookings_df.groupby(self.bookings_df['booked_start_at'].dt.hour).agg({
                'gross_revenue_eur': ['mean', 'count']
            })
            
            # Group shifts by hour and calculate average hourly rate
            shifts_hourly = self.shifts_df.groupby(self.shifts_df['shift_id'])['hourly_rate_eur'].mean()
            avg_shift_cost = shifts_hourly.mean()  # Use average across all shifts
            
            # Create peak hours analysis DataFrame
            peak_df = pd.DataFrame({
                'hour': bookings_hourly.index,
                'num_bookings': bookings_hourly[('gross_revenue_eur', 'count')],
                'avg_revenue': bookings_hourly[('gross_revenue_eur', 'mean')],
                'avg_shift_cost': avg_shift_cost
            })
            
            # Calculate revenue difference
            peak_df['revenue_difference'] = peak_df['avg_revenue'] - peak_df['avg_shift_cost']
            
            # Format hour ranges
            peak_df['time_range'] = peak_df['hour'].apply(lambda x: f"{x:02d}:00 - {(x+1):02d}:00")
            
            # Debug output
            print("\nPeak Hours Analysis Debug:")
            print(f"Columns in peak_df: {peak_df.columns.tolist()}")
            print(f"Number of rows: {len(peak_df)}")
            print("Sample of peak_df:")
            print(peak_df.head())
            
            return {
                'peak_hours_analysis': peak_df[['time_range', 'num_bookings', 'avg_revenue', 'avg_shift_cost', 'revenue_difference']],
                'hourly_metrics': {
                    'hour': peak_df['hour'].tolist(),
                    'bookings_count': peak_df['num_bookings'].tolist(),
                    'avg_revenue': peak_df['avg_revenue'].tolist(),
                    'avg_shift_cost': [avg_shift_cost] * len(peak_df)
                }
            }
            
        except Exception as e:
            print(f"Error in analyze_peak_hours: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
            # Analyze bookings by hour
            hourly_metrics = self.bookings_df.groupby('hour').agg({
                'booking_uuid': 'count',
                'gross_revenue_eur': 'mean',
                'booked_duration': 'mean'
            }).reset_index()
            
            # Rename columns
            hourly_metrics.columns = ['hour', 'total_bookings', 'avg_revenue', 'avg_duration']
            
            # Convert duration to minutes
            hourly_metrics['avg_duration'] = hourly_metrics['avg_duration'] / 60
            
            # Get average auction prices by hour
            if not self.auctions_df.empty and 'booking_uuid' in self.auctions_df.columns:
                # Merge auctions with bookings to get the hour
                auctions_with_hour = pd.merge(
                    self.auctions_df,
                    self.bookings_df[['booking_uuid', 'hour']],
                    on='booking_uuid',
                    how='left'
                )
                
                # Calculate average auction price by hour
                hourly_auctions = auctions_with_hour.groupby('hour')['auction_winning_price'].mean().reset_index()
                hourly_auctions.columns = ['hour', 'avg_auction_price']
                
                # Merge auction data with hourly metrics
                hourly_metrics = pd.merge(hourly_metrics, hourly_auctions, on='hour', how='left')
            else:
                hourly_metrics['avg_auction_price'] = 0
            
            return hourly_metrics
            
        except Exception as e:
            print(f"Error in analyze_peak_hours: {str(e)}")
            return None

    def analyze_cost_revenue(self):
        """Analyze total costs and revenue"""
        try:
            # Calculate total shift costs
            total_shift_cost = (self.shifts_df['shift_working_hours'] * 
                              self.shifts_df['hourly_rate_eur']).sum()
            
            # Calculate total revenue
            total_revenue = self.bookings_df['gross_revenue_eur'].sum()
            
            # Calculate profit/loss
            profit_loss = total_revenue - total_shift_cost
            
            # Calculate average revenue per booking
            avg_revenue_per_booking = (self.bookings_df['gross_revenue_eur'].mean()
                                     if len(self.bookings_df) > 0 else 0)
            
            # Calculate average cost per shift
            avg_cost_per_shift = ((self.shifts_df['shift_working_hours'] * 
                                 self.shifts_df['hourly_rate_eur']).mean()
                                if len(self.shifts_df) > 0 else 0)
            
            return {
                'total_shift_cost': total_shift_cost,
                'total_revenue': total_revenue,
                'profit_loss': profit_loss,
                'avg_revenue_per_booking': avg_revenue_per_booking,
                'avg_cost_per_shift': avg_cost_per_shift
            }
            
        except Exception as e:
            print(f"Error in analyze_cost_revenue: {str(e)}")
            return None

if __name__ == "__main__":
    analyzer = DispatchAnalyzer()
    analyzer.load_data(
        bookings_path='data/raw/disp_incoming_bookings.csv',
        shifts_path='data/raw/disp_pre_purchased_shifts.csv',
        auctions_path='data/raw/disp_historical_auction_data.csv'
    )
