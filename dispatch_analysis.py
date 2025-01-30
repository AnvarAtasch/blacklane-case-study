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
            self.bookings_df = pd.read_csv(bookings_path)
            self.shifts_df = pd.read_csv(shifts_path)
            self.auctions_df = pd.read_csv(auctions_path)
            
            print("\nData Loading Summary:")
            print(f"Bookings: {len(self.bookings_df)} records")
            print(f"Shifts: {len(self.shifts_df)} records")
            print(f"Auctions: {len(self.auctions_df)} records")
            
            # Convert datetime columns
            print("\nConverting datetime columns...")
            self.bookings_df['booked_start_at'] = pd.to_datetime(self.bookings_df['booked_start_at'])
            self.shifts_df['shift_date'] = pd.to_datetime(self.shifts_df['shift_date'])
            
            # Handle missing values in shifts before conversion
            print("\nHandling missing values in shifts...")
            if 'hourly_rate_eur' in self.shifts_df.columns:
                median_rate = self.shifts_df['hourly_rate_eur'].astype(float, errors='ignore').median()
                self.shifts_df['hourly_rate_eur'] = self.shifts_df['hourly_rate_eur'].fillna(median_rate)
            
            # Convert numeric columns with error handling
            print("\nConverting numeric columns...")
            numeric_cols = {
                'bookings_df': ['booked_duration', 'distance_km', 'gross_revenue_eur'],
                'shifts_df': ['shift_working_hours', 'hourly_rate_eur'],
                'auctions_df': ['auction_winning_price']
            }
            
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
    
    def analyze_supply(self):
        """Analyze supply patterns"""
        # Aggregate shifts by date
        daily_shifts = self.shifts_df.groupby('shift_date').agg({
            'shift_id': 'count',
            'shift_working_hours': 'sum',
            'hourly_rate_eur': 'mean'
        }).reset_index()
        
        return daily_shifts
    
    def analyze_demand(self):
        """Analyze demand patterns"""
        # Convert booked_start_at to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.bookings_df['booked_start_at']):
            self.bookings_df['booked_start_at'] = pd.to_datetime(self.bookings_df['booked_start_at'])
        
        # Extract date from datetime
        self.bookings_df['booking_date'] = self.bookings_df['booked_start_at'].dt.date
        
        # Ensure numeric types
        numeric_cols = ['gross_revenue_eur', 'booked_duration', 'booked_distance']
        for col in numeric_cols:
            if col in self.bookings_df.columns:
                self.bookings_df[col] = pd.to_numeric(self.bookings_df[col], errors='coerce')
        
        # Aggregate bookings by date with error handling
        try:
            daily_bookings = self.bookings_df.groupby('booking_date').agg({
                'booking_uuid': 'count',
                'gross_revenue_eur': 'sum',
                'booked_duration': 'mean',
                'booked_distance': 'mean'
            }).reset_index()
            
            # Fill any NaN values with 0
            daily_bookings = daily_bookings.fillna(0)
            
            return daily_bookings
            
        except Exception as e:
            print(f"Error in analyze_demand: {str(e)}")
            # Return a minimal dataframe if aggregation fails
            return pd.DataFrame({
                'booking_date': pd.date_range(
                    self.bookings_df['booked_start_at'].min(),
                    self.bookings_df['booked_start_at'].max(),
                    freq='D'
                ),
                'booking_uuid': 0,
                'gross_revenue_eur': 0,
                'booked_duration': 0,
                'booked_distance': 0
            })
    
    def analyze_auctions(self):
        """Analyze auction patterns"""
        # Calculate profit margin first
        self.auctions_df['profit_margin'] = (
            self.auctions_df['auction_winning_price'] - 
            self.auctions_df['auction_corridor_min_price']
        )
        
        # Calculate metrics separately to ensure consistent structure
        metrics = {
            'min_price': {
                'mean': self.auctions_df['auction_corridor_min_price'].mean(),
                'std': self.auctions_df['auction_corridor_min_price'].std()
            },
            'max_price': {
                'mean': self.auctions_df['auction_corridor_max_price'].mean(),
                'std': self.auctions_df['auction_corridor_max_price'].std()
            },
            'winning_price': {
                'mean': self.auctions_df['auction_winning_price'].mean(),
                'std': self.auctions_df['auction_winning_price'].std()
            },
            'profit_margin': {
                'mean': self.auctions_df['profit_margin'].mean(),
                'std': self.auctions_df['profit_margin'].std()
            }
        }
        
        # Convert to DataFrame with consistent structure
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        return metrics_df
    
    def analyze_auction_vs_shift_costs(self):
        """Analyze auction prices vs shift costs"""
        print("\nDEBUG: Starting auction vs shift cost analysis")
        
        # Join auctions with bookings to get timing information
        df_auction_enriched = (
            self.auctions_df
            .merge(self.bookings_df, on='booking_uuid', how='left')
        )
        print(f"Auction enriched shape: {df_auction_enriched.shape}")
        print("Auction enriched columns:", df_auction_enriched.columns.tolist())
        
        # Calculate average auction prices by hour
        auction_agg = (
            df_auction_enriched
            .assign(
                hour=pd.to_datetime(df_auction_enriched['booked_start_at']).dt.hour,
                day=pd.to_datetime(df_auction_enriched['booked_start_at']).dt.dayofweek
            )
            .groupby(['day', 'hour'])
            ['auction_winning_price']
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(['day', 'hour'])
        )
        print("\nAuction aggregation:")
        print(auction_agg.head())
        
        # Calculate average shift costs by hour
        shift_costs = (
            self.shifts_df
            .assign(
                hour=pd.to_datetime(self.shifts_df['shift_date']).dt.hour,
                day=pd.to_datetime(self.shifts_df['shift_date']).dt.dayofweek
            )
            .groupby(['day', 'hour'])
            ['hourly_rate_eur']
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(['day', 'hour'])
        )
        print("\nShift costs aggregation:")
        print(shift_costs.head())
        
        # Find peak hours (highest auction prices)
        peak_hours = (
            auction_agg
            .sort_values('mean', ascending=False)
            .head(5)
        )
        print("\nPeak hours by auction price:")
        print(peak_hours)
        
        return auction_agg, shift_costs, peak_hours
    
    def analyze_hours_distribution(self):
        """Analyze pre-purchased hours vs utilized hours by hour of day"""
        try:
            # Calculate utilized hours by hour of day from bookings
            utilized_hours = (
                self.bookings_df
                .assign(
                    hour_of_day=pd.to_datetime(self.bookings_df['booked_start_at']).dt.hour,
                    utilized_hours=lambda x: x['booked_duration'] / 3600  # Convert seconds to hours
                )
                .groupby('hour_of_day')
                ['utilized_hours']
                .sum()
                .reset_index()
            )
            
            # Calculate pre-purchased hours by hour of day from shifts
            prepurchased_hours = (
                self.shifts_df
                .assign(
                    hour_of_day=pd.to_datetime(self.shifts_df['shift_date']).dt.hour,
                    working_hours=lambda x: x['shift_working_hours']
                )
                .groupby('hour_of_day')
                ['working_hours']
                .sum()
                .reset_index()
            )
            
            # Ensure we have all hours (0-23)
            all_hours = pd.DataFrame({'hour_of_day': range(24)})
            
            utilized_hours = (
                all_hours
                .merge(utilized_hours, on='hour_of_day', how='left')
                .fillna(0)
            )
            
            prepurchased_hours = (
                all_hours
                .merge(prepurchased_hours, on='hour_of_day', how='left')
                .fillna(0)
            )
            
            # Calculate total metrics
            total_utilized = utilized_hours['utilized_hours'].sum()
            total_prepurchased = prepurchased_hours['working_hours'].sum()
            unused_hours = total_prepurchased - total_utilized
            utilization_rate = (total_utilized / total_prepurchased * 100) if total_prepurchased > 0 else 0
            
            # Print debug information
            print("\nDebug Information:")
            print(f"Total utilized hours (seconds): {self.bookings_df['booked_duration'].sum()}")
            print(f"Total utilized hours (hours): {total_utilized:.2f}")
            print(f"Total pre-purchased hours: {total_prepurchased:.2f}")
            print(f"Unused hours: {unused_hours:.2f}")
            print(f"Utilization rate: {utilization_rate:.2f}%")
            
            return {
                'utilized_hours': utilized_hours,
                'prepurchased_hours': prepurchased_hours,
                'total_utilized': total_utilized,
                'total_prepurchased': total_prepurchased,
                'unused_hours': unused_hours,
                'utilization_rate': utilization_rate
            }
            
        except Exception as e:
            print(f"Error in analyze_hours_distribution: {str(e)}")
            return None
    
    def analyze_supply_demand(self):
        """Analyze supply and demand patterns with detailed metrics"""
        try:
            # Calculate supply metrics
            supply_metrics = {}
            
            # Calculate total purchased hours
            total_hours = self.shifts_df['shift_working_hours'].sum()
            supply_metrics['total_hours'] = total_hours
            
            # Calculate hourly supply distribution
            hourly_supply = (
                self.shifts_df
                .assign(
                    hour=pd.to_datetime(self.shifts_df['shift_date']).dt.hour,
                    shift_cost=self.shifts_df['shift_working_hours'] * self.shifts_df['hourly_rate_eur']
                )
                .groupby('hour')
                .agg({
                    'shift_working_hours': 'sum',
                    'shift_cost': 'sum',
                    'hourly_rate_eur': ['mean', 'std']
                })
                .round(2)
                .reset_index()
            )
            
            # Calculate demand metrics
            demand_metrics = {}
            
            # Total bookings
            total_bookings = len(self.bookings_df)
            demand_metrics['total_bookings'] = total_bookings
            
            # Calculate hourly demand distribution
            hourly_demand = (
                self.bookings_df
                .assign(hour=pd.to_datetime(self.bookings_df['booked_start_at']).dt.hour)
                .groupby('hour')
                .agg({
                    'booking_uuid': 'count',
                    'gross_revenue_eur': ['sum', 'mean', 'std']
                })
                .round(2)
                .reset_index()
            )
            
            # Calculate auction metrics
            auction_metrics = {}
            
            # Ensure numeric type for auction_winning_price
            self.auctions_df['auction_winning_price'] = pd.to_numeric(self.auctions_df['auction_winning_price'], errors='coerce')
            self.auctions_df['auction_corridor_min_price'] = pd.to_numeric(self.auctions_df['auction_corridor_min_price'], errors='coerce')
            self.auctions_df['auction_corridor_max_price'] = pd.to_numeric(self.auctions_df['auction_corridor_max_price'], errors='coerce')
            
            # Overall auction statistics
            auction_metrics['avg_winning_price'] = self.auctions_df['auction_winning_price'].mean()
            auction_metrics['std_winning_price'] = self.auctions_df['auction_winning_price'].std()
            
            # Create a temporary DataFrame with hour information
            auctions_with_hour = (
                self.auctions_df
                .merge(
                    self.bookings_df[['booking_uuid', 'booked_start_at']],
                    on='booking_uuid',
                    how='left'
                )
            )
            auctions_with_hour['hour'] = pd.to_datetime(auctions_with_hour['booked_start_at']).dt.hour
            
            # Hourly auction statistics
            hourly_auctions = (
                auctions_with_hour
                .groupby('hour')
                .agg({
                    'auction_winning_price': ['mean', 'std', 'min', 'max'],
                    'auction_corridor_min_price': 'mean',
                    'auction_corridor_max_price': 'mean'
                })
                .round(2)
                .reset_index()
            )
            
            # Calculate utilization and efficiency metrics
            efficiency_metrics = {}
            
            # Hours per booking
            efficiency_metrics['hours_per_booking'] = total_hours / total_bookings if total_bookings > 0 else 0
            
            # Average revenue per hour
            total_revenue = self.bookings_df['gross_revenue_eur'].sum()
            efficiency_metrics['revenue_per_hour'] = total_revenue / total_hours if total_hours > 0 else 0
            
            # Calculate idle hours
            total_booking_hours = total_bookings * efficiency_metrics['hours_per_booking']
            efficiency_metrics['idle_hours'] = max(0, total_hours - total_booking_hours)
            efficiency_metrics['utilization_rate'] = (total_booking_hours / total_hours * 100) if total_hours > 0 else 0
            
            return {
                'supply_metrics': supply_metrics,
                'demand_metrics': demand_metrics,
                'auction_metrics': auction_metrics,
                'efficiency_metrics': efficiency_metrics,
                'hourly_supply': hourly_supply,
                'hourly_demand': hourly_demand,
                'hourly_auctions': hourly_auctions
            }
        except Exception as e:
            print(f"Error in analyze_supply_demand: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_shift_utilization(self):
        """Analyze shift utilization by comparing scheduled vs actual hours"""
        print("\nDEBUG: Starting shift utilization analysis")
        
        # Print initial data stats
        print(f"Total bookings: {len(self.bookings_df)}")
        print(f"Total revenue before grouping: €{self.bookings_df['gross_revenue_eur'].sum():.2f}")
        print(f"Total shifts: {len(self.shifts_df)}")
        print(f"Average hourly rate: €{self.shifts_df['hourly_rate_eur'].mean():.2f}")
        
        # Calculate scheduled hours per chauffeur
        scheduled_hours = (
            self.shifts_df
            .groupby('chauffeur_uuid')
            .agg({
                'shift_working_hours': 'sum',
                'hourly_rate_eur': 'mean'
            })
            .round(2)
        )
        
        print("\nScheduled hours stats:")
        print(f"Total scheduled hours: {scheduled_hours['shift_working_hours'].sum():.2f}")
        
        # Calculate actual hours worked from bookings (convert minutes to hours)
        actual_hours = (
            self.bookings_df
            .groupby('chauffeur_uuid')
            .agg({
                'booked_duration': lambda x: (x.sum() / 60),  # Convert minutes to hours
                'gross_revenue_eur': 'sum'
            })
            .round(2)
        )
        
        print("\nActual hours stats:")
        print(f"Total actual hours: {actual_hours['booked_duration'].sum():.2f}")
        print(f"Total revenue after grouping: €{actual_hours['gross_revenue_eur'].sum():.2f}")
        
        # Combine scheduled and actual hours
        utilization = (
            scheduled_hours
            .join(actual_hours, how='left')
            .fillna(0)
        )
        
        # Calculate utilization metrics
        utilization['unused_hours'] = utilization['shift_working_hours'] - utilization['booked_duration']
        utilization['utilization_rate'] = (utilization['booked_duration'] / utilization['shift_working_hours'] * 100).round(2)
        utilization['cost_per_hour'] = utilization['hourly_rate_eur']  # Already per hour
        utilization['revenue_per_hour'] = (utilization['gross_revenue_eur'] / utilization['shift_working_hours']).round(2)
        
        # Calculate total cost (hourly_rate * scheduled_hours)
        total_cost = (utilization['hourly_rate_eur'] * utilization['shift_working_hours']).sum()
        print(f"\nTotal cost calculation: €{total_cost:.2f}")
        
        # Calculate total revenue (sum of all gross revenue)
        total_revenue = utilization['gross_revenue_eur'].sum()
        print(f"Total revenue calculation: €{total_revenue:.2f}")
        
        # Calculate hourly utilization
        hourly_utilization = (
            self.bookings_df
            .assign(
                hour=pd.to_datetime(self.bookings_df['booked_start_at']).dt.hour,
                hours_used=self.bookings_df['booked_duration'] / 60  # Convert minutes to hours
            )
            .groupby('hour')
            .agg({
                'hours_used': ['sum', 'count'],
                'gross_revenue_eur': 'sum'
            })
            .round(2)
        )
        
        # Calculate overall metrics
        total_metrics = {
            'total_scheduled_hours': utilization['shift_working_hours'].sum(),
            'total_actual_hours': utilization['booked_duration'].sum(),
            'total_unused_hours': utilization['unused_hours'].sum(),
            'average_utilization_rate': (utilization['booked_duration'].sum() / utilization['shift_working_hours'].sum() * 100).round(2),
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'total_chauffeurs': len(utilization)
        }
        
        print("\nFinal metrics:")
        for key, value in total_metrics.items():
            print(f"{key}: {value}")
        
        # Filter out rows with invalid utilization rates (> 100% or <= 0%)
        valid_utilization = utilization[
            (utilization['utilization_rate'] > 0) & 
            (utilization['utilization_rate'] <= 100)
        ]
        
        # Calculate top and bottom performers from valid utilization data
        top_performers = (
            valid_utilization
            .sort_values('utilization_rate', ascending=False)
            .head(5)
        )
        
        bottom_performers = (
            valid_utilization
            .sort_values('utilization_rate', ascending=True)
            .head(5)
        )
        
        return {
            'utilization_data': utilization,
            'hourly_utilization': hourly_utilization,
            'total_metrics': total_metrics,
            'top_performers': top_performers,
            'bottom_performers': bottom_performers
        }
    
    def analyze_peak_hours(self):
        """Analyze peak hours with auction prices and shift costs"""
        try:
            # Create copies and ensure datetime columns
            bookings = self.bookings_df.copy()
            auctions = self.auctions_df.copy()
            shifts = self.shifts_df.copy()
            
            # Convert datetime columns
            bookings['booked_start_at'] = pd.to_datetime(bookings['booked_start_at'])
            
            # Create time ranges for peak hours analysis
            peak_hours = [
                ('09:00 - 09:59', '09:00', '09:59'),
                ('13:00 - 13:59', '13:00', '13:59'),
                ('14:00 - 14:59', '14:00', '14:59'),
                ('15:00 - 15:59', '15:00', '15:59'),
                ('18:00 - 18:59', '18:00', '18:59'),
                ('21:00 - 21:59', '21:00', '21:59')
            ]
            
            peak_analysis = []
            for time_range, start_time, end_time in peak_hours:
                # Filter bookings for the time range
                hour = int(start_time.split(':')[0])
                hour_bookings = bookings[bookings['booked_start_at'].dt.hour == hour]
                booking_uuids = hour_bookings['booking_uuid'].tolist()
                
                # Get corresponding auction prices
                hour_auctions = auctions[auctions['booking_uuid'].isin(booking_uuids)]
                avg_auction_price = hour_auctions['auction_winning_price'].astype(float).mean()
                
                # Get shift costs for the same hour
                hour_shifts = shifts[shifts['shift_date'].dt.hour == hour]
                avg_shift_cost = hour_shifts['hourly_rate_eur'].astype(float).mean()
                
                peak_analysis.append({
                    'time_range': time_range,
                    'num_bookings': len(hour_bookings),
                    'avg_auction_price': avg_auction_price,
                    'avg_shift_cost': avg_shift_cost,
                    'price_difference': avg_auction_price - avg_shift_cost if pd.notna(avg_auction_price) and pd.notna(avg_shift_cost) else 0
                })
            
            # Convert to DataFrame for easier handling
            peak_df = pd.DataFrame(peak_analysis)
            
            # Calculate hourly metrics for the graph
            hourly_metrics = []
            for hour in range(24):
                hour_bookings = bookings[bookings['booked_start_at'].dt.hour == hour]
                booking_uuids = hour_bookings['booking_uuid'].tolist()
                
                hour_auctions = auctions[auctions['booking_uuid'].isin(booking_uuids)]
                avg_auction = hour_auctions['auction_winning_price'].astype(float).mean()
                
                hour_shifts = shifts[shifts['shift_date'].dt.hour == hour]
                avg_shift = hour_shifts['hourly_rate_eur'].astype(float).mean()
                
                hourly_metrics.append({
                    'hour': hour,
                    'avg_auction_price': avg_auction if pd.notna(avg_auction) else 0,
                    'avg_shift_cost': avg_shift if pd.notna(avg_shift) else 0
                })
            
            hourly_df = pd.DataFrame(hourly_metrics)
            
            return {
                'peak_hours_analysis': peak_df,
                'hourly_metrics': hourly_df
            }
            
        except Exception as e:
            print(f"Error in analyze_peak_hours: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_patterns(self):
        """Create visualizations for supply-demand patterns"""
        # Time series of supply and demand
        daily_supply = self.analyze_supply()
        daily_demand = self.analyze_demand()
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_supply['shift_date'], daily_supply['shift_working_hours'], 
                label='Supply (Hours)')
        plt.plot(daily_demand['booking_date'], daily_demand['booking_uuid'], 
                label='Demand (Bookings)')
        plt.title('Supply vs Demand Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig('supply_demand_pattern.png')
        plt.close()

def main():
    """Main function to run the analysis"""
    # Initialize analyzer
    analyzer = DispatchAnalyzer()
    
    # Load data
    bookings_path = 'data/raw/disp_incoming_bookings.csv'
    shifts_path = 'data/raw/disp_pre_purchased_shifts.csv'
    auctions_path = 'data/raw/disp_historical_auction_data.csv'
    analyzer.load_data(bookings_path, shifts_path, auctions_path)
    
    # Calculate KPIs
    kpis = analyzer.calculate_kpis()
    
    # Perform auction vs shift cost analysis
    auction_agg, shift_costs, peak_hours = analyzer.analyze_auction_vs_shift_costs()
    
    # Print aggregated results
    print("\nAuction Prices by Time Period:")
    print(auction_agg)
    print("\nShift Costs by Day:")
    print(shift_costs)
    print("\nPeak Hours Analysis:")
    print(peak_hours)
    
    # Generate visualizations
    analyzer.visualize_patterns()
    
    # Create and save visualization
    fig = analyzer.plot_price_comparison(auction_agg, shift_costs)
    fig.savefig('auction_vs_shift_comparison.png', bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\nSupply Analysis Summary:")
    print(analyzer.analyze_supply().describe())
    
    print("\nDemand Analysis Summary:")
    print(analyzer.analyze_demand().describe())
    
    print("\nKPI Summary:")
    for kpi, value in kpis.items():
        print(f"{kpi}: {value:.2f}")

    # Perform supply and demand analysis
    supply_demand_results = analyzer.analyze_supply_demand()
    print("\nSupply and Demand Analysis Results:")
    for key, value in supply_demand_results.items():
        if isinstance(value, pd.DataFrame):
            print(f"\n{key}:")
            print(value)
        else:
            print(f"\n{key}: {value}")

    # Perform shift utilization analysis
    shift_utilization_results = analyzer.analyze_shift_utilization()
    print("\nShift Utilization Analysis Results:")
    for key, value in shift_utilization_results.items():
        if isinstance(value, pd.DataFrame):
            print(f"\n{key}:")
            print(value)
        else:
            print(f"\n{key}: {value}")

if __name__ == "__main__":
    main()
