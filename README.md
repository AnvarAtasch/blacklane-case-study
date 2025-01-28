# Blacklane Dispatching Optimization

This project implements a data-driven dispatching optimization system for Blacklane's two-sided marketplace between guests and chauffeur service providers. The system analyzes historical data and makes intelligent decisions about whether to use pre-purchased shift capacity or leverage the auction-based marketplace for incoming bookings.

## Project Structure

```
blacklane-case-study/
├── data/                      # Data directory
│   ├── raw/                   # Original CSV files
│   └── processed/             # Processed data files
├── src/                       # Source code
│   ├── analysis/             # Analysis modules
│   └── optimization/         # Optimization modules
├── notebooks/                # Jupyter notebooks for analysis
└── requirements.txt          # Python dependencies
```

## Data Sources

The project uses three main data sources:

1. **Pre-purchased Shifts** (`disp_pre_purchased_shifts.csv`)
   - Contains information about pre-purchased chauffeur shifts
   - Fields: shift_id, chauffeur_uuid, shift_date, shift_working_hours, hourly_rate_eur

2. **Incoming Bookings** (`disp_incoming_bookings.csv`)
   - Contains details about guest bookings
   - Fields: booking_uuid, booker_uuid, chauffeur_uuid, booked_start_at, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, booked_duration, booked_distance, gross_revenue_eur

3. **Historical Auction Data** (`disp_historical_auction_data.csv`)
   - Contains historical auction outcomes
   - Fields: booking_uuid, auction_corridor_min_price, auction_corridor_max_price, auction_winning_price

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the CSV files in the `data/raw/` directory.

## Usage

1. Run data analysis:
```bash
python src/analysis/dispatch_analysis.py
```

2. Run optimization model:
```bash
python src/optimization/dispatch_optimizer.py
```

## Key Features

1. **Data Analysis**
   - Supply and demand pattern analysis
   - Auction behavior analysis
   - KPI calculation and monitoring
   - Data visualization

2. **Optimization Engine**
   - Machine learning-based decision system
   - Real-time decision making
   - Profit optimization
   - Risk assessment

## Performance Metrics

The system tracks several key performance indicators:
- Supply utilization rate
- Revenue per hour
- Cost per hour
- Auction success rate
- Decision confidence scores
- Expected profit margins

## Technical Implementation

- Python 3.8+
- pandas for data processing
- scikit-learn for machine learning
- matplotlib/seaborn for visualization
- Jupyter notebooks for interactive analysis

## Author

Anvar Atash
