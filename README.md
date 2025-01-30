# Blacklane Case Study

## Overview
This repository contains a comprehensive analysis of taxi service data, focusing on cost comparison between pre-purchased shifts and auction participation.

Optimizing Dispatching: Balancing Supply, Demand,
and Marketplace Dynamics
Context Setting
Blacklane offers a two-sided marketplace between guests and logistics service
providers represented by chauffeurs. There are 3 parties involved in this marketplace:
Guests, chauffeur partners and Blacklane’s platform.
In our Data Products domain, we tackle many issues related to supply and demand.
Imagine that in our markets we have a set of pre-purchased chauffeur shifts (supply) for
the upcoming weeks. This capacity is purchased by Blacklane upfront at a
pre-negotiated price per chauffeur hour. We also have a set of incoming guest bookings
(demand) which are usually placed on our platform with a booking lead time of 1-14
days. Each booking has some pick-up and drop-off location.
For each incoming booking, we can decide if we use some of our pre-purchased shift
capacity for it or, alternatively, send the booking to our reverse auction-based
marketplace. In this marketplace, chauffeur companies compete with each other and
can claim the booking at an auction price. The auction starts at a preset minimum price
and then goes up in increments until the ride gets accepted at a certain price point by
some chauffeur.
Our dispatching data product decides for each incoming booking whether it will be used
to fill some of the pre-purchased chauffeur shift blocks or if it gets sent to the auction
marketplace.
Task
As the PM of our dispatching product, please propose a solution to the aforementioned
problem space. Please consider the following pointers in your solution:
1. Objective: Analyze the data we provided to you and identify opportunities. What
do you think is our objective in this problem space? What do we want to achieve
when designing our dispatching system? Explain the business objective and
describe the underlying marketplace dynamics at play.
2. 3. Solution Design: Please describe how your technical solution would look like.
Describe the behavior of an optimized dispatching system. How would you build
the decision-making component of this product?
Performance Metrics: Define key performance indicators (KPIs) that you would
use to measure the success of your solution.

## Dashboards
1. Cost Comparison Dashboard (`src/ui/cost_comparison.py`)
   - Analyzes cost differences between pre-purchased shifts and auction prices
   - Visualizes cost distribution and trends

2. Geographical Analysis Dashboard (`src/ui/geographical_analysis.py`)
   - Maps pickup and dropoff locations
   - Analyzes price variations by area
   - Identifies price hotspots

3. Time Analysis Dashboard (`src/ui/time_analysis.py`)
   - Examines price variations by time of day and day of week
   - Analyzes impact of weather and events on prices
   - Provides correlation analysis

## Data
Sample data files are located in `data/raw/`:
- `bookings.csv`: Booking information
- `auction_data.csv`: Auction results
- `pre_purchased_shifts.csv`: Pre-purchased shift data
- `weather_data.csv`: Weather conditions
- `events_data.csv`: Special events data

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run dashboards:
```bash
streamlit run src/ui/cost_comparison.py
streamlit run src/ui/geographical_analysis.py
streamlit run src/ui/time_analysis.py
```

## Analysis Results
The analysis covers three main hypotheses:
1. Cost Effectiveness of Pre-purchased Shifts vs Auctions
2. Impact of Booking Duration on Cost Differences
3. Geographical Influence on Auction Prices
4. Time and Event Impact on Auction Dynamics
