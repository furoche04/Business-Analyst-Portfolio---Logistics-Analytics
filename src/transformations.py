"""
Business Logic Transformations Module
====================================

This module implements business-specific data transformations for logistics analytics.
It demonstrates Business Analyst capabilities in:
- Business rule implementation and automation
- KPI calculation and metric derivation
- Data enrichment through reference lookups
- Performance analysis and benchmarking

Author: Karol Nosarzewski
Purpose: Business Analyst Portfolio - Business Logic Implementation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Configure module-level logger
logger = logging.getLogger(__name__)

# Business configuration constants
FINANCIAL_CATEGORIES = {
    'charge_columns': [
        "('OS_AMT_USD', 'BrokerageCharges')",
        "('OS_AMT_USD', 'TotalFreightCharges')",
        "('OS_AMT_USD', 'FuelSurcharge')",
        "('OS_AMT_USD', 'SecuritySurcharge')",
        "('OS_AMT_USD', 'Duty & Tax')",
        "('OS_AMT_USD', 'Other')"
    ],
    'high_cost_threshold_percentile': 90,
    'cost_variance_threshold': 0.25  # 25% variance threshold
}

PERFORMANCE_METRICS = {
    'on_time_tolerance_days': 1,
    'delay_categories': {
        'minor': (1, 3),      # 1-3 days late
        'moderate': (4, 7),   # 4-7 days late  
        'severe': (8, float('inf'))  # 8+ days late
    }
}

def calculate_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive financial KPIs for logistics performance analysis.
    
    Args:
        df: DataFrame with financial charge columns
        
    Returns:
        DataFrame: Enhanced with financial KPIs
        
    Business Metrics Added:
        - TotalCharges: Sum of all charge categories
        - ChargeBreakdown: Percentage distribution by category  
        - CostEfficiencyRating: Relative cost performance
        - HighCostFlag: Identifies premium shipments
    """
    logger.info("Calculating financial performance metrics...")
    
    enhanced_df = df.copy()
    charge_columns = FINANCIAL_CATEGORIES['charge_columns']
    
    # Verify required columns exist
    existing_charge_cols = [col for col in charge_columns if col in enhanced_df.columns]
    missing_cols = [col for col in charge_columns if col not in enhanced_df.columns]
    
    if missing_cols:
        logger.warning(f"Missing charge columns: {missing_cols}")
    
    if not existing_charge_cols:
        logger.error("No financial charge columns found for calculation")
        return enhanced_df
    
    # Calculate total charges
    enhanced_df['OS_AMT_USD'] = enhanced_df[existing_charge_cols].sum(axis=1, skipna=True)
    logger.info(f"Calculated total charges for {len(enhanced_df)} records")
    
    # Calculate charge breakdown percentages
    for col in existing_charge_cols:
        # Extract category name for cleaner column naming
        category_name = col.split("', '")[1].rstrip("')")
        percentage_col = f'{category_name}_Percentage'
        
        enhanced_df[percentage_col] = (
            (enhanced_df[col] / enhanced_df['OS_AMT_USD'] * 100)
            .round(2)
            .fillna(0)
        )
    
    # Identify high-cost shipments for special handling
    if len(enhanced_df) > 0:
        cost_threshold = enhanced_df['OS_AMT_USD'].quantile(
            FINANCIAL_CATEGORIES['high_cost_threshold_percentile'] / 100
        )
        enhanced_df['HighCostFlag'] = enhanced_df['OS_AMT_USD'] >= cost_threshold
        
        # Calculate cost efficiency rating (relative to median)
        median_cost = enhanced_df['OS_AMT_USD'].median()
        enhanced_df['CostEfficiencyRating'] = (
            enhanced_df['OS_AMT_USD'] / median_cost
        ).round(3)
        
        high_cost_count = enhanced_df['HighCostFlag'].sum()
        logger.info(f"Identified {high_cost_count} high-cost shipments (threshold: ${cost_threshold:.2f})")
    
    return enhanced_df

def enrich_location_data(df: pd.DataFrame, city_codes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich shipment data with standardized location information.
    
    Args:
        df: Main DataFrame with location codes
        city_codes_df: Reference DataFrame with city mappings
        
    Returns:
        DataFrame: Enhanced with location details
        
    Business Value:
        - Standardizes location naming for consistent reporting
        - Enables geographic analysis and routing optimization
        - Supports regional performance comparisons
    """
    logger.info("Enriching location data with standardized mappings...")
    
    enriched_df = df.copy()
    
    if 'Unlocode' not in city_codes_df.columns or 'City' not in city_codes_df.columns:
        logger.error("City codes reference data missing required columns (Unlocode, City)")
        return enriched_df
    
    # Create city mapping dictionary
    city_mapping = dict(zip(city_codes_df['Unlocode'], city_codes_df['City']))
    logger.info(f"Created city mapping for {len(city_mapping)} locations")
    
    # Apply location enrichment if debtor city column exists
    location_columns = ['Debtor City', 'Shipper City', 'Consignee City']
    
    for col in location_columns:
        if col in enriched_df.columns:
            enriched_col_name = f'{col} Name'
            enriched_df[enriched_col_name] = enriched_df[col].map(city_mapping)
            
            # Track mapping success rate
            mapped_count = enriched_df[enriched_col_name].notna().sum()
            total_count = len(enriched_df)
            mapping_rate = (mapped_count / total_count * 100) if total_count > 0 else 0
            
            logger.info(f"Mapped {col}: {mapped_count}/{total_count} ({mapping_rate:.1f}%)")
    
    return enriched_df

def calculate_transit_performance(df: pd.DataFrame, transit_time_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate transit time performance metrics against business standards.
    
    Args:
        df: Main DataFrame with shipment dates
        transit_time_df: Reference DataFrame with standard transit times
        
    Returns:
        DataFrame: Enhanced with performance metrics
        
    Business Metrics:
        - TransitTimeActual: Calculated actual transit days
        - TransitTimeStandard: Expected transit time from standards
        - PerformanceVariance: Actual vs standard comparison
        - OnTimeDeliveryFlag: Boolean performance indicator
    """
    logger.info("Calculating transit time performance metrics...")
    
    performance_df = df.copy()
    
    # Look for date columns
    date_columns = {
        'ship_date': ['First ATD'],
        'delivery_date': ['Last ATA']
    }
    
    ship_date_col = None
    delivery_date_col = None
    
    # Find available date columns
    for col_type, possible_names in date_columns.items():
        for col_name in possible_names:
            if col_name in performance_df.columns:
                if col_type == 'ship_date':
                    ship_date_col = col_name
                else:
                    delivery_date_col = col_name
                break
    
    if not ship_date_col or not delivery_date_col:
        logger.warning(f"Required date columns not found. Available: {list(performance_df.columns)}")
        return performance_df

    # Create 'port key' for merging
    performance_df['port key'] = performance_df['Origin'].astype(str) + performance_df['Destination'].astype(str)

    # Normalize mode for mapping
    performance_df['Mode'] = performance_df['Mode'].replace({'LSE': 'AIR', 'FCL': 'FCL'})

    # Merge with transit_time_df on port key and mode
    performance_df = performance_df.merge(
        transit_time_df,
        on=['port key', 'Mode'],
        how='left',
        suffixes=('', '_tt')
    )
    
    # Calculate actual transit time
    try:
        performance_df[ship_date_col] = pd.to_datetime(performance_df[ship_date_col])
        performance_df[delivery_date_col] = pd.to_datetime(performance_df[delivery_date_col])
        
        performance_df['TransitTimeActual'] = (
            performance_df[delivery_date_col] - performance_df[ship_date_col]
        ).dt.days
        
        # Create performance categories
        performance_df['DeliveryStatus'] = 'On Time'

        # Calculate delay delta: positive = early, negative = late
        performance_df['DelayDelta'] = performance_df['DTD'] - performance_df['TransitTimeActual']

        for category, (min_days, max_days) in PERFORMANCE_METRICS['delay_categories'].items():
            delay_mask = (
                (performance_df['DelayDelta'] >= min_days) & 
                (performance_df['DelayDelta'] < max_days if max_days != float('inf') else True)
            )
            performance_df.loc[delay_mask, 'DeliveryStatus'] = f'Delay - {category.title()}'
        
        # Calculate on-time delivery flag using DelayDelta
        tolerance = PERFORMANCE_METRICS['on_time_tolerance_days']
        performance_df['OnTimeDeliveryFlag'] = performance_df['DelayDelta'] >= -tolerance
        
        try:
            # Performance summary
            on_time_count = performance_df['OnTimeDeliveryFlag'].sum()
            total_count = len(performance_df)
            on_time_rate = (on_time_count / total_count * 100) if total_count > 0 else 0

            logger.info(f"Transit performance calculated: {on_time_count}/{total_count} on-time ({on_time_rate:.1f}%)")

        except Exception as e:
            logger.error(f"Error calculating transit performance: {e}")

        return performance_df
    except Exception as e:
        logger.error(f"Error in calculating transit time performance: {e}")
        return performance_df

def categorize_delay_reasons(df: pd.DataFrame, reason_codes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize and analyze delay reasons for operational insights.
    
    Args:
        df: Main DataFrame with delay reason codes
        reason_codes_df: Reference DataFrame with reason code mappings
        
    Returns:
        DataFrame: Enhanced with delay categorization
        
    Business Value:
        - Identifies root causes of delivery delays
        - Supports process improvement initiatives  
        - Enables targeted operational interventions
    """
    logger.info("Categorizing delay reasons for operational analysis...")
    
    categorized_df = df.copy()
    
    # Verify reason codes reference data
    required_cols = ['Reason Code', 'Control description', 'TMS Code']
    missing_cols = [col for col in required_cols if col not in reason_codes_df.columns]
    
    if missing_cols:
        logger.warning(f"Reason codes reference missing columns: {missing_cols}")
        # Use available columns
        available_cols = [col for col in required_cols if col in reason_codes_df.columns]
        if 'Reason Code' not in available_cols:
            logger.error("Cannot categorize without Reason Code column")
            return categorized_df
    
    # Create reason code mappings
    if 'Control description' in reason_codes_df.columns:
        description_mapping = dict(zip(reason_codes_df['Reason Code'], reason_codes_df['Control description']))
        categorized_df['DelayReasonDescription'] = categorized_df.get('DelayReasonCode', pd.Series()).map(description_mapping)
    
    if 'Category' in reason_codes_df.columns:
        category_mapping = dict(zip(reason_codes_df['Reason Code'], reason_codes_df['Category']))
        categorized_df['TMS Code'] = categorized_df.get('DelayReasonCode', pd.Series()).map(category_mapping)
        
        # Analyze delay category distribution
        if 'TMS Code' in categorized_df.columns:
            category_counts = categorized_df['TMS Code'].value_counts()
            logger.info(f"Delay category distribution: {category_counts.to_dict()}")
    
    return categorized_df

def apply_business_transformations(
    df: pd.DataFrame, 
    tt_df: pd.DataFrame, 
    rc_df: pd.DataFrame, 
    cc_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply comprehensive business transformations for logistics analytics.
    
    Args:
        df: Main shipment DataFrame
        tt_df: Transit time standards reference
        rc_df: Reason codes reference  
        cc_df: City codes reference
        
    Returns:
        DataFrame: Fully transformed with business KPIs and enrichments
        
    Process:
        1. Calculate financial metrics and cost analysis
        2. Enrich location data with standardized names
        3. Calculate transit time performance vs standards
        4. Categorize delay reasons for root cause analysis
        5. Generate executive summary metrics
    """
    logger.info("=" * 50)
    logger.info("APPLYING BUSINESS TRANSFORMATIONS")
    logger.info("=" * 50)
    
    try:
        transformed_df = df.copy()
        
        # Step 1: Financial Analysis
        logger.info("Step 1: Calculating financial performance metrics")
        transformed_df = calculate_financial_metrics(transformed_df)
        
        # Step 2: Location Enrichment  
        logger.info("Step 2: Enriching location data")
        transformed_df = enrich_location_data(transformed_df, cc_df)
        
        # Step 3: Transit Performance Analysis
        logger.info("Step 3: Calculating transit time performance")
        transformed_df = calculate_transit_performance(transformed_df, tt_df)
        
        # Step 4: Delay Reason Analysis
        logger.info("Step 4: Categorizing delay reasons")
        transformed_df = categorize_delay_reasons(transformed_df, rc_df)
        
        # Step 5: Generate business summary metrics
        logger.info("Step 5: Generating business summary")
        summary_metrics = generate_transformation_summary(transformed_df)
        
        logger.info("Business transformation completed successfully")
        logger.info(f"Enhanced dataset: {len(transformed_df)} records with {len(transformed_df.columns)} dimensions")
        
        return transformed_df
        
    except Exception as e:
        logger.error(f"Error in business transformations: {e}")
        raise

def generate_transformation_summary(df: pd.DataFrame) -> Dict:
    """
    Generate executive summary of transformation results.
    
    Args:
        df: Transformed DataFrame
        
    Returns:
        Dict: Key business metrics and insights
        
    Metrics:
        - total_shipments: Total number of shipments in the dataset.
        - transformation_date: Timestamp when the summary was generated.
        - financial_metrics: Dictionary with total revenue, average shipment cost, and count of high-cost shipments.
        - performance_metrics: Dictionary including:
            - on_time_deliveries: Number of shipments delivered on time (sum of 'OnTimeDeliveryFlag' column).
            - on_time_percentage: Percentage of shipments delivered on time.
        - data_quality: Dictionary with completeness percentage and count of enriched columns.
    """
    summary = {
        'total_shipments': len(df),
        'transformation_date': datetime.now().isoformat(),
        'financial_metrics': {},
        'performance_metrics': {},
        'data_quality': {}
    }
    
    # Financial summary
    if 'TotalCharges' in df.columns:
        summary['financial_metrics'] = {
            'total_revenue': df['OS_AMT_USD'].sum(),
            'average_shipment_cost': df['OS_AMT_USD'].mean(),
            # 'high_cost_shipments': df['HighCostFlag'].sum() if 'HighCostFlag' in df.columns else 0
        }
    
    return summary

# Legacy function for backward compatibility
def test_transform(df: pd.DataFrame, tt_df: pd.DataFrame, rc_df: pd.DataFrame, cc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy transformation function maintained for backward compatibility.
    Redirects to apply_business_transformations with enhanced functionality.
    """
    logger.info("Using legacy transformation function - redirecting to enhanced version")
    return apply_business_transformations(df, tt_df, rc_df, cc_df)
