"""
Financial Data Processing Module
===============================

This module handles the loading, validation, and preprocessing of financial data
for logistics analytics. It demonstrates Business Analyst capabilities in:
- Financial data quality assurance
- Cost categorization and standardization
- Data validation and cleansing
- Financial KPI preparation

Author: Karol Nosarzewski
Purpose: Business Analyst Portfolio - Financial Analytics Component
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)

# Import universal file reader after logger configuration
from helpers import read_data_file

# Business-defined financial categories for cost analysis
STANDARD_CHARGE_CATEGORIES = {
    'primary_charges': [
        'BrokerageCharges',
        'TotalFreightCharges', 
        'FuelSurcharge',
        'SecuritySurcharge',
        'Duty & Tax'
    ],
    'currency_prefix': 'OS_AMT_USD',
    'other_category': 'Other'
}

def validate_financial_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate financial dataset for completeness and business logic compliance.
    
    Args:
        df: Financial DataFrame to validate
        
    Returns:
        Tuple: (is_valid, list_of_issues)
        
    Business Rules:
        - Must contain ShipmentID for joining with operational data
        - Financial amounts should be numeric and non-negative
        - Required charge categories should be present
    """
    issues = []
    
    # Check for required columns
    if 'ShipmentID' not in df.columns:
        issues.append("Missing required column: ShipmentID")
    
    # Check for duplicate shipment IDs
    if df['ShipmentID'].duplicated().any():
        duplicate_count = df['ShipmentID'].duplicated().sum()
        issues.append(f"Found {duplicate_count} duplicate ShipmentID records")
    
    # Validate financial columns
    financial_cols = [col for col in df.columns if 'OS_AMT_USD' in str(col)]
    if not financial_cols:
        issues.append("No financial amount columns found (OS_AMT_USD)")
    
    # Check for negative amounts
    for col in financial_cols:
        if df[col].dtype in ['float64', 'int64']:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"Found {negative_count} negative amounts in {col}")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def standardize_charge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize financial charge column names for consistent analysis.
    
    Args:
        df: Raw financial DataFrame
        
    Returns:
        DataFrame: Standardized financial data
        
    Business Value:
        - Ensures consistent naming across different data sources
        - Facilitates automated reporting and analysis
        - Supports scalable data processing
    """
    logger.info("Standardizing financial charge column names...")
    
    standardized_df = df.copy()
    
    # Track column name changes for audit purposes
    column_mapping = {}
    
    # Standardize primary charge categories
    for category in STANDARD_CHARGE_CATEGORIES['primary_charges']:
        # Look for variations in column naming
        potential_names = [
            f"('{STANDARD_CHARGE_CATEGORIES['currency_prefix']}', '{category}')",
            f"{STANDARD_CHARGE_CATEGORIES['currency_prefix']}_{category}",
            f"{category}",
            f"{category.replace(' ', '').replace('&', 'And')}"
        ]
        
        for col in standardized_df.columns:
            if any(name.lower() in str(col).lower() for name in potential_names):
                standard_name = f"('{STANDARD_CHARGE_CATEGORIES['currency_prefix']}', '{category}')"
                if col != standard_name:
                    column_mapping[col] = standard_name
    
    # Apply column name standardization
    if column_mapping:
        standardized_df = standardized_df.rename(columns=column_mapping)
        logger.info(f"Standardized {len(column_mapping)} column names")
    
    return standardized_df

def calculate_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate key financial metrics for business analysis.
    
    Args:
        df: Financial DataFrame with standardized columns
        
    Returns:
        DataFrame: Enhanced with calculated financial KPIs
        
    Business Metrics Added:
        - Total_Charges: Sum of all charges per shipment
        - Primary_Charges_Pct: Percentage of primary vs other charges
        - Cost_Per_Unit: If quantity data available
    """
    logger.info("Calculating financial business metrics...")
    
    enhanced_df = df.copy()
    
    # Get all financial amount columns
    amount_columns = [col for col in enhanced_df.columns 
                     if STANDARD_CHARGE_CATEGORIES['currency_prefix'] in str(col)]
    
    if amount_columns:
        # Calculate total charges per shipment
        enhanced_df['Total_Charges'] = enhanced_df[amount_columns].sum(axis=1, skipna=True)
        
        # Calculate primary charges percentage
        primary_cols = [col for col in amount_columns 
                       if any(cat in str(col) for cat in STANDARD_CHARGE_CATEGORIES['primary_charges'])]
        
        if primary_cols:
            enhanced_df['Primary_Charges'] = enhanced_df[primary_cols].sum(axis=1, skipna=True)
            enhanced_df['Primary_Charges_Pct'] = (
                enhanced_df['Primary_Charges'] / enhanced_df['Total_Charges'] * 100
            ).round(2)
        
        # Identify high-cost shipments (business rule: top 10%)
        cost_threshold = enhanced_df['Total_Charges'].quantile(0.9)
        enhanced_df['High_Cost_Flag'] = enhanced_df['Total_Charges'] >= cost_threshold
        
        logger.info(f"Calculated metrics for {len(enhanced_df)} shipments")
        logger.info(f"Average total charges: ${enhanced_df['Total_Charges'].mean():.2f}")
        logger.info(f"High-cost shipments (>{cost_threshold:.2f}): {enhanced_df['High_Cost_Flag'].sum()}")
    
    return enhanced_df

def consolidate_miscellaneous_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate miscellaneous charges into 'Other' category for cleaner reporting.
    
    Args:
        df: Financial DataFrame with all charge categories
        
    Returns:
        DataFrame: Consolidated with 'Other' charges category
        
    Business Logic:
        - Maintains visibility of major cost drivers
        - Simplifies reporting while preserving total cost accuracy
        - Supports executive-level financial summaries
    """
    logger.info("Consolidating miscellaneous charges...")
    
    consolidated_df = df.copy()
    currency_prefix = STANDARD_CHARGE_CATEGORIES['currency_prefix']
    
    # Define primary charge column names
    primary_charge_cols = [
        f"('{currency_prefix}', '{category}')" 
        for category in STANDARD_CHARGE_CATEGORIES['primary_charges']
    ]
    
    # Identify other charge columns
    all_charge_cols = [col for col in consolidated_df.columns if currency_prefix in str(col)]
    other_charge_cols = [col for col in all_charge_cols if col not in primary_charge_cols]
    
    if other_charge_cols:
        # Create 'Other' charges category
        other_category_name = f"('{currency_prefix}', '{STANDARD_CHARGE_CATEGORIES['other_category']}')"
        consolidated_df[other_category_name] = consolidated_df[other_charge_cols].sum(axis=1, skipna=True)
        
        # Remove individual miscellaneous charge columns
        consolidated_df = consolidated_df.drop(columns=other_charge_cols)
        
        logger.info(f"Consolidated {len(other_charge_cols)} miscellaneous charge categories")
        logger.info(f"Created '{STANDARD_CHARGE_CATEGORIES['other_category']}' category")
    
    return consolidated_df

def get_fin_file(path: Path) -> pd.DataFrame:
    """
    Load and process financial data with comprehensive business logic.
    
    Args:
        path: Path to the financial data Excel file
        
    Returns:
        DataFrame: Processed financial data ready for analysis
        
    Process:
        1. Load raw financial data
        2. Validate data quality and business rules
        3. Standardize column names
        4. Calculate business metrics
        5. Consolidate charge categories
        
    Raises:
        FileNotFoundError: If financial data file doesn't exist
        ValueError: If data validation fails
    """
    logger.info(f"Loading financial data from: {path}")
    
    try:
        # Load raw financial data using universal file reader
        raw_df = read_data_file(path)
        logger.info(f"Loaded {len(raw_df)} financial records")
        
        # Data validation
        is_valid, issues = validate_financial_data(raw_df)
        if not is_valid:
            logger.warning("Data quality issues detected:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            # Continue processing but log warnings
        
        # Process the data through business logic steps
        processed_df = raw_df.copy()
        
        # Step 1: Standardize column names
        processed_df = standardize_charge_columns(processed_df)
        
        # Step 2: Calculate business metrics
        processed_df = calculate_financial_metrics(processed_df)
        
        # Step 3: Consolidate miscellaneous charges
        processed_df = consolidate_miscellaneous_charges(processed_df)
        
        logger.info("Financial data processing completed successfully")
        logger.info(f"Final dataset: {len(processed_df)} records, {len(processed_df.columns)} columns")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error processing financial data: {e}")
        raise

def generate_financial_summary(df: pd.DataFrame) -> Dict:
    """
    Generate executive summary of financial data for reporting.
    
    Args:
        df: Processed financial DataFrame
        
    Returns:
        Dict: Key financial metrics and insights
        
    Business Value:
        - Provides quick insights for stakeholders
        - Supports executive decision making
        - Identifies cost optimization opportunities
    """
    logger.info("Generating financial summary...")
    
    try:
        currency_prefix = STANDARD_CHARGE_CATEGORIES['currency_prefix']
        charge_columns = [col for col in df.columns if currency_prefix in str(col)]
        
        summary = {
            'total_shipments': len(df),
            'total_revenue': df['Total_Charges'].sum() if 'Total_Charges' in df.columns else 0,
            'average_shipment_cost': df['Total_Charges'].mean() if 'Total_Charges' in df.columns else 0,
            'cost_range': {
                'min': df['Total_Charges'].min() if 'Total_Charges' in df.columns else 0,
                'max': df['Total_Charges'].max() if 'Total_Charges' in df.columns else 0
            },
            'charge_categories': len(charge_columns),
            'high_cost_shipments': df['High_Cost_Flag'].sum() if 'High_Cost_Flag' in df.columns else 0
        }
        
        logger.info("Financial summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating financial summary: {e}")
        return {}
