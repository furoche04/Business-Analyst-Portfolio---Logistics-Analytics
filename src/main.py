"""
Logistics Data Processing Pipeline
==================================

This module orchestrates the end-to-end data processing workflow for logistics
and shipment analytics. It demonstrates key Business Analyst capabilities including:
- Multi-source data integration
- Complex business logic implementation  
- Performance metrics calculation
- Automated reporting generation

Author: Karol Nosarzewski
Purpose: Business Analyst Portfolio - Logistics Analytics
"""

import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional

# Custom modules for specialized processing
from financial import get_fin_file, generate_financial_summary
from helpers import generate_br_dataframe, remove_sensitive_columns, generate_data_profile, export_data_quality_report
from transformations import apply_business_transformations  # Renamed from test_transform
from save import save_comprehensive_reports

# Configure logging for pipeline monitoring
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define data directories using Path for cross-platform compatibility
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

def load_reference_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load reference datasets required for data enrichment and validation.
    
    Returns:
        Tuple of DataFrames: (transit_time, reason_codes, city_codes)
        
    Business Value:
        - Standardizes location and timing references
        - Enables consistent delay reason classification
        - Supports performance benchmarking
    """
    logger.info("Loading reference datasets...")
    
    try:
        tt_df = pd.read_excel(DATA_DIR / "TransitTime.xlsx")
        rc_df = pd.read_excel(DATA_DIR / "ReasonCodes.xlsx") 
        cc_df = pd.read_excel(DATA_DIR / "CityCodes.xlsx")
        
        logger.info(f"Loaded {len(tt_df)} transit time records")
        logger.info(f"Loaded {len(rc_df)} reason codes")
        logger.info(f"Loaded {len(cc_df)} city mappings")
        
        return tt_df, rc_df, cc_df
        
    except FileNotFoundError as e:
        logger.error(f"Reference file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        raise

def process_financial_data() -> pd.DataFrame:
    """
    Load and process financial data with charge categorization.
    
    Returns:
        DataFrame: Processed financial data with standardized charge categories
        
    Business Logic:
        - Consolidates miscellaneous charges into 'Other' category
        - Maintains visibility of major cost components
        - Supports financial analysis and cost optimization
    """
    logger.info("Processing financial data...")
    
    try:
        # Load and process financial data using enhanced module
        fin_df = get_fin_file(DATA_DIR / "Financial.xlsx")
        
        # Generate financial summary for reporting
        financial_summary = generate_financial_summary(fin_df)
        logger.info(f"Financial Summary - Total Shipments: {financial_summary.get('total_shipments', 'N/A')}")
        logger.info(f"Financial Summary - Average Cost: ${financial_summary.get('average_shipment_cost', 0):.2f}")
        
        return fin_df
        
    except Exception as e:
        logger.error(f"Error processing financial data: {e}")
        raise

def integrate_datasets(odw_df: pd.DataFrame, fin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate base report, operational, and financial datasets.
    
    Args:
        odw_df: Operational data warehouse DataFrame  
        fin_df: Financial data DataFrame
        
    Returns:
        DataFrame: Integrated dataset ready for analysis
        
    Business Value:
        - Creates single source of truth for analytics
        - Enables cross-functional analysis (ops + finance)
        - Supports comprehensive performance reporting
    """
    logger.info("Integrating datasets...")
    
    try:
        # Select key operational fields for analysis
        operational_fields = [
            'HAWB/HBL',
            'Master Bill',
            'Shippers Reference',
            'Consignee Ref.',
            'PickUpOrg',
            'PickUpName',
            'PickupCity',
            'PickupState',
            'PickupPostCode',
            'PickupCountry',
            'PickupUnloco',
            'ShipperOrg',
            'ShipperName',
            'Shipper City',
            'ShipperState',
            'ShipperPostCode',
            'ShipperCountry',
            'ShipperUnloco',
            'Consignee_Code',
            'ConsigneeName',
            'Consignee City',
            'ConsigneeState',
            'ConsigneePostCode',
            'ConsigneeCountry',
            'ConsigneeUnloco',
            'DeliverToOrg',
            'DeliverToName',
            'DeliverToCity',
            'DeliverToState',
            'DeliverToPostCode',
            'DeliverToCountry',
            'DeliverToUnloco',
            'Origin Company',
            'Origin Branch',
            'Destination Company',
            'Destination Branch',
            'Shipment ID',
            'Booking Number',
            'Container Number',
            'Transport Type',
            'ConsolID',
            'Consol Type',
            'Consol Service Level',
            'Mode','Type',
            'Is Master / Lead',
            'Origin',
            'Origin Country',
            'Destination',
            'Destination Country',
            'Seal#',
            'Container Type',
            'Voyage/Flight',
            'Carrier',
            'Vessel',
            'Pieces',
            'Pieces UQ',
            'Weight',
            'Weight UQ',
            'Volume',
            'Volume UQ',
            'Chargeable',
            'Chargeable UQ',
            'LoadMeters',
            'Goods Type',
            'Incoterm',
            'QA1.2 Issue Point',
            'QA1.3 Issue Party',
            'QA1.4 Issue Type',
            'QA1.5 Issue Reason',
            'QA1.6 Issue Notes',
            'QA1.7 Issue Resolved',
            'QA1.1 Issue Date',
            'QA2.2 Issue Point',
            'QA2.3 Issue Party',
            'QA2.4 Issue Type',
            'QA2.5 Issue Reason',
            'QA2.6 Issue Notes',
            'QA2.7 Issue Resolved',
            'Added (Created date)',
            'Empty Pickup at Container Depot - Origin',
            'Requested Pickup Date',
            'Estimated Pickup Date',
            'Actual Pickup Date',
            'Interim receipt',
            'Gate-In Port - Origin',
            'FCL Loaded',
            'Shipment ETD',
            'ETD First Load',
            'First ATD',
            'Shipment ETA',
            'ETA Last Disch',
            'Last ATA',
            'FCL Unload (Freight Unloaded)',
            'Storage Date',
            'Customs Reported Date',
            'Customs Cleared Date',
            'Gate-Out Port - Destination',
            'Estimated Delivery Date',
            'Actual Delivery Date',
            'Cargo Handover Date',
            'Empty Returned at Container Depot - Destination',
            'Invoice Number',
            'Invoice Date'
        ]
        
        odw_analysis = odw_df.drop_duplicates()
        logger.info(f"Selected {len(operational_fields)} operational dimensions")
        
        integrated_df = pd.merge(
            odw_analysis, fin_df,
            left_on="Shipment ID", 
            right_on="ShipmentID", 
            how="left"
        )
        
        # Clean up the integrated dataset
        integrated_df = integrated_df.reset_index(drop=True)
        
        logger.info(f"Successfully integrated {len(integrated_df)} records")
        return integrated_df
        
    except Exception as e:
        logger.error(f"Error during dataset integration: {e}")
        raise

def prepare_logistics_report() -> None:
    """
    Main pipeline orchestrator for logistics data processing and reporting.
    
    Process Flow:
        1. Load reference data for enrichment
        2. Process financial data with business logic
        3. Generate base operational reports  
        4. Integrate all data sources
        5. Apply business transformations
        6. Generate final reports and analytics
        
    Output:
        - Integrated raw dataset (for audit trail)
        - Transformed analytical dataset (for reporting)
        - Final formatted reports (for stakeholders)
    """
    
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("STARTING LOGISTICS DATA PROCESSING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Step 1: Load reference data
        logger.info("STEP 1: Loading reference datasets")
        tt_df, rc_df, cc_df = load_reference_data()
        
        # Step 2: Process financial data
        logger.info("STEP 2: Processing financial data")
        fin_df = process_financial_data()
        
        # Step 3: Generate base operational reports
        logger.info("STEP 3: Generating base operational reports") 
        odw_df = generate_br_dataframe(
            DATA_DIR / "ShipmentData.xlsx",
        )
        
        # Remove duplicates with logging
        original_count = len(odw_df)
        odw_df = odw_df.drop_duplicates()
        logger.info(f"Removed {original_count - len(odw_df)} duplicate records from base report")
        logger.info(f"Generated base report with {len(odw_df)} unique records")
        
        # Step 4: Integrate all datasets
        logger.info("STEP 4: Integrating datasets")
        integrated_df = integrate_datasets(odw_df, fin_df)
        
        # Save intermediate result for audit purposes
        audit_path = OUTPUT_DIR / "integrated_dataset_audit.xlsx"
        integrated_df.to_excel(audit_path, index=False)
        logger.info(f"Saved audit dataset: {audit_path}")
        
        # Step 5: Apply business transformations
        logger.info("STEP 5: Applying business transformations")
        analytical_df = integrated_df.copy()
        analytical_df = apply_business_transformations(
            analytical_df, tt_df, rc_df, cc_df
        )
        
        # Prepare final datasets
        final_integrated = integrated_df.drop(columns=['Last Z08 Reference'], errors='ignore')
        final_integrated = remove_sensitive_columns(final_integrated)
        
        # Generate data quality profiles for reporting
        profiles = [
            generate_data_profile(final_integrated, "Final Integrated Dataset"),
            generate_data_profile(analytical_df, "Analytical Dataset")
        ]
        
        # Export data quality report
        quality_report_path = OUTPUT_DIR / "data_quality_report.xlsx"
        export_data_quality_report(profiles, quality_report_path)
        
        # Save transformed analytical dataset
        analysis_path = OUTPUT_DIR / "analytical_dataset.xlsx"
        analytical_df.to_excel(analysis_path, index=False)
        logger.info(f"Saved analytical dataset: {analysis_path}")
        
        # Step 6: Generate final reports and analytics
        logger.info("STEP 6: Generating comprehensive business reports")
        
        # Create processing steps list for audit trail
        processing_steps = [
            "Reference data loading and validation",
            "Financial data processing with business logic",
            "Base operational report generation", 
            "Multi-source data integration",
            "Business transformations and KPI calculation",
            "Data quality profiling and reporting",
            "Comprehensive report generation"
        ]
        
        save_comprehensive_reports(final_integrated, analytical_df, OUTPUT_DIR, processing_steps)
        
        # Pipeline completion summary
        end_time = datetime.now()
        processing_duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Processing time: {processing_duration}")
        logger.info(f"Records processed: {len(integrated_df):,}")
        logger.info(f"Output location: {OUTPUT_DIR}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Check logs above for detailed error information")
        raise

if __name__ == "__main__":
    # Execute the main pipeline
    prepare_logistics_report()
