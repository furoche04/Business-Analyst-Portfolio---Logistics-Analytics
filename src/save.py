"""
Data Export and Reporting Module
================================

This module handles the export of processed data and generation of business reports.
It demonstrates Business Analyst capabilities in:
- Multi-format data export and distribution
- Executive reporting and dashboard preparation
- Audit trail maintenance and data lineage
- Stakeholder-specific output formatting

Author: Karol Nosarzewski
Purpose: Business Analyst Portfolio - Data Export and Reporting
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import json

# Configure module-level logger
logger = logging.getLogger(__name__)

# Report configuration
REPORT_FORMATS = {
    'excel': {
        'extension': '.xlsx',
        'engine': 'openpyxl',
        'supports_multiple_sheets': True
    },
    'csv': {
        'extension': '.csv',
        'supports_multiple_sheets': False
    },
    'parquet': {
        'extension': '.parquet',
        'supports_multiple_sheets': False,
        'compression': 'snappy'
    }
}

EXECUTIVE_SUMMARY_COLUMNS = [
    'Shipment ID', 'TotalCharges', 'OnTimeDeliveryFlag', 
    'DeliveryStatus', 'Debtor City Name', 'HighCostFlag'
]

def create_executive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create executive summary dataset with key business metrics.
    
    Args:
        df: Full transformed DataFrame
        
    Returns:
        DataFrame: Executive summary with essential KPIs
        
    Business Value:
        - Provides high-level overview for C-suite stakeholders
        - Focuses on actionable metrics and exceptions
        - Supports executive dashboard requirements
    """
    logger.info("Creating executive summary dataset...")
    
    # Select available summary columns
    available_summary_cols = [col for col in EXECUTIVE_SUMMARY_COLUMNS if col in df.columns]
    
    if not available_summary_cols:
        logger.warning("No standard summary columns found, creating basic summary")
        available_summary_cols = df.columns[:10].tolist()  # First 10 columns as fallback
    
    summary_df = df[available_summary_cols].copy()
    
    # Add calculated summary fields
    if 'TotalCharges' in df.columns:
        # Cost category classification
        cost_quantiles = df['TotalCharges'].quantile([0.33, 0.67, 1.0])
        summary_df['CostCategory'] = pd.cut(
            df['TotalCharges'], 
            bins=[0, cost_quantiles[0.33], cost_quantiles[0.67], cost_quantiles[1.0]],
            labels=['Low Cost', 'Medium Cost', 'High Cost'],
            include_lowest=True
        )
    
    if 'OnTimeDeliveryFlag' in df.columns:
        summary_df['PerformanceRating'] = df['OnTimeDeliveryFlag'].map({
            True: 'Excellent', 
            False: 'Needs Improvement'
        })
    
    logger.info(f"Executive summary created with {len(summary_df)} records and {len(summary_df.columns)} KPIs")
    return summary_df

def create_operational_dashboard_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create multiple datasets optimized for operational dashboards.
    
    Args:
        df: Full transformed DataFrame
        
    Returns:
        Dict: Multiple DataFrames for different dashboard views
        
    Dashboard Views:
        - Performance: Transit time and delivery metrics
        - Financial: Cost analysis and charge breakdowns  
        - Geographic: Location-based performance analysis
        - Exceptions: High-cost and delayed shipments
    """
    logger.info("Creating operational dashboard datasets...")
    
    dashboard_data = {}
    
    # Performance Dashboard Data
    performance_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                       ['performance', 'ontime', 'delivery', 'transit', 'delay'])]
    if performance_cols:
        base_cols = ['Shipment ID'] if 'Shipment ID' in df.columns else []
        dashboard_data['performance'] = df[base_cols + performance_cols].copy()
    
    # Financial Dashboard Data  
    financial_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                     ['charges', 'cost', 'amount', 'financial', 'revenue'])]
    if financial_cols:
        base_cols = ['Shipment ID'] if 'Shipment ID' in df.columns else []
        dashboard_data['financial'] = df[base_cols + financial_cols].copy()
    
    # Geographic Dashboard Data
    geographic_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                      ['city', 'location', 'origin', 'destination', 'debtor'])]
    if geographic_cols:
        base_cols = ['Shipment ID', 'TotalCharges'] if all(col in df.columns for col in ['Shipment ID', 'TotalCharges']) else []
        dashboard_data['geographic'] = df[base_cols + geographic_cols].copy()
    
    # Exceptions Dashboard Data (high-cost, delayed shipments)
    exception_filters = []
    
    if 'HighCostFlag' in df.columns:
        exception_filters.append(df['HighCostFlag'] == True)
    
    if 'OnTimeDeliveryFlag' in df.columns:
        exception_filters.append(df['OnTimeDeliveryFlag'] == False)
    
    if exception_filters:
        combined_filter = exception_filters[0]
        for filter_condition in exception_filters[1:]:
            combined_filter = combined_filter | filter_condition
        
        dashboard_data['exceptions'] = df[combined_filter].copy()
    
    logger.info(f"Created {len(dashboard_data)} dashboard datasets")
    for view_name, view_df in dashboard_data.items():
        logger.info(f"  - {view_name}: {len(view_df)} records, {len(view_df.columns)} columns")
    
    return dashboard_data

def export_to_excel_with_formatting(
    dataframes: Dict[str, pd.DataFrame], 
    output_path: Path,
    include_charts: bool = False
) -> None:
    """
    Export multiple DataFrames to Excel with professional formatting.
    
    Args:
        dataframes: Dictionary of DataFrames to export
        output_path: Path for the Excel file
        include_charts: Whether to include basic charts (future enhancement)
        
    Business Value:
        - Professional presentation for stakeholder distribution
        - Multiple views in single file for convenience
        - Formatted for immediate business use
    """
    logger.info(f"Exporting {len(dataframes)} datasets to Excel: {output_path}")
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            for sheet_name, df in dataframes.items():
                # Clean sheet name for Excel compatibility
                clean_sheet_name = sheet_name.replace(' ', '_')[:31]  # Excel limit
                
                # Export DataFrame
                df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
                
                # Get the workbook and worksheet for formatting
                workbook = writer.book
                worksheet = writer.sheets[clean_sheet_name]
                
                # Apply basic formatting
                # Header formatting
                for cell in worksheet[1]:  # First row (headers)
                    cell.font = cell.font.copy(bold=True)
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                logger.info(f"  - Exported {sheet_name}: {len(df)} records")
        
        logger.info(f"Excel export completed successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise

def create_data_lineage_report(processing_steps: List[str], output_path: Path) -> None:
    """
    Create data lineage and processing audit report.
    
    Args:
        processing_steps: List of processing steps performed
        output_path: Path for the lineage report
        
    Business Value:
        - Provides audit trail for compliance
        - Documents data transformation logic
        - Supports data governance initiatives
    """
    logger.info("Creating data lineage report...")
    
    lineage_info = {
        'report_generated': datetime.now().isoformat(),
        'processing_pipeline': processing_steps,
        'data_sources': [
            'Financial.xlsx - Financial charges and cost data',
            'BaseReport.xlsx - Core shipment information', 
            'ShipmentData.xlsx - Operational details',
            'TransitTime.xlsx - Performance standards',
            'ReasonCodes.xlsx - Delay classification',
            'CityCodes.xlsx - Location mappings'
        ],
        'transformations_applied': [
            'Financial metrics calculation (total charges, cost categories)',
            'Location enrichment with standardized names',
            'Transit time performance analysis',
            'Delay reason categorization',
            'Executive summary generation'
        ],
        'output_datasets': [
            'final_transformed.xlsx - Complete integrated dataset',
            'analytical_dataset.xlsx - Business-ready analytics',
            'executive_summary.xlsx - C-suite dashboard data',
            'operational_dashboards.xlsx - Department-specific views'
        ]
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(lineage_info, f, indent=2)
        
        logger.info(f"Data lineage report created: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating lineage report: {e}")

def save_comprehensive_reports(
    final_df: pd.DataFrame, 
    analytical_df: pd.DataFrame, 
    output_dir: Path,
    processing_steps: Optional[List[str]] = None
) -> None:
    """
    Save comprehensive business reports in multiple formats for different stakeholders.
    
    Args:
        final_df: Final integrated dataset
        analytical_df: Transformed analytical dataset
        output_dir: Directory for output files
        processing_steps: List of processing steps for audit trail
        
    Output Files:
        - Executive summary (Excel with multiple views)
        - Operational dashboards (Excel with department views)
        - Complete datasets (Excel and Parquet for different uses)
        - Data lineage report (JSON for audit)
        
    Business Value:
        - Serves multiple stakeholder needs with appropriate formats
        - Provides audit trail and data governance documentation
        - Enables immediate business use without additional processing
    """
    logger.info("=" * 60)
    logger.info("GENERATING COMPREHENSIVE BUSINESS REPORTS")
    logger.info("=" * 60)
    
    try:
        # Ensure output directory exists
        output_dir.mkdir(exist_ok=True)
        
        # 1. Executive Summary Report
        logger.info("Step 1: Creating executive summary report")
        executive_summary = create_executive_summary(analytical_df)
        
        executive_datasets = {
            'Executive_Summary': executive_summary,
            'Key_Metrics': analytical_df[[col for col in analytical_df.columns 
                                        if any(keyword in col.lower() for keyword in 
                                              ['total', 'flag', 'percentage', 'rating'])]][:100]  # Top 100 for exec view
        }
        
        executive_path = output_dir / "executive_summary.xlsx"
        export_to_excel_with_formatting(executive_datasets, executive_path)
        
        # 2. Operational Dashboard Data
        logger.info("Step 2: Creating operational dashboard datasets")
        dashboard_data = create_operational_dashboard_data(analytical_df)
        
        if dashboard_data:
            dashboard_path = output_dir / "operational_dashboards.xlsx"
            export_to_excel_with_formatting(dashboard_data, dashboard_path)
        
        # 3. Complete Datasets (legacy format maintained)
        logger.info("Step 3: Saving complete datasets")
        final_path = output_dir / "final_transformed.xlsx"
        analytical_path = output_dir / "analytical_dataset.xlsx"
        
        final_df.to_excel(final_path, index=False)
        analytical_df.to_excel(analytical_path, index=False)
        
        # 4. High-performance formats for data science/advanced analytics
        logger.info("Step 4: Exporting high-performance formats")
        parquet_path = output_dir / "analytical_dataset.parquet"
        analytical_df.to_parquet(parquet_path, compression='snappy')
        
        # 5. Data lineage and audit report
        logger.info("Step 5: Creating data lineage report")
        if processing_steps is None:
            processing_steps = [
                "Data loading and validation",
                "Financial data processing", 
                "Operational data integration",
                "Business transformations applied",
                "Quality assurance checks",
                "Report generation"
            ]
        
        lineage_path = output_dir / "data_lineage_report.json"
        create_data_lineage_report(processing_steps, lineage_path)
        
        # 6. Generate processing summary
        logger.info("Step 6: Generating processing summary")
        summary = {
            'processing_completed': datetime.now().isoformat(),
            'records_processed': len(final_df),
            'analytical_records': len(analytical_df),
            'output_files_created': [
                str(executive_path.name),
                str(dashboard_path.name) if dashboard_data else None,
                str(final_path.name),
                str(analytical_path.name),
                str(parquet_path.name),
                str(lineage_path.name)
            ]
        }
        
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Final summary log
        logger.info("=" * 60)
        logger.info("REPORT GENERATION COMPLETED SUCCESSFULLY")
        logger.info(f"Total files created: {len([f for f in summary['output_files_created'] if f])}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error generating comprehensive reports: {e}")
        raise

def save_file(final_df: pd.DataFrame, rep_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Legacy save function maintained for backward compatibility.
    Redirects to save_comprehensive_reports with enhanced functionality.
    
    Args:
        final_df: Final integrated dataset
        rep_df: Transformed analytical dataset (rep_df = report dataframe)
        output_dir: Output directory path
    """
    logger.info("Using legacy save function - redirecting to comprehensive reporting")
    
    # Maintain original simple file exports for compatibility
    final_path = output_dir / "final_transformed.xlsx"
    rep_path = output_dir / "report_transformed.xlsx"
    
    final_df.to_excel(final_path, index=False)
    rep_df.to_excel(rep_path, index=False)
    
    logger.info(f"Legacy files saved: {final_path.name}, {rep_path.name}")
    
    # Also generate comprehensive reports
    save_comprehensive_reports(final_df, rep_df, output_dir)

def export_to_multiple_formats(
    df: pd.DataFrame, 
    base_filename: str, 
    output_dir: Path,
    formats: List[str] = ['excel', 'csv', 'parquet']
) -> Dict[str, Path]:
    """
    Export DataFrame to multiple formats for different use cases.
    
    Args:
        df: DataFrame to export
        base_filename: Base name for output files
        output_dir: Directory for output files
        formats: List of formats to export ('excel', 'csv', 'parquet')
        
    Returns:
        Dict: Mapping of format to file path
        
    Business Value:
        - Supports different stakeholder technical requirements
        - Provides optimal format for each use case
        - Ensures accessibility across different tools
    """
    logger.info(f"Exporting {base_filename} to multiple formats: {formats}")
    
    output_paths = {}
    
    try:
        for format_name in formats:
            if format_name not in REPORT_FORMATS:
                logger.warning(f"Unsupported format: {format_name}")
                continue
            
            format_config = REPORT_FORMATS[format_name]
            file_path = output_dir / f"{base_filename}{format_config['extension']}"
            
            if format_name == 'excel':
                df.to_excel(file_path, index=False, engine=format_config['engine'])
            elif format_name == 'csv':
                df.to_csv(file_path, index=False)
            elif format_name == 'parquet':
                df.to_parquet(file_path, compression=format_config['compression'])
            
            output_paths[format_name] = file_path
            logger.info(f"  - {format_name}: {file_path.name}")
        
        return output_paths
        
    except Exception as e:
        logger.error(f"Error exporting to multiple formats: {e}")
        raise
