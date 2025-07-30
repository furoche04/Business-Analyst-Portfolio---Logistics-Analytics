"""
Data Processing Helper Functions
===============================

This module provides utility functions for data manipulation, validation, and 
preprocessing in logistics analytics. It demonstrates Business Analyst skills in:
- Data quality assessment and cleansing
- Operational data standardization
- Performance monitoring and validation
- Cross-system data integration

Author: Karol Nosarzewski
Purpose: Business Analyst Portfolio - Data Processing Utilities
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime

# Configure module-level logger
logger = logging.getLogger(__name__)

# Supported file formats and their configurations
SUPPORTED_FILE_FORMATS = {
    '.xlsx': {
        'reader': 'pd.read_excel',
        'engine': 'openpyxl',
        'description': 'Excel 2007+ format'
    },
    '.xlsb': {
        'reader': 'pd.read_excel', 
        'engine': 'pyxlsb',
        'description': 'Excel Binary format'
    },
    '.xls': {
        'reader': 'pd.read_excel',
        'engine': 'xlrd',
        'description': 'Excel Legacy format'
    },
    '.csv': {
        'reader': 'pd.read_csv',
        'encoding': 'utf-8',
        'description': 'Comma Separated Values'
    },
    '.parquet': {
        'reader': 'pd.read_parquet',
        'engine': 'pyarrow',
        'description': 'Apache Parquet format'
    },
    '.json': {
        'reader': 'pd.read_json',
        'description': 'JSON format'
    },
    '.txt': {
        'reader': 'pd.read_csv',
        'separator': '\t',
        'description': 'Tab-delimited text'
    }
}

# Business configuration for data processing
DATA_QUALITY_THRESHOLDS = {
    'max_missing_percentage': 50,  # Maximum allowed missing data percentage
    'min_records_required': 10,    # Minimum records for valid processing
    'duplicate_tolerance': 0.05    # Maximum allowed duplicate percentage
}

OPERATIONAL_COLUMNS = {
    'required_br_columns': [
        'Shipment ID'
    ],
    'required_odw_columns': [
        'Shipment ID'
    ],
    'sensitive_columns': [
        'Master / Lead Ref',
        'Internal Reference',
        'Customer Code',
        'SomeOtherColumn'
    ]
}

def detect_file_format(file_path: Path) -> Dict[str, Union[str, bool]]:
    """
    Detect file format and return appropriate reading configuration.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict: File format information and reading parameters
        
    Business Value:
        - Automatically handles different data source formats
        - Reduces manual configuration and errors
        - Supports diverse data integration scenarios
    """
    file_extension = file_path.suffix.lower()
    
    if file_extension in SUPPORTED_FILE_FORMATS:
        format_config = SUPPORTED_FILE_FORMATS[file_extension].copy()
        format_config['extension'] = file_extension
        format_config['supported'] = True
        
        logger.info(f"Detected format: {file_extension} - {format_config.get('description', 'Unknown format')}")
        return format_config
    else:
        logger.warning(f"Unsupported file format: {file_extension}")
        return {
            'extension': file_extension,
            'supported': False,
            'description': f'Unsupported format: {file_extension}'
        }

def read_data_file(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Universal data file reader that automatically detects format and applies appropriate engine.
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional parameters to pass to the pandas reader
        
    Returns:
        DataFrame: Loaded data
        
    Supported Formats:
        - Excel (.xlsx, .xlsb, .xls)
        - CSV (.csv)
        - Parquet (.parquet)
        - JSON (.json)
        - Tab-delimited text (.txt)
        
    Business Value:
        - Simplifies data loading across different source systems
        - Handles format-specific optimizations automatically
        - Provides consistent error handling and logging
    """
    logger.info(f"Loading data file: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Detect file format
    format_info = detect_file_format(file_path)
    
    if not format_info['supported']:
        raise ValueError(f"Unsupported file format: {format_info['extension']}")
    
    try:
        # Prepare reading parameters
        read_params = kwargs.copy()
        
        # Apply format-specific configurations
        if format_info['extension'] in ['.xlsx', '.xlsb', '.xls']:
            read_params['engine'] = format_info.get('engine', 'openpyxl')
            df = pd.read_excel(file_path, **read_params)
            
        elif format_info['extension'] == '.csv':
            # CSV-specific optimizations
            if 'encoding' not in read_params:
                read_params['encoding'] = format_info.get('encoding', 'utf-8')
            if 'low_memory' not in read_params:
                read_params['low_memory'] = False  # Better for data quality
            df = pd.read_csv(file_path, **read_params)
            
        elif format_info['extension'] == '.parquet':
            read_params['engine'] = format_info.get('engine', 'pyarrow')
            df = pd.read_parquet(file_path, **read_params)
            
        elif format_info['extension'] == '.json':
            df = pd.read_json(file_path, **read_params)
            
        elif format_info['extension'] == '.txt':
            # Tab-delimited text files
            if 'sep' not in read_params:
                read_params['sep'] = format_info.get('separator', '\t')
            df = pd.read_csv(file_path, **read_params)
            
        else:
            raise ValueError(f"No reader implementation for format: {format_info['extension']}")
        
        # Log successful load
        logger.info(f"Successfully loaded {len(df)} records from {file_path.name}")
        logger.info(f"Dataset dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        logger.error(f"File format: {format_info['extension']}, Size: {file_path.stat().st_size} bytes")
        raise

def load_reference_files(data_dir: Path, file_mapping: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple reference files with automatic format detection.
    
    Args:
        data_dir: Directory containing reference files
        file_mapping: Dictionary mapping logical names to filenames
                     e.g., {'transit_time': 'TransitTime.xlsx', 'city_codes': 'CityCodes.csv'}
        
    Returns:
        Dict: Mapping of logical names to loaded DataFrames
        
    Business Value:
        - Centralizes reference data loading with consistent error handling
        - Supports mixed file formats in single data directory
        - Provides clear mapping between business concepts and file sources
    """
    logger.info(f"Loading {len(file_mapping)} reference files from {data_dir}")
    
    loaded_files = {}
    loading_errors = []
    
    for logical_name, filename in file_mapping.items():
        try:
            file_path = data_dir / filename
            df = read_data_file(file_path)
            loaded_files[logical_name] = df
            
            logger.info(f"  ✓ {logical_name}: {len(df)} records from {filename}")
            
        except Exception as e:
            error_msg = f"Failed to load {logical_name} from {filename}: {e}"
            loading_errors.append(error_msg)
            logger.error(f"  ✗ {error_msg}")
    
    if loading_errors:
        logger.warning(f"Completed with {len(loading_errors)} errors out of {len(file_mapping)} files")
        # Log summary of what was successfully loaded
        logger.info(f"Successfully loaded: {list(loaded_files.keys())}")
    
    return loaded_files
    """
    Comprehensive data quality assessment for business datasets.
    
    Args:
        df: DataFrame to validate
        df_name: Name of the dataset for logging
        
    Returns:
        Tuple: (is_valid, quality_metrics, issues_list)
        
    Business Value:
        - Ensures data reliability for decision making
        - Identifies data quality issues early in pipeline
        - Provides metrics for data governance reporting
    """
    logger.info(f"Validating data quality for {df_name}...")
    
    issues = []
    quality_metrics = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_data_percentage': 0,
        'duplicate_records': 0,
        'data_types_count': df.dtypes.value_counts().to_dict()
    }
    
    # Check minimum record requirements
    if len(df) < DATA_QUALITY_THRESHOLDS['min_records_required']:
        issues.append(f"Insufficient records: {len(df)} < {DATA_QUALITY_THRESHOLDS['min_records_required']}")
    
    # Calculate missing data percentage
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    quality_metrics['missing_data_percentage'] = round(missing_percentage, 2)
    
    if missing_percentage > DATA_QUALITY_THRESHOLDS['max_missing_percentage']:
        issues.append(f"Excessive missing data: {missing_percentage:.1f}% > {DATA_QUALITY_THRESHOLDS['max_missing_percentage']}%")
    
    # Check for duplicate records
    duplicate_count = df.duplicated().sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
    quality_metrics['duplicate_records'] = duplicate_count
    quality_metrics['duplicate_percentage'] = round(duplicate_percentage, 2)
    
    if duplicate_percentage > DATA_QUALITY_THRESHOLDS['duplicate_tolerance'] * 100:
        issues.append(f"High duplicate rate: {duplicate_percentage:.1f}% > {DATA_QUALITY_THRESHOLDS['duplicate_tolerance']*100}%")
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        issues.append(f"Empty columns found: {empty_columns}")
        quality_metrics['empty_columns'] = empty_columns
    
    is_valid = len(issues) == 0
    
    # Log quality metrics
    logger.info(f"{df_name} Quality Metrics:")
    logger.info(f"  - Records: {quality_metrics['total_records']:,}")
    logger.info(f"  - Missing data: {quality_metrics['missing_data_percentage']}%")
    logger.info(f"  - Duplicates: {quality_metrics['duplicate_records']}")
    
    if issues:
        logger.warning(f"{df_name} Quality Issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return is_valid, quality_metrics, issues

def standardize_shipment_ids(df: pd.DataFrame, id_column: str = 'Shipment ID') -> pd.DataFrame:
    """
    Standardize shipment ID format for consistent data joining.
    
    Args:
        df: DataFrame containing shipment IDs
        id_column: Name of the shipment ID column
        
    Returns:
        DataFrame: With standardized shipment IDs
        
    Business Logic:
        - Removes leading/trailing whitespace
        - Converts to uppercase for consistency
        - Handles common formatting variations
    """
    if id_column not in df.columns:
        logger.warning(f"Shipment ID column '{id_column}' not found in DataFrame")
        return df
    
    standardized_df = df.copy()
    original_count = len(standardized_df)
    
    # Standardization steps
    standardized_df[id_column] = (
        standardized_df[id_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace('NAN', np.nan)
    )
    
    # Remove records with invalid shipment IDs
    invalid_mask = (
        standardized_df[id_column].isnull() | 
        (standardized_df[id_column] == '') |
        (standardized_df[id_column] == 'NONE')
    )
    
    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        logger.warning(f"Removed {invalid_count} records with invalid shipment IDs")
        standardized_df = standardized_df[~invalid_mask]
    
    logger.info(f"Standardized {original_count} -> {len(standardized_df)} records")
    return standardized_df

def generate_br_dataframe(odw_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate base report and operational data warehouse datasets using universal file reader.
    
    Args:
        br_path: Path to base report file (any supported format)
        odw_path: Path to operational data warehouse file (any supported format)
        
    Returns:
        Tuple: (base_report_df, operational_dw_df)
        
    Process:
        1. Auto-detect file formats and load with appropriate engines
        2. Validate data quality and business rules
        3. Standardize key identifiers
        4. Log processing metrics for audit
        
    Business Value:
        - Handles mixed file formats automatically
        - Ensures reliable data foundation for analytics
        - Provides audit trail for data lineage
        - Validates operational data consistency
    """
    logger.info("Loading base operational datasets with auto-format detection...")
    
    try:
        # # Load base report dataset with automatic format detection
        # logger.info(f"Loading base report from: {br_path}")
        # br_df = read_data_file(br_path)
        # logger.info(f"Loaded base report: {len(br_df)} records, {len(br_df.columns)} columns")
        
        # Load operational data warehouse dataset with automatic format detection
        logger.info(f"Loading ODW data from: {odw_path}")
        odw_df = read_data_file(odw_path)
        logger.info(f"Loaded ODW data: {len(odw_df)} records, {len(odw_df.columns)} columns")
        
        # Standardize shipment IDs for reliable joining
        # br_df = standardize_shipment_ids(br_df, 'Shipment ID')
        odw_df = standardize_shipment_ids(odw_df, 'Shipment ID')
        
        # # Validate required columns exist
        # for required_col in OPERATIONAL_COLUMNS['required_br_columns']:
        #     if required_col not in br_df.columns:
        #         logger.error(f"Required column missing in Base Report: {required_col}")
        #         raise ValueError(f"Missing required column: {required_col}")
        
        for required_col in OPERATIONAL_COLUMNS['required_odw_columns']:
            if required_col not in odw_df.columns:
                logger.error(f"Required column missing in ODW: {required_col}")
                raise ValueError(f"Missing required column: {required_col}")
        
        # Log final dataset metrics
        logger.info("Dataset loading completed successfully:")
        # logger.info(f"  - Base Report: {len(br_df):,} records")  
        logger.info(f"  - ODW Data: {len(odw_df):,} records")
        
        return odw_df
        
    except Exception as e:
        logger.error(f"Error loading operational datasets: {e}")
        raise

def remove_sensitive_columns(df: pd.DataFrame, additional_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove sensitive or unnecessary columns for portfolio demonstration.
    
    Args:
        df: DataFrame to clean
        additional_columns: Additional columns to remove beyond defaults
        
    Returns:
        DataFrame: Cleaned dataset suitable for portfolio
        
    Business Purpose:
        - Removes sensitive business information for portfolio sharing
        - Eliminates irrelevant columns for focused analysis
        - Maintains data utility while ensuring privacy compliance
    """
    logger.info("Removing sensitive and unnecessary columns...")
    
    cleaned_df = df.copy()
    
    # Combine default sensitive columns with additional ones
    columns_to_remove = OPERATIONAL_COLUMNS['sensitive_columns'].copy()
    if additional_columns:
        columns_to_remove.extend(additional_columns)
    
    # Remove columns that exist in the DataFrame
    existing_columns_to_remove = [col for col in columns_to_remove if col in cleaned_df.columns]
    
    if existing_columns_to_remove:
        cleaned_df = cleaned_df.drop(columns=existing_columns_to_remove)
        logger.info(f"Removed {len(existing_columns_to_remove)} sensitive columns: {existing_columns_to_remove}")
    else:
        logger.info("No sensitive columns found to remove")
    
    logger.info(f"Dataset cleaned: {len(df.columns)} -> {len(cleaned_df.columns)} columns")
    return cleaned_df

def remove_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function maintained for backward compatibility.
    Redirects to remove_sensitive_columns with improved functionality.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame: Cleaned dataset
    """
    return remove_sensitive_columns(df)

def generate_data_profile(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
    """
    Generate comprehensive data profile for business stakeholders.
    
    Args:
        df: DataFrame to profile
        dataset_name: Name for reporting purposes
        
    Returns:
        Dict: Comprehensive data profile with business insights
        
    Business Value:
        - Provides data overview for stakeholder communication
        - Identifies potential data quality improvements
        - Supports data governance initiatives
    """
    logger.info(f"Generating data profile for {dataset_name}...")
    
    try:
        profile = {
            'dataset_name': dataset_name,
            'overview': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'data_quality': {
                'missing_data_percentage': round((df.isnull().sum().sum() / df.size) * 100, 2),
                'duplicate_records': df.duplicated().sum(),
                'complete_records': len(df.dropna())
            },
            'column_analysis': {},
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Analyze each column
        for column in df.columns:
            col_info = {
                'data_type': str(df[column].dtype),
                'missing_count': df[column].isnull().sum(),
                'missing_percentage': round((df[column].isnull().sum() / len(df)) * 100, 2),
                'unique_values': df[column].nunique()
            }
            
            # Add specific analysis based on data type
            if df[column].dtype in ['int64', 'float64']:
                col_info.update({
                    'min_value': df[column].min(),
                    'max_value': df[column].max(),
                    'mean_value': round(df[column].mean(), 2),
                    'median_value': df[column].median()
                })
                profile['numeric_summary'][column] = col_info
            else:
                col_info.update({
                    'top_values': df[column].value_counts().head(3).to_dict()
                })
                profile['categorical_summary'][column] = col_info
            
            profile['column_analysis'][column] = col_info
        
        logger.info(f"Data profile generated for {dataset_name}")
        return profile
        
    except Exception as e:
        logger.error(f"Error generating data profile: {e}")
        return {}

def export_data_quality_report(profiles: List[Dict], output_path: Path) -> None:
    """
    Export comprehensive data quality report for stakeholders.
    
    Args:
        profiles: List of data profiles from generate_data_profile
        output_path: Path to save the quality report
        
    Business Value:
        - Provides executive summary of data quality status
        - Supports data governance and compliance reporting
        - Identifies areas for process improvement
    """
    logger.info("Generating data quality report...")
    
    try:
        # Create summary report
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Executive summary sheet
            summary_data = []
            for profile in profiles:
                summary_data.append({
                    'Dataset': profile['dataset_name'],
                    'Records': profile['overview']['total_records'],
                    'Columns': profile['overview']['total_columns'],
                    'Missing_Data_%': profile['data_quality']['missing_data_percentage'],
                    'Duplicates': profile['data_quality']['duplicate_records'],
                    'Complete_Records': profile['data_quality']['complete_records'],
                    'Memory_MB': profile['overview']['memory_usage_mb']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
            
            # Detailed analysis for each dataset
            for profile in profiles:
                sheet_name = profile['dataset_name'].replace(' ', '_')[:31]  # Excel sheet name limit
                
                # Column analysis details
                col_analysis = pd.DataFrame.from_dict(profile['column_analysis'], orient='index')
                col_analysis.to_excel(writer, sheet_name=f'{sheet_name}_Details')
        
        logger.info(f"Data quality report saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating data quality report: {e}")
        raise
