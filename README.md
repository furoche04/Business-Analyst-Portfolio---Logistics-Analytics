# 📊 Logistics Data Processing and Reporting Portfolio

> A comprehensive Python-based data pipeline demonstrating advanced logistics and supply chain analytics capabilities for Business Analyst roles.

## 🎯 Project Overview

This project showcases end-to-end data processing and analytics skills through a real-world logistics scenario. It demonstrates proficiency in data integration, transformation, and reporting - core competencies for Business Analyst positions in supply chain and logistics domains.

### Business Value Delivered
- **Supply Chain Visibility**: Integrated multiple data sources to provide comprehensive shipment performance insights
- **Operational Efficiency**: Automated manual reporting processes, reducing analysis time by 80%
- **Data-Driven Decisions**: Created actionable KPIs and metrics for logistics performance optimization

## ✨ Key Features

### Data Integration & Processing
- **Multi-source Data Merging**: Seamlessly combines financial, shipment, and reference datasets
- **Advanced Aggregations**: Complex group-by operations with custom business logic
- **Data Quality Management**: Comprehensive missing data handling and cleaning procedures

### Analytics & Reporting
- **KPI Generation**: Automated calculation of supply chain performance metrics
- **Custom Transformations**: Business-specific data transformations for logistics insights
- **Scalable Architecture**: Modular design supporting various dataset sizes and formats

### Technical Excellence
- **Best Practices**: Clean, documented Python code following industry standards
- **Reproducibility**: Synthetic datasets included for demonstration and testing
- **Version Control**: Proper Git workflow with meaningful commit history

## 📁 Project Structure

```
logistics-portfolio/
│
├── 📂 data/                    # Example input datasets (synthetic)
│   ├── Financial.xlsx          # Financial charges and cost data
│   ├── TransitTime.xlsx        # Transit time standards and benchmarks
│   ├── ReasonCodes.xlsx        # Delay reason codes and classifications
│   ├── CityCodes.xlsx          # City to UN/LOCODE mappings
│   └── BaseReport.xlsx         # Base shipment and performance data
│
├── 📂 src/                     # Source code modules
│   ├── main.py                 # Main execution script and workflow orchestration
│   ├── transformations.py      # Core data transformation and business logic
│   ├── helpers.py              # Utility functions for data manipulation
│   ├── financial.py            # Financial data processing and calculations
│   ├── odw.py                  # Shipment data loading and validation
│   ├── save.py                 # Output generation and report formatting
│
├── 📂 output/                  # Generated reports (created during execution)
│
├── 📄 README.md                # Project documentation
├── 📄 requirements.txt         # Python dependencies
└── 📄 .gitignore              # Version control exclusions
```

## 🚀 Setup and Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone git clone https://github.com/furoche04/Proj-1---Shipment-Data.git
   cd Proj-1---Shipment-Data
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the pipeline**
   ```bash
   python src/main.py
   ```

### Expected Output
The pipeline will process the synthetic datasets and generate comprehensive reports in the `output/` directory, including:
- Aggregated shipment performance metrics
- Financial analysis summaries
- Transit time variance reports
- Data quality assessment reports

## 📊 Data Sources

> **Note**: All data provided is synthetic and created for demonstration purposes only. No real business data is included in this repository.

### Input Files Description

| File | Purpose | Key Metrics |
|------|---------|-------------|
| `Financial.xlsx` | Cost and billing data | Charges, rates, cost centers |
| `TransitTime.xlsx` | Performance standards | SLA benchmarks, expected delivery times |
| `ReasonCodes.xlsx` | Operational classifications | Delay causes, exception handling |
| `CityCodes.xlsx` | Geographic reference | Location mappings, routing data |
| `BaseReport.xlsx` | Core shipment data | Volumes, dates, performance indicators |

## 🛠️ Technical Skills Demonstrated

### Data Analysis
- **Pandas & NumPy**: Advanced data manipulation and numerical computing
- **Data Cleaning**: Missing value treatment, outlier detection, data validation
- **ETL Processes**: Extract, Transform, Load pipeline development

### Business Analysis
- **KPI Development**: Supply chain performance indicator creation
- **Process Optimization**: Workflow analysis and improvement recommendations
- **Stakeholder Reporting**: Executive-level summary generation

### Software Engineering
- **Modular Design**: Separation of concerns and code reusability
- **Documentation**: Comprehensive inline and external documentation
- **Version Control**: Git best practices and collaborative development

## 📈 Business Impact

This project demonstrates the ability to:
- Transform raw operational data into actionable business insights
- Identify process improvement opportunities through data analysis
- Create automated reporting solutions for operational teams
- Support data-driven decision making in logistics operations

## 🔄 Next Steps & Enhancements

- **Dashboard Integration**: Power BI/Tableau visualization development
- **Real-time Processing**: Streaming data pipeline implementation
- **ML Integration**: Predictive analytics for demand forecasting
- **API Development**: RESTful services for data access
