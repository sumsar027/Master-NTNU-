# Master Thesis Repository

This repository contains the data, code, articles, and supporting material used in our master thesis. The thesis studies the relationship between risk measures, leverage, and post-crisis regulation in large U.S. banks, with particular attention to Value-at-Risk, capital structure, liquidity requirements, and regulatory constraints.

The repository is made publicly available from the submission date, June 1. Its purpose is to document the empirical work behind the thesis and make the data collection and analysis process more transparent.

## Repository Structure

### AI Declaration

This folder contains the declaration form concerning the use of AI tools in the thesis. AI tools were used for tasks such as language improvement, translation, grammar support, and coding assistance. The authors remain fully responsible for the final content, analysis, and conclusions presented in the thesis.

### Artikler

This folder contains articles and other academic material used during the writing process. Some of these sources are discussed directly in the literature review and theoretical framework, while others were used as background material.

### BalanceSheets

This folder contains balance sheet data used in the empirical analysis. These data were mainly collected from Refinitiv LSEG through access provided by NTNU Business School.

### CashFlows

This folder contains cash flow data downloaded and prepared for the analysis.

### FinancialSummary

This folder contains financial summary variables used to describe the banks in the sample and support the empirical analysis.

### GSIB

This folder contains data and supporting material related to the U.S. Global Systemically Important Banks included in the sample.

### IncomeStatement

This folder contains income statement data used as part of the broader bank-level dataset.

### Latex

This folder contains LaTeX-related files used in the writing and formatting of the thesis.

### RStudio

This folder contains R scripts used for data cleaning, data structuring, descriptive statistics, figures, and regression analysis.

### VIX

This folder contains data related to the VIX index, which is used as a broader market risk indicator.

### VaR

This folder contains the manually collected Value-at-Risk data. The VaR figures were collected from the banks’ own 10-Q and 10-K reports, accessed through the SEC EDGAR database. Each bank in the sample was searched manually, and the relevant VaR observations were extracted from the reported filings.

### balancesheetv2

This folder contains an updated version of the balance sheet data used during the data preparation process.

### python, python_improved, python_may, and python_v3

These folders contain Python scripts and working files used at different stages of the data collection and data cleaning process. Some scripts were exploratory, while others were used to improve or automate parts of the workflow.

## Data Sources

The VaR data were collected manually from the banks’ own quarterly and annual reports, mainly 10-Q and 10-K filings. These reports were accessed through the SEC EDGAR database.

Data related to capital structure, balance sheet composition, liquidity measures, and regulatory ratios were collected from Refinitiv LSEG, using access provided by NTNU Business School.

After collection, the data were cleaned, structured, and combined before being used in the empirical analysis.

## Notes on Reproducibility

The repository includes the datasets and code used in the thesis. However, some files may reflect different stages of the working process, including earlier versions, exploratory scripts, and updated data structures. The final empirical analysis is based on the cleaned and combined datasets described in the thesis.

## Authors

Rasmus Thorbjørnsen
Alexander Ski
