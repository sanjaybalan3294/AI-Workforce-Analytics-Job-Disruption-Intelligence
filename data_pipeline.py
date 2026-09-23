"""
data_pipeline.py
=============================================================================
Tier 1: Data Hygiene & Architecture Pipeline
Enterprise-Grade Data Cleaning, Validation, Entity Aggregation, and Feature Engineering.
=============================================================================
"""

import os
import re
import pandas as pd
import numpy as np


class DataHygienePipeline:
    """
    Executes comprehensive data quality audit, hygiene rules, and 
    domain-specific feature engineering on the AI Job Trends dataset.
    """

    def __init__(self, filepath: str = "ai_job_trends_dataset.csv"):
        self.filepath = filepath
        self.audit_report = {}

    def load_raw_data(self) -> pd.DataFrame:
        """Loads dataset handling potential encoding variations."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Source file not found at: {self.filepath}")

        # Attempt UTF-8 with fallback
        try:
            df = pd.read_csv(self.filepath, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(self.filepath, encoding="latin1")

        self.audit_report["raw_shape"] = df.shape
        return df

    def clean_column_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strips leading/trailing whitespace and normalizes column headers."""
        raw_cols = df.columns.tolist()
        df.columns = [c.strip() for c in df.columns]
        cleaned_cols = df.columns.tolist()
        self.audit_report["header_changes"] = {
            r: c for r, c in zip(raw_cols, cleaned_cols) if r != c
        }
        return df

    def normalize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans strings, strips whitespace, and corrects typographic apostrophes."""
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = df[col].astype(str).str.strip()
        if "Required Education" in df.columns:
            df["Required Education"] = df["Required Education"].str.replace("\u2019", "'", regex=False)
            df["Required Education"] = df["Required Education"].str.replace("’", "'", regex=False)
            df["Required Education"] = df["Required Education"].str.replace("`", "'", regex=False)
            # Handle possible encoding replacement characters
            df["Required Education"] = df["Required Education"].apply(
                lambda x: "Master's Degree" if "Master" in x else ("Bachelor's Degree" if "Bachelor" in x else x)
            )
        return df

    def audit_and_clean_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Audits nulls, duplicates, and verifies domain value constraints."""
        # 1. Null values
        missing_counts = df.isnull().sum().to_dict()
        self.audit_report["missing_values"] = {k: v for k, v in missing_counts.items() if v > 0}
        if sum(missing_counts.values()) > 0:
            # Impute numeric with median, categorical with mode
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ["int64", "float64"]:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])

        # 2. Duplicate detection
        dup_count = df.duplicated().sum()
        self.audit_report["duplicate_records"] = int(dup_count)
        if dup_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)

        # 3. Domain Range & Anomaly Validation
        anomalies = {}
        # Salary bound: > 0
        neg_salaries = (df["Median Salary (USD)"] <= 0).sum()
        if neg_salaries > 0:
            anomalies["invalid_salary"] = int(neg_salaries)
            df = df[df["Median Salary (USD)"] > 0]

        # Experience bound: [0, 50]
        invalid_exp = ((df["Experience Required (Years)"] < 0) | (df["Experience Required (Years)"] > 50)).sum()
        if invalid_exp > 0:
            anomalies["invalid_experience"] = int(invalid_exp)
            df["Experience Required (Years)"] = df["Experience Required (Years)"].clip(0, 50)

        # Percentages bounds: [0, 100]
        pct_cols = ["Remote Work Ratio (%)", "Automation Risk (%)", "Gender Diversity (%)"]
        for p_col in pct_cols:
            if p_col in df.columns:
                out_of_bounds = ((df[p_col] < 0) | (df[p_col] > 100)).sum()
                if out_of_bounds > 0:
                    anomalies[f"out_of_bounds_{p_col}"] = int(out_of_bounds)
                    df[p_col] = df[p_col].clip(0.0, 100.0)

        # Openings bounds: >= 0
        opening_cols = ["Job Openings (2024)", "Projected Openings (2030)"]
        for o_col in opening_cols:
            if o_col in df.columns:
                neg_open = (df[o_col] < 0).sum()
                if neg_open > 0:
                    anomalies[f"negative_{o_col}"] = int(neg_open)
                    df[o_col] = df[o_col].clip(lower=0)

        self.audit_report["anomalies_detected"] = anomalies
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates economically meaningful derived features and target indicators.
        """
        # Net Projected Employment Growth & Growth Rate
        df["Net_Projected_Growth"] = df["Projected Openings (2030)"] - df["Job Openings (2024)"]
        df["Growth_Rate_Pct"] = np.where(
            df["Job Openings (2024)"] > 0,
            (df["Net_Projected_Growth"] / df["Job Openings (2024)"]) * 100.0,
            0.0
        )

        # Compensation per year of experience (Capital return per experience unit)
        df["Salary_Per_Exp_Year"] = df["Median Salary (USD)"] / (df["Experience Required (Years)"] + 1)

        # Automation Exposure Index (Compound interaction of remote work and automation risk)
        df["Automation_Exposure_Index"] = (df["Automation Risk (%)"] * df["Remote Work Ratio (%)"]) / 100.0

        # Education Ordinal Rank
        edu_rank_map = {
            "High School": 1,
            "Associate Degree": 2,
            "Bachelor's Degree": 3,
            "Master's Degree": 4,
            "PhD": 5
        }
        df["Education_Rank"] = df["Required Education"].map(edu_rank_map).fillna(3).astype(int)

        # Primary Target: Decline Risk Flag (1 = Decreasing, 0 = Increasing)
        df["Decline_Risk_Flag"] = (df["Job Status"] == "Decreasing").astype(int)

        # Secondary Target: High Automation Risk Flag (1 = Risk >= 70%, 0 = Risk < 70%)
        df["High_Automation_Risk_Flag"] = (df["Automation Risk (%)"] >= 70.0).astype(int)

        return df

    def aggregate_entity_profiles(self, df: pd.DataFrame) -> dict:
        """
        Aggregates data at macro entity levels:
        1. Job Title Entity: Profile per distinct occupation.
        2. Industry x Location Entity: Profile per regional industrial hub.
        """
        # 1. Job Title Profile
        title_profile = df.groupby("Job Title").agg(
            Observation_Count=("Job Status", "count"),
            Mean_Salary_USD=("Median Salary (USD)", "mean"),
            Mean_Experience=("Experience Required (Years)", "mean"),
            Mean_2024_Openings=("Job Openings (2024)", "mean"),
            Mean_2030_Openings=("Projected Openings (2030)", "mean"),
            Mean_Automation_Risk=("Automation Risk (%)", "mean"),
            Mean_Remote_Ratio=("Remote Work Ratio (%)", "mean"),
            Mean_Gender_Diversity=("Gender Diversity (%)", "mean"),
            Decline_Rate=("Decline_Risk_Flag", "mean")
        ).reset_index()

        # 2. Industry x Location Profile
        hub_profile = df.groupby(["Industry", "Location"]).agg(
            Total_Postings=("Job Status", "count"),
            Average_Salary=("Median Salary (USD)", "mean"),
            Average_Automation_Risk=("Automation Risk (%)", "mean"),
            Decline_Proportion=("Decline_Risk_Flag", "mean")
        ).reset_index()

        return {
            "title_profile": title_profile,
            "hub_profile": hub_profile
        }

    def run_pipeline(self, output_cleaned_path: str = "cleaned_ai_job_trends.csv") -> pd.DataFrame:
        """Executes full hygiene, audit, and feature engineering pipeline."""
        print("[INFO] Starting Data Hygiene & Architecture Pipeline...")
        df = self.load_raw_data()
        df = self.clean_column_headers(df)
        df = self.normalize_text_fields(df)
        df = self.audit_and_clean_values(df)
        df = self.engineer_features(df)
        self.audit_report["final_shape"] = df.shape

        if output_cleaned_path:
            df.to_csv(output_cleaned_path, index=False)
            print(f"[INFO] Cleaned dataset persisted to: {output_cleaned_path}")

        print("[INFO] Data Quality Audit Report:")
        for k, v in self.audit_report.items():
            print(f"  - {k}: {v}")

        return df


if __name__ == "__main__":
    pipeline = DataHygienePipeline()
    df_clean = pipeline.run_pipeline()
    entity_profiles = pipeline.aggregate_entity_profiles(df_clean)
    print("\n[SUCCESS] Tier 1 Data Hygiene & Architecture successfully executed!")
    print(f"Cleaned dataset rows: {len(df_clean)}, features: {df_clean.shape[1]}")
    print(f"Distinct Job Titles profiled: {len(entity_profiles['title_profile'])}")
    print(f"Distinct Industry-Location Hubs profiled: {len(entity_profiles['hub_profile'])}")
