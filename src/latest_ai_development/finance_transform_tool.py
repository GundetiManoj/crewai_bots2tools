import os
import pandas as pd
import numpy as np
import torch
import re
from typing import List, Dict, Union, Any, ClassVar
from transformers import BartTokenizer, BartForConditionalGeneration
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel
import json
import difflib

MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME, local_files_only=True)

# -------------------- Schema for the Tool --------------------
# class ColumnCorrectionToolSchema(BaseModel):
#     data: str
#     kwargs: dict = {} 

# # -------------------- Tool for Column Correction --------------------
# class ColumnCorrectionTool(BaseTool):
#     name: str = "Column Correction Tool"
#     description: str = "Corrects ambiguous or inconsistent column names using a local HuggingFace model"
#     args_schema = ColumnCorrectionToolSchema

#     def _run(self, data: str, **kwargs: Any) -> Any:
#         # Pass the data (a JSON string) to the model.
#         inputs = tokenizer(data, return_tensors="pt", max_length=1024, truncation=True)
#         summary_ids = model.generate(inputs["input_ids"], max_length=256, min_length=5, length_penalty=2.0)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#         print("🔎 Model-generated summary for column mapping:", summary)

#         # Return a hardcoded mapping (this may be replaced by dynamic logic later)
#         return {
#             'date': 'Tran Date',
#             'transaction date': 'Tran Date',
#             'dr/cr': 'Dr/Cr',
#             'sl no': 'Sl No',
#             'amt': 'Amount',
#             'amount': 'Amount',
#             'Debit_INR': 'Debit',
#             'Credit_INR': 'Credit',
#             'debit_inr': 'Debit',
#             'credit_inr': 'Credit',
#             'dr': 'Debit',
#             'cr': 'Credit'
#         }


class ColumnCorrectionToolSchema(BaseModel):
    data: str
    kwargs: dict = {} 
class ColumnCorrectionTool(BaseTool):
    name: str = "Column Correction Tool"
    description: str = "Corrects ambiguous or inconsistent column names using similarity matching and NLP"
    args_schema = ColumnCorrectionToolSchema

    # Standard column names that are commonly used in financial data
    STANDARD_COLUMNS: Dict[str, List[str]] = {
        'transaction_date': ['tran date', 'date', 'transaction date', 'trans date', 'value date'],
        'debit_credit': ['dr/cr', 'debit/credit', 'transaction type', 'type'],
        'serial_number': ['sl no', 'serial no', 'sr no', 'id'],
        'amount': ['amt', 'amount', 'transaction amount', 'value'],
        'debit': ['dr', 'debit', 'debit_inr', 'debit amount'],
        'credit': ['cr', 'credit', 'credit_inr', 'credit amount']
    }

    def get_best_match(self, input_col: str, threshold: float = 0.6) -> str:
        """
        Find the best matching standard column name using string similarity
        """
        input_col = input_col.lower().strip()
        best_match = input_col
        highest_ratio = 0

        # First try exact matches in standard columns
        for standard_category, variations in self.STANDARD_COLUMNS.items():
            if input_col in variations:
                return self.get_formatted_name(standard_category)

        # If no exact match, use similarity matching
        for standard_category, variations in self.STANDARD_COLUMNS.items():
            for variation in variations:
                ratio = difflib.SequenceMatcher(None, input_col, variation).ratio()
                if ratio > highest_ratio and ratio >= threshold:
                    highest_ratio = ratio
                    best_match = self.get_formatted_name(standard_category)

        return best_match

    def get_formatted_name(self, category: str) -> str:
        """
        Convert standard category to properly formatted column name
        """
        formatting_map = {
            'transaction_date': 'Tran Date',
            'debit_credit': 'Dr/Cr',
            'serial_number': 'Sl No',
            'amount': 'Amount',
            'debit': 'Debit',
            'credit': 'Credit'
        }
        return formatting_map.get(category, category.title())

    def use_bart_for_complex_columns(self, column_name: str) -> str:
        """
        Use BART model for more complex column name understanding
        """
        prompt = f"Convert this column name '{column_name}' to a standard financial statement column name. Consider if it's related to date, amount, transaction type, or serial number."
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=32,
            min_length=2,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        suggested_name = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Clean up the model output
        suggested_name = suggested_name.strip().split()[0]  # Take first word
        return self.get_best_match(suggested_name)  # Validate against standard names

    def _run(self, data: str, **kwargs: Any) -> Dict[str, str]:
        """
        Main method to process column names
        """
        try:
            # Parse the input JSON data
            df_dict = json.loads(data)
            if isinstance(df_dict, list):
                df_dict = df_dict[0]  # Take first record to get column names
            
            column_mapping = {}
            
            for original_col in df_dict.keys():
                # Skip empty or None columns
                if not original_col or pd.isna(original_col):
                    continue
                
                # Try difflib matching first
                matched_name = self.get_best_match(original_col)
                
                # If difflib doesn't find a good match, use BART
                if matched_name == original_col.lower().strip():
                    matched_name = self.use_bart_for_complex_columns(original_col)
                
                column_mapping[original_col.lower().strip()] = matched_name
                
            print("🔄 Dynamic column mapping generated:", column_mapping)
            return column_mapping

        except Exception as e:
            print(f"❌ Error in column correction: {str(e)}")
            return {}

# -------------------- Data Transformer --------------------
class DataTransformer:
    DEFAULT_DATE_FORMAT = "%d-%m-%Y"
    AMOUNT_MULTIPLIER = 1000
    AMOUNT_MULTIPLIER_DOLLAR = 80

    @staticmethod
    def rename_columns_using_map(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        df.columns = [mapping.get(col.strip().lower(), col.strip()) for col in df.columns]
        print("✅ Renamed columns:", df.columns.tolist())
        return df

    @staticmethod
    def transform_data(df: pd.DataFrame) -> pd.DataFrame:
        transformations = [
            DataTransformer.flatten_table,
            DataTransformer.validate_column_data_types,
            DataTransformer.handle_direct_amount,
            DataTransformer.split_rows,
            DataTransformer.merge_rows,
            DataTransformer.handle_missing_values,
            DataTransformer.format_date_column,
            DataTransformer.calculate_amount,
            DataTransformer.merge_columns,
            DataTransformer.fill_empty_cells,
            DataTransformer.correct_spelling,
            DataTransformer.text_transformation,
            DataTransformer.currency_conversion,
            DataTransformer.validate_column_values,
            lambda x: DataTransformer.retain_required_columns(x, ["Sl No", "Tran Date", "Dr/Cr", "Amount"])
        ]
        for transform in transformations:
            try:
                df = transform(df)
                print(f"✅ {transform.__name__} applied")
            except Exception as e:
                print(f"❌ Error in {transform.__name__}: {e}")
        return df

    @staticmethod
    def flatten_table(df): 
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(col).strip() for col in df.columns.values]
        return df

    @staticmethod
    def validate_column_data_types(df):
        if "Sl No" in df.columns:
            df["Sl No"] = pd.to_numeric(df["Sl No"], errors="coerce").ffill().astype(int)
        if "Tran Date" in df.columns:
            df["Tran Date"] = pd.to_datetime(df["Tran Date"], errors="coerce").dt.strftime(DataTransformer.DEFAULT_DATE_FORMAT)
        for col in ["Debit", "Credit", "Amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    @staticmethod
    def retain_required_columns(df, required_columns):
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        return df[required_columns]

    @staticmethod
    def merge_columns(df):
        if {"Debit", "Credit"}.issubset(df.columns):
            df["Amount"] = df["Debit"].fillna(0) + df["Credit"].fillna(0)
            df.drop(columns=["Debit", "Credit"], inplace=True)
        return df

    @staticmethod
    def handle_direct_amount(df):
        if "Amount" not in df.columns and {"Debit", "Credit"}.issubset(df.columns):
            df = DataTransformer.merge_columns(df)
        elif "Amount" in df.columns:
            df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
        return df

    @staticmethod
    def split_rows(df):
        if "Tran Date" in df.columns:
            df["Tran Date"] = df["Tran Date"].astype(str).str.split("|")
            df = df.explode("Tran Date")
        return df

    @staticmethod
    def merge_rows(df):
        return df.ffill(axis=0)

    @staticmethod
    def handle_missing_values(df):
        if "Tran Date" in df.columns:
            df["Tran Date"] = df["Tran Date"].fillna(method="ffill")
        if "Dr/Cr" in df.columns:
            df["Dr/Cr"] = df["Dr/Cr"].fillna("Unknown")
        if "Amount" in df.columns:
            df["Amount"] = df["Amount"].fillna(0)
        return df

    @staticmethod
    def format_date_column(df):
        if "Tran Date" in df.columns:
            df["Tran Date"] = pd.to_datetime(df["Tran Date"], errors="coerce", dayfirst=True)\
                .dt.strftime(DataTransformer.DEFAULT_DATE_FORMAT)
        return df

    @staticmethod
    def calculate_amount(df):
        if {"Debit", "Credit"}.issubset(df.columns):
            df["Amount"] = (df["Credit"] - df["Debit"]).abs()
        return df

    @staticmethod
    def fill_empty_cells(df):
        df.fillna({"Dr/Cr": "Unknown", "Amount": 0, "Tran Date": "01-01-2000"}, inplace=True)
        return df

    @staticmethod
    def validate_column_values(df):
        if "Amount" in df.columns:
            df = df[df["Amount"] >= 0]
        return df

    @staticmethod
    def text_transformation(df):
        mapping = {"Dr": "Debit", "Cr": "Credit"}
        if "Dr/Cr" in df.columns:
            df["Dr/Cr"] = df["Dr/Cr"].replace(mapping)
        return df

    @staticmethod
    def correct_spelling(df):
        spell_map = {"Debbit": "Debit", "Creditt": "Credit"}
        if "Dr/Cr" in df.columns:
            df["Dr/Cr"] = df["Dr/Cr"].replace(spell_map)
        return df

    @staticmethod
    def currency_conversion(df):
        if "Amount" in df.columns:
            if any(re.search(r'\$', col) for col in df.columns):
                df["Amount"] *= DataTransformer.AMOUNT_MULTIPLIER_DOLLAR
            elif any(re.search(r'1000s', col) for col in df.columns):
                df["Amount"] *= DataTransformer.AMOUNT_MULTIPLIER
        return df

# -------------------- Main CrewAI Workflow --------------------
def main_workflow():
    input_path = "src/latest_ai_development/input.xlsx"
    output_path = "src/latest_ai_development/output.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    df = pd.read_excel(input_path)
    print("✅ Excel file read successfully.")

    # Step 1: Use Tool to fix column names
    corrector_tool = ColumnCorrectionTool()
    # Convert the dataframe to JSON string for the tool
    json_records = df.to_json(orient="records")
    # Pass an empty kwargs dictionary to satisfy validation requirements
    col_map = corrector_tool.run(json_records, kwargs={})
    print("✅ Column mapping received:", col_map)

    # Rename columns based on mapping from the tool
    df = DataTransformer.rename_columns_using_map(df, col_map)

    # Step 2: Transform the data
    transformed_df = DataTransformer.transform_data(df)

    # Save the final result
    transformed_df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data written to {output_path}")

# -------------------- CrewAI Definitions --------------------
corrector_tool = ColumnCorrectionTool()

agent = Agent(
    role="Excel Data Specialist",
    goal="Clean and standardize Excel data for further analysis",
    backstory="A detail-oriented expert who transforms messy tabular data into structured, analysis-ready datasets.",
    tools=[corrector_tool],
    verbose=True
)

task = Task(
    description="Read the input Excel file at path: src/latest_ai_development/Book2.xlsx, use LLM to correct column names, then apply standard transformations like date formatting, amount calculation, column merging, etc. Save the final clean output at path: {output_path}.",
    expected_output="A cleaned CSV file with columns: Sl No, Tran Date, Dr/Cr, Amount.",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential
)

if __name__ == "__main__":
    try:
        crew.kickoff()
    except Exception as crew_error:
        print("❌ Crew kickoff error:", crew_error)
    main_workflow()