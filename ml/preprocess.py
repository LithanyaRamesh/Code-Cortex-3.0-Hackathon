"""
Email Secure Lens - Data Preprocessing Module
Author: Email Secure Lens ML Team
Description:
    Provides reusable text cleaning and dataset preprocessing functions
    for email spam classification.
"""

import os
import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Cleans raw email text for spam classification.

    Steps:
    1. Ensures input is string.
    2. Converts text to lowercase.
    3. Removes invisible control characters / non-printable characters.
    4. Normalizes multiple whitespaces, tabs, and newlines to a single space.
    5. Strips leading and trailing whitespace.
    
    Preserves:
    - Words, numbers, punctuation, currency symbols ($, €), percentage signs (%),
      and URL indicators, as these are critical spam signals.

    Args:
        text (str): Raw email text string.

    Returns:
        str: Cleaned email text.
    """
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove control / non-printable characters (keep normal letters, numbers, punctuation)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # 3. Normalize multiple whitespace characters (spaces, tabs, newlines) into a single space
    text = re.sub(r"\s+", " ", text)

    # 4. Strip leading and trailing whitespace
    text = text.strip()

    return text


def preprocess_dataset(
    input_csv_path: str = None,
    output_csv_path: str = None
) -> dict:
    """
    Loads raw email dataset, removes duplicate rows, cleans the text column,
    ensures proper data types, and saves the cleaned dataset.

    Args:
        input_csv_path (str, optional): Path to input emails.csv.
        output_csv_path (str, optional): Path to save cleaned_emails.csv.

    Returns:
        dict: Summary statistics of the preprocessing operation.
    """
    # Auto-resolve paths relative to this script if not provided
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if input_csv_path is None:
        input_csv_path = os.path.join(current_dir, "emails.csv")
    if output_csv_path is None:
        output_csv_path = os.path.join(current_dir, "cleaned_emails.csv")

    print(f"Loading raw dataset from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    original_row_count = len(df)

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = original_row_count - len(df)

    # 2. Ensure column types
    # 'text' column as string
    df["text"] = df["text"].astype(str)
    # 'spam' column as integer
    df["spam"] = df["spam"].astype(int)

    # 3. Clean email text column using reusable clean_text function
    print("Cleaning email text...")
    df["text"] = df["text"].apply(clean_text)

    final_row_count = len(df)

    # 4. Calculate label distributions
    label_counts = df["spam"].value_counts().to_dict()
    legitimate_count = label_counts.get(0, 0)
    spam_count = label_counts.get(1, 0)

    # 5. Save cleaned dataset
    print(f"Saving cleaned dataset to: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)

    stats = {
        "original_row_count": original_row_count,
        "duplicate_rows_removed": duplicates_removed,
        "final_row_count": final_row_count,
        "spam_count": spam_count,
        "legitimate_count": legitimate_count,
    }

    return stats


if __name__ == "__main__":
    print("=" * 50)
    print("Email Secure Lens - Preprocessing Pipeline")
    print("=" * 50)

    stats = preprocess_dataset()

    print("\n--- Preprocessing Summary ---")
    print(f"Original row count     : {stats['original_row_count']}")
    print(f"Duplicate rows removed : {stats['duplicate_rows_removed']}")
    print(f"Final row count        : {stats['final_row_count']}")
    print(f"Spam count (label 1)   : {stats['spam_count']}")
    print(f"Legitimate count (0)   : {stats['legitimate_count']}")
    print("=" * 50)
    print("Preprocessing completed successfully!")
