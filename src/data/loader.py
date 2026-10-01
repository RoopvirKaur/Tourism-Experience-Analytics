"""
Data Loader Module for Tourism Experience Analytics.
Loads raw Excel files from the dataset directory into Pandas DataFrames without modifying raw source files.
"""

from pathlib import Path
from typing import Dict, Union, Optional
import pandas as pd


def get_default_data_dir() -> Path:
    """Returns the absolute path to the default dataset directory."""
    base_path = Path(__file__).resolve().parent.parent.parent
    return base_path / "dataset"


def load_excel_file(file_path: Path) -> pd.DataFrame:
    """Helper to safely read an Excel file into a DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    return pd.read_excel(file_path)


def load_raw_data(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all original dataset Excel workbooks.

    Args:
        data_dir: Optional path to the dataset folder. Defaults to project 'dataset/' directory.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of dataset names to raw DataFrames.
    """
    if data_dir is None:
        data_path = get_default_data_dir()
    else:
        data_path = Path(data_dir)

    raw_data: Dict[str, pd.DataFrame] = {}

    # File mapping: key -> relative path
    file_map = {
        "transaction": "Transaction.xlsx",
        "user": "User.xlsx",
        "city": "City.xlsx",
        "continent": "Continent.xlsx",
        "country": "Country.xlsx",
        "region": "Region.xlsx",
        "type": "Type.xlsx",
        "mode": "Mode.xlsx",
        "item": "Item.xlsx",
    }

    for key, filename in file_map.items():
        target = data_path / filename
        if target.exists():
            raw_data[key] = load_excel_file(target)
        else:
            raise FileNotFoundError(f"Required dataset file missing: {target}")

    # Check for Updated_Item.xlsx in additional folder (with underscore or spaces)
    updated_item_path = data_path / "Additional_Data_for_Attraction_Sites" / "Updated_Item.xlsx"
    if not updated_item_path.exists():
        # Fallback check for space in directory name if applicable
        updated_item_path = data_path / "Additional Data for Attraction Sites" / "Updated_Item.xlsx"

    if updated_item_path.exists():
        raw_data["updated_item"] = load_excel_file(updated_item_path)
    else:
        # Fallback to standard item dataframe if updated dataset isn't found
        raw_data["updated_item"] = raw_data["item"]

    return raw_data


if __name__ == "__main__":
    data = load_raw_data()
    print("Successfully loaded datasets:")
    for name, df in data.items():
        print(f" - {name}: {df.shape[0]} rows, {df.shape[1]} columns")
