"""
Data Preprocessor Module for Tourism Experience Analytics.
Performs data cleaning, missing value imputation, string standardization, rating validation,
and merges relational tables into consolidated master DataFrames.
"""

from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.loader import load_raw_data


def clean_lookup_tables(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Cleans lookup tables (City, Country, Region, Continent, Type, Mode).

    Args:
        raw_data: Dictionary of raw DataFrames.

    Returns:
        Dict[str, pd.DataFrame]: Cleaned lookup DataFrames.
    """
    cleaned: Dict[str, pd.DataFrame] = {}

    # 1. Continent
    df_cont = raw_data["continent"].copy()
    df_cont["Continent"] = df_cont["Continent"].astype(str).str.strip()
    df_cont.loc[df_cont["Continent"] == "-", "Continent"] = "Unknown"
    cleaned["continent"] = df_cont

    # 2. Region
    df_reg = raw_data["region"].copy()
    df_reg["Region"] = df_reg["Region"].astype(str).str.strip()
    df_reg.loc[df_reg["Region"] == "-", "Region"] = "Unknown"
    cleaned["region"] = df_reg

    # 3. Country
    df_ctr = raw_data["country"].copy()
    df_ctr["Country"] = df_ctr["Country"].astype(str).str.strip()
    df_ctr.loc[df_ctr["Country"] == "-", "Country"] = "Unknown"
    cleaned["country"] = df_ctr

    # 4. City
    df_city = raw_data["city"].copy()
    df_city["CityName"] = df_city["CityName"].astype(str).str.strip()
    df_city.loc[df_city["CityName"].isin(["-", "nan", "None", ""]), "CityName"] = "Unknown"
    df_city["CityId"] = pd.to_numeric(df_city["CityId"], errors="coerce").fillna(0).astype(int)
    cleaned["city"] = df_city

    # 5. Type (Attraction Types)
    df_type = raw_data["type"].copy()
    df_type["AttractionType"] = df_type["AttractionType"].astype(str).str.strip()
    cleaned["type"] = df_type

    # 6. Mode (Visit Modes)
    df_mode = raw_data["mode"].copy()
    df_mode["VisitMode"] = df_mode["VisitMode"].astype(str).str.strip()
    df_mode.loc[df_mode["VisitMode"] == "-", "VisitMode"] = "Unknown"
    cleaned["mode"] = df_mode

    return cleaned


def clean_item_data(raw_data: Dict[str, pd.DataFrame], cleaned_lookups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Cleans and merges attraction items using Updated_Item.xlsx (with fallback to Item.xlsx).

    Args:
        raw_data: Dictionary of raw DataFrames.
        cleaned_lookups: Dictionary of cleaned lookup DataFrames.

    Returns:
        pd.DataFrame: Cleaned and merged attraction master DataFrame.
    """
    df_updated = raw_data.get("updated_item", raw_data["item"]).copy()
    df_base = raw_data["item"].copy()

    # Combine updated items with base items to ensure no missing AttractionIds
    combined_items = pd.concat([df_updated, df_base], ignore_index=True)
    combined_items = combined_items.drop_duplicates(subset=["AttractionId"], keep="first")

    # Clean text columns
    combined_items["Attraction"] = combined_items["Attraction"].astype(str).str.strip()
    combined_items["AttractionAddress"] = combined_items["AttractionAddress"].astype(str).str.strip()

    # Merge attraction types
    df_type = cleaned_lookups["type"]
    combined_items = combined_items.merge(df_type, on="AttractionTypeId", how="left")
    combined_items["AttractionType"] = combined_items["AttractionType"].fillna("Other Attractions")

    # Merge attraction cities
    df_city = cleaned_lookups["city"]
    combined_items = combined_items.merge(
        df_city[["CityId", "CityName", "CountryId"]],
        left_on="AttractionCityId",
        right_on="CityId",
        how="left",
        suffixes=("", "_city_drop")
    )
    combined_items.rename(columns={"CityName": "AttractionCityName", "CountryId": "AttractionCountryId"}, inplace=True)
    if "CityId_city_drop" in combined_items.columns:
        combined_items.drop(columns=["CityId_city_drop"], inplace=True)

    combined_items["AttractionCityName"] = combined_items["AttractionCityName"].fillna("Unknown")

    return combined_items


def clean_user_data(raw_data: Dict[str, pd.DataFrame], cleaned_lookups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Cleans user demographic data and joins geographical hierarchy (Continent, Region, Country, City).

    Args:
        raw_data: Dictionary of raw DataFrames.
        cleaned_lookups: Dictionary of cleaned lookup DataFrames.

    Returns:
        pd.DataFrame: Cleaned and enriched user demographic DataFrame.
    """
    users = raw_data["user"].copy()

    # Handle missing CityId in User table
    users["CityId"] = pd.to_numeric(users["CityId"], errors="coerce").fillna(0).astype(int)
    users["ContinentId"] = pd.to_numeric(users["ContinentId"], errors="coerce").fillna(0).astype(int)
    users["RegionId"] = pd.to_numeric(users["RegionId"], errors="coerce").fillna(0).astype(int)
    users["CountryId"] = pd.to_numeric(users["CountryId"], errors="coerce").fillna(0).astype(int)

    # Merge Continents
    df_cont = cleaned_lookups["continent"]
    users = users.merge(df_cont, on="ContinentId", how="left")

    # Merge Regions
    df_reg = cleaned_lookups["region"]
    users = users.merge(df_reg[["RegionId", "Region"]], on="RegionId", how="left")

    # Merge Countries
    df_ctr = cleaned_lookups["country"]
    users = users.merge(df_ctr[["CountryId", "Country"]], on="CountryId", how="left")

    # Merge Cities
    df_city = cleaned_lookups["city"]
    users = users.merge(df_city[["CityId", "CityName"]], on="CityId", how="left")

    # Fill any remaining NaNs in text columns
    for col in ["Continent", "Region", "Country", "CityName"]:
        if col in users.columns:
            users[col] = users[col].fillna("Unknown")

    return users


def clean_transaction_data(
    raw_data: Dict[str, pd.DataFrame],
    cleaned_lookups: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Cleans transaction logs, validates ratings, and merges visit mode descriptions.

    Args:
        raw_data: Dictionary of raw DataFrames.
        cleaned_lookups: Dictionary of cleaned lookup DataFrames.

    Returns:
        pd.DataFrame: Cleaned transaction logs DataFrame.
    """
    tx = raw_data["transaction"].copy()

    # Validate ratings (clamp strictly to range [1.0, 5.0])
    tx["Rating"] = pd.to_numeric(tx["Rating"], errors="coerce")
    tx["Rating"] = tx["Rating"].clip(lower=1.0, upper=5.0)

    # Standardize VisitYear and VisitMonth
    tx["VisitYear"] = pd.to_numeric(tx["VisitYear"], errors="coerce").fillna(2022).astype(int)
    tx["VisitMonth"] = pd.to_numeric(tx["VisitMonth"], errors="coerce").fillna(1).astype(int)
    tx["VisitModeId"] = pd.to_numeric(tx["VisitMode"], errors="coerce").fillna(0).astype(int)

    # Merge VisitMode text description
    df_mode = cleaned_lookups["mode"]
    tx = tx.merge(df_mode, left_on="VisitModeId", right_on="VisitModeId", how="left", suffixes=("_orig", ""))

    # If VisitMode column was overwritten or duplicated, resolve string column 'VisitMode'
    if "VisitMode_orig" in tx.columns and "VisitMode" in tx.columns:
        tx.rename(columns={"VisitMode": "VisitModeName", "VisitMode_orig": "VisitMode"}, inplace=True)
    elif "VisitMode" in tx.columns and not pd.api.types.is_string_dtype(tx["VisitMode"]):
        # Map integer VisitMode to string name
        mode_map = dict(zip(df_mode["VisitModeId"], df_mode["VisitMode"]))
        tx["VisitModeName"] = tx["VisitMode"].map(mode_map).fillna("Unknown")

    return tx


def build_consolidated_dataset(data_dir: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main pipeline function that loads raw files, cleans them, and merges all relational tables
    into a consolidated master DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            1. Master Consolidated Transactions DataFrame
            2. Enriched User DataFrame
            3. Master Attraction Items DataFrame
    """
    raw_data = load_raw_data(data_dir=data_dir)
    cleaned_lookups = clean_lookup_tables(raw_data)

    df_items = clean_item_data(raw_data, cleaned_lookups)
    df_users = clean_user_data(raw_data, cleaned_lookups)
    df_tx = clean_transaction_data(raw_data, cleaned_lookups)

    # Merge transactions with User demographics
    master = df_tx.merge(df_users, on="UserId", how="left")

    # Merge transactions with Attraction Item details
    master = master.merge(df_items, on="AttractionId", how="left")

    # Final cleanup of any duplicate/redundant columns
    if "CityId_x" in master.columns:
        master.rename(columns={"CityId_x": "UserCityId"}, inplace=True)
    if "CityId_y" in master.columns:
        master.rename(columns={"CityId_y": "AttractionCityId_drop"}, inplace=True)
        master.drop(columns=["AttractionCityId_drop"], inplace=True, errors="ignore")

    return master, df_users, df_items


if __name__ == "__main__":
    print("Executing Data Preprocessing Pipeline...")
    master_df, users_df, items_df = build_consolidated_dataset()
    print(f"[OK] Consolidated Master DataFrame shape: {master_df.shape}")
    print(f"[OK] Enriched Users DataFrame shape: {users_df.shape}")
    print(f"[OK] Master Items DataFrame shape: {items_df.shape}")
    print("Master columns:", master_df.columns.tolist())
