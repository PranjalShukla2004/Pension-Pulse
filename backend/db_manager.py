"""
Module for managing database queries for precomputed prediction data.
"""

import sqlite3
from typing import Any, Dict, List, Tuple

def get_all_predictions(db_path: str) -> List[Tuple[Any, ...]]:
    """
    Retrieves all prediction records from the specified SQLite database.
    
    :param db_path: Path to the SQLite database file.
    :return: List of prediction records as tuples.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions')
    rows = cursor.fetchall()
    conn.close()
    return rows

def query_predictions(db_path: str, filters: Dict[str, float]) -> List[Tuple[Any, ...]]:
    """
    Queries prediction records based on provided filters.
    
    :param db_path: Path to the SQLite database file.
    :param filters: Dictionary of column filters (e.g., {"wealth": 500}).
    :return: List of filtered prediction records as tuples.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT * FROM predictions"
    conditions = []
    params = []
    for key, value in filters.items():
        conditions.append(f"{key} = ?")
        params.append(value)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    return rows
