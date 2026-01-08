import pandas as pd
from pathlib import Path

def load_employee_data():
    """
    Load employee data from a CSV file into a pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing employee data.
    """
    BASE_DIR = Path(__file__).resolve().parents[1]
    CSV_PATH = BASE_DIR / "data" / "employees.csv"
    df = pd.read_csv(CSV_PATH)
    return df

if __name__ == "__main__":
    data = load_employee_data()    
    print(data.head())
    print(data.info())
    print(data.describe())
    print(data['salary'].value_counts())
