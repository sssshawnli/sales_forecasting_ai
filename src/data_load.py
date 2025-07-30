import pandas as pd

# Load OCRD customer master data
def load_ocrd(file_path: str = "/home/shawn/pd_project/data/raw/OCRD.csv") -> pd.DataFrame:
    # Only load necessary columns
    use_cols = ['CardCode','CardType','City','Country']    

    # Set expected data types
    d_type = {
        'CardCode': str,
        'CardType': str,
        'City': str,
        'Country': str
    }

    # Read file in chunks for memory efficiency
    chunks = pd.read_csv(file_path, sep=',', usecols=use_cols, chunksize=100_000, dtype=d_type)

    frames = []

    # Filter only customer rows (CardType == 'C')
    for chunk in chunks:
        filtered = chunk[(chunk['CardType'] == 'C')]
        frames.append(filtered)

    # Combine all chunks and sort
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values('CardCode')

    return df

# Load ORDR sales order header data
def load_ordr(file_path: str = "/home/shawn/pd_project/data/raw/ORDR.csv") -> pd.DataFrame:
    use_cols = ['DocEntry','DocNum','CANCELED','DocStatus','DocDate','CardCode','NumAtCard','DocTotal']

    d_type = {
        'DocEntry': str,
        'DocNum': str,
        'CANCELED': str,
        'DocStatus': str,
        'DocDate': str,
        'CardCode': str,
        'NumAtCard': str,
        'DocTotal': float
    }

    chunks = pd.read_csv(file_path, sep=',', usecols=use_cols, chunksize=100_000, dtype=d_type)
    frames = []

    # Filter: exclude canceled, open, and zero-value orders
    for chunk in chunks:
        filtered = chunk[
            (chunk['CANCELED'] == 'N') &
            (chunk['DocTotal'] != 0.00) &
            (chunk['DocStatus'] != 'O')             
        ]
        frames.append(filtered)

    df = pd.concat(frames, ignore_index=True)

    # Exclude specific internal accounts
    excluded_cardcode = ["C00151", "C00104"]
    df = df[~df['CardCode'].isin(excluded_cardcode)]

    return df

# Load RDR1 sales order line data
def load_rdr1(file_path: str = "/home/shawn/pd_project/data/raw/RDR1.csv") -> pd.DataFrame:
    use_cols = ['DocEntry', 'TrgetEntry','LineNum', 'ItemCode','LineTotal']

    d_type = {
        'DocEntry': str, 
        'TrgetEntry': str,
        'LineNum': str, 
        'ItemCode': str, 
        'LineTotal': str
    }

    chunks = pd.read_csv(file_path, sep=',', usecols=use_cols, chunksize=100_000, dtype=d_type)
    frames = []

    # Convert LineTotal to numeric, drop non-numeric
    for chunk in chunks:
        chunk["LineTotal"] = pd.to_numeric(chunk["LineTotal"], errors="coerce")
        filtered = chunk.dropna(subset=["LineTotal"])
        frames.append(filtered)

    df = pd.concat(frames, ignore_index=True)
    return df

# Load ODLN delivery document header data
def load_odln(file_path: str = "/home/shawn/pd_project/data/raw/ODLN.csv") -> pd.DataFrame:
    use_cols = ['DocEntry','DocNum','CANCELED','DocStatus','DocDate','CardCode','NumAtCard','DocTotal']

    d_type = {
        'DocEntry': str,
        'DocNum': str,
        'CANCELED': str,
        'DocStatus': str,
        'DocDate': str,
        'CardCode': str,
        'NumAtCard': str,
        'DocTotal': float
    }

    chunks = pd.read_csv(file_path, sep=',', usecols=use_cols, chunksize=100_000, dtype=d_type)
    frames = []

    # Filter: only completed deliveries that were not canceled
    for chunk in chunks:
        filtered = chunk[
            (chunk['CANCELED'] == 'N') &
            (chunk['DocStatus'] == 'C')             
        ]
        frames.append(filtered)

    df = pd.concat(frames, ignore_index=True)

    # Exclude internal accounts
    excluded_cardcode = ["C00151", "C00104"]
    df = df[~df['CardCode'].isin(excluded_cardcode)]

    return df

# Load OITM item master data
def load_oitm(file_path: str = "/home/shawn/pd_project/data/raw/OITM.csv") -> pd.DataFrame:
    use_cols = ('ItemCode','Category')

    d_type = {
        'ItemCode': str,
        'Category': str
    }

    chunks = pd.read_csv(file_path, sep=',', usecols=use_cols, chunksize=100_000, dtype=d_type)
    frames = []

    for chunk in chunks:
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)

    # Exclude service categories or irrelevant items
    df = df[
        (df["Category"] != 'Bank Fees')  &
        (df["Category"] != 'Components') &
        (df["Category"] != 'FreeGood Shipping') &
        (df["Category"] != 'Items') &
        (df["Category"] != 'Merchant Charges') &
        (df["Category"] != 'Others') &
        (df["Category"] != 'Repair & Maintenance') &
        (df["Category"] != 'Machinery & Equip')
    ]
    return df
