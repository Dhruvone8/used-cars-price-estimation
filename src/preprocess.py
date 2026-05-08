import pandas as pd
import numpy as np


METRO_MAP = {
    'Navi Mumbai': 'Mumbai', 'Thane': 'Mumbai',
    'Gurgaon': 'Delhi', 'Noida': 'Delhi', 'Faridabad': 'Delhi',
    'Mohali': 'Chandigarh', 'Zirakpur': 'Chandigarh', 'Panchkula': 'Chandigarh',
    'Pimpri-Chinchwad': 'Pune',
    'Ernakulam': 'Kochi', 'Kollam': 'Kochi',
    'Ranga Reddy': 'Hyderabad', 'Warangal': 'Hyderabad',
    'Dak. Kannada': 'Mangalore',
}

OWNER_MAP = {'First': 1, 'Second': 2, 'Third': 3}

FUEL_FIX = {'CNG + CNG': 'CNG', 'Petrol + CNG': 'CNG'}

MODEL_FIX = {
    'Wagon':  'Wagon R',
    'Vitara': 'Vitara Brezza',
    '3':      '3 Series',
}

BRAND_MODEL_FIX = {
    ('Hyundai',       'Grand'): 'Grand i10',
    ('Maruti Suzuki', 'Grand'): 'Grand Vitara',
    ('Mercedes-Benz', 'New'):   'C-Class',
    ('Tata',          'New'):   'Safari',
}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # ── Brand ────────────────────────────────────────
    df['Brand'] = df['Brand'].replace({
        'Maruti': 'Maruti Suzuki',
        'Land':   'Land Rover',
        'MINI':   'Mini',
    })
    brand_counts = df['Brand'].value_counts()
    df = df[df['Brand'].isin(brand_counts[brand_counts >= 100].index)]

    # ── Model Base ───────────────────────────────────
    df['Model_Base'] = df['Model'].str.split().str[0]

    df['Model_Normalized'] = (df['Model_Base']
                              .str.lower()
                              .str.replace('-', '', regex=False)
                              .str.replace(' ', '', regex=False)
                              .str.strip())

    model_counts     = df['Model_Base'].value_counts()
    duplicate_groups = df.groupby('Model_Normalized')['Model_Base'].unique()
    duplicate_groups = duplicate_groups[duplicate_groups.apply(len) > 1]

    replace_map = {}
    for norm, variants in duplicate_groups.items():
        canonical = max(variants, key=lambda v: model_counts.get(v, 0))
        for variant in variants:
            if variant != canonical:
                replace_map[variant] = canonical

    df['Model_Base'] = df['Model_Base'].replace(replace_map)
    df.drop(columns=['Model_Normalized', 'Model'], inplace=True)

    # Fix display names
    df['Model_Base'] = df.apply(lambda row:
        BRAND_MODEL_FIX.get(
            (row['Brand'], row['Model_Base']),
            MODEL_FIX.get(row['Model_Base'], row['Model_Base'])
        )
    , axis=1)

    df = df[df['Model_Base'] != 'New']

    model_counts = df['Model_Base'].value_counts()
    df = df[df['Model_Base'].isin(model_counts[model_counts >= 75].index)]

    # ── Year ─────────────────────────────────────────
    df = df[df['Year'] >= 2004]

    # ── Fuel ─────────────────────────────────────────
    df['Fuel'] = df['Fuel'].replace(FUEL_FIX)
    df = df[df['Fuel'] != 'LPG']

    # ── Km ───────────────────────────────────────────
    df = df[(df['Km'] >= 500) & (df['Km'] <= 500000)]

    # ── Owner ────────────────────────────────────────
    df['Owner'] = df['Owner'].replace({'Fourth': 'Fourth & Above'})
    df = df[~df['Owner'].isin(['UnRegistered Car', 'Fourth & Above'])]
    df['Owner'] = df['Owner'].map(OWNER_MAP)

    # ── Location ─────────────────────────────────────
    df['Location'] = df['Location'].replace(METRO_MAP)
    city_counts = df['Location'].value_counts()
    df = df[df['Location'].isin(city_counts[city_counts >= 50].index)]

    # ── Seats ────────────────────────────────────────
    df = df[~df['Seats'].isin([0.0, 9.0])]

    # ── Feature Engineering ──────────────────────────
    df['Car_Age']     = 2024 - df['Year']
    df['Km_Per_Year'] = df['Km'] / df['Car_Age'].replace(0, 1)
    df['Log_Price']   = np.log1p(df['Price'])

    return df.reset_index(drop=True)


def encode(df: pd.DataFrame):
    from sklearn.preprocessing import LabelEncoder

    encoders = {}
    for col in ['Brand', 'Model_Base', 'Location']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    df = pd.get_dummies(
        df, columns=['Fuel', 'Transmission'],
        drop_first=True, dtype=int
    )

    df = df.rename(columns={
        'Car_Age':             'Car Age',
        'Km_Per_Year':         'Km Per Year',
        'Log_Price':           'Log Price',
        'Fuel_Diesel':         'Diesel',
        'Fuel_Petrol':         'Petrol',
        'Transmission_Manual': 'Manual Transmission',
        'Model_Base':          'Base Model',
    })

    return df, encoders


def get_feature_columns():
    return [
        'Brand', 'Km', 'Owner', 'Location', 'Engine', 'Seats',
        'Base Model', 'Car Age', 'Km Per Year',
        'Diesel', 'Petrol', 'Manual Transmission'
    ]