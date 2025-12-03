import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_and_split_data(
    file_path: str, 
    target_column: str, 
    test_size: float = 0.3, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads data from a file path and splits it into training and testing sets.

    Args:
        file_path (str): The path to the data file (e.g., 'data.csv').
        target_column (str): The name of the column to be used as the target variable (y).
        test_size (float): The proportion of the data to be used for the test set (default is 0.3).
        random_state (int): Seed for random number generator for reproducibility (default is 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
        (X_train, X_test, y_train, y_test)
    """
    print(f"--- Loading data from: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found.")
        raise ValueError(f"File '{file_path}' not found.")
        
    if target_column not in df.columns:
        print(f"ERROR: Target column '{target_column}' not found in the data.")
        raise ValueError(f"Target column '{target_column}' missing.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]


    # Encode target variable
    target_mapping = {
        "NO INJURY / DRIVE AWAY": 0,
        "INJURY AND / OR TOW DUE TO CRASH": 1
    }
    y = y.map(target_mapping)
    if y.isna().any():
        print("ATTENTION : Certaines valeurs de la target n'ont pas été trouvées dans le mapping (elles sont devenues NaN).")
    else:
        print("Target encoded successfully.")

    # Replace 'Unknown' placeholders with NaN
    missing_placeholders = [
        "UNKNOWN", "Unknown", "unknown", 
        "UNABLE TO DETERMINE", "NOT APPLICABLE", 
        "nan", "?", "-"
    ]
    
    X = X.replace(missing_placeholders, np.nan)

    X = X.drop(columns=["crash_date"])

    X = simplify_categories(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y 
    )

    print(f"Data shape (X_train): {X_train.shape}")
    print(f"Data shape (X_test): {X_test.shape}")
    
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), \
           y_train.reset_index(drop=True), y_test.reset_index(drop=True)

def simplify_categories(df):
    df = df.copy()

    # 1. First Crash Type (first_crash_type)
    def map_crash_type(val):
        veh_to_veh = [
            'TURNING', 'ANGLE', 'REAR END', 'SIDESWIPE SAME DIRECTION', 
            'SIDESWIPE OPPOSITE DIRECTION', 'HEAD ON', 'REAR TO FRONT', 'REAR TO SIDE'
        ]
        non_motorist = ['PEDESTRIAN', 'PEDALCYCLIST']
        
        if val in veh_to_veh:
            return 'VEHICLE_TO_VEHICLE'
        elif val in non_motorist:
            return 'VEHICLE_TO_PERSON_CYCLIST'
        else:
            return 'SINGLE_VEHICLE_STATIC'

    if 'first_crash_type' in df.columns:
        df['first_crash_type'] = df['first_crash_type'].apply(map_crash_type)

    # 2. Road Defect (road_defect)
    if 'road_defect' in df.columns:
        no_defect_list = ['NO DEFECTS', 'UNKNOWN']
        
        df['road_defect_present'] = df['road_defect'].apply(lambda x: 0 if x in no_defect_list else 1)
        df.drop(columns=['road_defect'], inplace=True)

    # 3. Traffic Control Device (traffic_control_device)
    def map_traffic_control(val):
        signalized = ['TRAFFIC SIGNAL', 'FLASHING CONTROL SIGNAL', 'LANE USE MARKING']
        sign_controlled = [
            'STOP SIGN/FLASHER', 'YIELD', 'OTHER REG. SIGN', 
            'PEDESTRIAN CROSSING SIGN', 'RAILROAD CROSSING GATE', 
            'SCHOOL ZONE', 'POLICE/FLAGMAN'
        ]
        no_control = ['NO CONTROLS']
        
        if val in signalized:
            return 'SIGNALIZED'
        elif val in sign_controlled:
            return 'SIGN_CONTROLLED'
        elif val in no_control:
            return 'NO_CONTROL'
        else:
            return 'UNKNOWN'

    if 'traffic_control_device' in df.columns:
        df['traffic_control_device'] = df['traffic_control_device'].apply(map_traffic_control)

    # 4. Weather Condition (weather_condition)
    def map_weather(val):
        winter = ['SNOW', 'SLEET/HAIL', 'FREEZING RAIN/DRIZZLE', 'BLOWING SNOW']
        
        if val == 'CLEAR':
            return 'CLEAR'
        elif val == 'RAIN':
            return 'RAIN'
        elif val in winter:
            return 'ADVERSE_WINTER'
        elif val == 'UNKOWN':
            return 'UNKNOWN'
        else: 
            return 'OTHER'

    if 'weather_condition' in df.columns:
        df['weather_condition'] = df['weather_condition'].apply(map_weather)

    # 5. Trafficway Type (trafficway_type)
    def map_trafficway(val):
        val_str = str(val).upper()
        if 'DIVIDED' in val_str:
            return 'DIVIDED'
        elif 'ONE-WAY' in val_str:
            return 'ONE_WAY'
        elif 'UNKNOWN' in val_str:
            return 'UNKNOWN'
        elif 'INTERSECTION' in val_str or 'ROUNDABOUT' in val_str:
            return 'INTERSECTION'
        else:
            return 'UNDIVIDED'

    if 'trafficway_type' in df.columns:
        df['trafficway_type'] = df['trafficway_type'].apply(map_trafficway)

    # 6. Roadway Surface Condition (roadway_surface_cond)
    def map_surface(val):
        frozen = ['SNOW OR SLUSH', 'ICE']
        
        if val == 'DRY':
            return 'DRY'
        elif val == 'WET':
            return 'WET'
        elif val in frozen:
            return 'FROZEN_HAZARDOUS'
        elif val == 'UNKNOWN':
            return 'UNKNOWN'
        else:
            return 'OTHER'

    if 'roadway_surface_cond' in df.columns:
        df['roadway_surface_cond'] = df['roadway_surface_cond'].apply(map_surface)

    # 7. Primary Contributory Cause (prim_contributory_cause)
    def map_cause(val):
        unknown_list = [
            'UNABLE TO DETERMINE', 
            'NOT APPLICABLE'
        ]
        
        distracted_list = [
            'DISTRACTION - FROM OUTSIDE VEHICLE',
            'DISTRACTION - FROM INSIDE VEHICLE',
            'DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)',
            'CELL PHONE USE OTHER THAN TEXTING',
            'TEXTING'
        ]
        
        external_list = [
            'WEATHER', 
            'EQUIPMENT - VEHICLE CONDITION',
            'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)',
            'EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST',
            'ROAD ENGINEERING/SURFACE/MARKING DEFECTS',
            'ROAD CONSTRUCTION/MAINTENANCE',
            'ANIMAL',
            'RELATED TO BUS STOP',
            'OBSTRUCTED CROSSWALKS',
            'BICYCLE ADVANCING LEGALLY ON RED LIGHT',
            'MOTORCYCLE ADVANCING LEGALLY ON RED LIGHT'
        ]
        if val in unknown_list:
            return 'UNKNOWN'
        elif val in distracted_list:
            return 'DISTRACTED_DRIVING'
        elif val in external_list:
            return 'EXTERNAL_FACTOR'
        else:
            return 'DRIVER_ERROR'

    if 'prim_contributory_cause' in df.columns:
        df['prim_contributory_cause'] = df['prim_contributory_cause'].apply(map_cause)

    return df
