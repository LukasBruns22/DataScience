import pandas as pd
import numpy as np 

def create_interaction_features(df):
    df = df.copy()

    if 'crash_day_of_week' in df.columns:
        df['is_weekend'] = df['crash_day_of_week'].isin([1, 7]).astype(int)

    if 'crash_hour' in df.columns:
        def get_rush_hour(h):
            if 7 <= h <= 9: return 1
            elif 16 <= h <= 18: return 1
            else: return 0
        df['is_rush_hour'] = df['crash_hour'].apply(get_rush_hour)


    is_bad_weather = df['weather_condition'].isin(['RAIN', 'SNOW', 'FREEZING RAIN/DRIZZLE', 'SLEET/HAIL']).astype('int') if 'weather_condition' in df.columns else False
    is_bad_road = df['roadway_surface_cond'].isin(['WET', 'SNOW OR SLUSH', 'ICE']).astype('int') if 'roadway_surface_cond' in df.columns else False
    
    if 'weather_condition' in df.columns and 'roadway_surface_cond' in df.columns:
        df['dangerous_conditions'] = (is_bad_weather | is_bad_road).astype(int)

    if 'lighting_condition' in df.columns:
        dark_conditions = ['DARKNESS, LIGHTED ROAD', 'DARKNESS'] 
        is_dark = df['lighting_condition'].apply(lambda x: 1 if 'DARKNESS' in str(x).upper() else 0)
        
        if 'weather_condition' in df.columns:
             df['poor_visibility'] = (is_dark & is_bad_weather).astype(int)

        
    return df

def create_advanced_features(X_train, X_test):
    # On travaille sur des copies pour ne pas toucher aux originaux par erreur
    df_train = X_train.copy()
    df_test = X_test.copy()
    
    for col in ['prim_contributory_cause', 'first_crash_type', 'trafficway_type']:
        if col in df_train.columns:
            freq_map = df_train[col].value_counts(normalize=True) # Normalize=True donne un %
            
            df_train[f'{col}_freq'] = df_train[col].map(freq_map).fillna(0)
            df_test[f'{col}_freq'] = df_test[col].map(freq_map).fillna(0)

    # ---------------------------------------------------------
    # 2. Aggregats (Groupby Statistics)
    # ---------------------------------------------------------
    # Exemple : Complexité moyenne par type de route
    if 'trafficway_type' in df_train.columns and 'num_units' in df_train.columns:
        # On calcule la moyenne de véhicules impliqués par type de route sur le TRAIN
        mean_units_map = df_train.groupby('trafficway_type')['num_units'].mean()
        
        df_train['avg_units_per_road'] = df_train['trafficway_type'].map(mean_units_map)
        df_test['avg_units_per_road'] = df_test['trafficway_type'].map(mean_units_map).fillna(mean_units_map.mean())
        
        # FEATURE ALGÉBRIQUE : Delta par rapport à la moyenne
        # Est-ce que CE crash a plus de voitures que d'habitude pour cette route ?
        df_train['units_above_avg'] = df_train['num_units'] - df_train['avg_units_per_road']
        df_test['units_above_avg'] = df_test['num_units'] - df_test['avg_units_per_road']

    
    # ---------------------------------------------------------
    # 4. Saisonnalité (Winter Driving)
    # ---------------------------------------------------------
    if 'crash_month' in df_train.columns:
        # Mois d'hiver à Chicago (Décembre, Janvier, Février, Mars)
        winter_months = [12, 1, 2, 3]
        df_train['is_winter'] = df_train['crash_month'].apply(lambda x: 1 if x in winter_months else 0)
        df_test['is_winter'] = df_test['crash_month'].apply(lambda x: 1 if x in winter_months else 0)


    return df_train, df_test

import pandas as pd
import numpy as np 

# ======================================================
# 1. SIMPLE INTERACTION FEATURES  (your first function)
# ======================================================
def create_interaction_features(df):
    df = df.copy()

    # Weekend
    if 'crash_day_of_week' in df.columns:
        df['is_weekend'] = df['crash_day_of_week'].isin([1, 7]).astype(int)

    # Rush hour
    if 'crash_hour' in df.columns:
        def get_rush_hour(h):
            if 7 <= h <= 9: return 1
            elif 16 <= h <= 18: return 1
            else: return 0
        df['is_rush_hour'] = df['crash_hour'].apply(get_rush_hour)

    # Dangerous conditions
    is_bad_weather = df['weather_condition'].isin(
        ['RAIN', 'SNOW', 'FREEZING RAIN/DRIZZLE', 'SLEET/HAIL']
    ).astype('int') if 'weather_condition' in df.columns else False

    is_bad_road = df['roadway_surface_cond'].isin(
        ['WET', 'SNOW OR SLUSH', 'ICE']
    ).astype('int') if 'roadway_surface_cond' in df.columns else False
    
    if isinstance(is_bad_weather, pd.Series) and isinstance(is_bad_road, pd.Series):
        df['dangerous_conditions'] = (is_bad_weather | is_bad_road).astype(int)

    # Poor visibility
    if 'lighting_condition' in df.columns:
        is_dark = df['lighting_condition'].apply(lambda x: 1 if 'DARKNESS' in str(x).upper() else 0)
        if isinstance(is_bad_weather, pd.Series):
            df['poor_visibility'] = (is_dark & is_bad_weather).astype(int)

    return df


# ======================================================
# 2. MORE ADVANCED FEATURES  (your second function)
# ======================================================
def create_advanced_features(X_train, X_test):
    df_train = X_train.copy()
    df_test = X_test.copy()
    
    # Frequency encoding
    for col in ['prim_contributory_cause', 'first_crash_type', 'trafficway_type']:
        if col in df_train.columns:
            freq_map = df_train[col].value_counts(normalize=True)
            df_train[f'{col}_freq'] = df_train[col].map(freq_map).fillna(0)
            df_test[f'{col}_freq'] = df_test[col].map(freq_map).fillna(0)

    # Group-based averages
    if 'trafficway_type' in df_train.columns and 'num_units' in df_train.columns:
        mean_units_map = df_train.groupby('trafficway_type')['num_units'].mean()
        
        df_train['avg_units_per_road'] = df_train['trafficway_type'].map(mean_units_map)
        df_test['avg_units_per_road'] = df_test['trafficway_type'].map(mean_units_map).fillna(mean_units_map.mean())
        
        df_train['units_above_avg'] = df_train['num_units'] - df_train['avg_units_per_road']
        df_test['units_above_avg'] = df_test['num_units'] - df_test['avg_units_per_road']

    # Winter season feature
    if 'crash_month' in df_train.columns:
        winter_months = [12, 1, 2, 3]
        df_train['is_winter'] = df_train['crash_month'].isin(winter_months).astype(int)
        df_test['is_winter'] = df_test['crash_month'].isin(winter_months).astype(int)

    return df_train, df_test


# ======================================================
# 3. WRAPPER FOR THE PIPELINE  (THIS IS THE IMPORTANT PART)
# ======================================================
def generate_features(X_train, X_test, strategy="simple"):
    """
    This is the function used by the tournament pipeline.
    
    MUST RETURN: X_train_new, X_test_new
    """

    if strategy == "simple":
        return create_interaction_features(X_train), create_interaction_features(X_test)

    elif strategy == "advanced":
        return create_advanced_features(X_train, X_test)

    # Default fallback
    return X_train, X_test
