import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# ==========================================
# 1. IMPROVED SEMANTIC BINARY ENCODING
# ==========================================

def build_hierarchy_paths(tree, current_path=None, paths_lookup=None):
    """
    Flattens the dictionary tree into a lookup of {Leaf_Value: [Root, Child, Leaf]}.
    This tells us which bits to turn on for a specific value.
    """
    if paths_lookup is None:
        paths_lookup = {}
    if current_path is None:
        current_path = []

    if isinstance(tree, dict):
        for key, value in tree.items():
            new_path = current_path + [key]
            build_hierarchy_paths(value, new_path, paths_lookup)
    elif isinstance(tree, list):
        for item in tree:
            paths_lookup[item] = current_path + [item]
    else:
        paths_lookup[tree] = current_path + [tree]
        
    return paths_lookup

def get_parent_nodes_only(paths_lookup):
    """Extract only parent/intermediate nodes, excluding leaf values."""
    parent_nodes = set()
    for path in paths_lookup.values():
        # Add all nodes except the last one (which is the leaf)
        if len(path) > 1:
            parent_nodes.update(path[:-1])
    return sorted(list(parent_nodes))

def get_leaf_nodes_only(paths_lookup):
    """Extract only leaf nodes (actual values)."""
    return sorted(list(paths_lookup.keys()))

def apply_semantic_binary_encoding(df, col_name, tree, prefix, mode='parents_only'):
    """
    Transforms a single categorical column into a Semantic Multi-Hot DataFrame.
    
    Parameters:
    -----------
    mode : str
        'parents_only' - Only activate parent/intermediate nodes (recommended)
        'leaves_only' - Only activate leaf nodes (equivalent to one-hot)
        'full_path' - Activate entire path (original, causes multicollinearity)
    """
    # 1. Build the lookup
    path_map = build_hierarchy_paths(tree)
    
    # 2. Determine which nodes to use based on mode
    if mode == 'parents_only':
        all_nodes = get_parent_nodes_only(path_map)
    elif mode == 'leaves_only':
        all_nodes = get_leaf_nodes_only(path_map)
    else:  # full_path
        all_nodes = sorted(set(node for path in path_map.values() for node in path))
    
    # 3. Create empty binary matrix
    binary_matrix = np.zeros((len(df), len(all_nodes)), dtype=int)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    # 4. Fill matrix based on mode
    temp_series = df[col_name].reset_index(drop=True)
    
    for i, val in enumerate(temp_series):
        if val in path_map:
            full_path = path_map[val]
            
            if mode == 'parents_only':
                # Only activate parent nodes (exclude the leaf itself)
                nodes_to_activate = full_path[:-1]
            elif mode == 'leaves_only':
                # Only activate the leaf node
                nodes_to_activate = [full_path[-1]]
            else:  # full_path
                # Activate entire path
                nodes_to_activate = full_path
            
            for node in nodes_to_activate:
                if node in node_to_idx:
                    binary_matrix[i, node_to_idx[node]] = 1
            
    # 5. Return as DataFrame
    new_cols = [f"{prefix}_{node}" for node in all_nodes]
    return pd.DataFrame(binary_matrix, columns=new_cols, index=df.index)

# ==========================================
# 2. MAIN PIPELINE FUNCTION
# ==========================================

def encode_and_impute(X_train, X_test, mvi_strategy='statistical', encoding_strategy='semantic', semantic_mode='parents_only'):
    """
    Applies MVI and Categorical Encoding.
    
    Parameters:
    -----------
    mvi_strategy : str
        'statistical' - mean for numeric, mode for categorical
        'constant' - -1 for numeric, 'Missing_Data' for categorical
    encoding_strategy : str
        'semantic' - Uses hierarchical encoding
        'onehot' - Standard one-hot encoding
    semantic_mode : str
        'parents_only' - Only encode parent categories (reduces multicollinearity)
        'leaves_only' - Only encode leaf values (similar to one-hot)
        'full_path' - Encode entire path (original, not recommended)
    """
    
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    # ==========================================
    # A. MISSING VALUE IMPUTATION
    # ==========================================
    num_cols = X_train_enc.select_dtypes(include=['number']).columns
    cat_cols = X_train_enc.select_dtypes(include=['object', 'category']).columns

    if mvi_strategy == 'statistical':
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            X_train_enc[num_cols] = num_imputer.fit_transform(X_train_enc[num_cols])
            X_test_enc[num_cols] = num_imputer.transform(X_test_enc[num_cols])
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train_enc[cat_cols] = cat_imputer.fit_transform(X_train_enc[cat_cols])
            X_test_enc[cat_cols] = cat_imputer.transform(X_test_enc[cat_cols])

    elif mvi_strategy == 'constant':
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='constant', fill_value=-1)
            X_train_enc[num_cols] = num_imputer.fit_transform(X_train_enc[num_cols])
            X_test_enc[num_cols] = num_imputer.transform(X_test_enc[num_cols])
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing_Data')
            X_train_enc[cat_cols] = cat_imputer.fit_transform(X_train_enc[cat_cols])
            X_test_enc[cat_cols] = cat_imputer.transform(X_test_enc[cat_cols])

    # ==========================================
    # B. ENCODING
    # ==========================================

    # --- 1. BINARY & CYCLICAL (Always Applied) ---
    binary_mapping = {'Y': 1, 'N': 0}
    if 'intersection_related_i' in X_train_enc.columns:
        X_train_enc['intersection_related_i'] = X_train_enc['intersection_related_i'].map(binary_mapping).fillna(0).astype(int)
        X_test_enc['intersection_related_i'] = X_test_enc['intersection_related_i'].map(binary_mapping).fillna(0).astype(int)

    cyclical_cols = {'crash_hour': 23, 'crash_month': 12, 'crash_day_of_week': 7}
    for col, max_val in cyclical_cols.items():
        if col in X_train_enc.columns:
            X_train_enc[f'{col}_sin'] = np.sin(2 * np.pi * X_train_enc[col] / max_val)
            X_train_enc[f'{col}_cos'] = np.cos(2 * np.pi * X_train_enc[col] / max_val)
            X_test_enc[f'{col}_sin'] = np.sin(2 * np.pi * X_test_enc[col] / max_val)
            X_test_enc[f'{col}_cos'] = np.cos(2 * np.pi * X_test_enc[col] / max_val)
            X_train_enc.drop(columns=[col], inplace=True)
            X_test_enc.drop(columns=[col], inplace=True)

    # --- 2. SEMANTIC HIERARCHIES ---
    if encoding_strategy == 'semantic':
        # Primary Cause Tree
        cause_tree = {
            'DRIVER_ERROR': {
                'NEGLIGENCE': ['FOLLOWING TOO CLOSELY', 'FAILING TO YIELD RIGHT-OF-WAY', 'IMPROPER BACKING', 
                              'IMPROPER LANE USAGE', 'IMPROPER OVERTAKING/PASSING', 'IMPROPER TURNING/NO SIGNAL', 
                              'FAILING TO REDUCE SPEED TO AVOID CRASH', 'DRIVING SKILLS/KNOWLEDGE/EXPERIENCE'],
                'RECKLESSNESS': ['DRIVING ON WRONG SIDE/WRONG WAY', 'EXCEEDING AUTHORIZED SPEED LIMIT', 
                                'OPERATING VEHICLE IN ERRATIC, RECKLESS, CARELESS, NEGLIGENT OR AGGRESSIVE MANNER', 
                                'DISREGARDING TRAFFIC SIGNALS', 'DISREGARDING STOP SIGN', 'TURNING RIGHT ON RED', 
                                'DISREGARDING ROAD MARKINGS', 'DISREGARDING OTHER TRAFFIC SIGNS'],
                'IMPAIRMENT': ['PHYSICAL CONDITION OF DRIVER', 'HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)', 
                              'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)', 
                              'DISTRACTION - FROM INSIDE VEHICLE', 'DISTRACTION - FROM OUTSIDE VEHICLE', 
                              'CELL PHONE USE OTHER THAN TEXTING', 
                              'DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)']
            },
            'VEHICLE_DEFECT': ['EQUIPMENT - VEHICLE CONDITION', 'BICYCLE ADVANCING LEGALLY ON RED LIGHT'],
            'EXTERNAL': {
                'ENVIRONMENT': ['WEATHER', 'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)', 'ANIMAL', 
                               'EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST'],
                'INFRASTRUCTURE': ['ROAD ENGINEERING/SURFACE/MARKING DEFECTS', 'ROAD CONSTRUCTION/MAINTENANCE', 
                                  'RELATED TO BUS STOP']
            },
            'UNKNOWN': ['UNABLE TO DETERMINE', 'NOT APPLICABLE', 'Missing_Data']
        }
        
        # Trafficway Tree
        traffic_tree = {
            'INTERSECTION': ['FOUR WAY', 'T-INTERSECTION', 'Y-INTERSECTION', 'FIVE POINT, OR MORE', 
                           'ROUNDABOUT', 'L-INTERSECTION', 'UNKNOWN INTERSECTION TYPE'],
            'NON_INTERSECTION': {
                'HIGH_SPEED': ['DIVIDED - W/MEDIAN (NOT RAISED)', 'DIVIDED - W/MEDIAN BARRIER', 'ONE-WAY', 'RAMP'],
                'STANDARD': ['NOT DIVIDED', 'CENTER TURN LANE', 'TRAFFIC ROUTE'],
                'MINOR': ['PARKING LOT', 'DRIVEWAY', 'ALLEY']
            },
            'OTHER': ['UNKNOWN', 'NOT REPORTED', 'Missing_Data']
        }

        # Apply Semantic Binary Logic to Cause
        if 'prim_contributory_cause' in X_train_enc.columns:
            cause_df_train = apply_semantic_binary_encoding(
                X_train_enc, 'prim_contributory_cause', cause_tree, 
                prefix='cause', mode=semantic_mode
            )
            cause_df_test = apply_semantic_binary_encoding(
                X_test_enc, 'prim_contributory_cause', cause_tree, 
                prefix='cause', mode=semantic_mode
            )
            
            X_train_enc = pd.concat([X_train_enc.drop(columns=['prim_contributory_cause']), cause_df_train], axis=1)
            X_test_enc = pd.concat([X_test_enc.drop(columns=['prim_contributory_cause']), cause_df_test], axis=1)

        # Apply Semantic Binary Logic to Traffic
        if 'trafficway_type' in X_train_enc.columns:
            traffic_df_train = apply_semantic_binary_encoding(
                X_train_enc, 'trafficway_type', traffic_tree, 
                prefix='traffic', mode=semantic_mode
            )
            traffic_df_test = apply_semantic_binary_encoding(
                X_test_enc, 'trafficway_type', traffic_tree, 
                prefix='traffic', mode=semantic_mode
            )
            
            X_train_enc = pd.concat([X_train_enc.drop(columns=['trafficway_type']), traffic_df_train], axis=1)
            X_test_enc = pd.concat([X_test_enc.drop(columns=['trafficway_type']), traffic_df_test], axis=1)

    # --- 3. STANDARD ORDINAL (Always Applied) ---
    damage_order = ['$500 OR LESS', '501 - 1500', 'OVER $1500']
    injury_order = ['NO INDICATION OF INJURY', 'REPORTED, NOT EVIDENT', 'NONINCAPACITATING INJURY', 
                   'INCAPACITATING INJURY', 'FATAL']
    
    if 'damage' in X_train_enc.columns:
        od = OrdinalEncoder(categories=[damage_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_enc['damage'] = od.fit_transform(X_train_enc[['damage']])
        X_test_enc['damage'] = od.transform(X_test_enc[['damage']])

    if 'most_severe_injury' in X_train_enc.columns:
        od = OrdinalEncoder(categories=[injury_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_enc['most_severe_injury'] = od.fit_transform(X_train_enc[['most_severe_injury']])
        X_test_enc['most_severe_injury'] = od.transform(X_test_enc[['most_severe_injury']])

    # --- 4. ONE-HOT FOR REMAINING ---
    remaining_cat = X_train_enc.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if remaining_cat:
        oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        oh.fit(X_train_enc[remaining_cat])
        
        tr_oh = pd.DataFrame(oh.transform(X_train_enc[remaining_cat]), 
                            columns=oh.get_feature_names_out(), index=X_train_enc.index)
        te_oh = pd.DataFrame(oh.transform(X_test_enc[remaining_cat]), 
                            columns=oh.get_feature_names_out(), index=X_test_enc.index)
        
        X_train_enc = pd.concat([X_train_enc.drop(columns=remaining_cat), tr_oh], axis=1)
        X_test_enc = pd.concat([X_test_enc.drop(columns=remaining_cat), te_oh], axis=1)

    return X_train_enc, X_test_enc