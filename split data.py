import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter

def split_data_with_overfitting_prevention(
    input_path='exoplanet_data_clean.csv', 
    test_size=0.20,  # Increased test size for better generalization testing
    val_size=0.20,   # Increased validation size
    random_state=42,
    output_dir='data/',
    balance_method='smote',  # 'smote', 'undersample', 'smoteenn', or None
    remove_highly_correlated=True,
    correlation_threshold=0.95,
    feature_variance_threshold=0.01):
    """
    Split the exoplanet data with overfitting prevention techniques.
    
    Overfitting Prevention Strategies:
    1. Larger validation and test sets
    2. Class balancing (SMOTE/undersampling)
    3. Remove highly correlated features
    4. Remove low variance features
    5. Stratified splitting
    
    Parameters:
    -----------
    input_path : str
        Path to the cleaned exoplanet data CSV
    test_size : float
        Proportion of data for test set (default: 0.20 = 20%)
    val_size : float
        Proportion of data for validation set (default: 0.20 = 20%)
    random_state : int
        Random seed for reproducibility
    output_dir : str
        Directory to save the split datasets
    balance_method : str
        Method to balance classes: 'smote', 'undersample', 'smoteenn', or None
    remove_highly_correlated : bool
        Whether to remove highly correlated features
    correlation_threshold : float
        Correlation threshold for feature removal (default: 0.95)
    feature_variance_threshold : float
        Minimum variance threshold for features (default: 0.01)
    
    Returns:
    --------
    tuple
        (train_df, val_df, test_df) - DataFrames for each split
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("SPLITTING DATA WITH OVERFITTING PREVENTION TECHNIQUES")
    print("="*70)
    
    # Load the cleaned data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"\nOriginal class distribution:")
    print(df['koi_disposition'].value_counts())
    
    # Filter out CANDIDATE data
    print("\nFiltering data...")
    print("Keeping only: CONFIRMED and FALSE POSITIVE")
    print("Excluding: CANDIDATE (unknown labels)")
    
    df_filtered = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    
    print(f"\nFiltered dataset shape: {df_filtered.shape}")
    print(f"Removed {len(df) - len(df_filtered)} CANDIDATE samples")
    
    print(f"\nFiltered class distribution:")
    for label, count in df_filtered['koi_disposition'].value_counts().items():
        pct = (count/len(df_filtered))*100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    # Separate features and target
    feature_cols = [col for col in df_filtered.columns 
                   if col not in ['kepid', 'koi_disposition']]
    
    X = df_filtered[feature_cols].copy()
    y = df_filtered['koi_disposition'].copy()
    kepid = df_filtered['kepid'].copy()
    
    # ===== OVERFITTING PREVENTION TECHNIQUE 1: Remove Low Variance Features =====
    print("\n" + "="*70)
    print("STEP 1: REMOVING LOW VARIANCE FEATURES")
    print("="*70)
    
    # Calculate variance for each feature
    variances = X.var()
    low_variance_features = variances[variances < feature_variance_threshold].index.tolist()
    
    if len(low_variance_features) > 0:
        print(f"\nRemoving {len(low_variance_features)} low variance features:")
        for feat in low_variance_features[:10]:  # Show first 10
            print(f"  - {feat} (variance: {variances[feat]:.6f})")
        if len(low_variance_features) > 10:
            print(f"  ... and {len(low_variance_features) - 10} more")
        
        X = X.drop(columns=low_variance_features)
        feature_cols = [col for col in feature_cols if col not in low_variance_features]
    else:
        print("\nNo low variance features found.")
    
    print(f"Features after variance filtering: {X.shape[1]}")
    
    # ===== OVERFITTING PREVENTION TECHNIQUE 2: Remove Highly Correlated Features =====
    if remove_highly_correlated:
        print("\n" + "="*70)
        print("STEP 2: REMOVING HIGHLY CORRELATED FEATURES")
        print("="*70)
        
        # Fill NaN values temporarily for correlation calculation
        X_temp = X.fillna(X.median())
        
        # Calculate correlation matrix
        corr_matrix = X_temp.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > correlation_threshold)]
        
        if len(to_drop) > 0:
            print(f"\nRemoving {len(to_drop)} highly correlated features (threshold: {correlation_threshold}):")
            for feat in to_drop[:10]:
                print(f"  - {feat}")
            if len(to_drop) > 10:
                print(f"  ... and {len(to_drop) - 10} more")
            
            X = X.drop(columns=to_drop)
            feature_cols = [col for col in feature_cols if col not in to_drop]
        else:
            print(f"\nNo highly correlated features found (threshold: {correlation_threshold})")
        
        print(f"Features after correlation filtering: {X.shape[1]}")
    
    # ===== STEP 3: Split Data (Stratified) =====
    print("\n" + "="*70)
    print("STEP 3: STRATIFIED DATA SPLITTING")
    print("="*70)
    
    train_ratio = 1.0 - test_size - val_size
    val_ratio_adjusted = val_size / (1.0 - test_size)
    
    print(f"\nSplit ratios (for better generalization):")
    print(f"  Train: {train_ratio*100:.1f}%")
    print(f"  Validation: {val_size*100:.1f}%")
    print(f"  Test: {test_size*100:.1f}%")
    
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test, kepid_train_val, kepid_test = train_test_split(
        X, y, kepid,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: separate validation from train
    X_train, X_val, y_train, y_val, kepid_train, kepid_val = train_test_split(
        X_train_val, y_train_val, kepid_train_val,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=y_train_val
    )
    
    print("\nInitial split completed.")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ===== OVERFITTING PREVENTION TECHNIQUE 3: Balance Training Data =====
    if balance_method:
        print("\n" + "="*70)
        print(f"STEP 4: BALANCING TRAINING DATA ({balance_method.upper()})")
        print("="*70)
        
        print(f"\nOriginal training class distribution:")
        print(Counter(y_train))
        
        # Fill NaN values before balancing (required by SMOTE)
        X_train_filled = X_train.fillna(X_train.median())
        
        if balance_method == 'smote':
            # SMOTE: Synthetic Minority Over-sampling Technique
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filled, y_train)
            print("\nUsing SMOTE (Synthetic Minority Over-sampling)")
            
        elif balance_method == 'undersample':
            # Random Under-sampling
            rus = RandomUnderSampler(random_state=random_state)
            X_train_balanced, y_train_balanced = rus.fit_resample(X_train_filled, y_train)
            print("\nUsing Random Under-sampling")
            
        elif balance_method == 'smoteenn':
            # SMOTE + Edited Nearest Neighbors
            smote_enn = SMOTEENN(random_state=random_state)
            X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_filled, y_train)
            print("\nUsing SMOTE + ENN (combination)")
        
        print(f"\nBalanced training class distribution:")
        print(Counter(y_train_balanced))
        
        # Convert back to DataFrame
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
        y_train_balanced = pd.Series(y_train_balanced, name='koi_disposition')
        
        # Note: We don't have kepid for synthetic samples, so we'll create dummy IDs
        kepid_train_balanced = pd.Series(range(len(y_train_balanced)), name='kepid')
        
        # Use balanced data
        X_train = X_train_balanced
        y_train = y_train_balanced
        kepid_train = kepid_train_balanced
    
    # ===== Create Final DataFrames =====
    print("\n" + "="*70)
    print("FINAL DATA SPLITS")
    print("="*70)
    
    # Reconstruct DataFrames
    train_df = pd.concat([kepid_train.reset_index(drop=True), 
                         X_train.reset_index(drop=True), 
                         y_train.reset_index(drop=True)], axis=1)
    
    val_df = pd.concat([kepid_val.reset_index(drop=True), 
                       X_val.reset_index(drop=True), 
                       y_val.reset_index(drop=True)], axis=1)
    
    test_df = pd.concat([kepid_test.reset_index(drop=True), 
                        X_test.reset_index(drop=True), 
                        y_test.reset_index(drop=True)], axis=1)
    
    print(f"\nTraining set: {len(train_df)} samples ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.2f}%)")
    print("Class distribution:")
    for label, count in train_df['koi_disposition'].value_counts().items():
        pct = (count/len(train_df))*100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\nValidation set: {len(val_df)} samples ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.2f}%)")
    print("Class distribution:")
    for label, count in val_df['koi_disposition'].value_counts().items():
        pct = (count/len(val_df))*100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\nTest set: {len(test_df)} samples ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.2f}%)")
    print("Class distribution:")
    for label, count in test_df['koi_disposition'].value_counts().items():
        pct = (count/len(test_df))*100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    # Save the splits to CSV files
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'validation.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("\n" + "="*70)
    print("FILES SAVED")
    print("="*70)
    print(f"Training set saved to: {train_path}")
    print(f"Validation set saved to: {val_path}")
    print(f"Test set saved to: {test_path}")
    
    # Save detailed summary
    summary_path = os.path.join(output_dir, 'split_summary_with_prevention.txt')
    with open(summary_path, 'w') as f:
        f.write("EXOPLANET DATA SPLIT SUMMARY (WITH OVERFITTING PREVENTION)\n")
        f.write("="*70 + "\n\n")
        f.write("OVERFITTING PREVENTION TECHNIQUES APPLIED:\n")
        f.write(f"1. Low variance features removed (threshold: {feature_variance_threshold})\n")
        f.write(f"2. Highly correlated features removed (threshold: {correlation_threshold})\n")
        f.write(f"3. Class balancing method: {balance_method if balance_method else 'None'}\n")
        f.write(f"4. Larger validation/test sets ({val_size*100:.0f}%/{test_size*100:.0f}%)\n")
        f.write(f"5. Stratified sampling\n\n")
        f.write(f"Random State: {random_state}\n")
        f.write(f"Original features: {len(df.columns) - 2}\n")
        f.write(f"Final features: {len(feature_cols)}\n")
        f.write(f"Features removed: {len(df.columns) - 2 - len(feature_cols)}\n\n")
        f.write(f"Train samples: {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.2f}%)\n")
        f.write(f"Validation samples: {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.2f}%)\n")
        f.write(f"Test samples: {len(test_df)} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.2f}%)\n\n")
        f.write("Class Distribution:\n")
        f.write("-"*50 + "\n")
        for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            f.write(f"\n{split_name}:\n")
            for label, count in split_df['koi_disposition'].value_counts().items():
                pct = (count/len(split_df))*100
                f.write(f"  {label}: {count} ({pct:.2f}%)\n")
    
    print(f"Detailed summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("DATA SPLIT WITH OVERFITTING PREVENTION COMPLETED!")
    print("="*70)
    
    return train_df, val_df, test_df


def verify_split(train_df, val_df, test_df):
    """
    Verify that the splits don't have any overlapping data.
    """
    print("\n" + "="*70)
    print("VERIFYING DATA SPLITS")
    print("="*70)
    
    # Check for overlaps using kepid
    train_ids = set(train_df['kepid'])
    val_ids = set(val_df['kepid'])
    test_ids = set(test_df['kepid'])
    
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)
    
    print("\nChecking for data leakage (overlapping kepid):")
    print(f"  Train-Validation overlap: {len(train_val_overlap)} samples")
    print(f"  Train-Test overlap: {len(train_test_overlap)} samples")
    print(f"  Validation-Test overlap: {len(val_test_overlap)} samples")
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("\n✓ No data leakage detected! All splits are independent.")
    else:
        print("\n✗ WARNING: Data leakage detected! Splits have overlapping samples.")
    
    print("="*70)


if __name__ == "__main__":
    # Split the data with overfitting prevention
    train_df, val_df, test_df = split_data_with_overfitting_prevention(
        input_path='exoplanet_data_clean.csv',
        test_size=0.20,              # 20% for test
        val_size=0.20,               # 20% for validation
        random_state=42,
        output_dir='data/',
        balance_method='smote',      # Options: 'smote', 'undersample', 'smoteenn', None
        remove_highly_correlated=True,
        correlation_threshold=0.95,
        feature_variance_threshold=0.01
    )
    
    # Verify the splits
    verify_split(train_df, val_df, test_df)
    
    print("\n" + "="*70)
    print("OVERFITTING PREVENTION SUMMARY")
    print("="*70)
    print("\nTechniques Applied:")
    print("  ✓ Removed low variance features")
    print("  ✓ Removed highly correlated features")
    print("  ✓ Balanced training data (SMOTE)")
    print("  ✓ Larger validation/test sets (20%/20%)")
    print("  ✓ Stratified sampling")
    print("\nYou can now use these files for ML model training:")
    print("  - data/train.csv (for training)")
    print("  - data/validation.csv (for hyperparameter tuning)")
    print("  - data/test.csv (for final evaluation)")