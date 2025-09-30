import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(input_path='exoplanet_data_clean.csv', 
               test_size=0.15, 
               val_size=0.15,
               random_state=42,
               output_dir='data/'):
    """
    Split the exoplanet data into train, validation, and test sets.
    Excludes CANDIDATE data and only uses CONFIRMED and FALSE POSITIVE samples.
    
    Parameters:
    -----------
    input_path : str
        Path to the cleaned exoplanet data CSV
    test_size : float
        Proportion of data for test set (default: 0.15 = 15%)
    val_size : float
        Proportion of data for validation set (default: 0.15 = 15%)
    random_state : int
        Random seed for reproducibility
    output_dir : str
        Directory to save the split datasets
    
    Returns:
    --------
    tuple
        (train_df, val_df, test_df) - DataFrames for each split
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("SPLITTING EXOPLANET DATA INTO TRAIN/VAL/TEST SETS")
    print("="*60)
    
    # Load the cleaned data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"\nOriginal class distribution:")
    print(df['koi_disposition'].value_counts())
    print()
    
    # Filter out CANDIDATE data - only keep CONFIRMED and FALSE POSITIVE
    print("Filtering data...")
    print("Keeping only: CONFIRMED and FALSE POSITIVE")
    print("Excluding: CANDIDATE (unknown labels)")
    
    df_filtered = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    
    print(f"\nFiltered dataset shape: {df_filtered.shape}")
    print(f"Removed {len(df) - len(df_filtered)} CANDIDATE samples")
    
    print(f"\nFiltered class distribution:")
    for label, count in df_filtered['koi_disposition'].value_counts().items():
        pct = (count/len(df_filtered))*100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    # Calculate actual split sizes
    # We'll split: train (70%), validation (15%), test (15%)
    train_ratio = 1.0 - test_size - val_size
    val_ratio_adjusted = val_size / (1.0 - test_size)
    
    print(f"\nSplit ratios:")
    print(f"  Train: {train_ratio*100:.1f}%")
    print(f"  Validation: {val_size*100:.1f}%")
    print(f"  Test: {test_size*100:.1f}%")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df_filtered,
        test_size=test_size,
        random_state=random_state,
        stratify=df_filtered['koi_disposition']  # Maintain class balance
    )
    
    # Second split: separate validation from train
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_df['koi_disposition']  # Maintain class balance
    )
    
    print("\n" + "="*60)
    print("SPLIT RESULTS")
    print("="*60)
    
    print(f"\nTraining set: {len(train_df)} samples ({len(train_df)/len(df_filtered)*100:.2f}%)")
    print("Class distribution:")
    for label, count in train_df['koi_disposition'].value_counts().items():
        pct = (count/len(train_df))*100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\nValidation set: {len(val_df)} samples ({len(val_df)/len(df_filtered)*100:.2f}%)")
    print("Class distribution:")
    for label, count in val_df['koi_disposition'].value_counts().items():
        pct = (count/len(val_df))*100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\nTest set: {len(test_df)} samples ({len(test_df)/len(df_filtered)*100:.2f}%)")
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
    
    print("\n" + "="*60)
    print("FILES SAVED")
    print("="*60)
    print(f"Training set saved to: {train_path}")
    print(f"Validation set saved to: {val_path}")
    print(f"Test set saved to: {test_path}")
    
    # Also save a summary file
    summary_path = os.path.join(output_dir, 'split_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EXOPLANET DATA SPLIT SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Random State: {random_state}\n")
        f.write(f"Original samples: {len(df)}\n")
        f.write(f"Filtered samples (CONFIRMED + FALSE POSITIVE): {len(df_filtered)}\n")
        f.write(f"Excluded CANDIDATE samples: {len(df) - len(df_filtered)}\n\n")
        f.write(f"Train samples: {len(train_df)} ({len(train_df)/len(df_filtered)*100:.2f}%)\n")
        f.write(f"Validation samples: {len(val_df)} ({len(val_df)/len(df_filtered)*100:.2f}%)\n")
        f.write(f"Test samples: {len(test_df)} ({len(test_df)/len(df_filtered)*100:.2f}%)\n\n")
        f.write("Class Distribution:\n")
        f.write("-"*40 + "\n")
        for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            f.write(f"\n{split_name}:\n")
            for label, count in split_df['koi_disposition'].value_counts().items():
                pct = (count/len(split_df))*100
                f.write(f"  {label}: {count} ({pct:.2f}%)\n")
    
    print(f"Summary saved to: {summary_path}")
    
    print("\n" + "="*60)
    print("DATA SPLIT COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return train_df, val_df, test_df


def verify_split(train_df, val_df, test_df):
    """
    Verify that the splits don't have any overlapping data.
    
    Parameters:
    -----------
    train_df, val_df, test_df : pandas.DataFrame
        The three split datasets
    """
    print("\n" + "="*60)
    print("VERIFYING DATA SPLITS")
    print("="*60)
    
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
    
    print("="*60)


if __name__ == "__main__":
    # Split the data
    train_df, val_df, test_df = split_data(
        input_path='exoplanet_data_clean.csv',
        test_size=0.15,      # 15% for test
        val_size=0.15,       # 15% for validation
        random_state=42,     # For reproducibility
        output_dir='data/'   # Output directory
    )
    
    # Verify the splits
    verify_split(train_df, val_df, test_df)
    
    print("\nYou can now use these files for ML model training:")
    print("  - data/train.csv (for training)")
    print("  - data/validation.csv (for hyperparameter tuning)")
    print("  - data/test.csv (for final evaluation)")