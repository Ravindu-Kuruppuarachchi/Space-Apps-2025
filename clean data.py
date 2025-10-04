import pandas as pd
import numpy as np

# List of features to extract from the CSV
FEATURES = [
    'kepid',  # Planet ID (note: in your CSV it's 'kepid')
    'koi_disposition',  # Label (CONFIRMED, FALSE POSITIVE, CANDIDATE)
    
    # False Positive Flags
    #'koi_fpflag_nt',  # Not Transit-Like False Positive Flag
    #'koi_fpflag_ss',  # Stellar Eclipse False Positive Flag
    #'koi_fpflag_co',  # Centroid Offset False Positive Flag
    #'koi_fpflag_ec',  # Ephemeris Match Contamination Flag
    
    # Orbital and Transit Parameters
    'koi_period',  # Orbital Period [days]
    'koi_time0bk',  # Transit Epoch [BKJD]
    'koi_time0',  # Transit Epoch [BJD]
    'koi_impact',  # Impact Parameter
    'koi_impact_err1',  # Impact Parameter Upper Unc.
    'koi_impact_err2',  # Impact Parameter Lower Unc.
    'koi_duration',  # Transit Duration [hrs]
    'koi_duration_err1',  # Transit Duration Upper Unc. [hrs]
    'koi_duration_err2',  # Transit Duration Lower Unc. [hrs]
    'koi_depth',  # Transit Depth [ppm]
    'koi_depth_err1',  # Transit Depth Upper Unc. [ppm]
    'koi_depth_err2',  # Transit Depth Lower Unc. [ppm]
    
    # Planet-Star Ratios
    'koi_ror',  # Planet-Star Radius Ratio
    'koi_ror_err1',  # Planet-Star Radius Ratio Upper Unc.
    'koi_ror_err2',  # Planet-Star Radius Ratio Lower Unc.
    'koi_srho',  # Fitted Stellar Density [g/cm**3]
    'koi_srho_err1',  # Fitted Stellar Density Upper Unc.
    'koi_srho_err2',  # Fitted Stellar Density Lower Unc.
    
    # Planetary Properties
    'koi_prad',  # Planetary Radius [Earth radii]
    'koi_prad_err1',  # Planetary Radius Upper Unc.
    'koi_prad_err2',  # Planetary Radius Lower Unc.
    'koi_sma',  # Orbit Semi-Major Axis [au]
    'koi_incl',  # Inclination [deg]
    'koi_teq',  # Equilibrium Temperature [K]
    'koi_insol',  # Insolation Flux [Earth flux]
    'koi_insol_err1',  # Insolation Flux Upper Unc.
    'koi_insol_err2',  # Insolation Flux Lower Unc.
    'koi_dor',  # Planet-Star Distance over Star Radius
    'koi_dor_err1',  # Planet-Star Distance over Star Radius Upper Unc.
    'koi_dor_err2',  # Planet-Star Distance over Star Radius Lower Unc.
    
    # Limb Darkening
    'koi_ldm_coeff2',  # Limb Darkening Coeff. 2
    'koi_ldm_coeff1',  # Limb Darkening Coeff. 1
    
    # Statistics
    'koi_max_sngle_ev',  # Maximum Single Event Statistic
    'koi_max_mult_ev',  # Maximum Multiple Event Statistic
    'koi_model_snr',  # Transit Signal-to-Noise
    'koi_count',  # Number of Planets
    'koi_num_transits',  # Number of Transits
    'koi_bin_oedp_sig',  # Odd-Even Depth Comparison Statistic
    
    # Stellar Properties
    'koi_steff',  # Stellar Effective Temperature [K]
    'koi_steff_err1',  # Stellar Effective Temperature Upper Unc.
    'koi_steff_err2',  # Stellar Effective Temperature Lower Unc.
    'koi_slogg',  # Stellar Surface Gravity [log10(cm/s**2)]
    'koi_slogg_err1',  # Stellar Surface Gravity Upper Unc.
    'koi_slogg_err2',  # Stellar Surface Gravity Lower Unc.
    'koi_srad',  # Stellar Radius [Solar radii]
    'koi_srad_err1',  # Stellar Radius Upper Unc.
    'koi_srad_err2',  # Stellar Radius Lower Unc.
    'koi_smass',  # Stellar Mass [Solar mass]
    'koi_smass_err1',  # Stellar Mass Upper Unc.
    'koi_smass_err2',  # Stellar Mass Lower Unc.
    'koi_fwm_stat_sig',  # FW Offset Significance [percent]
]

# Critical features that must be present for a row to be valid
CRITICAL_FEATURES = [
    'koi_disposition',  # Must have a label
    'koi_period',  # Orbital period is critical
    'koi_duration',  # Transit duration is critical
    'koi_depth',  # Transit depth is critical
]


def clean_data(df, missing_threshold=0.5, critical_features=None):
    """
    Clean the dataset by removing rows with inadequate data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to clean
    missing_threshold : float
        Maximum proportion of missing values allowed per row (0 to 1)
        Default: 0.5 (50% of features can be missing)
    critical_features : list
        List of features that must not be missing. Rows missing any of these
        will be removed regardless of missing_threshold
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe with rows removed
    """
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    
    initial_rows = len(df)
    print(f"\nInitial number of rows: {initial_rows}")
    
    # Step 1: Remove rows where critical features are missing
    if critical_features is None:
        critical_features = CRITICAL_FEATURES
    
    print(f"\nChecking critical features: {critical_features}")
    
    critical_cols = [col for col in critical_features if col in df.columns]
    if critical_cols:
        df_cleaned = df.dropna(subset=critical_cols).copy()
        removed_critical = initial_rows - len(df_cleaned)
        print(f"Rows removed due to missing critical features: {removed_critical}")
        print(f"Remaining rows: {len(df_cleaned)}")
    else:
        df_cleaned = df.copy()
        print("Warning: No critical features found in dataset")
    
    # Step 2: Remove rows with too many missing values
    print(f"\nApplying missing value threshold: {missing_threshold*100}%")
    
    # Calculate percentage of missing values per row (excluding ID and label columns)
    feature_cols = [col for col in df_cleaned.columns 
                   if col not in ['kepid', 'koi_disposition']]
    
    missing_per_row = df_cleaned[feature_cols].isnull().sum(axis=1) / len(feature_cols)
    
    # Keep rows where missing percentage is below threshold
    rows_to_keep = missing_per_row <= missing_threshold
    df_cleaned = df_cleaned[rows_to_keep].copy()
    
    removed_threshold = len(df) - removed_critical - len(df_cleaned)
    print(f"Rows removed due to exceeding missing threshold: {removed_threshold}")
    
    # Final statistics
    final_rows = len(df_cleaned)
    total_removed = initial_rows - final_rows
    removal_pct = (total_removed / initial_rows) * 100
    
    print(f"\n" + "-"*50)
    print(f"Final number of rows: {final_rows}")
    print(f"Total rows removed: {total_removed} ({removal_pct:.2f}%)")
    print(f"Data retention rate: {(final_rows/initial_rows)*100:.2f}%")
    
    # Show class distribution after cleaning
    if 'koi_disposition' in df_cleaned.columns:
        print(f"\nClass distribution after cleaning:")
        print(df_cleaned['koi_disposition'].value_counts())
        print(f"\nPercentage distribution:")
        for label, count in df_cleaned['koi_disposition'].value_counts().items():
            pct = (count/final_rows)*100
            print(f"  {label}: {count} ({pct:.2f}%)")
    
    return df_cleaned


def load_and_extract_data(csv_path, output_path='exoplanet_data_clean.csv',
                         missing_threshold=0.5, remove_inadequate=True):
    """
    Load the Kepler dataset, extract relevant features, and optionally clean data.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file
    output_path : str
        Path where the cleaned data will be saved
    missing_threshold : float
        Maximum proportion of missing values allowed per row (0 to 1)
    remove_inadequate : bool
        Whether to remove rows with inadequate data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing only the selected features (cleaned if requested)
    """
    print(f"Loading data from {csv_path}...")
    
    # Load the CSV file (handle potential comment lines starting with #)
    df = pd.read_csv(csv_path, comment='#', skipinitialspace=True)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Total columns in original dataset: {len(df.columns)}")
    
    # Check which features are available in the dataset
    available_features = [f for f in FEATURES if f in df.columns]
    missing_features = [f for f in FEATURES if f not in df.columns]
    
    print(f"\nAvailable features: {len(available_features)}/{len(FEATURES)}")
    
    if missing_features:
        print(f"\nWarning: The following features are missing from the dataset:")
        for feat in missing_features:
            print(f"  - {feat}")
    
    # Extract only the available features
    df_extracted = df[available_features].copy()
    
    print(f"\nExtracted dataset shape: {df_extracted.shape}")
    
    # Display basic information about the target variable
    if 'koi_disposition' in df_extracted.columns:
        print("\nTarget variable distribution (before cleaning):")
        print(df_extracted['koi_disposition'].value_counts())
        print(f"\nPercentage distribution:")
        for label, count in df_extracted['koi_disposition'].value_counts().items():
            pct = (count/len(df_extracted))*100
            print(f"  {label}: {count} ({pct:.2f}%)")
    
    # Display missing values information (before cleaning)
    print("\nMissing values per feature (before cleaning):")
    missing_counts = df_extracted.isnull().sum()
    missing_pct = (missing_counts / len(df_extracted)) * 100
    missing_info = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_pct.round(2)
    })
    print(missing_info[missing_info['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))
    
    # Clean the data if requested
    if remove_inadequate:
        df_extracted = clean_data(df_extracted, missing_threshold=missing_threshold)
        
        # Show missing values after cleaning
        print("\nMissing values per feature (after cleaning):")
        missing_counts_clean = df_extracted.isnull().sum()
        missing_pct_clean = (missing_counts_clean / len(df_extracted)) * 100
        missing_info_clean = pd.DataFrame({
            'Missing_Count': missing_counts_clean,
            'Missing_Percentage': missing_pct_clean.round(2)
        })
        print(missing_info_clean[missing_info_clean['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))
    
    # Save the extracted data
    df_extracted.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    
    return df_extracted


def get_data_statistics(df):
    """
    Display detailed statistics about the extracted dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The extracted dataset
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumeric features: {len(numeric_cols)}")
    
    print("\nBasic statistics for numeric features:")
    print(df[numeric_cols].describe())


if __name__ == "__main__":
    # Example usage
    # Replace 'kepler_data.csv' with your actual CSV file path
    CSV_FILE_PATH = 'kepler.csv'
    OUTPUT_FILE_PATH = 'exoplanet_data_clean.csv'
    
    # Extract and clean the data
    # Adjust missing_threshold as needed (0.5 means 50% of features can be missing)
    # Set remove_inadequate=False to skip data cleaning
    df = load_and_extract_data(
        CSV_FILE_PATH, 
        OUTPUT_FILE_PATH,
        missing_threshold=0.3,  # Adjust this value (0.0 to 1.0)
        remove_inadequate=True   # Set to False to keep all rows
    )
    
    # Display statistics
    get_data_statistics(df)
    
    print("\n" + "="*50)
    print("Data extraction and cleaning completed successfully!")
    print("="*50)
  