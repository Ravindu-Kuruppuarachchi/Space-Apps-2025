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


def load_and_extract_data(csv_path, output_path='exoplanet_data_clean.csv'):
    """
    Load the Kepler dataset and extract only the relevant features.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file
    output_path : str
        Path where the cleaned data will be saved
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing only the selected features
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
        print("\nTarget variable distribution (koi_disposition):")
        print(df_extracted['koi_disposition'].value_counts())
        print(f"\nPercentage distribution:")
        print(df_extracted['koi_disposition'].value_counts(normalize=True).round(4) * 100)
        
        # Show breakdown
        total = len(df_extracted)
        print(f"\nTotal samples: {total}")
        for label, count in df_extracted['koi_disposition'].value_counts().items():
            pct = (count/total)*100
            print(f"  {label}: {count} ({pct:.2f}%)")
    
    # Display missing values information
    print("\nMissing values per feature:")
    missing_counts = df_extracted.isnull().sum()
    missing_pct = (missing_counts / len(df_extracted)) * 100
    missing_info = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_pct
    })
    print(missing_info[missing_info['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))
    
    # Save the extracted data
    df_extracted.to_csv(output_path, index=False)
    print(f"\nExtracted data saved to: {output_path}")
    
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
    
    # Extract the data
    df = load_and_extract_data(CSV_FILE_PATH, OUTPUT_FILE_PATH)
    
    # Display statistics
    get_data_statistics(df)
    
    print("\n" + "="*50)
    print("Data extraction completed successfully!")
    print("="*50)
    print(f"\nYou can now use '{OUTPUT_FILE_PATH}' for ML model training.")