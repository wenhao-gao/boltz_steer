"""
After running:
python -m scripts.eval.run_physicalsim_metrics outputs_5_4_5_20/posebusters --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs_5_4_5_20/posebusters_gbd --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs_5_4_5_20/posebusters_vm --num-workers 16
There will be a "physical_checks.csv" file in each of the output directories.
This script will read the results and print pdbid for the structures that in both posebusters 
and posebusters_gbd are unvalid but valid in posebusters_vm
"""

import pandas as pd
import os
from pathlib import Path

def main():
    # Define the output directories
    dirs = {
        'posebusters': 'outputs_5_4_5_20/posebusters',
        'posebusters_gbd': 'outputs_5_4_5_20/posebusters_gbd', 
        'posebusters_vm': 'outputs_5_4_5_20/posebusters_vm'
    }
    
    # Read the physical_checks.csv files from each directory
    data = {}
    for name, dir_path in dirs.items():
        csv_path = os.path.join(dir_path, 'physical_checks.csv')
        if os.path.exists(csv_path):
            data[name] = pd.read_csv(csv_path)
            print(f"Loaded {len(data[name])} records from {csv_path}")
        else:
            print(f"Warning: {csv_path} not found")
            return
    
    # Extract PDB ID from cif_path column (first 4 characters of filename)
    def extract_pdbid(cif_path):
        if pd.isna(cif_path):
            return None
        filename = os.path.basename(cif_path)
        return filename[:4] if len(filename) >= 4 else None
    
    # Add pdbid column to each dataset
    for name in data:
        data[name]['pdbid'] = data[name]['cif_path'].apply(extract_pdbid)
        # Remove rows where pdbid couldn't be extracted
        data[name] = data[name].dropna(subset=['pdbid'])
    
    # Determine what column indicates validity
    print("\nColumns in posebusters data:")
    print(data['posebusters'].columns.tolist())
    
    # Look for columns that might indicate validity (common names)
    validity_columns = []
    for col in data['posebusters'].columns:
        if any(keyword in col.lower() for keyword in ['valid', 'pass', 'check', 'status']):
            validity_columns.append(col)
    
    if not validity_columns:
        print("Could not find validity columns. Available columns:")
        for col in data['posebusters'].columns:
            print(f"  - {col}")
        return
    
    print(f"\nPotential validity columns: {validity_columns}")
    
    # For now, let's assume the first validity column is the main one
    # You may need to adjust this based on the actual column names
    validity_col = validity_columns[0]
    
    # Get pdbids that are invalid in both posebusters and posebusters_gbd
    # but valid in posebusters_vm
    posebusters_invalid = set(data['posebusters'][data['posebusters'][validity_col] == False]['pdbid'].tolist())
    posebusters_gbd_invalid = set(data['posebusters_gbd'][data['posebusters_gbd'][validity_col] == False]['pdbid'].tolist())
    posebusters_vm_valid = set(data['posebusters_vm'][data['posebusters_vm'][validity_col] == True]['pdbid'].tolist())
    
    # Find intersection: invalid in both posebusters and posebusters_gbd, but valid in posebusters_vm
    target_pdbids = posebusters_invalid.intersection(posebusters_gbd_invalid).intersection(posebusters_vm_valid)
    
    print(f"\nFound {len(target_pdbids)} structures that are:")
    print(f"  - Invalid in posebusters: {len(posebusters_invalid)}")
    print(f"  - Invalid in posebusters_gbd: {len(posebusters_gbd_invalid)}")
    print(f"  - Valid in posebusters_vm: {len(posebusters_vm_valid)}")
    print(f"  - Invalid in both posebusters and posebusters_gbd, but valid in posebusters_vm: {len(target_pdbids)}")
    
    if target_pdbids:
        print(f"\nPDB IDs with detailed violation information:")
        print("=" * 80)
        
        for pdbid in sorted(target_pdbids):
            print(f"\nPDB ID: {pdbid}")
            print("-" * 40)
            
            # Get the row for this PDB ID from each dataset
            for dataset_name, dataset in data.items():
                pdbid_rows = dataset[dataset['pdbid'] == pdbid]
                if not pdbid_rows.empty:
                    row = pdbid_rows.iloc[0]  # Take first row if multiple
                    print(f"\n{dataset_name.upper()}:")
                    print(f"  Validity ({validity_col}): {row[validity_col]}")
                    
                    # Print all columns that might contain violation information
                    violation_columns = [col for col in dataset.columns 
                                      if any(keyword in col.lower() for keyword in 
                                            ['violation', 'error', 'warning', 'issue', 'problem', 'fail'])]
                    
                    if violation_columns:
                        print("  Violations:")
                        for col in violation_columns:
                            value = row[col]
                            if pd.notna(value) and value != '' and value != 0:
                                print(f"    {col}: {value}")
                    else:
                        # If no specific violation columns, show all non-standard columns
                        standard_cols = ['pdbid', 'cif_path', validity_col]
                        other_cols = [col for col in dataset.columns if col not in standard_cols]
                        if other_cols:
                            print("  Additional information:")
                            for col in other_cols:
                                value = row[col]
                                if pd.notna(value) and value != '' and value != 0:
                                    print(f"    {col}: {value}")
                else:
                    print(f"\n{dataset_name.upper()}: No data found for {pdbid}")
            
            print("=" * 80)
    else:
        print("\nNo structures found matching the criteria.")

if __name__ == "__main__":
    main()
