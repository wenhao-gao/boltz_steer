"""
After running:
python -m scripts.eval.run_physicalsim_metrics outputs/posebusters --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs/posebusters_steer --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs/posebusters_fks --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs/posebusters_gg --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs/posebusters_gbd --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs/posebusters_vm --num-workers 16
There will be a "physical_checks.csv" file in each of the output directories.
This script will summarize the results of these files and print as a table.
The table will have the following columns:
- Method
- Total Structures
- Valid Structures
- Valid Fraction
- Clash Free Fraction
- num_bond_length_violations
- num_bond_angle_violations
- num_internal_clash_violations
- num_chiral_atom_violations
- num_stereo_bond_violations
- num_planar_5_ring_violations
- num_planar_6_ring_violations
- num_planar_double_bond_violations
- num_chain_clashes_all

The table will have the following rows:
- No Steering (outputs/posebusters)
- Boltz Steering (outputs/posebusters_steer)
- FKS (outputs/posebusters_fks)
- GG (outputs/posebusters_gg)
- GBD (outputs/posebusters_gbd)
- VM (outputs/posebusters_vm)

The table will be printed to the console.
"""

import pandas as pd
import os
from pathlib import Path


def read_physical_checks_csv(csv_path):
    """Read and return the physical checks CSV file."""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None
    return pd.read_csv(csv_path)


def calculate_summary_stats(df):
    """Calculate summary statistics from the physical checks dataframe."""
    if df is None or df.empty:
        return {
            'total_structures': 0,
            'valid_structures': 0,
            'valid_fraction': 0.0,
            'clash_free_fraction': 0.0,
            'num_bond_length_violations': 0,
            'num_bond_angle_violations': 0,
            'num_internal_clash_violations': 0,
            'num_chiral_atom_violations': 0,
            'num_stereo_bond_violations': 0,
            'num_planar_5_ring_violations': 0,
            'num_planar_6_ring_violations': 0,
            'num_planar_double_bond_violations': 0,
            'num_chain_clashes_all': 0
        }
    
    total_structures = len(df)
    valid_structures = df['valid'].sum() if 'valid' in df.columns else 0
    valid_fraction = valid_structures / total_structures if total_structures > 0 else 0.0
    
    clash_free_structures = df['clash_free'].sum() if 'clash_free' in df.columns else 0
    clash_free_fraction = clash_free_structures / total_structures if total_structures > 0 else 0.0
    
    # Sum up all violations across all structures
    violation_columns = [
        'num_bond_length_violations',
        'num_bond_angle_violations', 
        'num_internal_clash_violations',
        'num_chiral_atom_violations',
        'num_stereo_bond_violations',
        'num_planar_5_ring_violations',
        'num_planar_6_ring_violations',
        'num_planar_double_bond_violations',
        'num_chain_clashes_all'
    ]
    
    violations = {}
    for col in violation_columns:
        if col in df.columns:
            violations[col] = df[col].sum()
        else:
            violations[col] = 0
    
    return {
        'total_structures': total_structures,
        'valid_structures': valid_structures,
        'valid_fraction': valid_fraction,
        'clash_free_fraction': clash_free_fraction,
        **violations
    }


def generate_latex_table(df):
    """Generate a LaTeX table from the summary dataframe."""
    # Start the table
    latex = "\\begin{table}[h!]\n"
    latex += "\\centering\n"
    latex += "\\caption{Physical Simulation Metrics Summary}\n"
    latex += "\\label{tab:physical_metrics}\n"
    latex += "\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n"
    latex += "\\hline\n"
    
    # Header row
    headers = [
        "Method", "Total", "Valid", "Valid Frac.", "Clash Free Frac.",
        "Bond Length", "Bond Angle", "Internal Clash", "Chiral Atom",
        "Stereo Bond", "Planar 5-Ring", "Planar 6-Ring", "Planar Double Bond", "Chain Clashes"
    ]
    latex += " & ".join(headers) + " \\\\\n"
    latex += "\\hline\n"
    
    # Data rows
    for _, row in df.iterrows():
        # Format the data for LaTeX
        method = row['Method'].replace('_', '\\_')
        total = int(row['Total Structures'])
        valid = int(row['Valid Structures'])
        valid_frac = row['Valid Fraction']
        clash_free_frac = row['Clash Free Fraction']
        
        # Violation counts
        violations = [
            int(row['num_bond_length_violations']),
            int(row['num_bond_angle_violations']),
            int(row['num_internal_clash_violations']),
            int(row['num_chiral_atom_violations']),
            int(row['num_stereo_bond_violations']),
            int(row['num_planar_5_ring_violations']),
            int(row['num_planar_6_ring_violations']),
            int(row['num_planar_double_bond_violations']),
            int(row['num_chain_clashes_all'])
        ]
        
        # Create the row
        row_data = [
            method, total, valid, valid_frac, clash_free_frac
        ] + violations
        
        latex += " & ".join(str(x) for x in row_data) + " \\\\\n"
    
    # End the table
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def main():
    """Main function to generate and print the summary table."""
    # Define the methods and their corresponding directories
    methods = {
        'No Steering': 'outputs/posebusters',
        'Boltz Steering': 'outputs/posebusters_steer', 
        'FKS': 'outputs/posebusters_fks',
        'GG': 'outputs/posebusters_gg',
        'GBD': 'outputs/posebusters_gbd',
        'VM': 'outputs/posebusters_vm'
    }
    
    # Collect summary data for each method
    summary_data = []
    
    for method_name, output_dir in methods.items():
        csv_path = os.path.join(output_dir, 'physical_checks.csv')
        print(f"Processing {method_name} from {csv_path}")
        
        df = read_physical_checks_csv(csv_path)
        stats = calculate_summary_stats(df)
        
        summary_data.append({
            'Method': method_name,
            'Total Structures': stats['total_structures'],
            'Valid Structures': stats['valid_structures'],
            'Valid Fraction': f"{stats['valid_fraction']:.3f}",
            'Clash Free Fraction': f"{stats['clash_free_fraction']:.3f}",
            'num_bond_length_violations': stats['num_bond_length_violations'],
            'num_bond_angle_violations': stats['num_bond_angle_violations'],
            'num_internal_clash_violations': stats['num_internal_clash_violations'],
            'num_chiral_atom_violations': stats['num_chiral_atom_violations'],
            'num_stereo_bond_violations': stats['num_stereo_bond_violations'],
            'num_planar_5_ring_violations': stats['num_planar_5_ring_violations'],
            'num_planar_6_ring_violations': stats['num_planar_6_ring_violations'],
            'num_planar_double_bond_violations': stats['num_planar_double_bond_violations'],
            'num_chain_clashes_all': stats['num_chain_clashes_all']
        })
    
    # Create and display the summary table
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*120)
    print("PHYSICAL SIMULATION METRICS SUMMARY")
    print("="*120)
    print(summary_df.to_string(index=False))
    print("="*120)
    
    # Generate LaTeX table
    print("\n" + "="*120)
    print("LATEX TABLE FORMAT")
    print("="*120)
    
    # Create LaTeX table
    latex_table = generate_latex_table(summary_df)
    print(latex_table)
    print("="*120)
    
    # Also save to CSV for reference
    output_csv = 'physical_metrics_summary.csv'
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSummary table also saved to: {output_csv}")
    
    # Save LaTeX table to file
    latex_output = 'physical_metrics_summary.tex'
    with open(latex_output, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table also saved to: {latex_output}")


if __name__ == "__main__":
    main()