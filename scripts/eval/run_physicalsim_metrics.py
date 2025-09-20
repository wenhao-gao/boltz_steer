"""Run physical-simulation metrics on a collection of CIF files."""

import argparse
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from boltz.data import const
from boltz.data.parse.mmcif_with_constraints import parse_mmcif


CCD_CACHE = None
MOL_DIR = None
CIF_ROOT = None


def compute_torsion_angles(coords, torsion_index):
    r_ij = coords[..., torsion_index[0], :] - coords[..., torsion_index[1], :]
    r_kj = coords[..., torsion_index[2], :] - coords[..., torsion_index[1], :]
    r_kl = coords[..., torsion_index[2], :] - coords[..., torsion_index[3], :]
    n_ijk = np.cross(r_ij, r_kj, axis=-1)
    n_jkl = np.cross(r_kj, r_kl, axis=-1)
    r_kj_norm = np.linalg.norm(r_kj, axis=-1)
    n_ijk_norm = np.linalg.norm(n_ijk, axis=-1)
    n_jkl_norm = np.linalg.norm(n_jkl, axis=-1)
    sign_phi = np.sign(
        r_kj[..., None, :] @ np.cross(n_ijk, n_jkl, axis=-1)[..., None]
    ).squeeze(axis=(-1, -2))
    phi = sign_phi * np.arccos(
        np.clip(
            (n_ijk[..., None, :] @ n_jkl[..., None]).squeeze(axis=(-1, -2))
            / (n_ijk_norm * n_jkl_norm),
            -1 + 1e-8,
            1 - 1e-8,
        )
    )
    return phi


def check_ligand_distance_geometry(
    structure, constraints, bond_buffer=0.25, angle_buffer=0.25, clash_buffer=0.2
):
    coords = structure.coords["coords"]
    rdkit_bounds_constraints = constraints.rdkit_bounds_constraints
    pair_index = rdkit_bounds_constraints["atom_idxs"].copy().astype(np.int64).T
    bond_mask = rdkit_bounds_constraints["is_bond"].copy().astype(bool)
    angle_mask = rdkit_bounds_constraints["is_angle"].copy().astype(bool)
    upper_bounds = rdkit_bounds_constraints["upper_bound"].copy().astype(np.float32)
    lower_bounds = rdkit_bounds_constraints["lower_bound"].copy().astype(np.float32)
    dists = np.linalg.norm(coords[pair_index[0]] - coords[pair_index[1]], axis=-1)
    bond_length_violations = (
        dists[bond_mask] <= lower_bounds[bond_mask] * (1.0 - bond_buffer)
    ) + (dists[bond_mask] >= upper_bounds[bond_mask] * (1.0 + bond_buffer))
    bond_angle_violations = (
        dists[angle_mask] <= lower_bounds[angle_mask] * (1.0 - angle_buffer)
    ) + (dists[angle_mask] >= upper_bounds[angle_mask] * (1.0 + angle_buffer))
    internal_clash_violations = dists[~bond_mask * ~angle_mask] <= lower_bounds[
        ~bond_mask * ~angle_mask
    ] * (1.0 - clash_buffer)
    num_ligands = sum(
        [
            int(const.chain_types[chain["mol_type"]] == "NONPOLYMER")
            for chain in structure.chains
        ]
    )
    return {
        "num_ligands": num_ligands,
        "num_bond_length_violations": bond_length_violations.sum(),
        "num_bonds": bond_mask.sum(),
        "num_bond_angle_violations": bond_angle_violations.sum(),
        "num_angles": angle_mask.sum(),
        "num_internal_clash_violations": internal_clash_violations.sum(),
        "num_non_neighbors": (~bond_mask * ~angle_mask).sum(),
    }


def check_ligand_stereochemistry(structure, constraints):
    coords = structure.coords["coords"]
    chiral_atom_constraints = constraints.chiral_atom_constraints
    stereo_bond_constraints = constraints.stereo_bond_constraints

    chiral_atom_index = chiral_atom_constraints["atom_idxs"].T
    true_chiral_atom_orientations = chiral_atom_constraints["is_r"]
    chiral_atom_ref_mask = chiral_atom_constraints["is_reference"]
    chiral_atom_index = chiral_atom_index[:, chiral_atom_ref_mask]
    true_chiral_atom_orientations = true_chiral_atom_orientations[chiral_atom_ref_mask]
    pred_chiral_atom_orientations = (
        compute_torsion_angles(coords, chiral_atom_index) > 0
    )
    chiral_atom_violations = (
        pred_chiral_atom_orientations != true_chiral_atom_orientations
    )

    stereo_bond_index = stereo_bond_constraints["atom_idxs"].T
    true_stereo_bond_orientations = stereo_bond_constraints["is_e"]
    stereo_bond_ref_mask = stereo_bond_constraints["is_reference"]
    stereo_bond_index = stereo_bond_index[:, stereo_bond_ref_mask]
    true_stereo_bond_orientations = true_stereo_bond_orientations[stereo_bond_ref_mask]
    pred_stereo_bond_orientations = (
        np.abs(compute_torsion_angles(coords, stereo_bond_index)) > np.pi / 2
    )
    stereo_bond_violations = (
        pred_stereo_bond_orientations != true_stereo_bond_orientations
    )

    return {
        "num_chiral_atom_violations": chiral_atom_violations.sum(),
        "num_chiral_atoms": chiral_atom_index.shape[1],
        "num_stereo_bond_violations": stereo_bond_violations.sum(),
        "num_stereo_bonds": stereo_bond_index.shape[1],
    }


def check_ligand_flatness(structure, constraints, buffer=0.25):
    coords = structure.coords["coords"]

    planar_ring_5_index = constraints.planar_ring_5_constraints["atom_idxs"]
    ring_5_coords = coords[planar_ring_5_index, :]
    centered_ring_5_coords = ring_5_coords - ring_5_coords.mean(axis=-2, keepdims=True)
    ring_5_vecs = np.linalg.svd(centered_ring_5_coords)[2][..., -1, :, None]
    ring_5_dists = np.abs((centered_ring_5_coords @ ring_5_vecs).squeeze(axis=-1))
    ring_5_violations = np.all(ring_5_dists <= buffer, axis=-1)

    planar_ring_6_index = constraints.planar_ring_6_constraints["atom_idxs"]
    ring_6_coords = coords[planar_ring_6_index, :]
    centered_ring_6_coords = ring_6_coords - ring_6_coords.mean(axis=-2, keepdims=True)
    ring_6_vecs = np.linalg.svd(centered_ring_6_coords)[2][..., -1, :, None]
    ring_6_dists = np.abs((centered_ring_6_coords @ ring_6_vecs)).squeeze(axis=-1)
    ring_6_violations = np.any(ring_6_dists >= buffer, axis=-1)

    planar_bond_index = constraints.planar_bond_constraints["atom_idxs"]
    bond_coords = coords[planar_bond_index, :]
    centered_bond_coords = bond_coords - bond_coords.mean(axis=-2, keepdims=True)
    bond_vecs = np.linalg.svd(centered_bond_coords)[2][..., -1, :, None]
    bond_dists = np.abs((centered_bond_coords @ bond_vecs)).squeeze(axis=-1)
    bond_violations = np.any(bond_dists >= buffer, axis=-1)

    return {
        "num_planar_5_ring_violations": ring_5_violations.sum(),
        "num_planar_5_rings": ring_5_violations.shape[0],
        "num_planar_6_ring_violations": ring_6_violations.sum(),
        "num_planar_6_rings": ring_6_violations.shape[0],
        "num_planar_double_bond_violations": bond_violations.sum(),
        "num_planar_double_bonds": bond_violations.shape[0],
    }


def check_steric_clash(structure, molecules, buffer=0.25):
    result = {}
    for type_i in const.chain_types:
        out_type_i = type_i.lower()
        out_type_i = out_type_i if out_type_i != "nonpolymer" else "ligand"
        result[f"num_chain_pairs_sym_{out_type_i}"] = 0
        result[f"num_chain_clashes_sym_{out_type_i}"] = 0
        for type_j in const.chain_types:
            out_type_j = type_j.lower()
            out_type_j = out_type_j if out_type_j != "nonpolymer" else "ligand"
            result[f"num_chain_pairs_asym_{out_type_i}_{out_type_j}"] = 0
            result[f"num_chain_clashes_asym_{out_type_i}_{out_type_j}"] = 0

    connected_chains = set()
    for bond in structure.bonds:
        if bond["chain_1"] != bond["chain_2"]:
            connected_chains.add(tuple(sorted((bond["chain_1"], bond["chain_2"]))))

    vdw_radii = []
    for res in structure.residues:
        mol = molecules[res["name"]]
        token_atoms = structure.atoms[
            res["atom_idx"] : res["atom_idx"] + res["atom_num"]
        ]
        atom_name_to_ref = {a.GetProp("name"): a for a in mol.GetAtoms()}
        token_atoms_ref = [atom_name_to_ref[a["name"]] for a in token_atoms]
        vdw_radii.extend(
            [const.vdw_radii[a.GetAtomicNum() - 1] for a in token_atoms_ref]
        )
    vdw_radii = np.array(vdw_radii, dtype=np.float32)

    for i, chain_i in enumerate(structure.chains):
        for j, chain_j in enumerate(structure.chains):
            if (
                chain_i["atom_num"] == 1
                or chain_j["atom_num"] == 1
                or j <= i
                or (i, j) in connected_chains
            ):
                continue
            coords_i = structure.coords["coords"][
                chain_i["atom_idx"] : chain_i["atom_idx"] + chain_i["atom_num"]
            ]
            coords_j = structure.coords["coords"][
                chain_j["atom_idx"] : chain_j["atom_idx"] + chain_j["atom_num"]
            ]
            dists = np.linalg.norm(coords_i[:, None, :] - coords_j[None, :, :], axis=-1)
            radii_i = vdw_radii[
                chain_i["atom_idx"] : chain_i["atom_idx"] + chain_i["atom_num"]
            ]
            radii_j = vdw_radii[
                chain_j["atom_idx"] : chain_j["atom_idx"] + chain_j["atom_num"]
            ]
            radii_sum = radii_i[:, None] + radii_j[None, :]
            is_clashing = np.any(dists < radii_sum * (1.00 - buffer))
            type_i = const.chain_types[chain_i["mol_type"]].lower()
            type_j = const.chain_types[chain_j["mol_type"]].lower()
            type_i = type_i if type_i != "nonpolymer" else "ligand"
            type_j = type_j if type_j != "nonpolymer" else "ligand"
            is_symmetric = (
                chain_i["entity_id"] == chain_j["entity_id"]
                and chain_i["atom_num"] == chain_j["atom_num"]
            )
            if is_symmetric:
                key = "sym_" + type_i
            else:
                key = "asym_" + type_i + "_" + type_j
            result["num_chain_pairs_" + key] += 1
            result["num_chain_clashes_" + key] += int(is_clashing)
    return result


def _relative_cif_path(cif_path):
    if CIF_ROOT is None:
        return str(cif_path)
    try:
        return str(cif_path.relative_to(CIF_ROOT))
    except ValueError:
        return str(cif_path)


def _process_single(cif_path_str):
    cif_path = Path(cif_path_str)
    parsed_structure = parse_mmcif(
        cif_path,
        CCD_CACHE,
        MOL_DIR,
    )
    structure = parsed_structure.data
    constraints = parsed_structure.residue_constraints

    record = {
        "cif_path": _relative_cif_path(cif_path),
    }
    record.update(check_ligand_distance_geometry(structure, constraints))
    record.update(check_ligand_stereochemistry(structure, constraints))
    record.update(check_ligand_flatness(structure, constraints))
    record.update(check_steric_clash(structure, molecules=CCD_CACHE))
    return record


def _load_ccd(ccd_path):
    with ccd_path.open("rb") as file:
        return pickle.load(file)


def _find_cifs(cif_root):
    if cif_root.is_file():
        return [cif_root]
    return sorted(cif_root.rglob("*.cif"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run physical-simulation metrics for CIF files in a directory",
    )
    parser.add_argument(
        "cif_root",
        type=Path,
        help="Directory containing CIF files (searched recursively) or a single CIF file",
    )
    parser.add_argument(
        "--ccd-path",
        type=Path,
        default=Path("/home/whgao/.boltz/ccd.pkl"),
        help="Path to the cached CCD pickle",
    )
    parser.add_argument(
        "--mols-dir",
        type=Path,
        default=Path("/home/whgao/.boltz/mols"),
        help="Directory containing cached RDKit molecules",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the aggregated CSV (defaults to <cif_root>/physical_checks.csv)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use (defaults to available CPUs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cif_root = args.cif_root.resolve()
    if not cif_root.exists():
        raise FileNotFoundError(f"CIF root {cif_root} does not exist")

    ccd_path = args.ccd_path.resolve()
    if not ccd_path.exists():
        raise FileNotFoundError(f"CCD cache not found at {ccd_path}")

    mols_dir = args.mols_dir.resolve()
    if not mols_dir.exists():
        raise FileNotFoundError(f"Molecule cache directory not found at {mols_dir}")

    cif_paths = _find_cifs(cif_root)
    if not cif_paths:
        raise FileNotFoundError(f"No CIF files found under {cif_root}")

    output_path = args.output
    if output_path is None:
        if cif_root.is_file():
            output_path = cif_root.parent / "physical_checks.csv"
        else:
            output_path = cif_root / "physical_checks.csv"
    output_path = output_path.resolve()

    global CCD_CACHE, MOL_DIR, CIF_ROOT
    CCD_CACHE = _load_ccd(ccd_path)
    MOL_DIR = mols_dir
    CIF_ROOT = cif_root if cif_root.is_dir() else cif_root.parent

    pool_size = args.num_workers or (os.cpu_count() or 1)
    pool_size = max(1, min(pool_size, len(cif_paths)))

    records = []
    if pool_size == 1:
        for cif_path in tqdm(cif_paths, desc="processing", unit="cif"):
            records.append(_process_single(str(cif_path)))
    else:
        with Pool(pool_size) as pool:
            with tqdm(total=len(cif_paths), desc="processing", unit="cif") as pbar:
                for record in pool.imap_unordered(_process_single, map(str, cif_paths)):
                    records.append(record)
                    pbar.update(1)

    df = pd.DataFrame.from_records(records)
    df["num_chain_clashes_all"] = df[
        [key for key in df.columns if "chain_clash" in key]
    ].sum(axis=1)
    df["num_pairs_all"] = df[[key for key in df.columns if "chain_pair" in key]].sum(axis=1)
    df["clash_free"] = df["num_chain_clashes_all"] == 0
    df["valid_ligand"] = (
        df[[key for key in df.columns if "violation" in key]].sum(axis=1) == 0
    )
    df["valid"] = (df["clash_free"]) & (df["valid_ligand"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary_lines = [
        f"total_structures={len(df)}",
        f"valid_structures={int(df['valid'].sum())}",
        f"valid_fraction={df['valid'].mean():.3f}",
        f"clash_free_fraction={df['clash_free'].mean():.3f}",
    ]
    violation_columns = [col for col in df.columns if "violation" in col]
    if "num_chain_clashes_all" in df:
        violation_columns.append("num_chain_clashes_all")
    if violation_columns:
        totals = df[violation_columns].sum()
        for col, value in totals.items():
            if isinstance(value, (int, np.integer)):
                summary_lines.append(f"{col}={int(value)}")
            else:
                summary_lines.append(f"{col}={float(value):.3f}")
    print("summary:" + " ".join(summary_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
