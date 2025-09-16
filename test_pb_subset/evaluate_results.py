from pathlib import Path

# from pxmeter.eval import evaluate


# ROOT = Path(__file__).resolve().parents[1]

# # 8SLG
# ref_cif = str(ROOT / "examples/test/8slg.cif")
# model_cif = str(
#     ROOT
#     / "test_pb_subset/boltz_results_pb_subset/predictions/8SLG_G5A/8SLG_G5A_model_0.cif"
# )
# metric_result = evaluate(ref_cif=ref_cif, model_cif=model_cif)
# print("8SLG:", metric_result.to_json_dict())  # noqa: T201

# # 8CNH
# ref_cif = str(ROOT / "examples/test/8cnh.cif")
# model_cif = str(
#     ROOT
#     / "test_pb_subset/boltz_results_pb_subset/predictions/8CNH_V6U/8CNH_V6U_model_0.cif"
# )
# metric_result = evaluate(ref_cif=ref_cif, model_cif=model_cif)
# print("8CNH:", metric_result.to_json_dict())  # noqa: T201

# # 7CNQ
# ref_cif = str(ROOT / "examples/test/7cnq.cif")
# model_cif = str(
#     ROOT
#     / "test_pb_subset/boltz_results_pb_subset/predictions/7CNQ_G8X/7CNQ_G8X_model_0.cif"
# )
# metric_result = evaluate(ref_cif=ref_cif, model_cif=model_cif)
# print("7CNQ:", metric_result.to_json_dict())  # noqa: T201

# # 5SAK
# ref_cif = str(ROOT / "examples/test/5sak.cif")
# model_cif = str(
#     ROOT
#     / "test_pb_subset/boltz_results_pb_subset/predictions/5SAK_ZRY/5SAK_ZRY_model_0.cif"
# )
# metric_result = evaluate(ref_cif=ref_cif, model_cif=model_cif)
# print("5SAK:", metric_result.to_json_dict())  # noqa: T201

# help(evaluate)

from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

import gemmi  # pip install gemmi


ROOT = Path("/home/whgao/boltz")  # <-- change if needed

def superposed_rmsd(ref_xyz: List[gemmi.Position], mod_xyz: List[gemmi.Position]) -> float:
    """Return best-fit RMSD (heavy atoms) using Kabsch superposition via gemmi."""
    assert len(ref_xyz) == len(mod_xyz) and len(ref_xyz) > 0
    sp = gemmi.superpose_positions(mod_xyz, ref_xyz)  # superpose model onto ref
    return sp.rmsd

def get_chain_first_ligand_residue(struct: gemmi.Structure, chain_id: str) -> gemmi.Residue:
    """Get the first hetero (ligand) residue from a given chain."""
    ch = struct[0].find_chain(chain_id)
    if ch is None:
        raise ValueError(f"Chain '{chain_id}' not found.")
    # ligand residues are hetero (het_flag != ' ')
    ligands = [res for res in ch if res.het_flag != ' ']
    if not ligands:
        # fallback: in some structures ligand may be marked as ' ' but non-AA by chem comp?
        ligands = [res for res in ch if res.name.strip() not in ("ALA","GLY","SER","THR","CYS","VAL","LEU","ILE","MET","PRO","PHE","TYR","TRP","HIS","LYS","ARG","ASP","GLU","ASN","GLN","SEC","PYL")]
    if not ligands:
        raise ValueError(f"No ligand residues found on chain '{chain_id}'.")
    return ligands[0]

def get_heavy_atom_positions_and_names(res: gemmi.Residue) -> Tuple[List[str], List[gemmi.Position]]:
    names, pos = [], []
    for a in res:
        # gemmi.Atom.element is gemmi.Element
        if a.element.atomic_number() > 1:  # heavy atom only
            names.append(a.name.strip())
            pos.append(a.pos)
    if not pos:
        raise ValueError("Ligand residue has no heavy atoms.")
    return names, pos

def compute_ligand_rmsd_from_cifs(
    ref_cif: str,
    model_cif: str,
    ref_lig_chain: str,
    model_lig_chain: Optional[str] = None,
) -> float:
    """Compute heavy-atom best-fit RMSD of ligand by matching atom names."""
    ref_st = gemmi.read_structure(ref_cif)
    mod_st = gemmi.read_structure(model_cif)
    ref_res = get_chain_first_ligand_residue(ref_st, ref_lig_chain)

    # If model chain mapping is not given, assume same chain ID as ref
    mod_chain_id = model_lig_chain or ref_lig_chain
    mod_res = get_chain_first_ligand_residue(mod_st, mod_chain_id)

    ref_names, ref_pos = get_heavy_atom_positions_and_names(ref_res)
    mod_names, mod_pos = get_heavy_atom_positions_and_names(mod_res)

    # Build name->index map for model to match ordering to ref
    mod_idx = {n: i for i, n in enumerate(mod_names)}
    paired_ref, paired_mod = [], []
    for i, n in enumerate(ref_names):
        if n in mod_idx:
            paired_ref.append(ref_pos[i])
            paired_mod.append(mod_pos[mod_idx[n]])
    if len(paired_ref) < 3:  # need >=3 atoms for stable superposition
        raise ValueError(f"Too few matched heavy atoms between ref({len(ref_pos)}) and model({len(mod_pos)}).")

    return superposed_rmsd(paired_ref, paired_mod)

def pretty_print_metrics(title: str, d: Dict):
    print(f"\n=== {title} ===")
    print(json.dumps(d, indent=2))

# ----------------- Configure your cases -----------------
CASES = [
    # (label, ref_cif_path, model_cif_path, interested_lig_asym_in_ref, optional_hint_for_model_chain_name)
    ("8SLG",
     str(ROOT / "examples/test/8slg.cif"),
     str(ROOT / "test_pb_subset/boltz_results_pb_subset/predictions/8SLG_G5A/8SLG_G5A_model_0.cif"),
     "C",
     "L_G5A"),  # model ligand chain known from your mapping
    ("8CNH",
     str(ROOT / "examples/test/8cnh.cif"),
     str(ROOT / "test_pb_subset/boltz_results_pb_subset/predictions/8CNH_V6U/8CNH_V6U_model_0.cif"),
     "B",
     "L_V6U"),
    ("7CNQ",
     str(ROOT / "examples/test/7cnq.cif"),
     str(ROOT / "test_pb_subset/boltz_results_pb_subset/predictions/7CNQ_G8X/7CNQ_G8X_model_0.cif"),
     "H",
     "L_G8X"),
    ("5SAK",
     str(ROOT / "examples/test/5sak.cif"),
     str(ROOT / "test_pb_subset/boltz_results_pb_subset/predictions/5SAK_ZRY/5SAK_ZRY_model_0.cif"),
     "E",
     "L_ZRY"),
]

# ----------------- Run PXMeter + RMSD fallback -----------------
from pxmeter.eval import evaluate

all_out = []
for label, ref_cif, model_cif, INTERESTED_LIG_ASYM, MODEL_LIG_CHAIN in CASES:
    # 1) Ask PXMeter to focus on the ligand of interest (some versions expose this arg)
    result = evaluate(
        ref_cif=ref_cif,
        model_cif=model_cif,
        interested_lig_label_asym_id=INTERESTED_LIG_ASYM  # str or list[str]
        # You can also pass run_config=... if your version supports ligand RMSD switch there
    )
    d = result.to_json_dict()

    # 2) Try to read ligand RMSD if PXMeter provides it (key names vary by version)
    lig_rmsd_keys = [
        ("ligand", "rmsd"),
        ("interface", "ligand_rmsd"),
        ("complex", "ligand_rmsd"),
        ("chain", "ligand_rmsd"),
    ]
    found = None
    # heuristic search across likely locations
    if "ligand" in d and isinstance(d["ligand"], dict) and "rmsd" in d["ligand"]:
        found = d["ligand"]["rmsd"]
    else:
        # some versions might tuck it into interface or top-level
        for k1, k2 in lig_rmsd_keys:
            if k1 in d and isinstance(d[k1], dict) and k2 in d[k1]:
                found = d[k1][k2]
                break

    # 3) If not found, compute RMSD from the CIFs with gemmi (heavy atoms, name-matched)
    if found is None:
        # If PXMeter returned a mapping, prefer that mapping to locate the model ligand chain
        model_chain_hint = MODEL_LIG_CHAIN
        try:
            mapping = d.get("ref_to_model_chain_mapping", {})
            if INTERESTED_LIG_ASYM in mapping:
                model_chain_hint = mapping[INTERESTED_LIG_ASYM]
        except Exception:
            pass

        try:
            rmsd = compute_ligand_rmsd_from_cifs(
                ref_cif=ref_cif,
                model_cif=model_cif,
                ref_lig_chain=INTERESTED_LIG_ASYM,
                model_lig_chain=model_chain_hint,
            )
            # stash it into the dict for convenience
            d.setdefault("ligand", {})
            d["ligand"]["rmsd_heavy_bestfit"] = rmsd
        except Exception as e:
            d.setdefault("ligand", {})
            d["ligand"]["rmsd_heavy_bestfit"] = None
            d["ligand"]["rmsd_error"] = f"{type(e).__name__}: {e}"
    else:
        d.setdefault("ligand", {})
        d["ligand"]["rmsd_reported_by_pxmeter"] = float(found)

    all_out.append((label, d))
    pretty_print_metrics(label, d)

# If you want a compact summary row per entry:
print("\n=== Summary (entry, complex_lddt, protein_chain_lddt, ligand_chain_lddt, ligand_RMSD) ===")
for label, d in all_out:
    complex_lddt = d.get("complex", {}).get("lddt")
    # guess protein & ligand chain IDs from ref_chain_info types
    protein_lddt = None
    ligand_lddt = None
    if "chain" in d and "ref_chain_info" in d:
        for ref_chain, info in d["ref_chain_info"].items():
            if info.get("entity_type","").startswith("polypeptide"):
                protein_lddt = d["chain"].get(ref_chain, {}).get("lddt", protein_lddt)
            elif info.get("entity_type","") == "ligand":
                ligand_lddt = d["chain"].get(ref_chain, {}).get("lddt", ligand_lddt)
    lig_rmsd = d.get("ligand", {}).get("rmsd_reported_by_pxmeter",
                d.get("ligand", {}).get("rmsd_heavy_bestfit", None))
    print(f"{label:5s} | {complex_lddt:.3f} | {protein_lddt if protein_lddt is not None else 'NA'} | "
          f"{ligand_lddt if ligand_lddt is not None else 'NA'} | {lig_rmsd if lig_rmsd is not None else 'NA'}")

