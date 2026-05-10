# =========================================
# MatPES JSON → ML Dataset (FINAL CLEAN VERSION)
# =========================================

import ijson
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from decimal import Decimal
import os
from tqdm import tqdm

# =========================================
# FIX: Convert Decimal → float recursively
# =========================================
def convert_decimals(obj):
    if isinstance(obj, list):
        return [convert_decimals(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


# =========================================
# Physics feature functions
# =========================================
def soc_proxy(Z_list):
    # relativistic scaling proxy (heuristic)
    return np.mean([z**4 for z in Z_list]) if len(Z_list) > 0 else 0


def electronegativity_range(EN_list):
    return (max(EN_list) - min(EN_list)) if len(EN_list) > 0 else 0


# =========================================
# Process single entry
# =========================================
def process_entry(row):

    try:
        structure = Structure.from_dict(row["structure"])

        Z = []
        EN = []
        magmoms = []

        for site in structure:
            el = site.specie

            Z.append(el.Z)

            if hasattr(el, "X") and el.X is not None:
                EN.append(el.X)

            magmoms.append(site.properties.get("magmom", 0))

        return {
            # --------------------------
            # Identifiers
            # --------------------------
            "material_id": row.get("matpes_id"),

            # --------------------------
            # Electronic properties
            # --------------------------
            "band_gap": float(row.get("bandgap", 0) or 0),
            "energy": float(row.get("energy", 0) or 0),
            "cohesive_energy": float(row.get("cohesive_energy_per_atom", 0) or 0),

            # --------------------------
            # Symmetry
            # --------------------------
            "space_group": row.get("symmetry", {}).get("number", 0),

            # --------------------------
            # Structural physics (IMPORTANT)
            # --------------------------
            "volume": float(row.get("volume", 0) or 0),
            "density": float(row.get("density", 0) or 0),
            "nsites": len(structure),

            # --------------------------
            # Chemistry descriptors
            # --------------------------
            "Z_mean": np.mean(Z),
            "Z_max": np.max(Z),
            "Z_var": np.var(Z),
            "num_elements": len(structure.composition.elements),

            # --------------------------
            # Physics proxies
            # --------------------------
            "SOC_proxy": soc_proxy(Z),
            "EN_range": electronegativity_range(EN),

            # --------------------------
            # Magnetism
            # --------------------------
            "magmom": sum(magmoms)

        }

    except Exception as e:
        print("ERROR processing entry:", e)
        return None


# =========================================
# Dataset builder (STREAMING)
# =========================================
def build_dataset(json_file, max_entries=100000):

    results = []

    with open(json_file, "rb") as f:
        objects = ijson.items(f, "item")

        for i, obj in enumerate(tqdm(objects)):

            # fix Decimal issue
            obj = convert_decimals(obj)

            processed = process_entry(obj)

            if processed is not None:
                results.append(processed)

            # progress log
            if i % 1000 == 0:
                print(f"Processed: {i} | Valid: {len(results)}")

            # optional limit
            if max_entries and i >= max_entries:
                break

    df = pd.DataFrame(results)

    print("\n=================================")
    print("FINAL DATASET CREATED")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values:\n", df.isna().sum())
    print("=================================\n")

    return df


# =========================================
# MAIN EXECUTION
# =========================================
if __name__ == "__main__":

    INPUT_JSON = "MatPES-R2SCAN-2025.2.json"   # CHANGE THIS
    OUTPUT_PARQUET = "materials_ml.parquet"
    OUTPUT_CSV = "materials_ml.csv"

    print("\n🚀 Starting dataset build...\n")

    df_ml = build_dataset(INPUT_JSON, max_entries=20000)

    # =========================================
    # LABELS (IMPORTANT FOR ML)
    # =========================================
    df_ml["is_metal"] = df_ml["band_gap"] < 0.1
    df_ml["is_magnetic"] = df_ml["magmom"].fillna(0) > 0.1

    # =========================================
    # SAVE DATASET
    # =========================================
    df_ml.to_parquet(OUTPUT_PARQUET, index=False)
    df_ml.to_csv(OUTPUT_CSV, index=False)

    print("\n💾 Saved files:")
    print(" -", OUTPUT_PARQUET)
    print(" -", OUTPUT_CSV)

    # =========================================
    # FILE SIZE CHECK
    # =========================================
    size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024**2)
    print(f"\n📦 Parquet size: {size_mb:.2f} MB")

    # =========================================
    # QUICK SANITY CHECK
    # =========================================
    print("\n📊 Class balance:")
    print(df_ml["is_metal"].value_counts(normalize=True))
