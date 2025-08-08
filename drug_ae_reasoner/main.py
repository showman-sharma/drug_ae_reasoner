# drug_ae_reasoner/main.py
import argparse
from .config import RX_PATH, CADEC_KG_PATH, OAE_INDEX_PATH, OAE_LABEL_MAP_PATH, OAE_GRAPH_PATH
from .utils.path_reasoner import find_top_drug_to_input_ae_paths

def main():
    parser = argparse.ArgumentParser(
        description="Trace semantic paths from a drug to adverse effects using CADEC and OAE."
    )
    parser.add_argument("--drug", type=str, required=True, help="Drug name (e.g., 'metformin')")
    parser.add_argument("--aes",  type=str, required=True, nargs="+",
                        help="List of adverse effect labels (e.g., nausea vomiting)")
    parser.add_argument("--cadec_ae_thresh", type=float, default=0.70, help="Threshold for CADEC-AE→OAE mapping")
    parser.add_argument("--input_ae_thresh", type=float, default=0.70, help="Threshold for input-AE→OAE mapping")
    parser.add_argument("--n_paths", type=int, default=5)
    parser.add_argument("--n_cadec", type=int, default=5)
    parser.add_argument("--n_input", type=int, default=5)
    parser.add_argument("--n_disconnect", type=int, default=3)
    args = parser.parse_args()

    print(f"[INFO] Running Drug-AE Path Reasoning for drug: {args.drug}")
    print(f"[INFO] Input AE terms: {args.aes}")

    connected, top_paths, fb_drug, fb_ae, verbalizations = find_top_drug_to_input_ae_paths(
        drug=args.drug,
        ae_input_list=args.aes,
        rx_path=RX_PATH,
        cadec_kg_path=CADEC_KG_PATH,
        oae_index_path=OAE_INDEX_PATH,
        oae_label_map_path=OAE_LABEL_MAP_PATH,
        oae_graph_path=OAE_GRAPH_PATH,
        n_paths=args.n_paths,
        n_cadec=args.n_cadec,
        n_input=args.n_input,
        cadec_ae_threshold=args.cadec_ae_thresh,
        input_ae_threshold=args.input_ae_thresh,
        n_disconnect=args.n_disconnect
    )

    if connected:
        print(f"[INFO] Found {len(top_paths)} real paths.")
    else:
        print(f"[INFO] No real paths found. Showing fallback paths instead.")
        print(f"[INFO] Fallbacks: Drug→AE={len(fb_drug)} | AE→Drug={len(fb_ae)}")

    print("\n--- Verbalized Reasoning Paths ---\n")
    for v in verbalizations:
        print(v)

if __name__ == "__main__":
    main()
