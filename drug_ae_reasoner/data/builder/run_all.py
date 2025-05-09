import subprocess

def run(script):
    print(f"\n[RUNNING] {script} ...")
    subprocess.run(["python", f"drug_ae_reasoner/data/builder/{script}.py"], check=True)

def main():
    run("build_cadec_kg")
    run("normalize_cadec_kg")
    run("build_oae_index")
    run("convert_owl_to_graph")
    print("\n✅ All steps complete.")

if __name__ == "__main__":
    main()
