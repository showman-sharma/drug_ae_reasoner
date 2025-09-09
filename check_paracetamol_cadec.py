# Quick check: Does 'paracetamol' have any outgoing AE edges in CADEC KG?
from drug_ae_reasoner.data.cadec_loader import get_cadec_drug_nodes, get_cadec_ae_pairs

drug = "paracetamol"

drug_nodes = get_cadec_drug_nodes(drug)
pairs = get_cadec_ae_pairs(drug_nodes)

print(f"Drug nodes for '{drug}':", drug_nodes)
print(f"Outgoing Drug→AE edges for '{drug}':")
for pair in pairs:
    print(pair)
if not pairs:
    print("No outgoing AE edges found for this drug in CADEC KG.")
