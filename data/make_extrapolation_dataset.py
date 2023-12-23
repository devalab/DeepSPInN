from rdkit import Chem
import pickle
from tqdm import tqdm


with open('../data/qm9_train_test_val_ir_nmr.pickle', 'rb') as handle:
    ir_all_datasets = pickle.load(handle)

all_small = list()
all_big = list()

for each_dataset_type in ["train", "test", "val"]:
    for each_datapoint in tqdm(ir_all_datasets[each_dataset_type]):
        smiles = each_datapoint[0]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol.GetNumAtoms() <= 7:
            all_small.append(each_datapoint)
        else:
            all_big.append(each_datapoint)

print(f"Small: {len(all_small)}")
print(f"Big: {len(all_big)}")

qm9_small_mols_big_mols = dict()
qm9_small_mols_big_mols["train"] = all_small
qm9_small_mols_big_mols["test"] = all_big

with open('qm9_small_train_big_test.pickle', 'wb') as handle:
    pickle.dump(qm9_small_mols_big_mols, handle)
