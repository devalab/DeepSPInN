import os
import sys
import pickle
import numpy as np
from rdkit import Chem

if len(sys.argv) == 1:
    print_mols = False
else:
    print_mols = bool(sys.argv[1])

for num_episodes in range(40):

    num_total = 0
    num_top_1 = 0
    num_top_3 = 0
    num_top_5 = 0
    num_top_10 = 0
    num_top_40 = 0

    all_mols = list()
    wrong_mols = list()
    wrong_mols_nums = list()

    #for filename in os.listdir('~/DeepSPInN/test/test_outputs'):
    test_path = './test_outputs/'
    ar_test_path = '/home2/arihanth.srikar/sriram/DeepSPInN/test/test_outputs/'

    all_files = os.listdir(test_path)
    all_files = [test_path+x for x in all_files if 'output' in x]
    ar_all_files = os.listdir(ar_test_path)
    ar_all_files = [ar_test_path+x for x in ar_all_files if 'output' in x]

    all_files = all_files + ar_all_files

    all_files = sorted(all_files, key=lambda x:int(x.split('_')[-1].split('.')[0]))

    for filename in all_files:
        # with open(os.path.join('~/DeepSPInN/test/test_outputs', filename), 'rb') as f: # open in readonly mode
        with open(filename, 'rb') as f: # open in readonly mode
            current_mol_log = pickle.load(f)
            
        mol_id = int(filename.split('_')[-1].split('.')[0])
        all_mols.append(mol_id)
        
        smiles_scores = dict()
        all_smiles = list()
        
        for episode_result in current_mol_log[:num_episodes]:
            reached_mol = Chem.CanonSmiles(episode_result[2][-1])
            all_smiles.append(reached_mol)
            smiles_scores[reached_mol] = episode_result[3]
        
        all_smiles = list(set(all_smiles))
        all_smiles = sorted(all_smiles, key=lambda x: smiles_scores[x], reverse=True)
        
        num_total += 1
        
        target_smiles = Chem.CanonSmiles(current_mol_log[0][0])
        if print_mols:
            print('Top hits for', target_smiles, filename.split('/')[-1])
        idx = 0
        while idx < len(all_smiles):
            hit = all_smiles[idx]
            
            if hit == target_smiles:
                if idx < 5 and print_mols:
                    print('\t', idx, hit.ljust(20), str(smiles_scores[hit]), '<---')
                elif print_mols:
                    print('\t', idx, hit.ljust(20), str(smiles_scores[hit]), 'vvv')
                if idx < 1:
                    num_top_1 += 1
                if idx < 3:
                    num_top_3 += 1
                if idx < 5:
                    num_top_5 += 1
                if idx < 10:
                    num_top_10 += 1
                if idx < 40:
                    num_top_40 += 1
            elif idx < 5 and print_mols:
                print('\t', idx, hit.ljust(20), str(smiles_scores[hit]))

            idx += 1

        if target_smiles not in all_smiles:
            wrong_mols.append(target_smiles)
            wrong_mols_nums.append(int(filename.split('_')[-1].split('.')[0]))
            if print_mols:
                print('Paste in cheminfo:')
                print(target_smiles, current_mol_log[0][1])
                print('\n'.join(all_smiles[:5]))

    print(f"Number of episodes: {num_episodes}")
    print('Top 1 (%):', 100.0* num_top_1/num_total)
    print('Top 3 (%):', 100.0* num_top_3/num_total)
    print('Top 5 (%):', 100.0* num_top_5/num_total)
    print('Top 10 (%):', 100.0* num_top_10/num_total)
    print('Top 40 (%):', 100.0* num_top_40/num_total)
