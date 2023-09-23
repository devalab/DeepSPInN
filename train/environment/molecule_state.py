from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
import numpy as np
import copy
import random
from collections import OrderedDict
from rdkit.Chem import rdMolDescriptors as rdDesc
from .molecular_graph import *
import torch
import warnings
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem
from rdkit import DataStructs
import ipdb

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")
NUM_ATOMS = 9
NUM_ACTIONS = NUM_ATOMS * NUM_ATOMS * 3


class MolState:

    def __init__(self, molForm, nmr_shift_split, smiles="C"):
        self.molForm = molForm
        self.target_nmr = nmr_shift_split
        self.rdmol = Chem.MolFromSmiles("C")
        
        self.AtomicNum = np.zeros(NUM_ATOMS)
        self.presentInRdmol = np.zeros(NUM_ATOMS)
        self.rdmolIdx = np.zeros(NUM_ATOMS)
        self.remVal = np.zeros(NUM_ATOMS)
        self.molGraphSpectra = np.zeros((NUM_ATOMS,68))
        
        self.setUpMapping()

        #Tracking 5 rings:
        self.ringDat = np.zeros((5,NUM_ATOMS,1))
        self.presentInRdmol[0] = 1 #Mapping the first atom in the rdmol to this carbon
        self.adjacency_matrix = np.ones((NUM_ATOMS,NUM_ATOMS,1)) # Puts zero wherever there is a bond
        self.valid_actions()

    def setUpMapping(self):
        index = 0
        carbonIndex = 0
        for i in range(4):
            for j in range(self.molForm[i]):
                self.remVal[index] = 4 - i

                if i == 0:
                    self.molGraphSpectra[index] = makeSpectrumFeature(self.target_nmr[carbonIndex][0], self.target_nmr[carbonIndex][1])
                    # self.remVal[index] -= self.target_nmr[carbonIndex][1]
                   
                    carbonIndex += 1

                self.AtomicNum[index] = i+6  #Atomic Num
                self.presentInRdmol[index] = 0    #Present in rdmol
                self.rdmolIdx[index] = 0    #RdmolIndex

                index += 1

    def __str__(self):
        return Chem.MolToSmiles(self.rdmol)

    @property
    def molGraph(self):
        return get_graph_from_molstate(self)

    @property
    def totalNumOfAtoms(self):
        return sum(self.molForm)

    @property
    def numInRdmol(self):
        return self.rdmol.GetNumAtoms()

    def render(self):
        """
        Displays 2D line structure of the current state
        """
        return Chem.Draw.MolToImage(self.rdmol)

    def invalidAction(self):
        raise Exception("Invalid Action has been chosen :(")

    def _returnBondType(self, bondOrder):
        bondTypes = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE
            ]
        return bondTypes[bondOrder]

    def _actionListToInt(self, action):
        """
        Action: [A1,A2,BT]
        BT -> [0,2]
        A2 -> [0,23]
        A1 -> [0,NUM_ATOMS]
        """
        return (action[2]) + 3*action[1] + 3*NUM_ATOMS*action[0]
    
    def _actionIntToList(self, action):
        """
        
        """
        return [ (action//(3*NUM_ATOMS)) , (action//3)%NUM_ATOMS , action%3]                     

    def valid_actions(self):
        zeroVec = np.zeros((NUM_ATOMS,NUM_ATOMS,3))
        valDat = self.remVal
        presentInRdMol = self.presentInRdmol

        node1ValBroadcast = zeroVec + (valDat*presentInRdMol).reshape(NUM_ATOMS,1,1)
        node2ValBroadcast = zeroVec + valDat.reshape(NUM_ATOMS,1)

        bonds = np.array([1, 2, 3]).reshape(1,1,3)
        node1Bools = node1ValBroadcast - bonds
        node2Bools = node2ValBroadcast - bonds
        ans = (node1Bools >= 0).astype(int) * (node2Bools >= 0).astype(int)

        #removing self bonds:
        self_filter = np.diag(np.ones(NUM_ATOMS))
        self_filter = np.logical_not(self_filter).astype(int)
        self_filter = np.expand_dims(self_filter, -1)
        ans = ans * self_filter

        #Removing already present bonds
        ans = ans * self.adjacency_matrix

        #Removing bonds from atoms belonging to the same ring:
        common_ring_matrix = np.matmul(self.ringDat,self.ringDat.transpose(0,2,1)) # 5xNUM_ATOMSx1 * 5x1xNUM_ATOMS = 5xNUM_ATOMSxNUM_ATOMS
        ring_mask = np.max(common_ring_matrix, axis=0, keepdims=True) # 5xNUM_ATOMSxNUM_ATOMS -> 1xNUM_ATOMSxNUM_ATOMS

        #reshaping to appropriate shape:
        A = np.logical_not(self.adjacency_matrix.transpose(2,0,1)).astype(int)

        ring_mask_3 = np.matmul(A, A)
        ring_mask_4 = np.matmul(ring_mask_3, A)
        ring_mask = ring_mask.transpose(1,2,0) # 1xNUM_ATOMSxNUM_ATOMS -> NUM_ATOMSxNUM_ATOMSx1
        ring_mask_3 = ring_mask_3.transpose(1,2,0)
        ring_mask_4 = ring_mask_4.transpose(1,2,0)

        ring_mask = np.logical_not(ring_mask).astype(int) # now it has 0 where they are in the same ring
        ring_mask_3 = np.logical_not(ring_mask_3).astype(int)
        ring_mask_4 = np.logical_not(ring_mask_4).astype(int)
        
        ans = ans * ring_mask * ring_mask_3 * ring_mask_4
        # ans = ans*ring_mask

        self.action_mask = ans.reshape(NUM_ACTIONS)
        return ans.reshape(NUM_ACTIONS)

    def invalidAction(self):
        raise Exception("Invalid Action has been chosen :(")

    def doStep(self, action: list):
        """
        Params: 
        + action : list of 3 dimentions having:
        [nodeIdx1, nodeIdx2, bondOrder]
        Adds a bond of order = bondOrder if bondOrder in [0, 1, 2]
        
        returns validOrNot(boolean), smiles if action is possible
        """
        nodeIdx1, nodeIdx2, bondOrder = action

        
        if self.presentInRdmol[nodeIdx2] == 0:
            #Second atom is an independent Node

            element = self.AtomicNum[nodeIdx2]
            RW = Chem.RWMol(self.rdmol)
            rdIndex = RW.AddAtom(Chem.Atom(int(element)))

            self.rdmolIdx[nodeIdx2] = rdIndex
            self.presentInRdmol[nodeIdx2] = 1
            RW.AddBond(int(self.rdmolIdx[nodeIdx1]), rdIndex, order=self._returnBondType(bondOrder))
            # validOrNot = Chem.SanitizeMol(RW, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

 
            #s = Chem.MolToSmiles(m, isomericSmiles=True)
            #m = Chem.MolFromSmiles(s)
            self.remVal[nodeIdx1] -= (bondOrder + 1)
            self.remVal[nodeIdx2] -= (bondOrder + 1)
            self.adjacency_matrix[nodeIdx1][nodeIdx2] = 0
            self.adjacency_matrix[nodeIdx2][nodeIdx1] = 0
            # if validOrNot:
            #     return self.invalidAction()
            
            self.rdmol = RW.GetMol()

            return
        
        # Both the atoms are already in the rmol but don't have a bond between them
        
        RW = Chem.RWMol(self.rdmol)
        RW.AddBond(int(self.rdmolIdx[nodeIdx1]), int(self.rdmolIdx[nodeIdx2]), order=self._returnBondType(bondOrder))
        # validOrNot = Chem.SanitizeMol(RW, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

        self.remVal[nodeIdx1] -= (bondOrder + 1)
        self.remVal[nodeIdx2] -= (bondOrder + 1)
        self.adjacency_matrix[nodeIdx1][nodeIdx2] = 0
        self.adjacency_matrix[nodeIdx2][nodeIdx1] = 0


        # if validOrNot:
        #     return self.invalidAction()

        
        self.rdmol = RW.GetMol()
        self.ringDat = np.zeros((5,NUM_ATOMS,1))

        ringDataObj = Chem.GetSymmSSSR(self.rdmol)
        for i in range(len(ringDataObj)):
            if i>=5:
                break
            listDat = list(ringDataObj[i]) #converting from ring object to list
            for j in listDat:
                trueIndex = int(np.where((self.rdmolIdx==j).astype(int) * (self.presentInRdmol==1).astype(int))[0])
                self.ringDat[i][trueIndex] = 1

        return
 
