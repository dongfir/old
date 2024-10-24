# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 21:54:08 2021

@author: Osama
"""

from torch.utils.data import Dataset
from Bio.PDB import Polypeptide
import numpy as np
import torch


    
class InterpepComplexes(Dataset):
    
    def __init__(self, mode,
                 encoded_data_directory = "../../datasets/interpep_data/"):
        
        self.mode = mode
        
        self.encoded_data_directory = encoded_data_directory
        
        self.train_dir = "../../datasets/interpep_data/train_examples.npy"
        
        self.test_dir = "../../datasets/interpep_data/test_examples.npy"
        
        self.val_dir = "../../datasets/interpep_data/val_examples.npy"
       
        
        self.test_list = np.load(self.test_dir)

        self.train_list = np.load(self.train_dir)
        
        self.val_list = np.load(self.val_dir)
      

        
        if mode == "train":
            self.num_data = len(self.train_list)
        elif mode == "val":
            self.num_data = len(self.val_list)
        elif mode == "test":
            self.num_data = len(self.test_list)
    

        
    def __getitem__(self, index):
       
        if self.mode == "train":
            item = self.train_list[index]
        elif self.mode == "val":
            item = self.val_list[index]
        elif self.mode == "test":
            item = self.test_list[index]
    
        file_dir = self.encoded_data_directory
        
        with np.load(file_dir + "fragment_data/" + item + ".npz") as data:
            temp_pep_sequence = data["target_sequence"]
            temp_binding_sites = data["binding_sites"]
            
            
        with np.load(file_dir + "receptor_data/" + item.split("_")[0] + "_" +\
                     item.split("_")[1] + ".npz") as data:
            temp_nodes = data["nodes"]
           
        
        binding = np.zeros(len(temp_nodes))
        if len(temp_binding_sites) != 0:
            binding[temp_binding_sites] = 1
        target = torch.LongTensor(binding)
        
        
        
        
        
        
       
        nodes = temp_nodes[:, 0:20]
        
        prot_sequence = np.argmax(nodes, axis=-1)
        
        
        
        prot_sequence = " ".join([Polypeptide.index_to_one(i) for i in prot_sequence])
        

       
        pep_sequence = temp_pep_sequence 
        
        pep_sequence = torch.argmax(torch.FloatTensor(pep_sequence), dim=-1)
 

    
        
        
        return pep_sequence, prot_sequence, target
            
    def __len__(self):
        return self.num_data
    
class PepBindComplexes(Dataset):
    
    def __init__(self, mode,
                 encoded_data_directory = "../../datasets/pepbind_data/"):
        
        self.mode = mode
        
        self.encoded_data_directory = encoded_data_directory
        
        self.train_dir = "../../datasets/pepbind_data/train_examples.npy"
        
        self.test_dir = "../../datasets/pepbind_data/test_examples.npy"
        
        self.val_dir = "../../datasets/pepbind_data/val_examples.npy"
       
        
        self.test_list = np.load(self.test_dir)

        self.train_list = np.load(self.train_dir)
        
        self.val_list = np.load(self.val_dir)
      
        
        if mode == "train":
            self.num_data = len(self.train_list)
        elif mode == "val":
            self.num_data = len(self.val_list)
        elif mode == "test":
            self.num_data = len(self.test_list)
    

        
    def __getitem__(self, index):
       
        if self.mode == "train":
            item = self.train_list[index]
            
             
        elif self.mode == "val":
            item = self.val_list[index]
            
            
        elif self.mode == "test":
            item = self.test_list[index]
            
            
    
        file_dir = self.encoded_data_directory
       
        
        with np.load(file_dir + "fragment_data/" + item + ".npz") as data:
            temp_pep_sequence = data["target_sequence"]
            temp_binding_sites = data["binding_sites"]
            
            
        with np.load(file_dir + "receptor_data/" + item.split("_")[0] + "_" +\
                     item.split("_")[1] + ".npz") as data:
            temp_nodes = data["nodes"]
           
        
        binding = np.zeros(len(temp_nodes))
        if len(temp_binding_sites) != 0:
            binding[temp_binding_sites] = 1
        target = torch.LongTensor(binding)
        
        
        
        
        
        
       
        nodes = temp_nodes[:, 0:20]
        
        prot_sequence = np.argmax(nodes, axis=-1)
        
        
        
        prot_sequence = " ".join([Polypeptide.index_to_one(i) for i in prot_sequence])
        

       
        pep_sequence = temp_pep_sequence 
        
        pep_sequence = torch.argmax(torch.FloatTensor(pep_sequence), dim=-1)
 

    
        
        
        return pep_sequence, prot_sequence, target
     
            
    def __len__(self):
        return self.num_data
    
class PeptideComplexes(Dataset):
    
    def __init__(self, mode,
                 encoded_data_directory = "../../datasets/pepnn_data/all_data/"):
        
        self.mode = mode
        
        self.encoded_data_directory = encoded_data_directory
        
        self.train_dir = "../../datasets/pepnn_data/train_examples.npy"
        
        self.test_dir = "../../datasets/pepnn_test_data/test_examples.npy"
        
        self.val_dir = "../../datasets/pepnn_data/val_examples.npy"
       
        
        self.example_weights = np.load("../../datasets/pepnn_data/example_weights.npy")
        
        self.test_list = np.load(self.test_dir)

        self.train_list = np.load(self.train_dir)
        
        self.val_list = np.load(self.val_dir)
      

        
        if mode == "train":
            self.num_data = len(self.train_list)
        elif mode == "val":
            self.num_data = len(self.val_list)
        elif mode == "test":
            self.num_data = len(self.test_list)
    

        
    def __getitem__(self, index):
       
    
        if self.mode == "train":
            item = self.train_list[index]
            
            weight = self.example_weights[item]
             
        elif self.mode == "val":
            item = self.val_list[index]
            
            weight = self.example_weights[item]
            
        elif self.mode == "test":
            item = self.test_list[index]
            
            weight = 1
    
        if self.mode != "test":
            file_dir = self.encoded_data_directory
        else:
            file_dir = "../../datasets/pepnn_test_data/all_data/"
        
        
        with np.load(file_dir + "fragment_data/" + item + ".npz") as data:
            temp_pep_sequence = data["target_sequence"]
            temp_binding_sites = data["binding_sites"]
            
            
        with np.load(file_dir + "receptor_data/" + item.split("_")[0] + "_" +\
                     item.split("_")[1] + ".npz") as data:
            temp_nodes = data["nodes"]
           
        
        binding = np.zeros(len(temp_nodes))
        if len(temp_binding_sites) != 0:
            binding[temp_binding_sites] = 1
        target = torch.LongTensor(binding)
        
        
        
        
        
        
       
        nodes = temp_nodes[:, 0:20]
        
        prot_sequence = np.argmax(nodes, axis=-1)
        
        
        
        prot_sequence = " ".join([Polypeptide.index_to_one(i) for i in prot_sequence])
        

       
        pep_sequence = temp_pep_sequence 
        
        pep_sequence = torch.argmax(torch.FloatTensor(pep_sequence), dim=-1)
 

    
        
        
        return pep_sequence, prot_sequence, target, weight
     
            
    def __len__(self):
        return self.num_data
    
    
class BitenetComplexes(Dataset):
    
    def __init__(self, encoded_data_directory = "../bitenet_data/all_data/"):

        
        self.encoded_data_directory = encoded_data_directory
        
       


        self.train_dir = "../../datasets/bitenet_data/examples.npy"
    
        
         

        self.full_list = np.load(self.train_dir)
        

        
        
        self.num_data = len(self.full_list)
     
        

        
    def __getitem__(self, index):
       
        item = self.full_list[index]
        
        file_dir = self.encoded_data_directory
        
        with np.load(file_dir + "fragment_data/" + item[:-1] + "_" + item[-1]  + ".npz") as data:
            temp_pep_sequence = data["target_sequence"]
            temp_binding_matrix = data["binding_matrix"]
        
            
        with np.load(file_dir + "receptor_data/" + item.split("_")[0] + "_" +\
                     item.split("_")[1][0] + ".npz") as data:
            temp_nodes = data["nodes"]
           
        
        binding_sum = np.sum(temp_binding_matrix, axis=0).T
        
        target = torch.LongTensor(binding_sum >= 1)
        
           
               
        nodes = temp_nodes[:, 0:20]
        
        prot_sequence = np.argmax(nodes, axis=-1)
        
        
        
        prot_sequence = " ".join([Polypeptide.index_to_one(i) for i in prot_sequence])
        

       
        pep_sequence = temp_pep_sequence 
        
        pep_sequence = torch.argmax(torch.FloatTensor(pep_sequence), dim=-1)
 
        
        
        
        return pep_sequence, prot_sequence, target
            
    def __len__(self):
        return self.num_data