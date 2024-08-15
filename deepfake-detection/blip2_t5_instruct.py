import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
print(os.environ["CUDA_VISIBLE_DEVICES"])

from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import lavis
from lavis.models import load_model_and_preprocess

import random 
random.seed(43)

import argparse

EXT = ['.jpg', '.jpeg', '.png']

# class TextInvDataset(Dataset):
#     def __init__(self, csv, vis_processors=None, txt_processors=None):
        
#         self.path_and_labels = pd.read_csv(csv, index_col="img_path")
#         self.vis_processors = vis_processors
#         self.txt_processors = txt_processors

#     def __len__(self):
        
#         return len(list(self.path_and_labels.index))

#     def __getitem__(self, index):

#         image_path = list(self.path_and_labels.index)[index]
#         image = Image.open(image_path).convert("RGB")
#         if self.vis_processors:
#             image = self.vis_processors(image)
        
#         label = self.path_and_labels.loc[image_path, "label"]
        
#         is_uncommon = "uncommon" in image_path

#         return image, label, is_uncommon

class TextInvDataset(Dataset):
    def __init__(self, roots, labels, vis_processors=None, txt_processors=None):
        
        self.path_and_labels = {
            'img_path': [],
            'label': []
        }
        
        assert len(roots) == len(labels), "Please assign a label for each image root."
        
        for root, label in zip(roots, labels):
            n_sample = 0
            for r, dirs, files in os.walk(root):
                for file in files:
                    if os.path.splitext(file)[-1] in EXT:
                        self.path_and_labels["img_path"].append(os.path.join(r, file))
                        self.path_and_labels["label"].append(label)
                        n_sample += 1
            print(f'Found {n_sample} images with label "{label}".')
        
        self.path_and_labels = pd.DataFrame.from_dict(self.path_and_labels)
        self.path_and_labels.set_index("img_path", inplace=True)
        
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors

    def __len__(self):
        
        return len(list(self.path_and_labels.index))

    def __getitem__(self, index):

        image_path = list(self.path_and_labels.index)[index]
        image = Image.open(image_path).convert("RGB")
        if self.vis_processors:
            image = self.vis_processors(image)
        
        label = self.path_and_labels.loc[image_path, "label"]

        return image, label


# This is for query lots of images

from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import time

class InstructBLIP():
    def __init__(self, name="blip2_vicuna_instruct_textinv", model_type="vicuna7b", is_eval=True, device="cpu") -> None:
        print(f'Loading model...')
        #self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name, model_type, is_eval, device)
        self.imgs = []
        self.labels = []
        
        # QA
        self.question = ""
        
        # results
        self.acc = None
        self.confusion_mat = None
        
        self.acc_3class = None
        self.confusion_mat_3class = None
        
        self.com_acc = None
        self.com_confusion_mat = None
        self.uncom_acc = None
        self.uncom_confusion_mat = None

    def LoadModels(self, model, vis_processors, txt_processors, device):
        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.device = device
        
    def LoadData(self, roots, labels):
        self.roots = [roots] if isinstance(roots, str) else roots
        self.text_labels = [labels] if isinstance(labels, str) else labels
        self.dataset = TextInvDataset(self.roots, self.text_labels, vis_processors=self.vis_processors["eval"])
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=8, shuffle=False, num_workers=8)    
    
    def QueryImgs_batch(self, question, true_string="yes", logPath='log.txt'):
        self.labels = []
        self.label_3class = []
        self.ans_list = []
        self.question = question
        
        for image, label in tqdm(self.dataloader):
            
            image = image.to(self.device)
            
            questions = [self.question] * image.shape[0]
            
            # samples = {"image": image, "text_input": questions}
            # ans = self.model.predict_answers(samples=samples, inference_method="generate", answer_list=["yes", "no"])
            # pred_label = [0 if a == true_string else 1 for a in ans]
            
            samples = {"image": image, "prompt": questions}
            candidates = ["yes", "no"]
            ans = self.model.predict_class(samples=samples, candidates=candidates)
            pred_label = [0 if candidates[list(a).index(0)]==true_string else 1 for a in ans]
            self.ans_list += pred_label
            
            label = [0 if l == true_string else 1 for l in label]
            self.labels += label
        
        self.acc = accuracy_score(self.labels, self.ans_list)
        self.confusion_mat = confusion_matrix(self.labels, self.ans_list, labels=[0,1])
        
        self.ans_list = np.array(self.ans_list)
        self.labels = np.array(self.labels)
        self.label_3class = np.array(self.label_3class)
        
        self.PrintResult(detailed=True, logPath=logPath)
        
        return self.acc, self.confusion_mat, self.ans_list, self.labels, self.label_3class
    
    def Query(self, image, question):
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        
        samples = {"image": image, "prompt": question}
        candidates = ["yes", "no"]
        ans = self.model.predict_class(samples=samples, candidates=candidates)
        pred_label = ["True" if candidates[list(a).index(0)]=="yes" else "Fake" for a in ans]
        return pred_label

    def PrintResult(self, detailed=False, acc=None, confusion_mat=None, ans_list=None, labels=None, logPath=None):
        
        if acc:
            self.acc = acc
        if confusion_mat:
            self.confusion_mat = confusion_mat
        if ans_list:
            self.ans_list = ans_list
        if labels:
            self.labels = labels
        
        if logPath:
            logfile = open(logPath, 'a')
        
        if detailed:
            
            print(f'[TIME]      : {time.ctime()}', file=logfile)
            print(f'[Finetuned] : {self.model.finetuned}', file=logfile)
            print(f'[Img roots] : {self.roots}', file=logfile)
            print(f'[Labels]    : {self.text_labels}', file=logfile)
            print(f'[Question]  : {self.question}\n', file=logfile)
            
            print(f'=== Overall ===', file=logfile)
            print(f'Acc: {self.acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.confusion_mat, logfile=logfile)
            print('\n', file=logfile)
            
            if 0 in self.labels:
                real_ans_list = self.ans_list[self.labels==0]
                real_label = [0] * len(real_ans_list)
                self.real_acc = accuracy_score(real_label, real_ans_list)
                self.real_confusion_mat = confusion_matrix(real_label, real_ans_list, labels=[0,1])
                print(f'=== Real images ===', file=logfile)
                print(f'Acc: {self.real_acc*100:.2f}%', file=logfile)
                self.PrintConfusion(self.real_confusion_mat, logfile=logfile)
                print('\n', file=logfile)
            else:
                print(f'=== No real images ===\n', file=logfile)
            
            
            if 1 in self.labels:
                fake_ans_list = self.ans_list[self.labels==1]
                fake_label = [1] * len(fake_ans_list)
                self.com_acc = accuracy_score(fake_label, fake_ans_list)
                self.com_confusion_mat = confusion_matrix(fake_label, fake_ans_list, labels=[0,1])
                print(f'=== Fake images ===', file=logfile)
                print(f'Acc: {self.com_acc*100:.2f}%', file=logfile)
                self.PrintConfusion(self.com_confusion_mat, logfile=logfile)
                print('\n', file=logfile)
            else:
                print(f'=== No fake images ===\n', file=logfile)
        else:
            print(f'Question: {self.question}\n', file=logfile)
            print(f'Acc: {self.acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.confusion_mat, logfile=logfile)
            print('\n', file=logfile)
        
        logfile.close()
    
    def PrintConfusion(self, mat, logfile):
        padding = ' '
        print(f'        | Pred real | Pred fake |', file=logfile)
        print(f'GT real | {mat[0, 0]:{padding}<{10}}| {mat[0, 1]:{padding}<{11}}|', file=logfile)
        print(f'GT fake | {mat[1, 0]:{padding}<{10}}| {mat[1, 1]:{padding}<{11}}|', file=logfile)
        
    def MultipleAns(self, ans1, ans2):
    
        # Q1: Is this photo common in real world?
        # Q2: Is this photo generated by a model?
        
        final_ans = []
        for ans in zip(ans1, ans2):
            if ans[0] == 0 and ans[1] == 0:
                final_ans.append(0)
            else:
                final_ans.append(1)
        
        acc = accuracy_score(self.labels, final_ans)
        confusion_mat = confusion_matrix(self.labels, final_ans)
        print(f'Accuracy: {acc*100:.2f}%')
        self.PrintConfusion(confusion_mat)
        
        self.ans_list = final_ans
        self.acc = acc
        self.confusion_mat = confusion_mat
        
        return acc, confusion_mat, final_ans
    
    
def print_combine_result(pretrained_ans, finetuned_ans, label, logPath):
    
    logfile = open(logPath, 'a')
    
    def _print_confusion(mat, logfile):
        padding = ' '
        print(f'        | Pred real | Pred fake |', file=logfile)
        print(f'GT real | {mat[0, 0]:{padding}<{10}}| {mat[0, 1]:{padding}<{11}}|', file=logfile)
        print(f'GT fake | {mat[1, 0]:{padding}<{10}}| {mat[1, 1]:{padding}<{11}}|', file=logfile)
    
    comb_ans = np.ceil((pretrained_ans + finetuned_ans)/2).astype(np.int64)
    
    comb_acc = accuracy_score(label, comb_ans)
    comb_confusion_mat = confusion_matrix(label, comb_ans, labels=[0,1])
    
    print(f'=== Overall (Comb) ===', file=logfile)
    print(f'Acc: {comb_acc*100:.2f}%', file=logfile)
    _print_confusion(comb_confusion_mat, logfile=logfile)
    print('\n', file=logfile)
    
    real_ans_list = comb_ans[label==0]
    real_label = [0] * len(real_ans_list)
    real_acc = accuracy_score(real_label, real_ans_list)
    real_confusion_mat = confusion_matrix(real_label, real_ans_list, labels=[0,1])
    print(f'=== Real images (Comb) ===', file=logfile)
    print(f'Acc: {real_acc*100:.2f}%', file=logfile)
    _print_confusion(real_confusion_mat, logfile=logfile)
    print('\n', file=logfile)
    
    
    com_ans_list = comb_ans[label==1]
    com_label = [1] * len(com_ans_list)
    com_acc = accuracy_score(com_label, com_ans_list)
    com_confusion_mat = confusion_matrix(com_label, com_ans_list, labels=[0,1])
    print(f'=== Common fake images (Comb) ===', file=logfile)
    print(f'Acc: {com_acc*100:.2f}%', file=logfile)
    _print_confusion(com_confusion_mat, logfile=logfile)
    print('\n', file=logfile)
    
    return comb_acc, comb_confusion_mat, comb_ans

def main():
    
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/log.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SDXL_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/IF_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_imbalance_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_balance_onlyCommon.txt'
    
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_60k_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_120k_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_90k_postfix_onlyCommon.txt'
    logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_lama_90k_postfix_onlyCommon.txt'
    
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_90k_prefix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_90k_replace_onlyCommon.txt'
    
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_9k_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_900_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_90_postfix_onlyCommon.txt'
    
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_30k_postfix_onlyCommon.txt'
    
    q1 = "Is this photo real?"
    q2 = "Is this photo real [*]?"
    
    file = open(logPath, 'a')
    file.close()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct_textinv", model_type="vicuna7b", is_eval=True, device=device)
    
    print(f'Load model OK!')
    
    instruct = InstructBLIP()
    instruct.LoadModels(model, vis_processors, txt_processors, device)
    
    print(f'Log path: {logPath}')
    print(f'Q1: {q1}')
    print(f'Q2: {q2}')
    
    roots_and_labels = [
        ["/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test/1_fake/SDXLInpaint/SDXLInpainted_binaryMask", "no"]
    ]

    for root, label in roots_and_labels:
        instruct.LoadData(roots=root, labels=label)

        question = q1
        acc, confusion_mat, pretrained_ans_list, labels, label_3class = instruct.QueryImgs_batch(question=question, true_string="yes", logPath=logPath)
        print(f'Question: {question}')
        print(f'     Acc: {acc*100:.2f}%\n')

        question = q2
        acc, confusion_mat, finetuned_ans_list, labels, label_3class = instruct.QueryImgs_batch(question=question, true_string="yes", logPath=logPath)
        print(f'Question: {question}')
        print(f'Acc: {acc*100:.2f}%')

        comb_acc, comb_confusion_mat, comb_ans = print_combine_result(pretrained_ans_list, finetuned_ans_list, labels, logPath=logPath)
        print(f'[Combination]')
        print(f'Acc: {comb_acc*100:.2f}%')
        

if __name__ == '__main__':
    main()
    
# csvfiles = [
#     "/eva_data0/denny/textual_inversion/debug_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_COCO_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_Flickr_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_SD2_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_SDXL_label.csv", 
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_IF_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_DALLE_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_SGXL_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_Control_COCO_label.csv",
#     # "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_lama_label.csv",
#     # "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_SD2IP_label.csv",
#     # "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_lte_label.csv",
#     # "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_SD2SR_label.csv",
#     # "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_deeperforensics_faceOnly_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_AdvAtk_Imagenet_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_Backdoor_Imagenet_label.csv",
#     # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_DataPoison_Imagenet_label.csv",
#     ]