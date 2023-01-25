import requests 
import sys
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np


def _pick_seq(fasta):    
    seq_list = fasta.text.split("\n")[1:-1]  
    seq = "".join(seq_list)
    return seq


def load_transcript_list(df,col="Transcript"):
    transes = df[col].values
    uni_trans_id = np.unique(transes)

    return uni_trans_id


def fetch_fasta(trans_id,success=True):
    server   = "https://rest.ensembl.org"
    ext_cds  = f"/sequence/id/{trans_id}?type=cds"
    ext_cdna = f"/sequence/id/{trans_id}?type=cdna"

    r_cds  = requests.get(server+ext_cds, headers={ "Content-Type" : "text/x-fasta"})
    r_cdna = requests.get(server+ext_cdna, headers={ "Content-Type" : "text/x-fasta"})

    
    if (not r_cds.ok) or not(r_cdna.ok):
        success=False
    
    cds = _pick_seq(r_cds)
    cdna = _pick_seq(r_cdna)
    
    return cds,cdna,success

def extract_utrs(cds,cdna):
    start_idx = cdna.find(cds)
    fiveprime = cdna[:start_idx]
    threeprime = cdna[start_idx+len(cds):]

    return fiveprime,threeprime,start_idx

def divided_list(section,trans_list):
    DIV=4
    sep = int(len(trans_list)/DIV)

    if section==0:
        return trans_list
    elif section==4:
        return trans_list[sep*(section-1):]
    else:
        return trans_list[sep*(section-1):sep*section]



def main(section,save_name):
    df = pd.read_csv("data/homo_sapience_utr.csv")
    
    seqs_dict= {
        "trans_id":[],
        "gene":[],
        "fiveprime":[],
        "threeprime":[],
        "cds":[]
        }
    no_match_trans=[]
    not_found_trans=[]
    
    all_trans_list = load_transcript_list(df)
    trans_list = divided_list(section,all_trans_list)

    for trans_id in tqdm(trans_list):
        cds,cdna,success = fetch_fasta(trans_id)
        if success==False:
            not_found_trans.append(trans_id)
            continue

        fiveprime,threeprime,start_idx = extract_utrs(cds,cdna)

        if start_idx==-1:
            no_match_trans.append(trans_id)
            continue
        gene = df[df["Transcript"]==trans_id]["Gene"].values[0]

        seqs_dict["trans_id"].append(trans_id)
        seqs_dict["gene"].append(gene)
        seqs_dict["fiveprime"].append(fiveprime)
        seqs_dict["threeprime"].append(threeprime)
        seqs_dict["cds"].append(cds)

        
    built_df = pd.DataFrame(seqs_dict)
    built_df.to_csv(save_name+str(section)+".csv")
    np.save("no_match_list"+str(section),np.array(no_match_trans))
    np.save("not_found_list"+str(section),np.array(not_found_trans))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section",required=True,type=int,help="specify 1~4")
    parser.add_argument("--save_name",default="ensembl_trans_seq",type=str)
    args = parser.parse_args()

    main(section=args.section,save_name=args.save_name)
