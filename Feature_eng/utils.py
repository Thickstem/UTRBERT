import sys, os
from Bio import SeqIO
import Bio.SeqUtils.CodonUsage
import subprocess
from multiprocessing import Pool
import gzip
from Bio.Seq import Seq


##seq is seq object from bio.python
class Seq2Feature:
    def __init__(self, head_cds_len: int, tail_cds_len: int):
        self.head_cds_length = (
            head_cds_len  ##assumption last portion of sequence is cds
        )
        self.tail_cds_length = tail_cds_len
        # self.three = three  # Whether including three prime utr. Bool

    def codonFreq(self, seq):
        codon_str = seq.translate()
        tot = len(codon_str)
        feature_map = dict()
        for a in codon_str:
            a = "codon_" + a
            if a not in feature_map:
                feature_map[a] = 0
            feature_map[a] += 1.0 / tot
        feature_map["uAUG"] = codon_str.count("M")  # number of start codon
        feature_map["uORF"] = codon_str.count("*")  # number of stop codon
        return feature_map

    def singleNucleotide_composition(self, seq, three=False):
        dna_str = str(seq).upper()
        N_count = dict()  # add one pseudo count
        N_count["C"] = 1
        N_count["G"] = 1
        N_count["A"] = 1
        N_count["T"] = 1
        for a in dna_str:
            if a not in N_count:
                N_count[a] = 0
            N_count[a] += 1
        feature_map = dict()
        feature_map["CGperc"] = float(N_count["C"] + N_count["G"]) / len(dna_str)
        feature_map["CGratio"] = abs(float(N_count["C"]) / N_count["G"] - 1)
        feature_map["ATratio"] = abs(float(N_count["A"]) / N_count["T"] - 1)
        if three == True:
            feature_map["utrlen_m80"] = abs(len(dna_str) - 80 - self.tail_cds_length)
        else:
            feature_map["utrlen_m80"] = abs(len(dna_str) - 80 - self.head_cds_length)

        return feature_map

    def RNAfold_energy(self, sequence, *args):
        rnaf = subprocess.Popen(
            ["RNAfold", "--noPS"] + list(args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Universal Newlines effectively allows string IO.
            universal_newlines=True,
        )
        rnafold_output, folderr = rnaf.communicate(sequence)
        output_lines = rnafold_output.strip().splitlines()
        sequence = output_lines[0]
        structure = output_lines[1].split(None, 1)[0].strip()
        energy = float(output_lines[1].rsplit("(", 1)[1].strip("()").strip())
        return energy

    def RNAfold_energy_Gquad(self, sequence, *args):
        rnaf = subprocess.Popen(
            ["RNAfold", "--noPS"] + list(args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Universal Newlines effectively allows string IO.
            universal_newlines=True,
        )
        rnafold_output, folderr = rnaf.communicate(sequence)
        output_lines = rnafold_output.strip().splitlines()
        sequence = output_lines[0]
        structure = output_lines[1].split(None, 1)[0].strip()
        energy = float(output_lines[1].rsplit("(", 1)[1].strip("()").strip())
        return energy

    def foldenergy_feature(self, seq) -> dict:
        dna_str = str(seq)
        feature_map = dict()
        feature_map["energy_5cap"] = self.RNAfold_energy(dna_str[:100])
        feature_map["energy_whole"] = self.RNAfold_energy(dna_str)
        feature_map["energy_last30bp"] = self.RNAfold_energy(
            dna_str[(len(dna_str) - 30) : len(dna_str)]
        )
        feature_map["energy_Gquad_5utr"] = self.RNAfold_energy_Gquad(
            dna_str[: (len(dna_str) - self.head_cds_length)]
        )
        feature_map["energy_Gquad_5cap"] = self.RNAfold_energy_Gquad(dna_str[:50])
        feature_map["energy_Gquad_last50bp"] = self.RNAfold_energy_Gquad(
            dna_str[(len(dna_str) - 50) : len(dna_str)]
        )
        return feature_map

    def foldenergy_feature_3utr(self, seq) -> dict:
        dna_str = str(seq)
        feature_map = dict()
        feature_map["energy_5cap"] = self.RNAfold_energy(dna_str[:100])
        feature_map["energy_whole"] = self.RNAfold_energy(dna_str)
        feature_map["energy_last30bp"] = self.RNAfold_energy(
            dna_str[(len(dna_str) - 30) : len(dna_str)]
        )
        feature_map["energy_Gquad_3utr"] = self.RNAfold_energy_Gquad(
            dna_str[self.tail_cds_length :]
        )  # only for 3utr seq
        feature_map["energy_Gquad_5cap"] = self.RNAfold_energy_Gquad(dna_str[:50])
        feature_map["energy_Gquad_last50bp"] = self.RNAfold_energy_Gquad(
            dna_str[(len(dna_str) - 50) : len(dna_str)]
        )
        return feature_map

    def Kmer_feature(self, seq, klen=6) -> dict:
        feature_map = dict()
        seq = seq.upper()
        for k in range(1, klen + 1):
            for st in range(len(seq) - klen):
                kmer = seq[st : (st + k)]
                featname = "kmer_" + str(kmer)
                if featname not in feature_map:
                    feature_map[featname] = 0
                feature_map[featname] += 1.0 / (len(seq) - k + 1)
        return feature_map

    def oss(self, cmd):
        print(cmd)
        os.system(cmd)

    def run(self, seq) -> dict:
        ##codon
        ret = list(self.codonFreq(seq).items())
        ##DNA CG composition
        ret += list(self.singleNucleotide_composition(seq).items())
        ## Kmer feature
        ret += list(self.Kmer_feature(seq).items())

        return ret

    def run_with_energy(self, seq, three):
        print(f"Features with Energy")
        ##codon
        ret = list(self.codonFreq(seq).items())
        ##DNA CG composition
        ret += list(self.singleNucleotide_composition(seq, three).items())
        ##RNA folding
        if three == True:
            ret += list(self.foldenergy_feature_3utr(seq).items())
        else:
            ret += list(self.foldenergy_feature(seq).items())
        ##Kmer features
        ret += list(self.Kmer_feature(seq).items())
        return ret

    def run_with_energy_wrapper(self, args):
        return self.run_with_energy(*args)
