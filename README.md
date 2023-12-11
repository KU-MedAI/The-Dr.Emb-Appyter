# Dr.Emb Appyter: A Web Platform for Drug Discovery using Embedding Vectors

Dr.Emb Appyter is a web platform which searches closely located compounds of query compounds in the embedding space for drug discovery. 
We provide the [user guide](https://docs.google.com/uc?export=download&id=1S-NRfg8Sr7AkVumpcVbdYZ2HmQOOC_w8&confirm=t) for those who experience challenges when navigating Dr.Emb Appyter.

## Example input files
We provide two example input files, antiviral_drugs.txt for multiple query compounds and custom_library.tsv for custom library. <br>
1. antiviral_drugs.txt <br>
Information on four antiviral drugs, including their names and smiles. <br>
2. custom_library.tsv <br>
A library containing about 700 compounds, with mechanism of action (MoA) information indicated in 'Description' column. <br>

## Embedding vectors of libraries using each embedding method
Download (10 GB): [Embedding vetors of Libraries](https://docs.google.com/uc?export=download&id=1DfrbfQ8ranFIa4MfV-cKaTypteQDhcYp&confirm=t)

## Start
``` {bash}
git clone https://github.com/KU-MedAI/The-Dr.Emb-Appyter.git
```
``` {bash}
# Embedding vectors of libraries (10 GB)
wget https://docs.google.com/uc?export=download&id=1DfrbfQ8ranFIa4MfV-cKaTypteQDhcYp&confirm=t
```
``` {bash}
unzip -d Library.zip
```
``` {bash}
appyter dr_emb.ipynb --extras=toc --extras==toggle-code
```

