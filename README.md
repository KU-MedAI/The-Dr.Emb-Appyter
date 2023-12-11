# The Dr.Emb Appyter: A Web Platform for Drug Discovery using Embedding Vectors

The Dr.Emb Appyter is a web platform which searches closely located compounds of query compounds in the embedding space for drug discovery. 
We provide the [user guide](https://docs.google.com/uc?export=download&id=1GHDrT_EIGfNbG_TWeYtwV59HYc8q8zrq) for those who experience challenges when navigating the Dr.Emb Appyter.

## Example input files
We provide two example input files, antiviral_drugs.txt for multiple query compounds and custom_library.tsv for custom library

## Embedding vectors of libraries using each embedding method
Download: [Embedding vetors of Libraries](https://docs.google.com/uc?export=download&id=1DfrbfQ8ranFIa4MfV-cKaTypteQDhcYp&confirm=t)

## Start
``` {bash}
git clone https://github.com/KU-MedAI/The-Dr.Emb-Appyter.git
```
``` {bash}
wget https://docs.google.com/uc?export=download&id=1DfrbfQ8ranFIa4MfV-cKaTypteQDhcYp&confirm=t # Embedding vectors of libraries
```
``` {bash}
unzip -d Library.zip
```
``` {bash}
appyter dr_emb.ipynb --extras=toc --extras==toggle-code
```

