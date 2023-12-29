# Dr.Emb Appyter: A Web Platform for Drug Discovery using Embedding Vectors

Dr.Emb Appyter is a web platform which searches closely located compounds of query compounds in the embedding space for drug discovery. 
We provide the <a href="https://drive.google.com/file/d/1S-NRfg8Sr7AkVumpcVbdYZ2HmQOOC_w8/view?usp=sharing" target="_blank" rel="noopener noreferrer">user guide</a> for those who experience challenges when navigating Dr.Emb Appyter.

## Example input files
We provide two example input files, antiviral_drugs.txt for multiple query compounds and custom_library.tsv for custom library. <br>
1. static/antiviral_drugs.txt <br>
Information on four antiviral drugs, including their names and smiles. <br>
2. static/custom_library.tsv <br>
A library containing about 700 compounds, with mechanism of action (MoA) information indicated in 'Description' column. <br>

## List of files to download
1. [Embedding vetors of Default Libraries](https://docs.google.com/uc?export=download&id=1bZpepqycN9LPPLXDqX8georOCYsAj_zD&confirm=t) (10GB)
2. [Pretrained Mol2vec model](https://docs.google.com/uc?export=download&id=1Co4rwFTR0jPVMq_0ee5JP_1v_AufhT3Z&confirm=t) (75MB)

## Start
### Common Preparation
``` {bash}
git clone https://github.com/KU-MedAI/The-Dr.Emb-Appyter.git
cd The-Dr.Emb-Appyter
```
``` {bash}
# File Download
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bZpepqycN9LPPLXDqX8georOCYsAj_zD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bZpepqycN9LPPLXDqX8georOCYsAj_zD" -O Library.zip && rm -rf /tmp/cookies.txt && unzip Library.zip && rm -r Library.zip
wget 'https://docs.google.com/uc?export=download&id=1Co4rwFTR0jPVMq_0ee5JP_1v_AufhT3Z' -O methods/mol2vec_model_300dim.pkl
```

### Local
``` {bash}
pip install --no-cache-dir -r requirements.txt
```
``` {bash}
appyter dr_emb.ipynb --extras=toggle-code --extras=toc --extras=hide-code
```

### Docker
``` {bash}
docker build -t dremb:1.0 .
docker run --privileged --name dremb dremb:1.0
```
``` {bash}
appyter dr_emb_docker.ipynb --extras=toggle-code --extras=toc --extras=hide-code
```