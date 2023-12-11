import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import deepchem as dc
import os
import sys
import torch
import methods.moable.model
import pickle
import faiss
from tqdm import tqdm
from itertools import combinations
from IPython.display import HTML, display, Markdown, IFrame, FileLink, Image, HTML
from scipy.spatial import distance

sys_path = "/home/songhyeon/Research/Embedding_appyter"
sys.path.append(sys_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def library_npl(input_db):
    if input_db == 'kcb':
        library_df = pd.read_csv('/home/songhyeon/Research/Embedding_appyter/Library/kcb_final.tsv', sep='\t', index_col=0)
    elif input_db == 'zinc':
        library_df = pd.read_csv('/home/songhyeon/Research/Embedding_appyter/Library/ZINC_named+waited.tsv', sep='\t', index_col=0)
    elif input_db == 'mce':
        library_df = pd.read_csv('/home/songhyeon/Research/Embedding_appyter/Library/MCE_library.tsv', sep='\t', index_col=0)
    elif input_db == 'selleck':
        library_df = pd.read_csv('/home/songhyeon/Research/Embedding_appyter/Library/Selleckchem_library.tsv', sep='\t', index_col=0)
    
    library_df.rename(columns={'Name':'drug2_name','SMILES':'drug2_smiles'}, inplace=True)
    library_df1 = library_df['drug2_name']
    library_L = []
    
    for i in range(len(library_df1)):
        library_L.append(int(i))
        
    library_npl = np.array(library_L)
    return library_df, library_npl

def custom_npl(library_df):
    library_df.columns.values[0] = 'drug_name'
    library_df.columns.values[1] = 'drug_smiles'
    library_df1 = library_df['drug_name']
    library_L = []
    
    for i in range(len(library_df1)):
        library_L.append(int(i))
        
    library_npl = np.array(library_L)
    return library_npl

def embed_vecotr_lib(input_db, embed_method):
    if embed_method == 'ReSimNet':
        embedding_vectors = pd.read_pickle(f'/home/songhyeon/Research/Embedding_appyter/Library/{embed_method}_{input_db}/{embed_method}_{input_db}_7.pkl')
    else:
        embedding_vectors = pd.read_pickle(f'/home/songhyeon/Research/Embedding_appyter/Library/{embed_method}_{input_db}/{embed_method}_{input_db}.pkl')
    return embedding_vectors


def smiles2fp(smilesstr):
    mol = Chem.MolFromSmiles(smilesstr)
    fp_obj = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048,
                                                   useChirality=True)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_obj, arr)

    return arr


def drug_embeddings(drug_dict):
    result_dict = dict()
    global model
    model = methods.moable.model.DrugEncoder()
    model.load_state_dict(torch.load('/home/songhyeon/Research/Embedding_appyter/methods/moable/models/moable.pth'))
    model.to(device)
    model.eval()
    
    for key in drug_dict:
        smiles = drug_dict[key]
        ecfp = torch.from_numpy(smiles2fp(smiles)).to(device)
        ecfp = ecfp.reshape(1,-1)
        embedding = model(ecfp.float()).cpu().detach().numpy().flatten()
        magnitude = np.linalg.norm(embedding)
        embedding = embedding / magnitude
        result_dict[key] = embedding

    return result_dict


def pretrained_MACAW(input_db):
    with open(f'/home/songhyeon/Research/Embedding_appyter/methods/MACAW/MACAW_{input_db}_pre.pkl', 'rb') as model_file:
        mcw = pickle.load(model_file)
        
    return mcw


def custom_embedding(drug_dict, embed_method, output_path, output_file):
    embed_dict = {} 
    if embed_method == 'MoAble':
        embed_dict = drug_embeddings(drug_dict)
        with open(output_path + output_file,'wb') as f:
            pickle.dump(embed_dict, f)
            
    # Sequence
    elif embed_method == 'Mol2vec':
        featurizer = dc.feat.Mol2VecFingerprint('/home/songhyeon/Research/Embedding_appyter/methods/mol2vec/mol2vec_model_300dim.pkl')
        for name, smiles in drug_dict.items():
            with open(output_path + output_file, 'wb') as f:
                embed_dict[name] = featurizer.featurize(smiles)
                pickle.dump(embed_dict, f)
        
    # Fingerprint
    elif embed_method == 'ECFP':
        for name, smiles in drug_dict.items():
            with open(output_path + output_file, 'wb') as f:
                embed_dict[name] = smiles2fp(smiles)
                pickle.dump(embed_dict, f)


    elif embed_method == 'MACCSKeys':
        featurizer = dc.feat.MACCSKeysFingerprint()
        for name, smiles in drug_dict.items():
            with open(output_path + output_file, 'wb') as f:
                embed_dict[name] = featurizer.featurize(smiles)
                pickle.dump(embed_dict, f)

    return embed_dict


def jaccard_finder(input_db, embed_dict, embed_method, queries, topk_candidates):
    topk_similarities = {}
    if input_db == 'custom' and embed_method in ['ECFP', 'MACCSKeys']:
        library_ecfp = embed_dict
    elif input_db != 'custom' and embed_method in ['ECFP', 'MACCSKeys']:
        library_ecfp = pd.read_pickle(f"/home/songhyeon/Research/Embedding_appyter/Library/{embed_method}_{input_db}/{embed_method}_{input_db}.pkl")
    else:
        print('Jaccard similarity is ECFP, MACCSKeys Only')

    if embed_method == 'MACCSKeys':
        for drug, values in library_ecfp.items():
            library_ecfp[drug] = values.flatten()
        
    for query_name, query_ecfp in queries.items():
        similarities = {}
        for key, ecfp in library_ecfp.items():
            jaccard_similarity = 1 - distance.jaccard(query_ecfp, ecfp)
            similarities[key] = jaccard_similarity

        topk = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:topk_candidates])
        topk_similarities[query_name] = topk

    return topk_similarities 


def create_result_dataframe(results):
    data = []
    for query_name, topk in results.items():
        for key, similarity in topk.items():
            data.append([query_name, key, similarity])
    
    columns = ["Query", "Library Compound", "Jaccard Similarity"]
    result_df = pd.DataFrame(data, columns=columns)
    
    return result_df


def jaccard_dataframes(input_db, custom_embed_dict, embed_method, embed_dict, topk_candidate):
    results = jaccard_finder(input_db, custom_embed_dict, embed_method, embed_dict, topk_candidate)
    dataframes = {}
    for query_name, topk in results.items():
        data = []
        for key, similarity in topk.items():
            data.append([key, similarity])
        columns = ["drug_name", "Jaccard Similarity"]
        dataframes[query_name] = pd.DataFrame(data, columns=columns)
    return dataframes


def resimnet_finder(input_db, npl, output_embed_filename, topk_candidate, name, resimnet_model):
    embedding_vectors_directory = f"/home/songhyeon/Research/Embedding_appyter/Library/ReSimNet_{input_db}/"
    embedding_vectors_filenames = os.listdir(embedding_vectors_directory)
    result_df_list = list()

    try:
        faiss_index
        del faiss_index
    except:
        pass

    for i in range(10):        
        if resimnet_model != "All" and str(i) not in resimnet_model:
            continue
            
        for embedding_file_index, embedding_vectors_filename in enumerate(tqdm(embedding_vectors_filenames)):            
            if f"_{i}" in embedding_vectors_filename: 
                # load ZINC15 embedding vectors
                with open(embedding_vectors_directory+embedding_vectors_filename, "rb") as f:
                    embedding_vectors_dict = pickle.load(f)
                    
                # add embedding vectors to faiss
                embedding_vectors_df = pd.DataFrame.from_dict(embedding_vectors_dict).T
                embedding_vectors = np.ascontiguousarray(np.float32(embedding_vectors_df.values))
                    
                faiss.normalize_L2(embedding_vectors)
            
                try:
                    faiss_index
                except:
                    faiss_index = faiss.IndexFlatIP(embedding_vectors.shape[1])
                    faiss_index = faiss.IndexIDMap2(faiss_index)
                    
                faiss_index.add_with_ids(embedding_vectors, npl)

        print(f"{name}: Searching from {faiss_index.ntotal} candidates...")
            # load query embedding vector
        with open(output_embed_filename, "rb") as f:
            query_embedding_vectors = pickle.load(f)

        query_embedding_vectors = np.ascontiguousarray(np.float32(pd.DataFrame.from_dict(query_embedding_vectors).values)).T
        faiss.normalize_L2(query_embedding_vectors)

        Similarity, Index = faiss_index.search(query_embedding_vectors, topk_candidate)
        
        embedding_df = embedding_vectors_df.reset_index()
        embedding_df = embedding_df['index']
        emb_list = list(embedding_df[Index[0]])
        result_tmp_df = pd.DataFrame(index=[str(x) for x in emb_list])
        result_df_list.append(result_tmp_df)
                
    return Similarity, Index, result_df_list
    

def custom_finder(custom_dict, embed_method, npl, sim_method, output_embed_filename, topk_candidate, name):
    result_df_list = list()

    try:
        faiss_index
        del faiss_index
    except:
        pass
    
    for embedding_file_index, embedding_vectors_dict in enumerate(tqdm([custom_dict])):
        if embed_method in ['Mol2vec','MACCSKeys']:
            data_array = np.array([v[0] for v in custom_dict.values()])
            embedding_vectors_df = pd.DataFrame(data_array, columns=[i for i in range(data_array.shape[1])], index=custom_dict.keys())
        else:
            embedding_vectors_df = pd.DataFrame.from_dict(embedding_vectors_dict).T
            
        embedding_vectors = np.ascontiguousarray(np.float32(embedding_vectors_df.values))
                                    
        faiss.normalize_L2(embedding_vectors)
        
        try:
            faiss_index
        except:
            faiss_index = None
            
        if sim_method == 'Cosine':
            faiss_index = faiss.IndexFlatIP(embedding_vectors.shape[1])
            faiss_index = faiss.IndexIDMap2(faiss_index)
            faiss_index.add_with_ids(embedding_vectors, npl)

        elif sim_method == 'Euclidean':
            faiss_index = faiss.IndexFlatL2(embedding_vectors.shape[1])
            faiss_index.add(embedding_vectors)

    print(f"{name}: Searching from {faiss_index.ntotal} candidates...")
    
    with open(output_embed_filename, "rb") as f:
        query_embedding_vectors = pickle.load(f)

    query_embedding_vectors = np.float32([np.squeeze(arr) for arr in query_embedding_vectors.values()])
    faiss.normalize_L2(query_embedding_vectors)

    Similarity, Index = faiss_index.search(query_embedding_vectors, topk_candidate)
    
    embedding_df = embedding_vectors_df.reset_index()
    embedding_df = embedding_df['index']
    emb_list = list(embedding_df[Index[0]])
    result_tmp_df = pd.DataFrame(index=[str(x) for x in emb_list])
    result_df_list.append(result_tmp_df)

    return Similarity, Index, result_df_list

            
def finder(input_db, embed_method, npl, sim_method, output_embed_filename, topk_candidate, name):
    embedding_vectors_directory = f"/home/songhyeon/Research/Embedding_appyter/Library/{embed_method}_{input_db}/"
    embedding_vectors_filenames = os.listdir(embedding_vectors_directory)
    result_df_list = list()

    try:
        faiss_index
        del faiss_index
    except:
        pass
    
    for embedding_file_index, embedding_vectors_filename in enumerate(tqdm(embedding_vectors_filenames)):            
        with open(embedding_vectors_directory+embedding_vectors_filename, "rb") as f:
            embedding_vectors_dict = pickle.load(f)
            
            if embed_method == 'MACCSKeys' or 'Mol2vec':
                for key, value in embedding_vectors_dict.items():
                    embedding_vectors_dict[key] = value[0]

        embedding_vectors_df = pd.DataFrame.from_dict(embedding_vectors_dict).T
        embedding_vectors = np.ascontiguousarray(np.float32(embedding_vectors_df.values))
                
        faiss.normalize_L2(embedding_vectors)
        
        try:
            faiss_index
        except:
            faiss_index = None
            
        if sim_method == 'Cosine':
            faiss_index = faiss.IndexFlatIP(embedding_vectors.shape[1])
            faiss_index = faiss.IndexIDMap2(faiss_index)
            faiss_index.add_with_ids(embedding_vectors, npl)

        elif sim_method == 'Euclidean':
            faiss_index = faiss.IndexFlatL2(embedding_vectors.shape[1])
            faiss_index.add(embedding_vectors)

    print(f"{name}: Searching from {faiss_index.ntotal} candidates...")
    
    with open(output_embed_filename, "rb") as f:
        query_embedding_vectors = pickle.load(f)

    query_embedding_vectors = np.float32([np.squeeze(arr) for arr in query_embedding_vectors.values()])
    faiss.normalize_L2(query_embedding_vectors)

    Similarity, Index = faiss_index.search(query_embedding_vectors, topk_candidate)
    
    embedding_df = embedding_vectors_df.reset_index()
    embedding_df = embedding_df['index']
    emb_list = list(embedding_df[Index[0]])
    result_tmp_df = pd.DataFrame(index=[str(x) for x in emb_list])
    result_df_list.append(result_tmp_df)
                
    return Similarity, Index, result_df_list


def MA_finder(input_db, embed_method, npl, sim_method, output_embed_filename, topk_candidate, name):
    embedding_vectors_directory = f"/home/songhyeon/Research/Embedding_appyter/Library/{embed_method}_{input_db}/"
    embedding_vectors_filenames = os.listdir(embedding_vectors_directory)
    result_df_list = list()

    try:
        faiss_index
        del faiss_index
    except:
        pass
    
    for embedding_file_index, embedding_vectors_filename in enumerate(tqdm(embedding_vectors_filenames)):            
        with open(embedding_vectors_directory+embedding_vectors_filename, "rb") as f:
            embedding_vectors_dict = pickle.load(f)

        embedding_vectors_df = pd.DataFrame.from_dict(embedding_vectors_dict).T
        embedding_vectors = np.ascontiguousarray(np.float32(embedding_vectors_df.values))
                
        faiss.normalize_L2(embedding_vectors)
        
        try:
            faiss_index
        except:
            faiss_index = None
            
        if sim_method == 'Cosine':
            faiss_index = faiss.IndexFlatIP(embedding_vectors.shape[1])
            faiss_index = faiss.IndexIDMap2(faiss_index)
            faiss_index.add_with_ids(embedding_vectors, npl)

        elif sim_method == 'Euclidean':
            faiss_index = faiss.IndexFlatL2(embedding_vectors.shape[1])
            faiss_index.add(embedding_vectors)

    print(f"{name}: Searching from {faiss_index.ntotal} candidates...")
    
    with open(output_embed_filename, "rb") as f:
        query_embedding_vectors = pickle.load(f)

    query_embedding_vectors = np.float32([np.squeeze(arr) for arr in query_embedding_vectors.values()])
    faiss.normalize_L2(query_embedding_vectors)

    Similarity, Index = faiss_index.search(query_embedding_vectors, topk_candidate)
    
    embedding_df = embedding_vectors_df.reset_index()
    embedding_df = embedding_df['index']
    emb_list = list(embedding_df[Index[0]])
    result_tmp_df = pd.DataFrame(index=[str(x) for x in emb_list])
    result_df_list.append(result_tmp_df)
                
    return Similarity, Index, result_df_list   


def make_clickable(smiles, site="pubchem"):
    if site == "pubchem":
        url = f"https://pubchem.ncbi.nlm.nih.gov/#query={smiles}&input_type=smiles"
    else:
        url = f"https://zinc15.docking.org/substances/{smiles}/" #smiles is actually ZINC15 ID
    return '<a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(url,smiles)


def create_download_link(output_embed_filename):  
    html = "<a href=\"./{}\" target='_blank'>{}</a>".format(output_embed_filename, f"Download ReSimNet {name} embedding vectors")
    return HTML(html)


def calculate_cosine_similarity(vectors):
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine_similarity = np.dot(normalized_vectors, normalized_vectors.T)
    return cosine_similarity


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def jaccard_similarity(v1, v2):
    intersection = np.logical_and(v1, v2)
    union = np.logical_or(v1, v2)
    return intersection.sum() / union.sum()


def find_duplicate_names(dictionary):
    name_to_keys = {}

    for key, value in dictionary.items():
        for name in value:
            if name in name_to_keys:
                name_to_keys[name].append(key)
            else:
                name_to_keys[name] = [key]

    result_dict = {name: keys for name, keys in name_to_keys.items() if len(keys) > 1}

    return result_dict