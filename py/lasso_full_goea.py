#!/usr/bin/env python
# coding: utf-8

# In[4]:


# use kernel py3-6
#import goatools
from goatools import obo_parser
import re
import json
import numpy as np
import pandas as pd
from goatools.go_enrichment import GOEnrichmentStudy
import matplotlib.pyplot as plt
import os


# In[49]:


def matchgff2(feature, gff_file='/home/t44p/PW_rawdata/Transciptome_GenomeAnnotation/Xele_annotated2_gff_export2.gff', obo_path="/home/t44p/PW_rawdata/go_obo/go.obo", namespace=None, depth_threshold=0, goea=False):
    """
    Searches a GFF (General Feature Format) file for specific features and retrieves associated Gene Ontology (GO) terms,
    with optional filtering by GO term namespace and depth.

    Parameters:
    ----------
    feature : list or iterable
        An iterable of strings representing the features to search for in the GFF file.

    gff_file : str, optional
        The file path to the GFF file. Defaults to a predefined path.

    obo_path : str, optional
        The file path to the Gene Ontology .obo file. Defaults to a predefined path.

    namespace : list of str or None, optional
        List of GO term namespaces to filter term count results. Valid options are 'biological_process', 
        'molecular_function', 'cellular_component', or None. Defaults to None (no filtering).

    depth_threshold : int, optional
        Minimum depth of GO terms to include in the results. Defaults to 0 (no filtering).

    goea : bool, optional
        Whether to perform Gene Ontology Enrichment Analysis. Defaults to False.

    Returns:
    -------
    tuple
        A tuple containing:
        1. A dictionary with features as keys and lists of lines from the GFF file as values.
        2. A dictionary mapping features to their associated GO terms.
        3. A dictionary of GO term counts.
        4. GOEA results (if goea is True), filtered by the specified depth threshold for all namespaces

    Notes:
    -----
    - The function uses regular expressions for precise matching of features.
    - It extracts GO IDs from matched lines and retrieves their corresponding names and namespaces.
    - It is possible for the rawcounts to filter for namespaces 
    - If `goea` is True, the function performs GO Enrichment Analysis and returns the results filtered by the specified depth threshold but not for namespaces. 
    - The Enrichment analysis will be performed across all namespaces, filtering parameter "namespace" will have no effect on the enrichment analysis
    
    """

    valid_namespaces = {'biological_process', 'molecular_function', 'cellular_component', None}
    # Check if namespace is a list containing only valid elements
    if isinstance(namespace, list) and not all(ns in valid_namespaces for ns in namespace):
        raise ValueError("Invalid namespace provided. Valid options are 'biological_process', "
                         "'molecular_function', 'cellular_component', or a list containing any of these. "
                         "You can also use None for no filtering.")

    with open(gff_file, 'r') as file:
        go_ontology = obo_parser.GODag(obo_path)
        
        lines_where_feat_found = {}
        go_ids = {}
        background_genes = []
        go_term_count = {}

         # Find the depth of each GO term
        go_depths = {go_id: go_term.depth for go_id, go_term in go_ontology.items()}

        # construct background genes
        if goea:
            for line in file:
                if not line.lstrip().startswith('#'):
                    background_genes.append(line.split('\t')[0])

        for feat in feature:
            file.seek(0)  # reset file pointer to the beginning for each feature
            lines_where_feat_found[feat] = []
            go_ids[feat] = {}
            pattern = re.compile(re.escape(feat) + r'\t')  # exact match followed by a tab
            for line in file:
                if pattern.search(line):
                    lines_where_feat_found[feat].append(line.strip())  # Store the line (as a string) if feature is found
                    # Extract GO id
                    match = re.search(r"Ontology_id=([GO:\d,]+)", line.strip())
                    if match:
                        ids = match.group(1).split(',')
                        # Map Terms to Ids and Count Occurrences
                        for id in ids:
                            term = go_ontology.get(id)
                            if term is not None:
                                go_ids[feat][id] = {'name': term.name, 'namespace': term.namespace}

                                if namespace is None or term.namespace in namespace and go_ontology[id].depth >= depth_threshold:
                                    # Count Occurrences
                                    if id in go_term_count:
                                        go_term_count[id] = (term.name, go_term_count[id][1] + 1, term.namespace)
                                    else:
                                        go_term_count[id] = (term.name, 1, term.namespace)
                            else:
                                go_ids[feat][id] = {'name': None, 'namespace': None}
                                if id not in go_term_count:
                                    go_term_count[id] = (None, 1)
        if goea:
            print("GO Enrichment Analysis >>")
            goea_obj = GOEnrichmentStudy(
                background_genes,
                go_ids,  # This needs to be a dict mapping gene IDs to a set of GO IDs
                go_ontology,
                propagate_counts=False,
                alpha=0.05,  # significance level for the statistical test
                methods=['fdr_bh']  # correction method for multiple testing
            )
            goea_result = goea_obj.run_study(go_ids.keys())

            # filter based on depth
            filtered_goea_results = [res for res in goea_result if res.goterm.depth >= depth_threshold]
            return lines_where_feat_found, go_ids, go_term_count, filtered_goea_results


        return lines_where_feat_found, go_ids, go_term_count
    

def tabulate(term_count_dict, sort=True):
    """
    Prints a tabulated view of the term counts from a dictionary, such as the one returned by matchgff2.

    Parameters:
    ----------
    term_count_dict : dict
        A dictionary where keys are Gene Ontology (GO) IDs, and values are tuples containing the GO term name, 
        the count of occurrences, and optionally the namespace. 
        The structure is typically: {GO_ID: (GO_Term, Count, Namespace)}.

    sort : bool, optional
        Whether to sort the output based on the count of occurrences of each GO term. Defaults to True.

    Description:
    ------------
    This function iterates through the term_count_dict and prints each GO term's count, ID, name, and namespace 
    in a tabular format. It can optionally sort these terms based on their count in descending order.

    Note:
    -----
    The function is primarily a utility for visualizing the output of the matchgff2 function. It does not return any value 
    but prints the information directly to the console.

    Example Usage:
    --------------
    go_term_count = {'GO:0000001': ('term1', 5, 'biological_process'), 'GO:0000002': ('term2', 3, 'cellular_component')}
    tabulate(go_term_count, sort=True)
    """
    print(f"count\tGO ID\tGO Term\tnamespace")

    # Conditionally sort the dictionary if required
    items = sorted(term_count_dict.items(), key=lambda x: x[1][1], reverse=True) if sort else term_count_dict.items()

    for goid, values in items:
        count = values[1]
        term = values[0] if values[0] is not None else "N/A"
        namespace = values[2] if len(values) > 2 else "N/A"

        print(f"{count}\t{goid}\t{term}\t{namespace}")
    #return 


def parse_obo_file(obo_path):
    go_terms = {}
    with open(obo_path, 'r') as file:
        for line in file:
            if line.startswith('id: '):
                current_id = line.strip().split(' ')[1]
            elif line.startswith('namespace: '):
                current_namespace = line.strip().split(' ')[1]
                go_terms[current_id] = current_namespace
    return go_terms

def goea_results_to_file(goea_results, obo_path, path='./lasso_models/10xKfold_lasso_output/goea/', to_csv=True, to_excel=False):
    """
    Processes and saves Gene Ontology Enrichment Analysis (GOEA) results in CSV and/or Excel format.

    Parameters:
    - goea_results: dict
      - A dictionary with prefixes for file names as keys and lists of GOEA result objects as values.
    - obo_path: str
      - Path to the .obo file containing GO terms.
    - path: str, optional
      - Directory path for saving output files. Defaults to './lasso_models/10xKfold_lasso_output/goea/'.
    - to_csv: bool, optional
      - If True, saves results in CSV format. Defaults to True.
    - to_excel: bool, optional
      - If True, saves results in Excel format. Defaults to False.

    Functionality:
    - Parses the GO terms from the .obo file.
    - Creates a DataFrame from the GOEA results for each namespace ('BP', 'MF', 'CC').
    - Constructs an enrichment matrix with GO IDs as rows and goea_results keys as columns.
    - Saves individual DataFrames and the enrichment matrix in the specified format (CSV/Excel).

    Note:
    - 'openpyxl' library is required for saving to Excel files.
    - The function handles namespaces and enrichment status ('e' for enriched) for GOEA results.
    - Ensures all GO IDs are represented in the enrichment matrix, populating missing values with zeros.
    - Generates file names based on the provided prefixes and namespaces.
    - Prints file paths of saved files for user reference.
    """
    # Load GO terms from the obo file
    go_terms = parse_obo_file(obo_path)

    # Mapping for suffixes
    suffix_mapping = {'BP': 'biop', 'MF': 'molf', 'CC': 'cellc'}

    for ns in ['BP', 'MF', 'CC']:
        
        # mapping for enrichment matrix population
        namespace_mapping = {
        'BP': 'biological_process',
        'MF': 'molecular_function',
        'CC': 'cellular_component'
        }
        # populating enrichment matrix
        enrichment_matrix = {key: {go_id: 0 for go_id in go_terms if go_terms[go_id] == namespace_mapping[ns]} for key in goea_results}

        for file_prefix, res in goea_results.items():
            suffix = suffix_mapping[ns]

            # Filtering relevant records
            relevant_records = [r for r in res if r.NS == ns]

            # Creating a DataFrame
            data = {
                "GO_ID": [r.goterm.id for r in relevant_records],
                "level": [r.goterm.level for r in relevant_records],
                "depth": [r.goterm.depth for r in relevant_records],
                "GO_Term": [r.goterm.name for r in relevant_records],
                "study_count": [r.study_count for r in relevant_records],
                "study_n": [r.study_n for r in relevant_records],
                "pop_count": [r.pop_count for r in relevant_records],
                "pop_n": [r.pop_n for r in relevant_records],
                "enrichment": [r.enrichment for r in relevant_records],
                "adj_p_fdr_bh": [r.p_fdr_bh for r in relevant_records]
            }

            for record in relevant_records:
                if record.enrichment == 'e':
                    enrichment_matrix[file_prefix][record.goterm.id] = 1

            df = pd.DataFrame(data)
            # Saving to CSV
            if to_csv:
                file_name = f"{path}{file_prefix}_{suffix}.csv"
                df.to_csv(file_name, index=False)
                print(f"Saved to {file_name}")
            if to_excel:
                path_suffix = f"{path}goea_{suffix}.xlsx"
                if os.path.exists(path_suffix):
                    x_mode = 'a'
                else:
                    x_mode = 'w'
                    
                with pd.ExcelWriter(path_suffix, mode=x_mode, engine='openpyxl') as writer:
                    sheet = file_prefix
                    df.to_excel(writer, sheet_name=sheet, index=False)

        # Ensure that all GO IDs are represented and convert the matrix to DataFrame
        enrichment_df = pd.DataFrame(enrichment_matrix)
        
        # The DataFrame should now be fully populated with 0s and 1s. Converting to integer type should not be an issue.
        enrichment_df = enrichment_df.fillna(0).astype(int)

        # Save the DataFrame
        enrichment_file_name = f"{path}enrichment_matrix_{suffix}.csv"
        enrichment_df.to_csv(enrichment_file_name)
        print(f"Enrichment matrix saved to {enrichment_file_name}")


# In[ ]:


dict_rem = parse_obo_file()


# # GOEA for all files 

# In[5]:


path_gc = "/home/t44p/PW_rawdata/results/full_lasso/gcms/"
path_lc ="/home/t44p/PW_rawdata/results/full_lasso/lcms/"


lasso_goea = {}


# Iterate over each file in the directory
for p in [path_gc, path_lc]:
    tmp = 0

    for file in os.listdir(p):
        if file.endswith(".json") and not(file.startswith('gcms_dict_nXcv') or file.startswith('lcms_dict_nXcv')):
            file_path = os.path.join(p, file)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            
            # Extract mean scores and fold scores
            file_name = os.path.splitext(file)[0]
            file_name_stripped = file_name.rsplit('_nXcv', 1)[0]
            print(file_name_stripped)


            # perform GOEA
            data_matched, data_goids, data_term_count, data_goea_all = matchgff2(data['selected_features'], namespace=['molecular_function'], depth_threshold=2, goea=True)
            lasso_goea[file_name_stripped] = data_goea_all
            

            tmp += 1
            if tmp == 2:
              break




# In[6]:


goea_results_to_file(lasso_goea, path='/home/t44p/PW_rawdata/results/full_lasso/goea/', obo_path="/home/t44p/PW_rawdata/go_obo/go.obo", to_excel=True)


# # GCMS GOEA by Grouping

# In[8]:




# Define the categories
categories = {
    "Sugars_Carbohydrates": [
        "Cellobiose_361_204_rt14_40", 
        "raffinose_437_451_rt16_91", 
        "glucose_160_rt9_81", 
        "inositol_myo_305_265_rt10_71", 
        "sucrose_437_361_rt13_77", 
        "hexose_307_217_rt9_58", 
        "glucose_1_phosphate_217_rt9_16", 
        "trehalose_alpha_alpha_191_169_", 
        "fructose_307_217_rt9_48", 
        "glucose_160_319_rt9_68", 
        "galactinol_204_191_rt15_38", 
        "Ribulose_5_phosphate_357_299_1", 
        "glucose_6_phosphate_160_387_rt", 
        "Cellobiose_361_204_or_maltose_", 
        "_6_phospho_gluconate"
    ],
    "Amino_Acids_and_Derivatives": [
        "tyrosine_218_280_rt10_78",
        "serine_204_218_rt5_84", 
        "isoleucin_158_233_rt_5_21", 
        "asparagine_116_188_rt9_00", 
        "ornithine_142_174_rt9_34", 
        "alanine_3_cyano_141_100_rt6_78",
        "arginine_157_256_rt9_92", 
        "threonine_219_291_rt6_01", 
        "tryptophan_202_291_rt12_94", 
        "alanine_116_218_rt3_38", 
        "leucine_158_232_rt4_97", 
        "valine_144_218_rt4_42", 
        "asparagine_188_216_rt7_84", 
        "methionine_176_128_rt7_76", 
        "asparagine_188_216_rt7_45", 
        "lysine_156_174_rt10_07", 
        "glycine_102_147_rt3_70", 
        "aspartic_acid_232_218_rt7_48", 
        "histidine_154_254_rt11_10", 
        "glutamine_156_245_rt9_80", 
        "glycine_174_248_rt_5_31", 
        "homoserine_218_128_rt6_64", 
        "proline_142_130_rt_5_53", 
        "alanine_beta_248_290_rt6_44", 
        "ornithine_142_348_rt8_03"
    ],
    "Nucleotides_and_Derivatives": [
        "adenine_264_279_rt11_12", 
        "adenosine_5_monophosphate_169_"
    ],
    "Organic_Acids": [
        "threonic_acid_292_220_rt7_49", 
        "citric_acid_273_375_rt9_72", 
        "quinic_acid_255_345_rt9_45", 
        "Oxalic_acid_219_147_3_94", 
        "dehydroascorbic_acid_316_173_r", 
        "erythronic_acid_292_rt7_24", 
        "pyroglutamic_acid_156_258_rt8_", 
        "nonanoic_acid_117_215_rt6_19", 
        "malic_acid_233_245_rt7_22", 
        "glyceric_acid_292_189_rt5_63", 
        "butyric_acid_4_amino_174_304_r", 
        "quinic_acid_3_caffeoyl_trans_3", 
        "galactonic_acid_1_4_lacton_217"
    ],
    "Amines_Amides": [
        "dopamine_174_426_rt11_11", 
        "ethanolamine_174_rt4_63", 
        "guanidine_146_171_rt4_33"
    ],
    "Vitamins_and_Cofactors": [
        "nicotinic_acid_180_136_rt6_32", 
        "pantethaine_4_03_220_235"
    ],
    "Sterols": [
        "beta_Sitosterol_1TMS_129_18_44"
    ],
    "Others": [
        "phosphoric_acid_314_299_rt_5_4",
        "glycerol_117_205_rt4_75",
        "pyridine_2_hydroxy_152_167_rt3", 
        "inositol_1_phosphate_myo_299_3", 
        "urea_189_204_rt5_56", 
        "glycerol_3_phosphate_357_445_r", 
        "Hydroxylamine_133_3_07_No_need", 
        "glutamic_acid_246_363_rt8_31"
    ]
}


# In[45]:




# Define the categories
categories2 = {
    "Sugars_Carbohydrates": [
        "Cellobiose_361_204_rt14_40",
        "trehalose_alpha_alpha_191_169_"
    ],
    "Others": [
        "phosphoric_acid_314_299_rt_5_4",
        "glycerol_117_205_rt4_75"
    ]
}


# In[46]:



def goea_for_category(category, metabolites, path):
    #excel_writer = pd.ExcelWriter(f'{outpath}{category}.xlsx', engine='xlsxwriter')
    goea_results = {}
    print(category)
    for m in metabolites:
        print(os.path.join(path, f"{m}_nXcv.json"))
        file_path = os.path.join(path, f"{m}_nXcv.json")
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        

        # perform GOEA
        data_matched, data_goids, data_term_count, data_goea_all = matchgff2(data['selected_features'], namespace=['molecular_function'], depth_threshold=2, goea=True)
        goea_results[m] = data_goea_all
    
    return goea_results


# In[50]:


path_gc = "/home/t44p/PW_rawdata/results/full_lasso/gcms/"


for category, metabolites in categories2.items():
    # check if output directories exist otherwise createt them
    category_path = os.path.join(path_gc, 'goea', f"{category}/")
    if not os.path.exists(category_path):
        try:
            # If the path does not exist, create the directory
            os.makedirs(category_path)
            print(f"Directory created: {category_path}")
        except Exception as e:
            # If there is an error in creating the directory, raise an error
            raise Exception(f"Error creating directory: {category_path}. Error: {e}")
    else:
        print(f"Path already exists: {category_path}")

    print(category_path)

    # do GOEA for each metabolite in the category and save them into 
    # dictionary where each key is the prefix before _nXcv.json
    lasso_goea = goea_for_category(category, metabolites, path=path_gc)

    # save the goea results to .csv and excel tables and create enrichment matrix .csv
    goea_results_to_file(lasso_goea, path=category_path, obo_path="/home/t44p/PW_rawdata/go_obo/go.obo", to_excel=True)

    

