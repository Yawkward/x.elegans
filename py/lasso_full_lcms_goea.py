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
import os


# In[49]:


def matchgff2(feature, gff_file='/work/yhesse/PW_rawdata/Transciptome_GenomeAnnotation/Xele_annotated2_gff_export2.gff', obo_path="/work/yhesse/PW_rawdata/go_obo/go.obo", namespace=None, depth_threshold=0, goea=False):
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


lcms_categories2 = {
    "Amino_Acids_and_Derivatives": [
        "Argininosuccinic_acid_L", "Aspartyphenylalanine_1_L", "gamma_Glutamylisoleucine_", 
        "gamma_Glutamylvaline_", "Glycyl_L_phenylalanine_L", "Histidine_L", 
        "L_gamma_glutamyl_L_isoleucine", "N_gamma_L_Glutamyl_L_methionin", "Phenylalanylglutamic_acid_", 
        "Prolylglycine_L", "Prolyphenylalanine_or_Leucymet", "Tyrosine_L", "Kynurenine_free_base_L", "N_Decanoylglycine_",
        "Phenylalanyaspartatic_acid_L"
    ],
    "Quinic_Acid_Derivatives": [
        "_3_5_dicaffeoul_quinic_acid_L", "_1_3_Dicaffeoylquinic_acid_", "_4_O_p_Coumaroylquinic_acid_", 
        "_4_5_Di_O_caffeoylquinic_acid_", "Coumaroyl_quinic_acid_", "Caffeoylshikimic_acid_L","Quinic_acid_161_05_", 
        "Quinic_acid_derivative_3_56", "Quinic_acid_derivative_with_he"
    ],
    "Phenolic_Compounds": [
        "Caffeic_acid_1_L", "Caffeic_acid_glucoside_L", "Caffeoylglucarate_L", 
        "Chlorogenic_acid_", "Conduritol_B_Epoxide_L", "Dehydro_Ascorbic_acid_L", "Ferulic_acid_4_O_glucuronide_L", 
        "Feruloyl_di_glucoside_L", "Feruloylglucose_L", "Flavonoid_glycoside_", "Galloylglucose_L", 
        "Gentesic_acid_glucoside_L", "Kaempferol_3_O_glucoside_7_O_r", "Kaempferol_3_O_rhamnoside_7_O_", 
        "Kaempferol_3_O_rutinoside_", "Kaempferol_7_3G_glucosylgentio", "Kaempferol_7_O_Glucoside_1_L", 
        "Kaempferol_glucose_xylose_L", "Kaempferol_rhamnose_glucose", "Naringenin_chalcone_L", "p_coumaroyl_di_glucoside_L", 
        "p_coumaroyl_glucoside_L", "Phenolic_glycoside_", "Quercetin_3_7_di_glucoside_", "Quercetin_3_glucoside_3", 
        "Quercetin_7_4_O_diglucoside_", "Quercetin_glc_glc_rha", "Quercetin_glucose_glucose_rham", 
        "Quercetin_glucose", "Quercetin_glucose_xylose", "Quercitin_3_7_diglucoside_6_rh", "Quercitin_3_O_Glucoside_2_L", 
        "Resveratrol_3_4_diglucoside_L", "Sinapoylglucose_2_L", "Sinapoylglucose_L","Catechol_", "_5_Chlorogenic_acid", "chlorgenic_acid_L", 
        "hydroxyjasmonic_acid_glucoside", "Hydroxy_jasmonic_acid_glucosid"
    ],
    "Sugars_and_Sugar_Derivatives": [
        "alpha_D_Galacturonic_acid_1_ph", "alpha_L_Fucose_1_Phosphate", "Dihydrophaseic_acid_glucoside_", 
        "Dihydroxyacetone_phosphate_L", "Fructose_1_6_biosphasphate_L", "Glucaric_acid_1_4_lactone_L", 
        "Gluconic_acid_lactone_L", "Gluconic_acid_L", "Glutamic_acid_L", "Inosine_L", "Isorhamnetin_3_O_glucoside_L", 
        "Isorhamnetin_3_O_rutinoside", "Lactobionic_acid_L", "Maltotriose_", "naringenin_7_O_glucoside_1_L", 
        "naringenin_7_O_glucoside_3_L", "Pantothenic_acid_L", "Phaseoloidin", 
        "Pyroglutamic_acid_3_L", "Ribulose_5_phosphate_L", "Tartaric_acid_L", "Trehalose", 
        "Trehalose_phenolic_acid_", "Vanilloloside", "Vanilloside_L", "Zeatin_glucoside_L", "_1_O_Feruloyl_glucose_L"
    ],
    "Nucleotides_and_Derivatives": [
        "_5_Deoxy_5_Methylthioadenosine", "Guanosine_", "Nicotinamide_adenine_dinucleot", 
        "Oxidized_glutathione_",
    ],
    "Terpenes_and_Triterpenes": [
        "Diterpene_8_2", "Terpene_10_26", "Terpene_9_53", "Triterpene_10_43", "Triterpene_10_78", 
        "Triterpene_8_14", "Triterpene_8_21", "Triterpene_9_79"
    ],
    "Saponins": [
        "Saponin_10_182", "Saponin_10_183", "Saponin_10_264", "Saponin_10_47", "Saponin_10_56", 
        "Saponin_10_67780171", "Saponin_10_72", "Saponin_8_02", "Saponin_8_42", "Saponin_8_83", 
        "Saponin_8_8", "Saponin_9_18", "Saponin_9_88", "Soyasaponin_A2_L"
    ],
    "Porphyrins": [
        "porphobilinogen_2_L"
    ],
    "Flavonoids": [
        "_Rutin"
    ],
    "Other": [
        "_10_Formyltetrahydrofolate_L", "_3_Deoxy_D_manno_2_octulosonic", "_5_hydroxy_Ferulic_acid_Glucos2", 
        "_5_hydroxy_ferulic_acid_glucos", "_6_phosphogluconic_acid_L", "_7_Epi_12_hydroxyjasmonic_acid", 
        "_D_Glycero_alpha_D_Manno_Hepto", "_R_2_Phenylglycin", "Azelaic_acid_L", "Azukisaponin_VI_1_L", 
        "CGA_hexose_", "Urocanic_acid_L"
    ],
    "unidentified": [
        "_1021_486521_9_141280832",
        "_1021_48667462_9_66538870707",
        "_1063_497231_10_58247297",
        "_1063_529871_9_415000766",
        "_1141_528589_8_043846139",
        "_1151_551493_10_40677464",
        "_1165_527084_8_995631942",
        "_1183_537778_8_600427106",
        "_1183_538043_8_673358308",
        "_1191_542641_10_3213884",
        "_1209_556345_10_14653119",
        "_1241_580464_8_916526509",
        "_1253_57885907_8_99579782362",
        "_1283_593009_9_259004534",
        "_1283_593348_9_160768845",
        "_1313_601235_9_01121774",
        "_303_0720727_3_335904942",
        "_312_0943959_2_614228811",
        "_351_1294837_5_809380415",
        "_366_9939681_1_170199352",
        "_374_1568694_3_674168475",
        "_380_1288743_6_36896135",
        "_427_1824019_5_95992104",
        "_429_1765684_7_873841266",
        "_441_1978503_6_218123415",
        "_475_1821542_6_560220101",
        "_487_2029578_5_338775921",
        "_496_1502126_3_792994891",
        "_512_1445532_3_132648409",
        "_523_1663837_5_238225973",
        "_531_2448764_7_209307815",
        "_567_2828405_6_133036019",
        "_641_171822_6_885700966",
        "_658_1575839_4_528042834",
        "_671_2775803_7_254060113",
        "_821_3258533_9_228249892",
        "_857_418157679_9_32212355859",
        "_857_418693067_9_24167206032",
        "_931_4518066_8_63280459",
        "_933_4699157_8_265868462",
        "_947_486080952_9_72692719903",
        "_963_4798371_8_624190173",
        "_987_4809544_10_24318525"
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
        
        # if no feature have been selected
        if len(data['selected_features']) == 0:
            print(f"Warning: {file_path} has no selected_features\n")
            continue

        # perform GOEA
        data_matched, data_goids, data_term_count, data_goea_all = matchgff2(data['selected_features'], namespace=['molecular_function'], depth_threshold=2, goea=True)
        goea_results[m] = data_goea_all
    
    return goea_results


# In[50]:

path_gc = "/work/yhesse/PW_rawdata/results/full_lasso/lcms"


for category, metabolites in lcms_categories2.items():
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
    goea_results_to_file(lasso_goea, path=category_path, obo_path="/work/yhesse/PW_rawdata/go_obo/go.obo", to_excel=True)

    

