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

