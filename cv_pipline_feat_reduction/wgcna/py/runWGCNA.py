import PyWGCNA

geneExp = '/work/yhesse/PW_rawdata/tr_gc_mutual/tr_mut_transposed_modified.csv'
pyWGCNA_5xFAD = PyWGCNA.WGCNA(name='xele', 
                              species='xerophyta elegans', 
                              geneExpPath=geneExp, 
                              outputPath='/work/yhesse/jobs/xele_ml/wgcna',
                              save=True)
pyWGCNA_5xFAD.geneExpr.to_df().head(5)

# runWGCNA()
#   Preprocess and find modules
pyWGCNA_5xFAD.runWGCNA()

#saveWGCNA()

#   Saves the current WGCNA in pickle format with the .p extension
pyWGCNA_5xFAD.saveWGCNA()
