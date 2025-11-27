import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import spearmanr, pearsonr
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem


#BUILD MODELS
fp = pd.read_csv('EGFR_IC50_dataset_MACCS.csv') #'MCL1_Ki_dataset_MACCS.csv'

fp.set_index('0', inplace=True)
#fp =fp.groupby(fp.index).median() # meidan  
fp = fp[~fp.index.duplicated(keep='first')] #keep first
fp.dropna(how='any', inplace=True)

y=fp.iloc[:, -1].copy()
X=fp.iloc[:,:-1].copy()
y=y.astype(float)

##FOR TASK 1
##batch1=['CHEMBL1169870', 'CHEMBL3746212', 'CHEMBL1241860', 'CHEMBL1821890', 'CHEMBL4211322']
##batch2=['CHEMBL205454','CHEMBL168555','CHEMBL3740492','CHEMBL122522','CHEMBL3221552']
##batch3=['CHEMBL328295','CHEMBL3740923','CHEMBL110577','CHEMBL1173665','CHEMBL482489']
##examples=batch1
#train_index = X.drop(examples).index
#test_index=X.loc[examples].index
##Xtr= X.loc[train_index]
##Xts=X.loc[test_index]
##ytr=y.loc[Xtr.index]
##yts=y.loc[Xts.index]
Xtr= X.copy()
ytr=y.copy()

model=RandomForestRegressor(n_estimators =100)
#model=GradientBoostingRegressor(max_depth=10)#RandomForestRegressor(n_estimators=100, bootstrap=True)#GradientBoostingRegressor(max_depth=10)

model.fit(Xtr,ytr)


#TESTING NEW SMILES
smiles =['Nc1ncnc2c1c(-c1ccc(F)c(O)c1)nn2C1CCCC1',
         'C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1']# example SMILES for QSAR prediciton

#generate dataframe with new smiles and run predictions
ls=[]
for s in smiles:
    m=Chem.MolFromSmiles(s)
    fp_sm = list(MACCSkeys.GenMACCSKeys(m))
    ls.append(fp_sm)
df_test=pd.DataFrame(ls)
df_test.columns=Xtr.columns

yp=model.predict(df_test)
print(yp)
