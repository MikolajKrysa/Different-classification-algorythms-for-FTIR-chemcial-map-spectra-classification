# -*- coding: utf-8 -*-
"""
Code for preprocessed FTIR chemical maps 
Code suitable for the same amount of samples from different groups (eg. 3 control samples 3 diabetic samples )
"""
import os
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time
          
spectra_dict='C:/Users/annasrokabartnicka.FA-1-NO-I001796/Documents/Rzeczy pracowe/Cukrzyca/Solaris/Baseline corrected maps/To_analysis/'
samples=os.listdir(spectra_dict)
print('Samples are:')
print(samples)
#%% loading the samples 
group_names=['Liver_DM','Liver_control']
nr_of_same_group=int(len(samples)/len(group_names))

#importing the data
i=0
for line in samples:
    curr_group_name=group_names[int(i/nr_of_same_group)]
    if i==0:
        df=[pd.read_excel(spectra_dict+samples[i])]
        sample_classification=[pd.Series([1 for x in range(0, len(df[0]))], 
                                      name=df[0]['Sample Name'][0])]
        group_classification=[pd.Series([1 for x in range(0, len(df[0]))], 
                                      name=curr_group_name)]
        group_class_string=[pd.Series([curr_group_name for x in range(0, len(df[0]))], 
                                      name=curr_group_name)]
        to_sample_index=len(sample_classification[0])
        to_group_index=len(group_classification[0])
    else:
        to_add=pd.read_excel(spectra_dict+samples[i])
        df.append(to_add)
        sample_classification.append(pd.Series([1 for x in range(to_sample_index, to_sample_index+len(to_add))], 
                                            name=to_add['Sample Name'][0], 
                                            index=[x for x in range(to_sample_index, to_sample_index+len(to_add))]))
        group_classification.append(pd.Series([1 for x in range(to_group_index, to_group_index+len(to_add))], 
                                            name=curr_group_name, 
                                            index=[x for x in range(to_group_index, to_group_index+len(to_add))]))
        group_class_string.append(pd.Series([curr_group_name for x in range(to_group_index, to_group_index+len(to_add))], 
                                            name=curr_group_name, 
                                            index=[x for x in range(to_group_index, to_group_index+len(to_add))]))
        to_sample_index+=len(to_add)
        to_group_index+=len(to_add)
    print('loaded ', line, ' sample')
    i+=1
df=pd.concat(df,axis=0)
sample_classification=pd.concat(sample_classification,axis=1)
sample_classification=sample_classification.fillna(0)
sample_classification=sample_classification.groupby(sample_classification.columns, axis=1).sum()
group_classification=pd.concat(group_classification,axis=1)
group_classification=group_classification.fillna(0)
group_classification=group_classification.groupby(group_classification.columns, axis=1).sum()
group_class_string=pd.concat(group_class_string,axis=0)
sample_class_string=df['Sample Name']
df=df.drop(['map_x','map_y','Start Time','Sample Name'], axis='columns')
df.index=list(map(lambda x: x, range(0, len(df))))


sample_names=sample_classification.columns.tolist()

#creating separate list of dataframes of separate samples (order like in sample_names)
for line in sample_names:
    if line==sample_names[0]:
        separate_samples=[df[pd.Index(sample_class_string).get_loc(line)]]
    else:
        separate_samples.append(df[pd.Index(sample_class_string).get_loc(line)])

#creating a list of samples separated for the groups        
i=0
k=0
while i<len(sample_names):
    j=0
    while j<(len(sample_names)/len(group_names)):
        if j==0:
            curr_samples=[sample_names[i]]
        else:
            curr_samples.append(sample_names[i])
        i+=1
        j+=1
    if k==0:
        groups_of_samples=[curr_samples]
    else:
        groups_of_samples.append(curr_samples)
    k+=1

#creating separate groups for separate classification (samples form DM group seprate from control)
i=0
for line in group_names:
    if line==group_names[0]:
        vals_to_indices=[0]
    vals_to_indices.append((group_class_string.values==line).sum()+vals_to_indices[i])
    i+=1
i=0
while i<len(group_names):
    if i==0:
        separate_groups=[df[vals_to_indices[i]:vals_to_indices[i+1]]]
        separate_sample_class=[sample_classification[vals_to_indices[i]:vals_to_indices[i+1]].loc[:,groups_of_samples[i]]]
        separate_sample_class_str=[sample_class_string[vals_to_indices[i]:vals_to_indices[i+1]]]
    else:
        separate_groups.append(df[vals_to_indices[i]:vals_to_indices[i+1]])
        separate_sample_class.append(sample_classification[vals_to_indices[i]:vals_to_indices[i+1]].loc[:,groups_of_samples[i]])
        separate_sample_class_str.append(sample_class_string[vals_to_indices[i]:vals_to_indices[i+1]])
    i+=1

#getting x_axis (wavenumbers/raman shift)
for line in df.columns:
    if line==df.columns[0]:
        x_axis=[float(line)]
    else:
        x_axis.append(float(line))

for line in group_names:
    if line==group_names[0]:
        conf_matrix_cols=['predicted '+line]
        conf_matrix_index=['true '+line]
    else:
        conf_matrix_cols.append('predicted '+line)
        conf_matrix_index.append('true '+line) 
unique_groups=group_names
#RF_sample_classification=sample_class_string
#RF_group_classification=group_class_string

#%% CCA
#k-fold cross-validation
skf = StratifiedKFold(n_splits=5)
for k, (train_index, test_index) in enumerate(skf.split(df, group_class_string)):
    t1=time.perf_counter()
    cca = CCA(n_components=2)
    cca.fit(df.iloc[train_index], group_classification.iloc[train_index])
    prediction=pd.DataFrame(cca.predict(df.iloc[test_index]), columns=group_classification.columns)
    j=0
    while j<len(prediction.iloc[:]):
        i=0
        for line in prediction.iloc[j]:
            if line==max(prediction.iloc[j]):
                if j==0:
                    predicted_test=[prediction.columns[i]]
                else:
                    predicted_test.append(prediction.columns[i])
            i+=1
        j+=1
    #create a confusion matrix
    conf_matrix=pd.DataFrame(confusion_matrix(group_class_string.iloc[test_index],predicted_test, 
                                              labels=unique_groups), 
                           columns=conf_matrix_cols, 
                           index=conf_matrix_index)
    if k==0:
        conf_matrices=[conf_matrix]
    else:
        conf_matrices.append(conf_matrix)
    i=0
    for line in conf_matrix:
        if i==0:
            CC=[conf_matrix.iloc[i][i]/sum(conf_matrix.iloc[i])]
        else:
            CC.append(conf_matrix.iloc[i][i]/sum(conf_matrix.iloc[i]))
        i+=1
    conf_matrix['%CC']=CC
    if k==0:
        Non_reduced_CCs=[CC]
    else:
        Non_reduced_CCs.append(CC)
    if len(unique_groups)>2:
        whole_CC=conf_matrix['%CC'].mean()
        if k==0:
            CCs=[whole_CC]
        else:
            CCs.append(whole_CC) 
    else:
        sensitivity_g1=(conf_matrix.iloc[0][0])/(conf_matrix.iloc[0][0]+conf_matrix.iloc[0][1])
        specificity_g1=(conf_matrix.iloc[1][1])/(conf_matrix.iloc[1][0]+conf_matrix.iloc[1][1])
        precision_g1=(conf_matrix.iloc[0][0])/(conf_matrix.iloc[0][0]+conf_matrix.iloc[1][0])
        F1_g1=2*((precision_g1*sensitivity_g1)/(precision_g1+sensitivity_g1))
        if k==0:
            sensitivities_g1=[sensitivity_g1]
            specificities_g1=[specificity_g1]
            precisions_g1=[precision_g1]
            F1s_g1=[F1_g1]
        else:
            sensitivities_g1.append(sensitivity_g1)
            specificities_g1.append(specificity_g1)
            precisions_g1.append(precision_g1)
            F1s_g1.append(F1_g1)
    loading=cca.x_loadings_
    if k==0:
        many_loadings=[loading]
    else:
        many_loadings.append(loading)
    t2=time.perf_counter()
    if len(unique_groups)>2:
        print('the %CC of fold '+str(k)+' is: '+str(whole_CC)+' it took '+str(t2-t1) +
              ' seconds')  
    else:
        print('the F1 of fold '+str(k)+' is: '+str(F1_g1)+' it took '+str(t2-t1) +
              ' seconds')
master_conf_matrix=pd.concat(conf_matrices)
master_conf_matrix=master_conf_matrix.groupby(master_conf_matrix.index)
master_conf_matrix=master_conf_matrix.mean()
Mean_non_reduced_CC=pd.DataFrame(Non_reduced_CCs, columns=unique_groups).mean()
if len(unique_groups)>2:
    master_CC=mean(CCs)
else:
    master_sensitivity_g1=mean(sensitivities_g1)
    master_specificity_g1=mean(specificities_g1)
    master_precision_g1=mean(precisions_g1)
    master_F1_g1=mean(F1s_g1)
i=0
while i<len(loading[0]):
    j=0
    for line in many_loadings:
        if j==0:
            temp_loadings=[line[:,i]]
        else:
            temp_loadings.append(line[:,i])
        j+=1
    if i==0:
        master_loadings=[pd.DataFrame(temp_loadings).mean()]
    else:
        master_loadings.append(pd.DataFrame(temp_loadings).mean())
    i+=1
master_loadings=pd.DataFrame(master_loadings).T

for line in df.columns:
    if line==df.columns[0]:
        x_axis=[float(line)]
    else:
        x_axis.append(float(line))
master_loadings.index=x_axis

if len(unique_groups)>2:
    print('Overall CC is ',master_CC,' in CCA')
    master_CC_CCA=master_CC
else:
    print('Overall F1 is of ',sample_names[0],' is ',master_F1_g1,' in CCA')
    print('Overall sensitivity is of ',sample_names[0],' is ',master_sensitivity_g1,' in CCA')
    print('Overall specificity is of ',sample_names[0],' is ',master_specificity_g1,' in CCA')
    print('Overall precision is of ',sample_names[0],' is ',master_precision_g1,' in CCA')
    master_F1_g1_CCA=master_F1_g1
    master_sensitivity_g1_CCA=master_sensitivity_g1
    master_specificity_g1_CCA=master_specificity_g1
    master_precision_g1_CCA=master_precision_g1
print(master_conf_matrix)
master_conf_matrix_CCA=master_conf_matrix
master_loadings_CCA=master_loadings

#preparing the samples to be plotted (component 1 vs component 2)
X_train_r, Y_train_r = cca.transform(df.iloc[train_index], group_classification.iloc[train_index])
X_test_r, Y_test_r = cca.transform(df.iloc[test_index], group_classification.iloc[test_index])
#creating sorted X_test_r in accordance to unique names
i=1
while i<=len(X_train_r[0]):
    if i==1:
        CV_names=['CV '+str(i)]
    else:
        CV_names.append('CV '+str(i))
    i+=1   
X_train_r=pd.DataFrame(X_train_r, index=group_class_string.iloc[train_index],
                       columns=CV_names)
X_test_r=pd.DataFrame(X_test_r, index=group_class_string.iloc[test_index],
                       columns=CV_names)
sorted_X_test_r=[]
sorted_X_train_r=[]
for line in unique_groups:
    sorted_X_test_r.append(X_test_r[X_test_r.index==line])
    sorted_X_train_r.append(X_train_r[X_train_r.index==line])
#creating figure
fig, ax = plt.subplots()
i=0
for line in unique_groups:
    ax.scatter(sorted_X_test_r[i]['CV 1'], sorted_X_test_r[i]['CV 2'],label=line)
    #confidence_ellipse(sorted_X_test_r[i]['CV 1'], sorted_X_test_r[i]['CV 2'],ax,n_std=50.0,alpha=0.5, facecolor='pink', edgecolor='purple')
    i+=1
plt.legend()
plt.title('CCA')
#%%PLS-DA
#k-fold cross-validation
skf = StratifiedKFold(n_splits=5)
for k, (train_index, test_index) in enumerate(skf.split(df, group_class_string)):
    t1=time.perf_counter()
    pls = PLSRegression(n_components=2)
    pls.fit(df.iloc[train_index], group_classification.iloc[train_index])
    prediction=pd.DataFrame(pls.predict(df.iloc[test_index]), columns=group_classification.columns)
    j=0
    while j<len(prediction.iloc[:]):
        i=0
        for line in prediction.iloc[j]:
            if line==max(prediction.iloc[j]):
                if j==0:
                    predicted_test=[prediction.columns[i]]
                else:
                    predicted_test.append(prediction.columns[i])
            i+=1
        j+=1
    #create a confusion matrix
    conf_matrix=pd.DataFrame(confusion_matrix(group_class_string.iloc[test_index],predicted_test, 
                                              labels=unique_groups), 
                           columns=conf_matrix_cols, 
                           index=conf_matrix_index)
    if k==0:
        conf_matrices=[conf_matrix]
    else:
        conf_matrices.append(conf_matrix)
    i=0
    for line in conf_matrix:
        if i==0:
            CC=[conf_matrix.iloc[i][i]/sum(conf_matrix.iloc[i])]
        else:
            CC.append(conf_matrix.iloc[i][i]/sum(conf_matrix.iloc[i]))
        i+=1
    conf_matrix['%CC']=CC
    if k==0:
        Non_reduced_CCs=[CC]
    else:
        Non_reduced_CCs.append(CC)
    if len(unique_groups)>2:
        whole_CC=conf_matrix['%CC'].mean()
        if k==0:
            CCs=[whole_CC]
        else:
            CCs.append(whole_CC) 
    else:
        sensitivity_g1=(conf_matrix.iloc[0][0])/(conf_matrix.iloc[0][0]+conf_matrix.iloc[0][1])
        specificity_g1=(conf_matrix.iloc[1][1])/(conf_matrix.iloc[1][0]+conf_matrix.iloc[1][1])
        precision_g1=(conf_matrix.iloc[0][0])/(conf_matrix.iloc[0][0]+conf_matrix.iloc[1][0])
        F1_g1=2*((precision_g1*sensitivity_g1)/(precision_g1+sensitivity_g1))
        if k==0:
            sensitivities_g1=[sensitivity_g1]
            specificities_g1=[specificity_g1]
            precisions_g1=[precision_g1]
            F1s_g1=[F1_g1]
        else:
            sensitivities_g1.append(sensitivity_g1)
            specificities_g1.append(specificity_g1)
            precisions_g1.append(precision_g1)
            F1s_g1.append(F1_g1)
    loading=pls.x_loadings_
    if k==0:
        many_loadings=[loading]
    else:
        many_loadings.append(loading)
    t2=time.perf_counter()
    if len(unique_groups)>2:
        print('the %CC of fold '+str(k)+' is: '+str(whole_CC)+' it took '+str(t2-t1) +
              ' seconds')  
    else:
        print('the F1 of fold '+str(k)+' is: '+str(F1_g1)+' it took '+str(t2-t1) +
              ' seconds')
master_conf_matrix=pd.concat(conf_matrices)
master_conf_matrix=master_conf_matrix.groupby(master_conf_matrix.index)
master_conf_matrix=master_conf_matrix.mean()
Mean_non_reduced_CC=pd.DataFrame(Non_reduced_CCs, columns=unique_groups).mean()
if len(unique_groups)>2:
    master_CC=mean(CCs)
else:
    master_sensitivity_g1=mean(sensitivities_g1)
    master_specificity_g1=mean(specificities_g1)
    master_precision_g1=mean(precisions_g1)
    master_F1_g1=mean(F1s_g1)
i=0
while i<len(loading[0]):
    j=0
    for line in many_loadings:
        if j==0:
            temp_loadings=[line[:,i]]
        else:
            temp_loadings.append(line[:,i])
        j+=1
    if i==0:
        master_loadings=[pd.DataFrame(temp_loadings).mean()]
    else:
        master_loadings.append(pd.DataFrame(temp_loadings).mean())
    i+=1
master_loadings=pd.DataFrame(master_loadings).T

for line in df.columns:
    if line==df.columns[0]:
        x_axis=[float(line)]
    else:
        x_axis.append(float(line))
master_loadings.index=x_axis

if len(unique_groups)>2:
    print('Overall CC is ',master_CC,' in PLS-DA')
    master_CC_PLSDA=master_CC
else:
    print('Overall F1 is of ',sample_names[0],' is ',master_F1_g1,' in PLS-DA')
    print('Overall sensitivity is of ',sample_names[0],' is ',master_sensitivity_g1,' in PLS-DA')
    print('Overall specificity is of ',sample_names[0],' is ',master_specificity_g1,' in PLS-DA')
    print('Overall precision is of ',sample_names[0],' is ',master_precision_g1,' in PLS-DA')
    master_F1_g1_PLSDA=master_F1_g1
    master_sensitivity_g1_PLSDA=master_sensitivity_g1
    master_specificity_g1_PLSDA=master_specificity_g1
    master_precision_g1_PLSDA=master_precision_g1
print(master_conf_matrix)
master_conf_matrix_PLSDA=master_conf_matrix
master_loadings_PLSDA=master_loadings

X_train_r, Y_train_r = pls.transform(df.iloc[train_index], group_classification.iloc[train_index])
X_test_r, Y_test_r = pls.transform(df.iloc[test_index], group_classification.iloc[test_index])

#creating sorted X_test_r in accordance to unique names
i=1
while i<=len(X_train_r[0]):
    if i==1:
        component_names=['Component '+str(i)]
    else:
        component_names.append('Component '+str(i))
    i+=1
    
X_train_r=pd.DataFrame(X_train_r, index=group_class_string.iloc[train_index],
                       columns=component_names)
X_test_r=pd.DataFrame(X_test_r, index=group_class_string.iloc[test_index],
                       columns=component_names)
sorted_X_test_r=[]
sorted_X_train_r=[]
for line in unique_groups:
    sorted_X_test_r.append(X_test_r[X_test_r.index==line])
    sorted_X_train_r.append(X_train_r[X_train_r.index==line])
fig, ax = plt.subplots()
i=0
for line in unique_groups:
    ax.scatter(sorted_X_test_r[i]['Component 1'], sorted_X_test_r[i]['Component 2'],label=line)
    i+=1
plt.legend()
plt.title('PLS-DA')

#%% RF
#k-fold, stratified cross validation
skf = StratifiedKFold(n_splits=5)
for k, (train_index, test_index) in enumerate(skf.split(df, group_class_string)):
    t1=time.perf_counter()
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(df.iloc[train_index], group_class_string.iloc[train_index])
    pred_test_class=clf.predict(df.iloc[test_index])
    conf_matrix=pd.DataFrame(confusion_matrix(group_class_string.iloc[test_index],
                                               pred_test_class, labels=unique_groups), 
                           columns=conf_matrix_cols, 
                           index=conf_matrix_index)
    if k==0:
        conf_matrices=[conf_matrix]
    else:
        conf_matrices.append(conf_matrix)
    i=0
    for line in conf_matrix:
        if i==0:
            CC=[conf_matrix.iloc[i][i]/sum(conf_matrix.iloc[i])]
        else:
            CC.append(conf_matrix.iloc[i][i]/sum(conf_matrix.iloc[i]))
        i+=1
    conf_matrix['%CC']=CC
    if k==0:
        Non_reduced_CCs=[CC]
    else:
        Non_reduced_CCs.append(CC)
    if len(unique_groups)>2:
        score=clf.score(df.iloc[test_index], group_class_string.iloc[test_index]) # %correctly classified
        if k==0:
            scores=[score]
        else:
            scores.append(score)
    else:
        sensitivity_g1=(conf_matrix.iloc[0][0])/(conf_matrix.iloc[0][0]+conf_matrix.iloc[0][1])
        specificity_g1=(conf_matrix.iloc[1][1])/(conf_matrix.iloc[1][0]+conf_matrix.iloc[1][1])
        precision_g1=(conf_matrix.iloc[0][0])/(conf_matrix.iloc[0][0]+conf_matrix.iloc[1][0])
        F1_g1=2*((precision_g1*sensitivity_g1)/(precision_g1+sensitivity_g1))
        if k==0:
            sensitivities_g1=[sensitivity_g1]
            specificities_g1=[specificity_g1]
            precisions_g1=[precision_g1]
            F1s_g1=[F1_g1]
        else:
            sensitivities_g1.append(sensitivity_g1)
            specificities_g1.append(specificity_g1)
            precisions_g1.append(precision_g1)
            F1s_g1.append(F1_g1)
    feature_importances=clf.feature_importances_
    if k==0:
        many_f_importances=[feature_importances]
    else:
        many_f_importances.append(feature_importances)
    t2=time.perf_counter()
    if len(unique_groups)>2:
        print('the score of fold '+str(k)+' is: '+str(score)+' it took '+str(t2-t1) +
              ' seconds')  
    else:
        print('the F1 of fold '+str(k)+' is: '+str(F1_g1)+' it took '+str(t2-t1) +
              ' seconds')
master_conf_matrix=pd.concat(conf_matrices)
master_conf_matrix=master_conf_matrix.groupby(master_conf_matrix.index)
master_conf_matrix=master_conf_matrix.mean()
Mean_non_reduced_CC=pd.DataFrame(Non_reduced_CCs, columns=unique_groups).mean()
if len(unique_groups)>2:
    master_score=mean(scores)
else:
    master_sensitivity_g1=mean(sensitivities_g1)
    master_specificity_g1=mean(specificities_g1)
    master_precision_g1=mean(precisions_g1)
    master_F1_g1=mean(F1s_g1)
master_feature_importances=pd.DataFrame(many_f_importances).mean()

if len(unique_groups)>2:
    print('Overall CC is ',master_score,' in RF')
    master_CC_RF=master_score
else:
    print('Overall F1 is of ',sample_names[0],' is ',master_F1_g1,' in RF')
    print('Overall sensitivity is of ',sample_names[0],' is ',master_sensitivity_g1,' in RF')
    print('Overall specificity is of ',sample_names[0],' is ',master_specificity_g1,' in RF')
    print('Overall precision is of ',sample_names[0],' is ',master_precision_g1,' in RF')
    master_F1_g1_RF=master_F1_g1
    master_sensitivity_g1_RF=master_sensitivity_g1
    master_specificity_g1_RF=master_specificity_g1
    master_precision_g1_RF=master_precision_g1
master_conf_matrix_RF=master_conf_matrix
master_feature_importances_RF=master_feature_importances

#%%  Ada-Boost
#k-fold, stratified cross validation
skf = StratifiedKFold(n_splits=5)
for k, (train_index, test_index) in enumerate(skf.split(df, group_class_string)):
    t1=time.perf_counter()
    clf = AdaBoostClassifier()
    clf.fit(df.iloc[train_index], group_class_string.iloc[train_index])
    pred_test_class=clf.predict(df.iloc[test_index])
    conf_matrix=pd.DataFrame(confusion_matrix(group_class_string.iloc[test_index],
                                               pred_test_class, labels=unique_groups), 
                           columns=conf_matrix_cols, 
                           index=conf_matrix_index)
    if k==0:
        conf_matrices=[conf_matrix]
    else:
        conf_matrices.append(conf_matrix)
    i=0
    for line in conf_matrix:
        if i==0:
            CC=[conf_matrix.iloc[i][i]/sum(conf_matrix.iloc[i])]
        else:
            CC.append(conf_matrix.iloc[i][i]/sum(conf_matrix.iloc[i]))
        i+=1
    conf_matrix['%CC']=CC
    if k==0:
        Non_reduced_CCs=[CC]
    else:
        Non_reduced_CCs.append(CC)
    if len(unique_groups)>2:
        score=clf.score(df.iloc[test_index], group_class_string.iloc[test_index]) # %correctly classified
        if k==0:
            scores=[score]
        else:
            scores.append(score)
    else:
        sensitivity_g1=(conf_matrix.iloc[0][0])/(conf_matrix.iloc[0][0]+conf_matrix.iloc[0][1])
        specificity_g1=(conf_matrix.iloc[1][1])/(conf_matrix.iloc[1][0]+conf_matrix.iloc[1][1])
        precision_g1=(conf_matrix.iloc[0][0])/(conf_matrix.iloc[0][0]+conf_matrix.iloc[1][0])
        F1_g1=2*((precision_g1*sensitivity_g1)/(precision_g1+sensitivity_g1))
        if k==0:
            sensitivities_g1=[sensitivity_g1]
            specificities_g1=[specificity_g1]
            precisions_g1=[precision_g1]
            F1s_g1=[F1_g1]
        else:
            sensitivities_g1.append(sensitivity_g1)
            specificities_g1.append(specificity_g1)
            precisions_g1.append(precision_g1)
            F1s_g1.append(F1_g1)
    feature_importances=clf.feature_importances_
    if k==0:
        many_f_importances=[feature_importances]
    else:
        many_f_importances.append(feature_importances)
    t2=time.perf_counter()
    if len(unique_groups)>2:
        print('the score of fold '+str(k)+' is: '+str(score)+' it took '+str(t2-t1) +
              ' seconds')  
    else:
        print('the F1 of fold '+str(k)+' is: '+str(F1_g1)+' it took '+str(t2-t1) +
              ' seconds')
master_conf_matrix=pd.concat(conf_matrices)
master_conf_matrix=master_conf_matrix.groupby(master_conf_matrix.index)
master_conf_matrix=master_conf_matrix.mean()
Mean_non_reduced_CC=pd.DataFrame(Non_reduced_CCs, columns=unique_groups).mean()
if len(unique_groups)>2:
    master_score=mean(scores)
else:
    master_sensitivity_g1=mean(sensitivities_g1)
    master_specificity_g1=mean(specificities_g1)
    master_precision_g1=mean(precisions_g1)
    master_F1_g1=mean(F1s_g1)
master_feature_importances=pd.DataFrame(many_f_importances).mean()

if len(unique_groups)>2:
    print('Overall CC is ',master_score,' in AB')
    master_CC_AB=master_score
else:
    print('Overall F1 is of ',sample_names[0],' is ',master_F1_g1,' in AB')
    print('Overall sensitivity is of ',sample_names[0],' is ',master_sensitivity_g1,' in AB')
    print('Overall specificity is of ',sample_names[0],' is ',master_specificity_g1,' in AB')
    print('Overall precision is of ',sample_names[0],' is ',master_precision_g1,' in AB')
    master_F1_g1_AB=master_F1_g1
    master_sensitivity_g1_AB=master_sensitivity_g1
    master_specificity_g1_AB=master_specificity_g1
    master_precision_g1_AB=master_precision_g1
master_conf_matrix_AB=master_conf_matrix
master_feature_importances_AB=master_feature_importances
