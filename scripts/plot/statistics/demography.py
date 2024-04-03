"""
This script is used to plot the demography of the population.
"""

import pandas as pd
import numpy as np 
import scipy.stats as stats
from collections import Counter

import scipy.stats as stats

data = [ [58, 4, 5, 0], [40, 1, 25, 4] ]
chi2, p, dof, expected = stats.chi2_contingency(data)

print(f"Chi2 statistic: {chi2}, p-value: {p}")
print("Expected frequencies:", expected)

# Read the Excel file
demography_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/fNIRS x MDD Data_Demographics_Clinical.xlsx'

total_num_of_people = 72


""" 
because the excel file is not well organized, we need to find the correct column for each demographic data
"""
dict_demography = {
    'age': 'Demographic Data',
    'sex': 'Unnamed: 3',
    'ethinicity': 'Unnamed: 4',
    'handness': 'Unnamed: 5',
    'education': 'Unnamed: 6',
}

dict_demography_treatment_response = {
    'age': 'Demographic Data',
    'sex': 'Unnamed: 3',
    'ethinicity': 'Unnamed: 4',
    'handness': 'Unnamed: 5',
    'education': 'Unnamed: 6',
    'pretreatment HAM-D score':'HAM-D Questionnaire (T1)',
    'posttreatment HAM-D score':'HAM-D Questionnaire (T8)'
    
}

excel_data = pd.read_excel(demography_path, sheet_name='Summary T0T8_fNIRS Analysis')
all_involved_subject_id = []
# MDD = excel_data[excel_data['Subject ID'][:2] == 'PT']

def find_group_demography(type, specify_group=None):
    res ={}
    
    if specify_group is not None:
        for sub in specify_group:
            for key in dict_demography_treatment_response:
                if key not in res:
                    res[key] = []
                subject = excel_data['Subject ID'] == sub
                value = excel_data[subject][dict_demography_treatment_response[key]].iloc[0]
                if pd.isna(value):
                    res[key].append(value)
                else:
                    res[key].append(int(value))
        return res

    for i in range(0, 1+total_num_of_people):
        # find 'CT001' or 'CT010' or 'PT001' or 'PT010'
        sub_id = type + '00' + str(i) if i < 10 else type + '0' + str(i)
        sub = excel_data['Subject ID'] == sub_id
        
        # sub will be an array of [True, False, False, ...]
        if True in sub.values:
            if type == 'PT': all_involved_subject_id.append(sub_id)
            for key in dict_demography:
                if key not in res:
                    res[key] = []
                value = excel_data[sub][dict_demography[key]].iloc[0]

                if pd.isna(value):
                   res[key].append(value)
                else:
                    res[key].append(int(value))
    return res 

def identify_responders_nonresponders():
    responders_id, nonresponders_id = [], []
    for subject in excel_data['Subject ID'][2:]:
        # print('subject - >', subject)
        
        hamd_of_id_t1 = excel_data[excel_data['Subject ID'] == subject]['HAM-D Questionnaire (T1)'].iloc[0]
        hamd_of_id_t8 = excel_data[excel_data['Subject ID'] == subject]['HAM-D Questionnaire (T8)'].iloc[0]
        if type(hamd_of_id_t8) is not int:
            # print(f"subject: {subject} | hamd_of_id_t8: {hamd_of_id_t8} is not a number")
            continue
        reduction_percentage = (hamd_of_id_t8 - hamd_of_id_t1) / hamd_of_id_t1
        if reduction_percentage <= -0.5:
            responders_id.append(subject)
        else:
            nonresponders_id.append(subject)
    return responders_id, nonresponders_id

def address_nan_value(data):
    for key, value in data.items():

        # find the nan value
        nan_indice = pd.isna(value)
        # get the median value 
        median = np.nanmedian(value)
        for i, v in enumerate(value):
            if pd.isna(v):
                value[i] = int(median)
        # data[key] = value
    return data 

def show_metrics(data, name):
    print('show metrics of ', name)
    for key in dict_demography:
        print(key, 'mean + std', np.mean(data[key]), np.std(data[key]))

def compare_two_groups(g1, g2, name):
    print(name)
    if name == 'responders vs nonresponders':   
        compare_dict_demography = dict_demography_treatment_response
    else:
        compare_dict_demography = dict_demography
    for key in compare_dict_demography:
        if key in ['age', 'education', 'pretreatment HAM-D score', 'posttreatment HAM-D score']:
            t_stat, p_value = stats.ttest_ind(g1[key], g2[key])  # For normally distributed 
            print(f"{key} t_stat: {t_stat}, p_value: {p_value}")
        if key in ['sex', 'ethinicity', 'handness']:
            all_counter = Counter(g1[key] + g2[key])
            g1_counter = Counter(g1[key])
            g2_counter = Counter(g2[key])
            g1_group = [g1_counter[i] for i, _ in all_counter.items()]
            g2_group = [g2_counter[i] for i, _ in all_counter.items()]
            stat, p_value, _, _ = stats.chi2_contingency([g1_group, g2_group])  # Use chi2_contingency for categorical variables
            print(f"{key} chi2_stat: {stat}, p_value: {p_value}")

"""
HC, MDD, RESPOND, NONRESPOND have forms like this 
{
    'age': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    'sex': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], # 1 is Male, 2 is Female 
    'ethinicity': [1, 2, 3, 4, 1, 2, 3, 3, 4, 4], # 1 is Chinese, 2 is Malay, 3 is Indian, 4 is Others 
    'Handedness': [1, 2, 3, 3, 1, 2, 3, 2, 1, 2], # 1 is Right, 2 is Left, 3 is Ambidextrous
    'education': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21], # years of education
}
"""
HC = find_group_demography('CT')
MDD = find_group_demography('PT')
MDD = address_nan_value(MDD)
# print(excel_data)
resp, nonresp = identify_responders_nonresponders()
RESPOND = find_group_demography(None, resp)
NONRESPOND = address_nan_value(find_group_demography(None, nonresp))
print('responders', RESPOND)
print('nonresponders', NONRESPOND)

show_metrics(HC,'HC')
print('-------------------')
show_metrics(MDD,'MDD')
print('-------------------')
show_metrics(RESPOND,'RESPOND')
print('-------------------')
show_metrics(NONRESPOND,'NONRESPOND')


print('-------------------')
compare_two_groups(HC, MDD, 'HCs vs MDDs')
print('-------------------')
compare_two_groups(RESPOND, NONRESPOND, 'responders vs nonresponders')



"""Cortical haemodynamic response during the verbal fluency task in patients with bipolar disorder and borderline personality disorder: a preliminary functional near- infrared spectroscopy study

To determine if activation during the VFT occurred for each diagnostic group at each ROI, Student’s paired t- test was used to compare mean oxy-haemoglobin during the pre-task baseline period and task period. The effect of diagnostic group on categorical variables was deter- mined using chi-square test, while Student’s t-test or one-way analysis of variance (ANOVA) with Bonferroni corrected post-hoc pairwise comparisons, was used to determine the effect of diagnostic group on continuous variables. Categorical variables were gender, ethnicity, handedness, family psychiatric history, past admission to psychiatric ward and treatment with psychotropic drugs. Psychotropic drugs were further classified into antide- pressants, anxiolytics and sedatives, antipsychotics and mood stabilisers (Supplementary Table 1). Continuous variables were age, years of education, number of words generated, mean oxy-haemoglobin and mean deoxy- haemoglobin at each ROI, GAF score, HAM-D score, YMRS score, BPQ score, age at psychiatric illness onset, duration of psychiatric illness and equivalent doses of antidepressants, anxiolytics and sedatives, as well as anti- psychotics. Equivalent doses were calculated based on published mean dose ratios. Reference drugs were fluox- etine, diazepam and chlorpromazine, for antidepressants, anxiolytics and sedatives, and antipsychotics, respectively

"""

continues = ['age', 'education', 'pretreatment HAM-D score', 'posttreatment HAM-D score']
categorical = ['sex', 'ethinicity', 'handness']

ethinicity = {
    1: 'Chinese',
    2: 'Malay',
    3: 'Indian',
    4: 'Others'
}
handness = {
    1: 'Right',
    2: 'Left',
    3: 'Ambidextrous'
}

sex_category = {
    1: 'Male',
    2: 'Female'
    
}


def print_demo_numer(data):
    for key, value in data.items():
        if key in continues:
            mean = np.mean(value)
            std = np.std(value)
            print("{} mean = {:.2f}, std = {:.2f}".format(key, mean, std))
        if key in categorical:
            counter = Counter(value)
            Total = sum(counter.values())

            # for ehinicity:
            if key == 'ethinicity':
                for k, v in counter.items():
                    print(f"{ethinicity[k]} N = {v}, % = {v/Total*100:.2f}")
                    
            # for handness
            if key == 'handness':
                for k, b in counter.items():
                    print(f"{handness[k]} N = {b}, % = {b/Total*100:.2f}")
                    
            # fir sex_category
            if key =='sex':
                for k, b in counter.items():
                    print(f"{sex_category[k]} N = {b}, % = {b/Total*100:.2f}")
            print('-')
print("For responders, total subject is ", len(RESPOND['age']))
print_demo_numer(RESPOND)

print('-------------------')
print("For nonresponders, total subject is ", len(NONRESPOND['age']))
print_demo_numer(NONRESPOND)
