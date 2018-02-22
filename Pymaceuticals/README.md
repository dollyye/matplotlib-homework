
# coding: utf-8

# In[1]:


# Observed Trends:
# 1. Only the Capomulin and Ramicane treatments were effective in lowering tumor volume over time.
# 2. Propriva demonstrated the lowest survival rate out of all the treatments.
# 3. Ketapril was ineffective in lowering tumor volume as the treatment resulted in the greatest percentage increase in tumor volume over time.


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[3]:


# Raw csv data
clin_trial_data = pd.read_csv("raw_data/clinicaltrial_data.csv")
mouse_drug_data = pd.read_csv("raw_data/mouse_drug_data.csv")

# Merge together
trial_mouse_data = pd.merge(clin_trial_data, mouse_drug_data, how='left', on='Mouse ID')
trial_mouse_data.head()


# # Tumor Response to Treatment

# In[4]:


# Create pivot table showing Tumor Volume over time for each treatment
treatment_tumor = trial_mouse_data.pivot_table(values="Tumor Volume (mm3)", index="Timepoint", 
                                               columns="Drug", aggfunc=np.mean)
treatment_tumor


# In[5]:


# Plot Tumor Response to Treatment with errorbars
Timepoint = treatment_tumor.index
plt.figure(figsize=(12,8))
plt.errorbar(Timepoint, treatment_tumor['Capomulin'], yerr=treatment_tumor['Capomulin'].sem(), 
            marker ='o', linestyle='--', label='Capomulin')
plt.errorbar(Timepoint, treatment_tumor['Ceftamin'], yerr=treatment_tumor['Ceftamin'].sem(),
             marker='^', linestyle='--', label='Ceftamin')
plt.errorbar(Timepoint, treatment_tumor['Infubinol'], yerr=treatment_tumor['Infubinol'].sem(),
             marker='*', linestyle='--', label='Infubinol')
plt.errorbar(Timepoint, treatment_tumor['Ketapril'], yerr=treatment_tumor['Ketapril'].sem(),
             marker='s', linestyle='--', label='Ketapril')
plt.errorbar(Timepoint, treatment_tumor['Naftisol'], yerr=treatment_tumor['Naftisol'].sem(), 
             marker='x', linestyle='--', label='Naftisol')
plt.errorbar(Timepoint, treatment_tumor['Placebo'], yerr=treatment_tumor['Placebo'].sem(), 
             marker='v', linestyle='--', label='Placebo')
plt.errorbar(Timepoint, treatment_tumor['Propriva'], yerr=treatment_tumor['Propriva'].sem(), 
             marker='1', linestyle='--', label='Propriva')
plt.errorbar(Timepoint, treatment_tumor['Ramicane'], yerr=treatment_tumor['Ramicane'].sem(), 
             marker='d', linestyle='--', label='Ramicane')
plt.errorbar(Timepoint, treatment_tumor['Stelasyn'], yerr=treatment_tumor['Stelasyn'].sem(), 
             marker='p', linestyle='--', label='Stelasyn')
plt.errorbar(Timepoint, treatment_tumor['Zoniferol'], yerr=treatment_tumor['Zoniferol'].sem(), 
             marker='h', linestyle='--', label='Zoniferol')
plt.legend(loc='best')
plt.title('Tumor Response to Treatment')
plt.xlabel('Time (Days)')
plt.ylabel('Tumor Volume (mm3)')
plt.xlim(0,Timepoint.max())
plt.ylim(20,80)
plt.grid()


# #  Metastatic Response to Treatment

# In[6]:


# Create pivot table showing Metastatic Sites over time for each treatment
treatment_metastatic = trial_mouse_data.pivot_table(values='Metastatic Sites', index='Timepoint', 
                                                    columns='Drug', aggfunc=np.mean)
treatment_metastatic


# In[7]:


# Plot Metastatic Spread During Treatment with errorbars
Duration = treatment_metastatic.index
plt.figure(figsize=(12,8))
plt.errorbar(Duration, treatment_metastatic['Capomulin'], yerr=treatment_metastatic['Capomulin'].sem(), 
             marker ='o', linestyle='--', label='Capomulin')
plt.errorbar(Duration, treatment_metastatic['Ceftamin'], yerr=treatment_metastatic['Ceftamin'].sem(),
             marker='^', linestyle='--', label='Ceftamin')
plt.errorbar(Duration, treatment_metastatic['Infubinol'], yerr=treatment_metastatic['Infubinol'].sem(),
             marker='*', linestyle='--', label='Infubinol')
plt.errorbar(Duration, treatment_metastatic['Ketapril'], yerr=treatment_metastatic['Ketapril'].sem(),
             marker='s', linestyle='--', label='Ketapril')
plt.errorbar(Duration, treatment_metastatic['Naftisol'], yerr=treatment_metastatic['Naftisol'].sem(), 
             marker='x', linestyle='--', label='Naftisol')
plt.errorbar(Duration, treatment_metastatic['Placebo'], yerr=treatment_metastatic['Placebo'].sem(), 
             marker='v', linestyle='--', label='Placebo')
plt.errorbar(Duration, treatment_metastatic['Propriva'], yerr=treatment_metastatic['Propriva'].sem(), 
             marker='1', linestyle='--', label='Propriva')
plt.errorbar(Duration, treatment_metastatic['Ramicane'], yerr=treatment_metastatic['Ramicane'].sem(), 
             marker='d', linestyle='--', label='Ramicane')
plt.errorbar(Duration, treatment_metastatic['Stelasyn'], yerr=treatment_metastatic['Stelasyn'].sem(), 
             marker='p', linestyle='--', label='Stelasyn')
plt.errorbar(Duration, treatment_metastatic['Zoniferol'], yerr=treatment_metastatic['Zoniferol'].sem(), 
             marker='h', linestyle='--', label='Zoniferol')
plt.legend(loc='best')
plt.title('Metastatic Spread During Treatment')
plt.xlabel('Treatment Duration (Days)')
plt.ylabel('Metastatic Sites')
plt.xlim(0,Duration.max())
plt.ylim(0,4)
plt.grid()


# # Survival Rates

# In[8]:


# Creat pivot table showing survival count over time for each treatment
survival_count = trial_mouse_data.pivot_table(values='Mouse ID', index='Timepoint', 
                                              columns='Drug', aggfunc='count')
survival_count


# In[9]:


# Calculate percentage survival rate
survival_rate = survival_count.copy()
survival_rate = round(survival_rate.apply(lambda n: n / n.max()*100, axis=0),2)
survival_rate


# In[10]:


# Plot Survival During Treatment
Time = survival_rate.index
plt.figure(figsize=(12,8))
plt.errorbar(Time, survival_rate['Capomulin'], marker ='o', linestyle='--', label='Capomulin')
plt.errorbar(Time, survival_rate['Ceftamin'], marker='^', linestyle='--', label='Ceftamin')
plt.errorbar(Time, survival_rate['Infubinol'], marker='*', linestyle='--', label='Infubinol')
plt.errorbar(Time, survival_rate['Ketapril'], marker='s', linestyle='--', label='Ketapril')
plt.errorbar(Time, survival_rate['Naftisol'], marker='x', linestyle='--', label='Naftisol')
plt.errorbar(Time, survival_rate['Placebo'], marker='v', linestyle='--', label='Placebo')
plt.errorbar(Time, survival_rate['Propriva'], marker='8', linestyle='--', label='Propriva')
plt.errorbar(Time, survival_rate['Ramicane'], marker='d', linestyle='--', label='Ramicane')
plt.errorbar(Time, survival_rate['Stelasyn'], marker='p', linestyle='--', label='Stelasyn')
plt.errorbar(Time, survival_rate['Zoniferol'], marker='h', linestyle='--', label='Zoniferol')
plt.legend(loc='best')
plt.title('Survival During Treatment')
plt.xlabel('Time (Days)')
plt.ylabel('Survival Rate (%)')
plt.xlim(0,Time.max())
plt.ylim(20,102)
plt.grid()


# # Summary Bar Graph

# In[11]:


# Take the last row divided by the first row less 1 to calculate % Change for each treatment
tumor_change = ((treatment_tumor.iloc[-1]/treatment_tumor.iloc[0]-1)*100).to_frame("% Change")
tumor_change


# In[12]:


# Plot Tumor Change Over 45 Day Treatment
x = tumor_change.index
y = tumor_change["% Change"]

plt.figure(figsize=(12,8))
colors = ["red" if y >=0 else "green" for y in y]

g = sns.barplot(x,y,palette=colors)
plt.title("Tumor Change Over 45 Day Treatment")
plt.ylabel("% Tumor Volume Change")
plt.grid()

# Show data labels
ax = g.axes
for p in ax.patches:
    ax.annotate(s="{:.0f}%".format(p.get_height()), xy=((p.get_x() + p.get_width() / 2., p.get_height()*.9)),
                ha='center', va='center', color='white', xytext=(0, 5), 
                textcoords='offset points', weight='bold')  

