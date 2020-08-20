# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:56:32 2020

@author: LZ
"""
####################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as pystats
import statistics
import matplotlib.gridspec as gridspec
from datetime import datetime
df = pd.read_excel('IG Excess Ret Updated V4.xlsx')

print(df.columns)
df.dropna(inplace=True)


#%%
df['Excess Return % 3-mo (Treasury)'] = df['Excess Return % 1-mo (Treasury)'].rolling(3, win_type='boxcar', center=True).sum().shift(-1)
df['Excess Return % 3-mo (Agg)'] = df['Excess Return % 1-mo (Agg)'].rolling(3, win_type='boxcar', center=True).sum().shift(-1)
df['Excess Return % 6-mo (Treasury)'] = df['Excess Return % 1-mo (Treasury)'].rolling(6, win_type='boxcar', center=True).sum().shift(-3)
df['Excess Return % 6-mo (Agg)'] = df['Excess Return % 1-mo (Agg)'].rolling(6, win_type='boxcar', center=True).sum().shift(-3)
df['Excess Return % 9-mo (Treasury)'] = df['Excess Return % 1-mo (Treasury)'].rolling(9, win_type='boxcar', center=True).sum().shift(-5)
df['Excess Return % 9-mo (Agg)'] = df['Excess Return % 1-mo (Agg)'].rolling(9, win_type='boxcar', center=True).sum().shift(-5)
df['Excess Return % 12-mo (Treasury)'] = df['Excess Return % 1-mo (Treasury)'].rolling(12, win_type='boxcar', center=True).sum().shift(-7)
df['Excess Return % 12-mo (Agg)'] = df['Excess Return % 1-mo (Agg)'].rolling(12, win_type='boxcar', center=True).sum().shift(-7)
#%%
df['OAS Decile'] = pd.qcut(df['OAS'], 10, labels=False)
#%%
df=df.sort_values(['OAS Decile','Date'])

#%%
df=df.reset_index(drop=True)
#N = 3
#dtest=df.groupby(df.index // N).sum()

#%% IG OAS - Quad
xl=pd.ExcelFile(r'OAS.xlsx')
sheetnamelist = xl.sheet_names
for i, v in enumerate(sheetnamelist):
    if v == 'OAS':
        OAS = i
    if v == 'Quad':
        Quad = i


        
d1 = xl.sheet_names[OAS]
d2 = xl.sheet_names[Quad]

oas = xl.parse(d1)
oas['YM'] = pd.DatetimeIndex(oas['Date']).year.astype(str)+'-' +pd.DatetimeIndex(oas['Date']).month.astype(str)
oas['YQ'] = pd.PeriodIndex(oas['Date'], freq = 'Q').astype(str)
oas = oas.iloc[:,1:]
#oas.set_index('YM', inplace=True)

quad = xl.parse(d2)
quad['YQ'] = quad['YQ'].astype(str)


oas_withquad = pd.merge(oas, quad, left_on = 'YQ', right_on = 'YQ', how = 'inner')
oas_withquad['YM with quad'] = oas_withquad['YM']+" - "+oas_withquad['Quadrant']
#oas_withquad.set_index('YM with quad', inplace=True)

#%%
oas_withquad = oas_withquad.loc[:,['YM', 'YM with quad', 'Quadrant']]
#%%
df['YM'] = pd.DatetimeIndex(df['Date']).year.astype(str)+'-' +pd.DatetimeIndex(df['Date']).month.astype(str)
#%%
merge = pd.merge(df, oas_withquad, how='left', left_on='YM', right_on='YM')


#%%
def charts(df_3m_T):
    df_3m_T.dropna(inplace=True)
    dec = df_3m_T.columns[-2]
    var = df_3m_T.columns[-1]
    with PdfPages(r'Export IG '+var+'.pdf') as export_pdf:
        # decile summary
        decile_list = pd.Series(np.array(['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5',
               'Decile 6','Decile 7','Decile 8','Decile 9','Decile 10'])).to_frame()
        var = df_3m_T.columns[-1]
        aaa=df.groupby(by='OAS Decile')[var].mean().round(2).to_frame()
        ccc=df.groupby(by='OAS Decile')[var].median().round(2).to_frame()
        bbb=df.groupby(by='OAS Decile')[var].sum().round(2).to_frame()
        pdlist = [decile_list,aaa,ccc,bbb]
        new=pd.concat(pdlist, axis=1)
        new.columns =['OAS DECILE', 'Mean', 'Median', 'SUM']   
        
        new_sort_mean = new.sort_values('Mean').loc[:,['OAS DECILE','Mean']]
        new_sort_median = new.sort_values('Median').loc[:,['OAS DECILE','Median']]
        new_sort_sum = new.sort_values('SUM').loc[:,['OAS DECILE','SUM']]
        
        values_mean = new_sort_mean.values.tolist()
        values_median = new_sort_median.values.tolist()
        values_sum = new_sort_sum.values.tolist()
                
        fig = plt.figure(figsize = (25, 14), dpi = 60)
        gs = fig.add_gridspec(3, 1)
        
        ax_mean = fig.add_subplot(gs[0, :])
        table_col_s = ['OAS DECILE', 'Mean']               
        ax_mean.table(cellText = values_mean, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_mean.axis('off')
        ax_mean.set_title('Decile Ranked by Mean')
        
        ax_median = fig.add_subplot(gs[1, :])
        table_col_s = ['OAS DECILE', 'Median']               
        ax_median.table(cellText = values_median, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_median.axis('off')
        ax_median.set_title('Decile Ranked by Median')
        
        ax_sum = fig.add_subplot(gs[2, :])
        table_col_s = ['OAS DECILE', 'SUM']               
        ax_sum.table(cellText = values_sum, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_sum.axis('off')
        ax_sum.set_title('Decile Ranked by Sum')
        export_pdf.savefig() 
        
        # decile summary by quadrant - quad 1 
        df_3m_T_quad1 = df_3m_T[df_3m_T['Quadrant'] == 'Quadrant 1']
        decile_list1 = pd.Series(np.array(['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5',
               'Decile 6','Decile 7','Decile 8','Decile 9','Decile 10'])).to_frame()
        var = df_3m_T_quad1.columns[-1]
        aaa1=df_3m_T_quad1.groupby(by='OAS Decile')[var].mean().round(2).to_frame()
        ccc1=df_3m_T_quad1.groupby(by='OAS Decile')[var].median().round(2).to_frame()
        bbb1=df_3m_T_quad1.groupby(by='OAS Decile')[var].sum().round(2).to_frame()
        pdlist_1 = [decile_list1,aaa1,ccc1,bbb1]
        new1=pd.concat(pdlist_1, axis=1)
        new1.columns =['OAS DECILE', 'Mean', 'Median', 'SUM']   
        
        new_sort_mean1 = new1.sort_values('Mean').loc[:,['OAS DECILE','Mean']]
        new_sort_median1 = new1.sort_values('Median').loc[:,['OAS DECILE','Median']]
        new_sort_sum1 = new1.sort_values('SUM').loc[:,['OAS DECILE','SUM']]
        
        values_mean1 = new_sort_mean1.values.tolist()
        values_median1 = new_sort_median1.values.tolist()
        values_sum1 = new_sort_sum1.values.tolist()
                
        fig = plt.figure(figsize = (25, 14), dpi = 60)
        gs = fig.add_gridspec(3, 1)
        
        ax_mean = fig.add_subplot(gs[0, :])
        table_col_s = ['OAS DECILE', 'Mean']               
        ax_mean.table(cellText = values_mean1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_mean.axis('off')
        ax_mean.set_title('Decile Ranked by Mean - Quadrant 1')
        
        ax_median = fig.add_subplot(gs[1, :])
        table_col_s = ['OAS DECILE', 'Median']               
        ax_median.table(cellText = values_median1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_median.axis('off')
        ax_median.set_title('Decile Ranked by Median - Quadrant 1')
        
        ax_sum = fig.add_subplot(gs[2, :])
        table_col_s = ['OAS DECILE', 'SUM']               
        ax_sum.table(cellText = values_sum1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_sum.axis('off')
        ax_sum.set_title('Decile Ranked by Sum - Quadrant 1')
        export_pdf.savefig() 
        
        
        # decile summary by quadrant - quad 2
        df_3m_T_quad1 = df_3m_T[df_3m_T['Quadrant'] == 'Quadrant 2']
        decile_list1 = pd.Series(np.array(['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5',
               'Decile 6','Decile 7','Decile 8','Decile 9','Decile 10'])).to_frame()
        var = df_3m_T_quad1.columns[-1]
        aaa1=df_3m_T_quad1.groupby(by='OAS Decile')[var].mean().round(2).to_frame()
        ccc1=df_3m_T_quad1.groupby(by='OAS Decile')[var].median().round(2).to_frame()
        bbb1=df_3m_T_quad1.groupby(by='OAS Decile')[var].sum().round(2).to_frame()
        pdlist_1 = [decile_list1,aaa1,ccc1,bbb1]
        new1=pd.concat(pdlist_1, axis=1)
        new1.columns =['OAS DECILE', 'Mean', 'Median', 'SUM']   
        
        new_sort_mean1 = new1.sort_values('Mean').loc[:,['OAS DECILE','Mean']]
        new_sort_median1 = new1.sort_values('Median').loc[:,['OAS DECILE','Median']]
        new_sort_sum1 = new1.sort_values('SUM').loc[:,['OAS DECILE','SUM']]
        
        values_mean1 = new_sort_mean1.values.tolist()
        values_median1 = new_sort_median1.values.tolist()
        values_sum1 = new_sort_sum1.values.tolist()
                
        fig = plt.figure(figsize = (25, 14), dpi = 60)
        gs = fig.add_gridspec(3, 1)
        
        ax_mean = fig.add_subplot(gs[0, :])
        table_col_s = ['OAS DECILE', 'Mean']               
        ax_mean.table(cellText = values_mean1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_mean.axis('off')
        ax_mean.set_title('Decile Ranked by Mean - Quadrant 2')
        
        ax_median = fig.add_subplot(gs[1, :])
        table_col_s = ['OAS DECILE', 'Median']               
        ax_median.table(cellText = values_median1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_median.axis('off')
        ax_median.set_title('Decile Ranked by Median - Quadrant 2')
        
        ax_sum = fig.add_subplot(gs[2, :])
        table_col_s = ['OAS DECILE', 'SUM']               
        ax_sum.table(cellText = values_sum1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_sum.axis('off')
        ax_sum.set_title('Decile Ranked by Sum - Quadrant 2')
        export_pdf.savefig() 
        
        
        # decile summary by quadrant - quad 3 
        df_3m_T_quad1 = df_3m_T[df_3m_T['Quadrant'] == 'Quadrant 3']
        decile_list1 = pd.Series(np.array(['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5',
               'Decile 6','Decile 7','Decile 8','Decile 9','Decile 10'])).to_frame()
        var = df_3m_T_quad1.columns[-1]
        aaa1=df_3m_T_quad1.groupby(by='OAS Decile')[var].mean().round(2).to_frame()
        ccc1=df_3m_T_quad1.groupby(by='OAS Decile')[var].median().round(2).to_frame()
        bbb1=df_3m_T_quad1.groupby(by='OAS Decile')[var].sum().round(2).to_frame()
        pdlist_1 = [decile_list1,aaa1,ccc1,bbb1]
        new1=pd.concat(pdlist_1, axis=1)
        new1.columns =['OAS DECILE', 'Mean', 'Median', 'SUM']   
        
        new_sort_mean1 = new1.sort_values('Mean').loc[:,['OAS DECILE','Mean']]
        new_sort_median1 = new1.sort_values('Median').loc[:,['OAS DECILE','Median']]
        new_sort_sum1 = new1.sort_values('SUM').loc[:,['OAS DECILE','SUM']]
        
        values_mean1 = new_sort_mean1.values.tolist()
        values_median1 = new_sort_median1.values.tolist()
        values_sum1 = new_sort_sum1.values.tolist()
                
        fig = plt.figure(figsize = (25, 14), dpi = 60)
        gs = fig.add_gridspec(3, 1)
        
        ax_mean = fig.add_subplot(gs[0, :])
        table_col_s = ['OAS DECILE', 'Mean']               
        ax_mean.table(cellText = values_mean1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_mean.axis('off')
        ax_mean.set_title('Decile Ranked by Mean - Quadrant 3')
        
        ax_median = fig.add_subplot(gs[1, :])
        table_col_s = ['OAS DECILE', 'Median']               
        ax_median.table(cellText = values_median1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_median.axis('off')
        ax_median.set_title('Decile Ranked by Median - Quadrant 3')
        
        ax_sum = fig.add_subplot(gs[2, :])
        table_col_s = ['OAS DECILE', 'SUM']               
        ax_sum.table(cellText = values_sum1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_sum.axis('off')
        ax_sum.set_title('Decile Ranked by Sum - Quadrant 3')
        export_pdf.savefig() 
        
        # decile summary by quadrant - quad 4
        df_3m_T_quad1 = df_3m_T[df_3m_T['Quadrant'] == 'Quadrant 4']
        decile_list1 = pd.Series(np.array(['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5',
               'Decile 6','Decile 7','Decile 8','Decile 9','Decile 10'])).to_frame()
        var = df_3m_T_quad1.columns[-1]
        aaa1=df_3m_T_quad1.groupby(by='OAS Decile')[var].mean().round(2).to_frame()
        ccc1=df_3m_T_quad1.groupby(by='OAS Decile')[var].median().round(2).to_frame()
        bbb1=df_3m_T_quad1.groupby(by='OAS Decile')[var].sum().round(2).to_frame()
        pdlist_1 = [decile_list1,aaa1,ccc1,bbb1]
        new1=pd.concat(pdlist_1, axis=1)
        new1.columns =['OAS DECILE', 'Mean', 'Median', 'SUM']   
        
        new_sort_mean1 = new1.sort_values('Mean').loc[:,['OAS DECILE','Mean']]
        new_sort_median1 = new1.sort_values('Median').loc[:,['OAS DECILE','Median']]
        new_sort_sum1 = new1.sort_values('SUM').loc[:,['OAS DECILE','SUM']]
        
        values_mean1 = new_sort_mean1.values.tolist()
        values_median1 = new_sort_median1.values.tolist()
        values_sum1 = new_sort_sum1.values.tolist()
                
        fig = plt.figure(figsize = (25, 14), dpi = 60)
        gs = fig.add_gridspec(3, 1)
        
        ax_mean = fig.add_subplot(gs[0, :])
        table_col_s = ['OAS DECILE', 'Mean']               
        ax_mean.table(cellText = values_mean1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_mean.axis('off')
        ax_mean.set_title('Decile Ranked by Mean - Quadrant 4')
        
        ax_median = fig.add_subplot(gs[1, :])
        table_col_s = ['OAS DECILE', 'Median']               
        ax_median.table(cellText = values_median1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_median.axis('off')
        ax_median.set_title('Decile Ranked by Median - Quadrant 4')
        
        ax_sum = fig.add_subplot(gs[2, :])
        table_col_s = ['OAS DECILE', 'SUM']               
        ax_sum.table(cellText = values_sum1, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
        ax_sum.axis('off')
        ax_sum.set_title('Decile Ranked by Sum - Quadrant 4')
        export_pdf.savefig() 
        
        for length,i in enumerate(df_3m_T[dec].unique().tolist()):
            df_temp = df_3m_T[df_3m_T[dec] == i]
            # for each decile
            fig = plt.figure(figsize = (30, 25), dpi = 60)
            gs = fig.add_gridspec(2, 4)
            ax1 = fig.add_subplot(gs[:-1, :])
            
            tb=df_temp.iloc[:,-1]

        # Calculation for the statistics table
            all_mean = statistics.mean(tb)
            all_standard_error = pystats.sem(tb)
            all_median = statistics.median(tb)
            all_standard_deviation = statistics.stdev(tb)
            all_sample_variance = statistics.variance(tb)
            all_kurtosis = pystats.kurtosis(tb)
            all_skewness = pystats.skew(tb)
            all_min = min(tb)
            all_max = max(tb)
            all_range = all_max - all_min
            all_sum = sum(tb)
            all_count = len(tb)
        
            # Construct the Stats table
            table_data = [all_mean, all_standard_error, all_median,
                          all_standard_deviation, all_sample_variance, all_kurtosis, all_skewness,
                          all_range, all_min, all_max, all_sum, all_count]
            for j in range(len(table_data)): # Round the results to 4 decimals
                table_data[j] = round(table_data[j], 4)
            table_index = ['Mean', 'Standard Error', 'Median', 'Standard Deviation', 
                           'Sample Variance', 'Kurtosis', 'Skewness', 'Range', 'Minimum', 'Maximum', 'Sum', 
                           'Count']
            table_values = []
            for j in range(len(table_data)):
                temp_list = []
                temp_list.append(table_index[j])
                temp_list.append(table_data[j])
                table_values.append(temp_list)
            table_col_s = ['Statistics', 'Value']


            #print(df_temp)
            df_temp.plot(ax= ax1, kind='bar', x='YM with quad', y=var)
            ax1.set_xlabel("")
            length = length +1
            ax1.set_title(('OAS Decile- {}').format(length))
            ax1.grid()
            min1=df_temp[var].values.min()-0.01
            max1=df_temp[var].max()+0.01
            bin1=(max1-min1)/25
            plt.yticks(np.arange(min1, max1, step=bin1))
            plt.xticks(rotation=90)
            
            #Bbox
            ax2 = fig.add_subplot(gs[-1, 0])
            df_temp.boxplot(ax= ax2, column=var)
            
            #Statistics
            ax3 = fig.add_subplot(gs[-1, 1])
            ax3.table(cellText = table_values, colLabels = table_col_s, cellLoc = 'left', loc = 'center')
            ax3.axis('off')
            
            # Quadrant Count
            ax4 = fig.add_subplot(gs[-1, 2:])
            testt=df_temp['Quadrant'].value_counts().to_frame()
            testt1=df_temp.groupby(['Quadrant'])[var].mean().round(2)
            testt2=df_temp.groupby(['Quadrant'])[var].median().round(2)
            testt3=df_temp.groupby(['Quadrant'])[var].sum().round(2)
            pdlist2 = [testt,testt1,testt2,testt3]
            new2=pd.concat(pdlist2, axis=1)
            quadrantinfo = ['Quadrant 1', 'Quadrant 2', 'Quadrant 3', 'Quadrant 4']
            new2.insert (0, "Quadrant Info", quadrantinfo)
            new2_1 = new2.values.tolist()    
            new2columns =['Quadrant','Count', 'Mean', 'Median', 'SUM']  
            ax4.table(cellText = new2_1, colLabels = new2columns, cellLoc = 'left', loc = 'center')
            ax4.axis('off')
            ax4.set_title('')

            export_pdf.savefig()  
            plt.close()
        

#%%
df_3m_T = merge[['YM with quad', 'Quadrant','OAS Decile','Excess Return % 3-mo (Treasury)']]
df_3m_agg = merge[['YM with quad', 'Quadrant','OAS Decile','Excess Return % 3-mo (Agg)']]
df_6m_T = merge[['YM with quad', 'Quadrant','OAS Decile','Excess Return % 6-mo (Treasury)']]
df_6m_agg = merge[['YM with quad', 'Quadrant','OAS Decile','Excess Return % 6-mo (Agg)']]
df_9m_T = merge[['YM with quad', 'Quadrant','OAS Decile','Excess Return % 9-mo (Treasury)']]
df_9m_agg = merge[['YM with quad', 'Quadrant','OAS Decile','Excess Return % 9-mo (Agg)']]
df_12m_T = merge[['YM with quad', 'Quadrant','OAS Decile','Excess Return % 12-mo (Treasury)']]
df_12m_agg = merge[['YM with quad', 'Quadrant','OAS Decile','Excess Return % 12-mo (Agg)']]

#%%
charts(df_3m_T)
#%%
charts(df_3m_agg)
charts(df_6m_T)
charts(df_6m_agg)
#%%
charts(df_9m_T)
#%%
charts(df_9m_agg)
#%%
charts(df_12m_T)
#%%
charts(df_12m_agg)
#%% 
# =============================================================================
# OAS Deciel distribution by Quadrant
# =============================================================================
merge1 = merge[['OAS', 'YM', 'YM with quad', 'Quadrant']]
merge1['Decile'] = pd.qcut(merge1['OAS'], 10, labels=False)
merge1['Decile'] = merge1['Decile']+1
merge1['Year'] = merge1['YM'].str[:4]
merge1['Month'] = merge1['YM'].str[5:].astype(int)
merge1.sort_values(['Year', 'Month'], inplace=True)
merge1.set_index('YM with quad',inplace=True)

#%%
merge1_1 = merge1.loc[:][['OAS', 'Quadrant', 'Decile']]
merge1_1['Decile'] = 'Decile ' + merge1_1['Decile'].astype(str)
merge1_1.dropna(inplace = True)
#%% Quadrant Distribution
quad_info = merge1_1['Quadrant'].value_counts().to_frame()
quad_info['sum'] = merge1_1['Quadrant'].value_counts().sum()
quad_info['%'] = ((quad_info['Quadrant']/quad_info['sum'])*100).round(2).astype(str)+'%'
quad_info.reset_index(inplace = True)
quad_info.columns = ['Quadrant', 'Count', 'Sum', '%']
quad_info1 = quad_info.values.tolist()

#%% Decile Distribution within Each Quadrant
merge1_1_1 = merge1_1.loc[:,['Quadrant', 'Decile']]
merge1_1_1['Quad number'] = merge1_1['Quadrant'].str[9:].astype(int)
merge1_1_1['Decile number'] = merge1_1['Decile'].str[7:].astype(int)
merge1_1_1.sort_values(['Quad number', 'Decile number'], inplace = True)


#%% quadrant 1
merge1_1_1_quad1 = merge1_1_1[merge1_1_1['Quadrant'] == 'Quadrant 1']
merge1_1_1_quad1_decile = merge1_1_1_quad1['Decile'].value_counts().to_frame()
merge1_1_1_quad1_decile['sum'] = merge1_1_1_quad1['Decile'].value_counts().sum()
merge1_1_1_quad1_decile['%'] = ((merge1_1_1_quad1_decile['Decile']/merge1_1_1_quad1_decile['sum'])*100).round(2).astype(str)+'%'
merge1_1_1_quad1_decile.reset_index(inplace = True)
merge1_1_1_quad1_decile.columns = ['Decile', 'Count', 'Sum', '%']
merge1_1_1_quad11_decile = merge1_1_1_quad1_decile.values.tolist()


#%% quadrant 2
merge1_1_1_quad2 = merge1_1_1[merge1_1_1['Quadrant'] == 'Quadrant 2']
merge1_1_1_quad2_decile = merge1_1_1_quad2['Decile'].value_counts().to_frame()
merge1_1_1_quad2_decile['sum'] = merge1_1_1_quad2['Decile'].value_counts().sum()
merge1_1_1_quad2_decile['%'] = ((merge1_1_1_quad2_decile['Decile']/merge1_1_1_quad2_decile['sum'])*100).round(2).astype(str)+'%'
merge1_1_1_quad2_decile.reset_index(inplace = True)
merge1_1_1_quad2_decile.columns = ['Decile', 'Count', 'Sum', '%']
merge1_1_1_quad21_decile = merge1_1_1_quad2_decile.values.tolist()

#%% Quadrant 3
merge1_1_1_quad3 = merge1_1_1[merge1_1_1['Quadrant'] == 'Quadrant 3']
merge1_1_1_quad3_decile = merge1_1_1_quad3['Decile'].value_counts().to_frame()
merge1_1_1_quad3_decile['sum'] = merge1_1_1_quad3['Decile'].value_counts().sum()
merge1_1_1_quad3_decile['%'] = ((merge1_1_1_quad3_decile['Decile']/merge1_1_1_quad3_decile['sum'])*100).round(2).astype(str)+'%'
merge1_1_1_quad3_decile.reset_index(inplace = True)
merge1_1_1_quad3_decile.columns = ['Decile', 'Count', 'Sum', '%']
merge1_1_1_quad31_decile = merge1_1_1_quad3_decile.values.tolist()

#%% Quadrant 4
merge1_1_1_quad4 = merge1_1_1[merge1_1_1['Quadrant'] == 'Quadrant 4']
merge1_1_1_quad4_decile = merge1_1_1_quad4['Decile'].value_counts().to_frame()
merge1_1_1_quad4_decile['sum'] = merge1_1_1_quad4['Decile'].value_counts().sum()
merge1_1_1_quad4_decile['%'] = ((merge1_1_1_quad4_decile['Decile']/merge1_1_1_quad4_decile['sum'])*100).round(2).astype(str)+'%'
merge1_1_1_quad4_decile.reset_index(inplace = True)
merge1_1_1_quad4_decile.columns = ['Decile', 'Count', 'Sum', '%']
merge1_1_1_quad41_decile = merge1_1_1_quad4_decile.values.tolist()

#%%
with PdfPages(r'IG OAS - All Decile by Quandrant.pdf') as export_pdf:
    fig = plt.figure(figsize = (55, 18), dpi = 60)
    gs = fig.add_gridspec(1, 4)
    ax = fig.add_subplot(gs[:2, :])
    colors = {'Decile 1': '#FF000C', 'Decile 2': '#fc5e02', 'Decile 3': '#fa9e03', 'Decile 4': '#fddb6d', 'Decile 5': '#4b8bbe',
              'Decile 6': '#34a853', 'Decile 7': '#17bcb1', 'Decile 8': '#22a8f0', 'Decile 9': '#195eca', 'Decile 10': '#f084bc'}
    merge1_1.plot(use_index=True, y='OAS', ax = ax, kind='bar',
                                 color= merge1_1['Decile'].map(colors))
    
    ax.set_title('IG OAS - All Decile by Quandrant - All'.format(i))
    ax.set_xlabel('')
    #ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid()
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    ax.legend(handles, labels, bbox_to_anchor=(0.58, 0.5, 0.5, 0.5))
    export_pdf.savefig()
    
    # Qudrant
    fig = plt.figure(figsize = (55, 18), dpi = 60)
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[:, :])
    the_table = ax1.table(cellText = quad_info1, colLabels = quad_info.columns, cellLoc = 'left', loc = 'center', colWidths = [0.1,0.1,0.1,0.1], fontsize=35)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(30)
    the_table.scale(3, 3)
    ax1.title.set_text('')
    ax1.axis('off')
    export_pdf.savefig()
    # ------------------
    
    # Decile  
    ## Quad 1 
    fig = plt.figure(figsize = (55, 18), dpi = 60)
    gs = fig.add_gridspec(1, 4)
    ax1 = fig.add_subplot(gs[:, :1])
    the_table = ax1.table(cellText = merge1_1_1_quad11_decile, colLabels = merge1_1_1_quad1_decile.columns, cellLoc = 'left', loc = 'center', colWidths = [0.08,0.08,0.08,0.08], fontsize=35)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(30)
    the_table.scale(3, 3)
    ax1.set_title('Quadrant 1',fontsize= 30)
    ax1.axis('off')
    
    ## Quad 2
    ax2 = fig.add_subplot(gs[:, 1:2])
    the_table = ax2.table(cellText = merge1_1_1_quad21_decile, colLabels = merge1_1_1_quad2_decile.columns, cellLoc = 'left', loc = 'center', colWidths = [0.08,0.08,0.08,0.08], fontsize=35)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(30)
    the_table.scale(3, 3)
    ax2.set_title('Quadrant 2',fontsize= 30)
    ax2.axis('off')

    ## Quad 3
    ax3 = fig.add_subplot(gs[:, 2:3])
    the_table = ax3.table(cellText = merge1_1_1_quad31_decile, colLabels = merge1_1_1_quad3_decile.columns, cellLoc = 'left', loc = 'center', colWidths = [0.08,0.08,0.08,0.08], fontsize=35)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(30)
    the_table.scale(3, 3)
    ax3.set_title('Quadrant 3',fontsize= 30)
    ax3.axis('off')

    ## Quad 4
    ax4 = fig.add_subplot(gs[:, 3:])
    the_table = ax4.table(cellText = merge1_1_1_quad41_decile, colLabels = merge1_1_1_quad4_decile.columns, cellLoc = 'left', loc = 'center', colWidths = [0.08,0.08,0.08,0.08], fontsize=35)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(30)
    the_table.scale(3, 3)
    ax4.set_title('Quadrant 4',fontsize= 30)
    ax4.axis('off')
    export_pdf.savefig()




#%%
# =============================================================================
#  Summary page for forecasting - OAS
# =============================================================================
summary = merge.loc[:,['Date', 'OAS','Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)', 
                       'OAS Decile', 'YM', 'YM with quad','Quadrant']]



#%%
summary_mean = summary.groupby('OAS Decile')['Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)'].mean().round(2)



summary_median = summary.groupby('OAS Decile')['Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)'].median().round(2)

summary_sum = summary.groupby('OAS Decile')['Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)'].sum().round(2)

summary_now = summary.sort_values('Date')[['OAS Decile', 'Quadrant']].iloc[-1, :]

#%%
summary_mean_his = summary.loc[:,['Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                                  'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                                  'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                                  'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)']].mean().round(2).to_frame()
summary_median_his = summary.loc[:,['Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                                    'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                                    'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                                    'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)']].median().round(2).to_frame()
summary_sum_his = summary.loc[:,['Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                                 'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                                 'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                                 'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)']].sum().round(2).to_frame()





#%%
summary_mean_his.reset_index(inplace = True)
summary_mean_his.columns = ['Name', 'Mean Return']
summary_mean_his1 = summary_mean_his.values.tolist()
#%%
summary_median_his.reset_index(inplace = True)
summary_median_his.columns = ['Name', 'Median Return']
summary_median_his1 = summary_median_his.values.tolist()
#%%
summary_sum_his.reset_index(inplace = True)
summary_sum_his.columns = ['Name', 'Sum Return']
summary_sum_his1 = summary_sum_his.values.tolist()
#%%
sum_result1 = pd.merge(summary_mean_his, summary_median_his, how = 'inner', left_on = 'Name', right_on = 'Name')
sum_result = pd.merge(sum_result1, summary_sum_his, how = 'inner', left_on = 'Name', right_on = 'Name')
sum_result.columns = ['Name', 'Mean Return', 'Median', 'Sum']
sum_result2 = sum_result.values.tolist()

#%% the acutal one should be +1
summary_now_1 = summary_now[['OAS Decile']][0]
summary_now_quad = summary_now[['Quadrant']][0]
#%%
decilename = summary_now_1 + 1
#%%

summary_mean_1 = summary_mean.iloc[['{}'.format(summary_now_1)]]
summary_median_1 = summary_median.iloc[['{}'.format(summary_now_1)]]
summary_sum_1 = summary_sum.iloc[['{}'.format(summary_now_1)]]

#%%
summary_mean_2 = summary_mean.iloc[:,:]
for v in summary_mean.columns:
    summary_mean_2[v] = summary_mean[v].rank(method='min', ascending=False)
summary_mean_2 = summary_mean_2.iloc[['{}'.format(summary_now_1)]]
#%%
summary_median_2 = summary_median.iloc[:,:]
for v in summary_median.columns:
    summary_median_2[v] = summary_median[v].rank(method='min', ascending=False)
summary_median_2 = summary_median_2.iloc[['{}'.format(summary_now_1)]]
#%%
summary_sum_2 = summary_sum.iloc[:,:]
for v in summary_sum.columns:
    summary_sum_2[v] = summary_sum[v].rank(method='min', ascending=False)
summary_sum_2 = summary_sum_2.iloc[['{}'.format(summary_now_1)]]

#%%
frames = [summary_mean_1, summary_mean_2, summary_median_1, summary_median_2, summary_sum_1, summary_sum_2]
result = pd.concat(frames)

#%%
index_list = ['Mean', 'Mean Rank', 'Median', 'Median Rank', 'Sum', 'Sum Rank']
result.index = index_list
#%%
result.reset_index(inplace = True)
#%%
result.columns = ['Measure','Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)']
#%%
result_1 = result.values.tolist()

#%%
# =============================================================================
#  Sub-Summary page for forecasting
# =============================================================================
summary = merge.loc[:,['Date', 'OAS','Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)', 
                       'OAS Decile', 'YM', 'YM with quad','Quadrant']]
#%%
summary = summary[summary['Quadrant'] == 'Quadrant 4']
#%%
subsummary_mean = summary.groupby(['OAS Decile'])['Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)'].mean().round(2)

subsummary_median = summary.groupby(['OAS Decile'])['Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)'].median().round(2)

subsummary_sum = summary.groupby(['OAS Decile'])['Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)'].sum().round(2)

#%%

subsummary_mean_1 = subsummary_mean.iloc[['{}'.format(summary_now_1)]]
subsummary_median_1 = subsummary_median.iloc[['{}'.format(summary_now_1)]]
subsummary_sum_1 = subsummary_sum.iloc[['{}'.format(summary_now_1)]]
#%%
subsummary_mean_2 = subsummary_mean.iloc[:,:]
for v in subsummary_mean.columns:
    subsummary_mean_2[v] = subsummary_mean[v].rank(method='min', ascending=False).astype(int)
subsummary_mean_2 = subsummary_mean_2.iloc[['{}'.format(summary_now_1)]]

#%%
subsummary_median_2 = subsummary_median.iloc[:,:]
for v in subsummary_median.columns:
    subsummary_median_2[v] = subsummary_median[v].rank(method='min', ascending=False).astype(int)
subsummary_median_2 = subsummary_median_2.iloc[['{}'.format(summary_now_1)]]
#%%
subsummary_sum_2 = subsummary_sum.iloc[:,:]
for v in subsummary_sum.columns:
    subsummary_sum_2[v] = subsummary_sum[v].rank(method='min', ascending=False).astype(int)
subsummary_sum_2 = subsummary_sum_2.iloc[['{}'.format(summary_now_1)]]

#%%
#tuple_dec_quad = (summary_now_1, 'Quadrant 4')
#for num, indexname in enumerate(subsummary_mean.index):
#    if tuple_dec_quad == indexname:
#        subsummary_mean_1 = subsummary_mean.iloc[[num]]

#%%
frames2 = [subsummary_mean_1, subsummary_mean_2, subsummary_median_1, subsummary_median_2, subsummary_sum_1, subsummary_sum_2]
result2 = pd.concat(frames2)
#%%
index_list2 = ['Mean', 'Mean Rank', 'Median', 'Median Rank', 'Sum', 'Sum Rank']
result2.index = index_list2
#%%
result2.reset_index(inplace = True)
#%%
result2.columns = ['Measure','Excess Return % 3-mo (Agg)', 'Excess Return % 3-mo (Treasury)',
                       'Excess Return % 6-mo (Treasury)','Excess Return % 6-mo (Agg)', 
                       'Excess Return % 9-mo (Treasury)','Excess Return % 9-mo (Agg)', 
                       'Excess Return % 12-mo (Treasury)','Excess Return % 12-mo (Agg)']
#%%
result2_1 = result2.values.tolist()

#%%
with PdfPages(r'IG Summary under OAS Decile and Quadrant.pdf') as export_pdf:
    fig = plt.figure(figsize = (82, 40), dpi = 60)
    gs = fig.add_gridspec(5, 1)
    ax1 = fig.add_subplot(gs[:2, :])
    the_table = ax1.table(cellText = sum_result2, colLabels = sum_result.columns, cellLoc = 'left', loc = 'center', colWidths = [0.05,0.04,0.04,0.04], fontsize=35)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(30)
    the_table.scale(3, 3)
    ax1.set_title('Summary for IG', fontsize = 30)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[2:4, :])
    the_table2 = ax2.table(cellText = result_1, colLabels = result.columns, cellLoc = 'left', loc = 'center', colWidths = [0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04], fontsize=35)
    the_table2.auto_set_font_size(False)
    the_table2.set_fontsize(30)
    the_table2.scale(3, 3)
    ax2.set_title('Currrent Decile - {}'.format(decilename), fontsize = 30)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[4:, :])
    the_table3 = ax3.table(cellText = result2_1, colLabels = result2.columns, cellLoc = 'left', loc = 'center', colWidths = [0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04], fontsize=35)
    the_table3.auto_set_font_size(False)
    the_table3.set_fontsize(30)
    the_table3.scale(3, 3)
    ax3.set_title('Currrent Decile - {} - Quadrant 4'.format(decilename), fontsize = 30)
    ax3.axis('off')
    
    export_pdf.savefig()


