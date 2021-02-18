# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:53:52 2020

@author: aresovsk
"""

import numpy as np

def seasonal_sigma(df, ndays, dt_col):
    
    sd = np.empty([0, 1])
    rs = np.empty([0, 1])    
    for i in range((len(df)-ndays),len(df)):
        i_percentage(i, df, ndays)
        if (df[dt_col].iloc[i].timetuple().tm_yday >= 46 and 
                df[dt_col].iloc[i].timetuple().tm_yday <= 320): 
            for j in range(0,len(df)):
                if (df[dt_col].iloc[j].timetuple().tm_yday >= (df[dt_col].iloc[i].timetuple().tm_yday - 45) and 
                        df[dt_col].iloc[j].timetuple().tm_yday <= (df[dt_col].iloc[i].timetuple().tm_yday + 45)):
                    rs = np.append(rs, df['dC'].iloc[j])
            std = np.std(rs)
            sd = np.append(sd, std)
            rs = np.empty([0, 1])
        elif df[dt_col].iloc[i].timetuple().tm_yday < 46:
            for j in range(0,len(df)):
                if (df[dt_col].iloc[j].timetuple().tm_yday <= df[dt_col].iloc[i].timetuple().tm_yday or 
                        df[dt_col].iloc[j].timetuple().tm_yday >= (df[dt_col].iloc[i].timetuple().tm_yday + 320) or
                        df[dt_col].iloc[j].timetuple().tm_yday <= (df[dt_col].iloc[i].timetuple().tm_yday + 45)):
                    rs = np.append(rs, df['dC'].iloc[j])
            std = np.std(rs)
            sd = np.append(sd, std)
            rs = np.empty([0, 1])
        elif df[dt_col].iloc[i].timetuple().tm_yday > 320:
            for j in range(0,len(df)):
                if (df[dt_col].iloc[j].timetuple().tm_yday >= df[dt_col].iloc[i].timetuple().tm_yday or 
                        df[dt_col].iloc[j].timetuple().tm_yday <= (df[dt_col].iloc[i].timetuple().tm_yday + 320) or
                        df[dt_col].iloc[j].timetuple().tm_yday >= (df[dt_col].iloc[i].timetuple().tm_yday - 45)):
                    rs = np.append(rs, df['dC'].iloc[j])
            std = np.std(rs)
            sd = np.append(sd, std)
            rs = np.empty([0, 1])
  
    return(sd)
    
    
def seasonal_sigma_noleap(df, ndays, dt_col):
    
    sd = np.empty([0, 1])
    rs = np.empty([0, 1])  
    assign_sd = False
    for i in range((len(df)-ndays),len(df)):
        year_percentage(i, df, ndays)
        if i == (len(df)-ndays):
            month = df[dt_col].iloc[i].month
            day = df[dt_col].iloc[i].day
        if ((df[dt_col].iloc[i].month == month) and (df[dt_col].iloc[i].day == day) and 
            (i != (len(df)-ndays))):
            assign_sd = True
            print ("1 year complete, assigning sigma values")
            for j in range(i, len(df)):
                std = sd[j-365]
                sd = np.append(sd, std)
        if bool(assign_sd):
          print("done")
          break
        if (df[dt_col].iloc[i].timetuple().tm_yday >= 46 and 
            df[dt_col].iloc[i].timetuple().tm_yday <= 320): 
            for j in range(0,len(df)):
                if (df[dt_col].iloc[j].timetuple().tm_yday >= (df[dt_col].iloc[i].timetuple().tm_yday - 45) and 
                        df[dt_col].iloc[j].timetuple().tm_yday <= (df[dt_col].iloc[i].timetuple().tm_yday + 45)):
                    rs = np.append(rs, df['dC'].iloc[j])
            std = np.std(rs)
            sd = np.append(sd, std)
            rs = np.empty([0, 1])
        elif df[dt_col].iloc[i].timetuple().tm_yday < 46:
            for j in range(0,len(df)):
                if (df[dt_col].iloc[j].timetuple().tm_yday <= df[dt_col].iloc[i].timetuple().tm_yday or 
                        df[dt_col].iloc[j].timetuple().tm_yday >= (df[dt_col].iloc[i].timetuple().tm_yday + 320) or
                        df[dt_col].iloc[j].timetuple().tm_yday <= (df[dt_col].iloc[i].timetuple().tm_yday + 45)):
                    rs = np.append(rs, df['dC'].iloc[j])
            std = np.std(rs)
            sd = np.append(sd, std)
            rs = np.empty([0, 1])
        elif df[dt_col].iloc[i].timetuple().tm_yday > 320:
            for j in range(0,len(df)):
                if (df[dt_col].iloc[j].timetuple().tm_yday >= df[dt_col].iloc[i].timetuple().tm_yday or 
                        df[dt_col].iloc[j].timetuple().tm_yday <= (df[dt_col].iloc[i].timetuple().tm_yday + 320) or
                        df[dt_col].iloc[j].timetuple().tm_yday >= (df[dt_col].iloc[i].timetuple().tm_yday - 45)):
                    rs = np.append(rs, df['dC'].iloc[j])
            std = np.std(rs)
            sd = np.append(sd, std)
            rs = np.empty([0, 1])
  
    return(sd)    
    
    
#--------------------------------------------------
    
def i_percentage(i, df, ndays):
    percentage = np.linspace(5, 100, 20)
    for p in percentage:
        if i - (len(df) - ndays) == round(ndays/100 * p):
            print(p, "% complete")
            
            
#--------------------------------------------------
    
def year_percentage(i, df, ndays):   
    percentage = np.linspace(10, 100, 10)   
    for p in percentage:
        if ((round((i/365)*100) == p) and (round(((i-1)/365)*100) == (p-1))):
            print(p, "% complete")
                     
             
    
