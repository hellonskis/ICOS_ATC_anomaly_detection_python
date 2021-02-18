# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:39:42 2020

@author: aresovsk
"""

import re
import numpy as np
import math
import datetime

##########################################################################################
## Functions to convert ISO-formatted dates to decimal date
## Accounts for negative years (BCE) and for very large dates.
RE_YEARMONTHDAY = re.compile(r'^(\-?\+?)(\d+)\-(\d\d)\-(\d\d)$')

def iso2dec(isodate):
    datepieces = re.match(RE_YEARMONTHDAY, isodate)
    if not datepieces:
        raise ValueError("Invalid date format {}".format(isodate))

    (plusminus, yearstring, monthstring, daystring) = datepieces.groups()
    if not _isvalidmonth(monthstring) or not _isvalidmonthday(yearstring, monthstring, daystring):
        raise ValueError("Invalid date {}".format(isodate))

    decbit = _propotionofdayspassed(yearstring, monthstring, daystring)
    if plusminus == '-':
        decbit = 1 - decbit

    yeardecimal = int(yearstring) + decbit
    if plusminus == '-':
        yeardecimal *= -1

    return round(yeardecimal, 6)


def dec2iso(decdate):
    # strip the integer/year part
    # find how many days were in this year, multiply back out to get the day-of-year number
    if decdate >= 0:
        yearint = int(math.floor(decdate))
        plusminus = ''
    else:
        yearint = int(math.ceil(decdate))
        plusminus = '-'

    yearstring = str(abs(yearint))
    daysinyear = _daysinyear(yearstring)
    targetday = round(daysinyear * (decdate % 1), 1)

    # count up days months at a time, until we reach our target month
    # the the remainder days is the day of the month, offset by 1 cuz we count from 0
    dayspassed = 0
    for monthstring in ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'):
        dtm = _daysinmonth(yearstring, monthstring)
        if dayspassed + dtm < targetday:
            dayspassed += dtm
        else:
            break

    daynumber = int(math.floor(targetday - dayspassed + 1))
    daystring = "{:02d}".format(daynumber)

    return "{}{}-{}-{}".format(plusminus, yearstring, monthstring, daystring)


def _propotionofdayspassed(yearstring, monthstring, daystring):
    # count the number of days to get to this day of this month
    dayspassed = 0
    for tms in ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'):
        if tms < monthstring:
            dayspassed += _daysinmonth(yearstring, tms)
    dayspassed += int(daystring)

    # subtract 1 cuz day 0 is January 1 and not January 0
    # add 0.5 to get us 12 noon
    dayspassed -= 1
    dayspassed += 0.5

    # divide by days in year, to get decimal portion since noon of Jan 1
    daysinyear = _daysinyear(yearstring)
    return dayspassed / daysinyear


def _isvalidmonth(monthstring):
    validmonths = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12')
    return monthstring in validmonths


def _isvalidmonthday(yearstring, monthstring, daystring):
    days = int(daystring)
    return days > 0 and days <= _daysinmonth(yearstring, monthstring)


def _daysinmonth(yearstring, monthstring):
    monthdaycounts = {
        '01': 31,
        '02': 28,  # February
        '03': 31,
        '04': 30,
        '05': 31,
        '06': 30,
        '07': 31,
        '08': 31,
        '09': 30,
        '10': 31,
        '11': 30,
        '12': 31,
    }

    if _isleapyear(yearstring):
        monthdaycounts['02'] = 29

    return monthdaycounts[monthstring]


def _daysinyear(yearstring):
    return 366 if _isleapyear(yearstring) else 365


def _isleapyear(yearstring):
    yearnumber = int(yearstring)
    isleap = yearnumber % 4 == 0 and (yearnumber % 100 != 0 or yearnumber % 400 == 0)
    return isleap

#------------------------------------------------------------       
## Author: Greg Allensworth
## https://github.com/OpenHistoricalMap/decimaldate-python/blob/master/decimaldate.py
    
###########################################################################################
## Function to convert dateTime64/timestamp objects to decimal dates
def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

###########################################################################################
## Function to specify number of significant digits in a numeric object
def to_precision(x,p):
    x = float(x)
    if x == 0.:
        return "0." + "0"*(p-1)
    out = []
    if x < 0:
        out.append("-")
        x = -x
    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)
    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)
    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1
    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1
    m = "%.*g" % (p, n)
    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)
    return "".join(out)

#------------------------------------------------------------   
## Author: Randle Taylor 
## https://github.com/randlet/to-precision/blob/master/to_precision.py
    
###########################################################################################
## A simple implementation of the LOESS algorithm using numpy based on NIST.   

def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y


class Loess(object):

    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n-1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)

#------------------------------------------------------------            
## Author: João Paulo Figueira 
## https://github.com/joaofig/pyloess/blob/master/pyloess/Loess.py 
        
###########################################################################################    
## Function to fill gaps in seasonal and synoptic anomaly plots 

def plot_clean(df, pos, neg, none):
    
    for row in df.iloc[0:(len(df)-1)].itertuples():
        if ((math.isnan(df[pos].iat[row.Index]) & math.isnan(df[none].iat[row.Index + 1]) 
           & math.isnan(df[neg].iat[row.Index + 1])) | 
           (math.isnan(df[pos].iat[row.Index]) & math.isnan(df[none].iat[row.Index - 1]) 
           & math.isnan(df[neg].iat[row.Index - 1]))):
               df[pos].iat[row.Index] = 0
        elif ((math.isnan(df[neg].iat[row.Index]) & math.isnan(df[none].iat[row.Index + 1]) 
             & math.isnan(df[pos].iat[row.Index + 1])) |  
             (math.isnan(df[neg].iat[row.Index]) & math.isnan(df[none].iat[row.Index - 1]) 
             & math.isnan(df[pos].iat[row.Index - 1]))):
               df[neg].iat[row.Index] = 0 
    
#------------------------------------------------------------            
## Author: Alex Resovsky
               
###########################################################################################