ICOS_ATC_anomaly_detection
![image](https://user-images.githubusercontent.com/45566769/112658884-23907980-8e54-11eb-8560-f9b540ca1043.png)


The file ADA_co2.py can be used for extracting synoptic scale and seasonal anomalies from ICOS CO2 time series. Data from the OPE station are provided, but 
any ICOS station can be used if the user has downloaded the historical data from the ICOS carbon portal and the NRT growing data for the past year. The code will 
concatenate these datasets into one file, and then apply the CCGvu package (Thoning et al., 1989) to extract a harmonic fit to the data, a polynomial function, 
and a smooth curve defined as the polynomial plus a short-term residual filter. The difference between the smooth curve and the harmonic is stored as delta-C.

The delta-C values are then used to defined "envelopes" around the CCGvu harmonic curve. These are defined at each time step (days, in the example) as the standard 
deviation of all delta-C values within 90 (+45/-45) days of the current calendar date over all years in the record (9, in the example). A smooth curve is then fit 
to the daily data, and any segment where the smooth curve is outside the sigma envelope is considered to be a measurement anomaly.

This procedure is performed at 30 and 90 days. For the 30-day curve, anomalies are considered to represent synoptic scale events. For the 90-day curve, they are 
considered to represent seasonal anomalies. Note that the name of the station, the dates, and the paths may need to be changed from those in the example.
