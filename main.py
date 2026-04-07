import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

MAIN_FILE="main.xlsx"
ADDITIONAL_FILE="additional.csv"
SHEET_NAME="Update 2024 (V6.1)"

def clean_main_city_name(city_name): #Cleans inconsistent naming
    if pd.isna(city_name):
        return np.nan
    city_name=str(city_name)
    city_name=city_name.split("/")[0]
    city_name=re.sub(r"\s+[A-Z]{2}$","",city_name)
    return city_name.strip()

def annotate_all_points(df,x_col,y_col,label_col): #Annotate city names on plots
    for i,row in df.iterrows():
        plt.annotate(row[label_col],(row[x_col],row[y_col]),fontsize=8)

def station_group(station_type): #Clean inconsistent station types
    if pd.isna(station_type):
        return np.nan
    station_type=str(station_type).lower()
    if "suburban" in station_type:
        return "Suburban"
    if "urban" in station_type:
        return "Urban"
    if "rural" in station_type:
        return "Rural"
    return np.nan

def add_trendline(x,y): #Add a trendline to plots
    x=pd.Series(x).astype(float)
    y=pd.Series(y).astype(float)
    valid=pd.concat([x,y],axis=1).dropna()
    slope,intercept,i,j,k_=stats.linregress(valid.iloc[:,0],valid.iloc[:,1])
    x_line=np.linspace(valid.iloc[:,0].min(),valid.iloc[:,0].max(),200)
    y_line=slope*x_line+intercept
    plt.plot(x_line,y_line,color="red",label="Trendline")

def pearson_r(x,y): #Calculate Pearson correlation
    x=pd.Series(x).astype(float)
    y=pd.Series(y).astype(float)
    valid=pd.concat([x,y],axis=1).dropna()
    return stats.pearsonr(valid.iloc[:,0],valid.iloc[:,1])[0]

def scipy_skew(series): #Returns unbiased estimator
    series=pd.Series(series).dropna()
    return stats.skew(series,bias=False)

def zscore_series(series): #Returns a z score series, for calculating the combined relative index
    series=pd.Series(series).astype(float)
    return pd.Series(stats.zscore(series,nan_policy="omit"),index=series.index)

def summarize_with_ci(df,group_col,value_col,order): #Returns the confidence interval
    summary=df.groupby(group_col)[value_col].agg(["mean","std","count"]).reset_index()
    summary=summary[summary[group_col].isin(order)].copy()
    summary[group_col]=pd.Categorical(summary[group_col],categories=order,ordered=True)
    summary=summary.sort_values(group_col)
    summary["sem"]=summary["std"]/np.sqrt(summary["count"])
    summary["ci95"]=summary.apply(lambda row:stats.t.ppf(0.975,row["count"]-1)*row["sem"],axis=1)
    return summary

#Reading
main_raw=pd.read_excel(MAIN_FILE,sheet_name=SHEET_NAME)
additional_raw=pd.read_csv(ADDITIONAL_FILE)

#Clean main dataset
main_us_2020=main_raw[(main_raw["country_name"]=="United States of America")&(main_raw["year"]==2020)].copy()
main_us_2020["city_clean"]=main_us_2020["city"].apply(clean_main_city_name)

city_air=main_us_2020.groupby("city_clean")[["pm25_concentration","pm10_concentration","no2_concentration","population","latitude","longitude"]].mean().reset_index()
city_air=city_air.rename(columns={"pm25_concentration":"pm25_mean","pm10_concentration":"pm10_mean","no2_concentration":"no2_mean"})

station_info=main_us_2020.groupby("city_clean")["type_of_stations"].first().reset_index()
city_air=city_air.merge(station_info,on="city_clean",how="left")
city_air["station_group"]=city_air["type_of_stations"].apply(station_group)

#Clean additional dataset
health_total=additional_raw[(additional_raw["geo_level"]=="city")&(additional_raw["group_name"]=="Total")].copy()

health_total=health_total[["geo_name","metric_name","est"]]
health_total=health_total.drop_duplicates(subset=["geo_name","metric_name"])

health_city=health_total.pivot(index="geo_name",columns="metric_name",values="est").reset_index()
health_city.columns.name=None

health_city=health_city.rename(columns={"geo_name":"city_clean","Cardiovascular Disease Deaths":"cvd_deaths","Life Expectancy - City-Level":"life_expectancy"})

#Merge
merged=city_air.merge(health_city,on="city_clean",how="inner")
analysis_df=merged.copy()

#Statistics
city_air_cols=["pm25_mean","pm10_mean","no2_mean","population","latitude","longitude"]
health_cols=["life_expectancy","cvd_deaths"]
merged_cols=["pm25_mean","pm10_mean","no2_mean","population","latitude","longitude","life_expectancy","cvd_deaths"]

analysis_stats=pd.DataFrame({"mean":analysis_df[merged_cols].mean(),"std":analysis_df[merged_cols].std(),"min":analysis_df[merged_cols].min(),"max":analysis_df[merged_cols].max(),"skew":analysis_df[merged_cols].apply(scipy_skew)})

print("\nSummary statistics for analysis_df:")
print(analysis_stats)
correlation_matrix=analysis_df[merged_cols].corr()
print("\nCorrelation matrix for analysis_df:")
print(correlation_matrix)

#Uncomment this to render the plots in lightmode
#plt.rcParams['figure.facecolor'] = 'white'

# Plot 1 cpm vs life
plot_df_1=analysis_df[["pm25_mean","life_expectancy","population","city_clean"]].dropna().copy()
x1=plot_df_1["pm25_mean"]
y1=plot_df_1["life_expectancy"]
r1=pearson_r(x1,y1)
sizes1=np.sqrt(plot_df_1["population"])/np.sqrt(plot_df_1["population"]).max()*1000+30
plt.figure()
plt.scatter(x1,y1,s=sizes1,alpha=0.7,label="Cities")
add_trendline(x1,y1)
annotate_all_points(plot_df_1,"pm25_mean","life_expectancy","city_clean")
plt.title(f"Plot 1: PM2.5 vs Life Expectancy (bubble size = population, r = {r1:.3f})")
plt.xlabel("Average PM2.5 concentration in 2020")
plt.ylabel("Life Expectancy")
plt.legend()
plt.grid()
plt.savefig("plot1_pm25_vs_life_expectancy.png")
plt.show()

# Plot 2 no2 vs life
plot_df_2=analysis_df[["no2_mean","life_expectancy","population","city_clean"]].dropna().copy()
x2=plot_df_2["no2_mean"]
y2=plot_df_2["life_expectancy"]
r2=pearson_r(x2,y2)
sizes2=np.sqrt(plot_df_2["population"])/np.sqrt(plot_df_2["population"]).max()*1000+30
plt.figure()
plt.scatter(x2,y2,s=sizes2,alpha=0.7,label="Cities")
add_trendline(x2,y2)
annotate_all_points(plot_df_2,"no2_mean","life_expectancy","city_clean")
plt.title(f"Plot 2: NO2 vs Life Expectancy (bubble size = population, r = {r2:.3f})")
plt.xlabel("Average NO2 concentration in 2020")
plt.ylabel("Life Expectancy")
plt.legend()
plt.grid()
plt.savefig("plot2_no2_vs_life_expectancy.png")
plt.show()

# Plot 3 pm vs life
plot_df_3=analysis_df[["pm10_mean","life_expectancy","population"]].dropna().copy()
x3=plot_df_3["pm10_mean"]
y3=plot_df_3["life_expectancy"]
r3=pearson_r(x3,y3)
sizes3=np.sqrt(plot_df_3["population"])/np.sqrt(plot_df_3["population"]).max()*1000+30
plt.figure()
plt.scatter(x3,y3,s=sizes3,alpha=0.7,label="Cities")
add_trendline(x3,y3)
plt.title(f"Plot 3: PM10 vs Life Expectancy (bubble size = population, r = {r3:.3f})")
plt.xlabel("Average PM10 concentration in 2020")
plt.ylabel("Life Expectancy")
plt.legend()
plt.grid()
plt.savefig("plot3_pm10_vs_life_expectancy.png")
plt.show()

# Plot 4 pm2.5 vs death and cities are labeled
plot_df_4=analysis_df[["pm25_mean","cvd_deaths","population","city_clean"]].dropna().copy()
x4=plot_df_4["pm25_mean"]
y4=plot_df_4["cvd_deaths"]
r4=pearson_r(x4,y4)
sizes4=np.sqrt(plot_df_4["population"])/np.sqrt(plot_df_4["population"]).max()*1000+30
plt.figure()
plt.scatter(x4,y4,s=sizes4,alpha=0.7,label="Cities")
add_trendline(x4,y4)
annotate_all_points(plot_df_4,"pm25_mean","cvd_deaths","city_clean")
plt.title(f"Plot 4: PM2.5 vs Cardiovascular Disease Deaths (bubble size = population, r = {r4:.3f})")
plt.xlabel("Average PM2.5 concentration in 2020")
plt.ylabel("Cardiovascular disease deaths")
plt.legend()
plt.grid()
plt.savefig("plot4_pm25_vs_cvd_deaths.png")
plt.show()

# Plot 5 no2 vs death and cities are labeled
plot_df_5=analysis_df[["no2_mean","cvd_deaths","population","city_clean"]].dropna().copy()
x5=plot_df_5["no2_mean"]
y5=plot_df_5["cvd_deaths"]
r5=pearson_r(x5,y5)
sizes5=np.sqrt(plot_df_5["population"])/np.sqrt(plot_df_5["population"]).max()*1000+30
plt.figure()
plt.scatter(x5,y5,s=sizes5,alpha=0.7,label="Cities")
add_trendline(x5,y5)
annotate_all_points(plot_df_5,"no2_mean","cvd_deaths","city_clean")
plt.title(f"Plot 5: NO2 vs Cardiovascular Disease Deaths (bubble size = population, r = {r5:.3f})")
plt.xlabel("Average NO2 concentration in 2020")
plt.ylabel("Cardiovascular disease deaths")
plt.legend()
plt.grid()
plt.savefig("plot5_no2_vs_cvd_deaths.png")
plt.show()

# Plot 6 pm10 vs death and cities are labeled
plot_df_6=analysis_df[["pm10_mean","cvd_deaths","population","city_clean"]].dropna().copy()
x6=plot_df_6[("pm10_mean")]
y6=plot_df_6["cvd_deaths"]
r6=pearson_r(x6,y6)
sizes6=np.sqrt(plot_df_6["population"])/np.sqrt(plot_df_6["population"]).max()*1000+30
plt.figure()
plt.scatter(x6,y6,s=sizes6,alpha=0.7,label="Cities")
add_trendline(x6,y6)
annotate_all_points(plot_df_6,"pm10_mean","cvd_deaths","city_clean")
plt.title(f"Plot 6: PM10 vs Cardiovascular Disease Deaths (bubble size = population, r = {r6:.3f})")
plt.xlabel("Average PM10 concentration in 2020")
plt.ylabel("Cardiovascular disease deaths")
plt.legend()
plt.grid()
plt.savefig("plot6_pm10_vs_cvd_deaths.png")
plt.show()

# Plot 7 combined relative pollution index vs life expectancy
plot_df_7=analysis_df[["pm25_mean","pm10_mean","no2_mean","life_expectancy","population"]].dropna().copy()
plot_df_7["pm25_z"]=zscore_series(plot_df_7["pm25_mean"])
plot_df_7["pm10_z"]=zscore_series(plot_df_7["pm10_mean"])
plot_df_7["no2_z"]=zscore_series(plot_df_7["no2_mean"])
plot_df_7["pollution_index"]=(plot_df_7["pm25_z"]+plot_df_7["pm10_z"]+plot_df_7["no2_z"])/3
x7=plot_df_7["pollution_index"]
y7=plot_df_7["life_expectancy"]
r7=pearson_r(x7,y7)
sizes7=np.sqrt(plot_df_7["population"])/np.sqrt(plot_df_7["population"]).max()*1000+30
plt.figure()
plt.scatter(x7,y7,s=sizes7,alpha=0.75,label="Cities")
add_trendline(x7,y7)
plt.title(f"Plot 7: Combined Pollution Index vs Life Expectancy (r = {r7:.3f})")
plt.xlabel("Combined pollution index (standardized PM2.5, PM10, NO2)")
plt.ylabel("Life Expectancy")
plt.legend()
plt.grid()
plt.savefig("plot7_pollution_index_vs_life_expectancy.png")
plt.show()

# Plot 8 combined pollution index vs cardiovascular disease death
plot_df_8=analysis_df[["pm25_mean","pm10_mean","no2_mean","cvd_deaths","population"]].dropna().copy()
plot_df_8["pm25_z"]=zscore_series(plot_df_8["pm25_mean"])
plot_df_8["pm10_z"]=zscore_series(plot_df_8["pm10_mean"])
plot_df_8["no2_z"]=zscore_series(plot_df_8["no2_mean"])
plot_df_8["pollution_index"]=(plot_df_8["pm25_z"]+plot_df_8["pm10_z"]+plot_df_8["no2_z"])/3
x8=plot_df_8["pollution_index"]
y8=plot_df_8["cvd_deaths"]
r8=pearson_r(x8,y8)
sizes8=np.sqrt(plot_df_8["population"])/np.sqrt(plot_df_8["population"]).max()*1000+30
plt.figure()
plt.scatter(x8,y8,s=sizes8,alpha=0.75,label="Cities")
add_trendline(x8,y8)
plt.title(f"Plot 8: Combined Pollution Index vs Cardiovascular Disease Deaths (r = {r8:.3f})")
plt.xlabel("Combined pollution index (standardized PM2.5, PM10, NO2)")
plt.ylabel("Cardiovascular disease deaths")
plt.legend()
plt.grid()
plt.savefig("plot8_pollution_index_vs_cvd_deaths.png")
plt.show()

# Plot 9 PM 2.5 by Station Type
plot_df_9=main_raw.copy()
plot_df_9["station_group"]=plot_df_9["type_of_stations"].apply(station_group)
plot_df_9=plot_df_9[plot_df_9["station_group"].isin(["Urban","Suburban","Rural"])].copy()
plot_df_9=plot_df_9[["station_group","pm25_concentration"]].dropna()
station_pm25=summarize_with_ci(plot_df_9,group_col="station_group",value_col="pm25_concentration",order=["Urban","Suburban","Rural"])
plt.figure()
bars9=plt.bar(station_pm25["station_group"],station_pm25["mean"],yerr=station_pm25["ci95"],capsize=6,alpha=0.8)
for bar,n in zip(bars9,station_pm25["count"]):
    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f"n={int(n)}",ha="center",va="bottom")
plt.title("Plot 9: Average PM2.5 by Station Type (all air-quality data, 95% CI)")
plt.xlabel("Station type")
plt.ylabel("Average PM2.5 concentration")
plt.grid(axis="y")
plt.savefig("plot9_pm25_by_station_type.png")
plt.show()

# Plot 10 NO2 by Station Type
plot_df_10=main_raw.copy()
plot_df_10["station_group"]=plot_df_10["type_of_stations"].apply(station_group)
plot_df_10=plot_df_10[plot_df_10["station_group"].isin(["Urban","Suburban","Rural"])].copy()
plot_df_10=plot_df_10[["station_group","no2_concentration"]].dropna()
station_pm25=summarize_with_ci(plot_df_10,group_col="station_group",value_col="no2_concentration",order=["Urban","Suburban","Rural"])
plt.figure()
bars10=plt.bar(station_pm25["station_group"],station_pm25["mean"],yerr=station_pm25["ci95"],capsize=6,alpha=0.8)
for bar,n in zip(bars10,station_pm25["count"]):
    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f"n={int(n)}",ha="center",va="bottom")
plt.title("Plot 10: Average NO2 by Station Type (all air-quality data, 95% CI)")
plt.xlabel("Station type")
plt.ylabel("Average NO2 concentration")
plt.grid(axis="y")
plt.savefig("plot10_NO2_by_station_type.png")
plt.show()

# Plot 11 PM10 by Station Type
plot_df_11=main_raw.copy()
plot_df_11["station_group"]=plot_df_11["type_of_stations"].apply(station_group)
plot_df_11=plot_df_11[plot_df_11["station_group"].isin(["Urban","Suburban","Rural"])].copy()
plot_df_11=plot_df_11[["station_group","pm10_concentration"]].dropna()
station_pm25=summarize_with_ci(plot_df_11,group_col="station_group",value_col="pm10_concentration",order=["Urban","Suburban","Rural"])
plt.figure()
bars11=plt.bar(station_pm25["station_group"],station_pm25["mean"],yerr=station_pm25["ci95"],capsize=6,alpha=0.8)
for bar,n in zip(bars11,station_pm25["count"]):
    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f"n={int(n)}",ha="center",va="bottom")
plt.title("Plot 11: Average PM10 by Station Type (all air-quality data, 95% CI)")
plt.xlabel("Station type")
plt.ylabel("Average PM10 concentration")
plt.grid(axis="y")
plt.savefig("plot11_pm10_by_station_type.png")
plt.show()

#Plot Summary
print("\nMain correlations:")
print(f"PM2.5 vs Life Expectancy: {r1:.3f}")
print(f"NO2 vs Life Expectancy: {r2:.3f}")
print(f"PM10 vs Life Expectancy: {r3:.3f}")
print(f"PM2.5 vs CVD Deaths: {r4:.3f}")
print(f"NO2 vs CVD Deaths: {r5:.3f}")
print(f"PM10 vs CVD Deaths: {r6:.3f}")
print(f"Combined Relative Pollution Index vs Life Expectancy: {r7:.3f}")
print(f"Combined Relative Pollution Index vs Life CVD Deaths: {r8:.3f}")

print(f"Matched cities in merged dataset: {analysis_df['city_clean'].nunique()}")
print("\nStation type counts in merged US dataset from original 2020 field:")
print(analysis_df["station_group"].value_counts(dropna=False))
print("\nStation type counts used in plot 9, 10, and 11:")
print(station_pm25[["station_group","count"]])
