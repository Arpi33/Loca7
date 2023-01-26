# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 01:45:01 2023

@author: MP PC
"""

import pandas as pd
import numpy as np
import math
import statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cityblock # manhattan distance
import copy
from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
import seaborn as sns
#import warnings
import streamlit as st
from streamlit_folium import st_folium
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
#warnings.filterwarnings('ignore')


def read_data():
    
    global df_yp,df_brickz,df_total_pop,df_detailed_pop
    
    # yellow pages
    df_yp = pd.read_csv('final_poi_data.csv')

    # brickz
    df_brickz = pd.read_csv('brickz.csv')

    # Population
    df_total_pop = pd.read_csv("population.csv",low_memory=False)
    #population by gender
    df_detailed_pop = pd.read_csv("population_by_gender.csv",low_memory=False)
    #return df_yp,df_brickz,df_total_pop,df_detailed_pop
    
#read_data()

# Vectorized version of haversine func
# To calculate distance
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# Obtain POI
def getPOI(point_name, lat, lng, loc_data, radius):
    # Speed up by limitting the loc_date lat lng near to the specified lat lng
    offset = 0.01 * (math.floor(radius / 1000) + 1)
    
    # Subset out the loc_data dataset
    loc_data_pruned = loc_data.query('latitude < @lat + @offset & latitude > @lat-@offset & longitude < @lng +@offset & longitude > @lng-@offset')
    
    loc_data_temp = loc_data_pruned.assign(point_name = point_name, point_lat = lat, point_lng = lng)
    matched = haversine_np(loc_data_temp.point_lng, loc_data_temp.point_lat, loc_data_temp.longitude, loc_data_temp.latitude) * 1000 <= radius
    df = loc_data_temp[matched].copy()
    yield df



def get_top_poi(df, radius = 2000):
    poi_gen = df.apply(lambda x: getPOI(x.name, x.lat, x.lng, df_yp, radius), axis = 1)
    yp_poi_df = pd.concat([pd.concat([y for x in poi_gen for y in x], ignore_index = True)], 
                          ignore_index = True)
    #yp_poi_df["name"] = yp_poi_df[["name"]].progress_apply(cleanString, axis = 1)
    yp_poi_df = yp_poi_df[yp_poi_df['Name'] != 'Point of Interest/Null']
    #display(yp_poi_df)
    yp_poi_df = yp_poi_df.drop_duplicates(subset = ["Name", "point_lat", "point_lng"])
    yp_poi_df = yp_poi_df.groupby(["point_lat", "point_lng", "Name"]).size().reset_index(name = "counts")
    _ = yp_poi_df.groupby(['Name'])['counts'].sum().sort_values(ascending=False).index.tolist()
    yp_poi_df = pd.pivot_table(yp_poi_df, index = ["point_lat", "point_lng"], aggfunc = np.sum, columns = ["Name"], values = "counts", fill_value=0)
    yp_poi_df = yp_poi_df.reset_index().rename_axis(None, axis = 1)
    yp_poi_df_final = pd.DataFrame()
    yp_poi_df_final = yp_poi_df[_]
    yp_poi_df_final['lat'] = ''
    yp_poi_df_final['lng'] = ''
    yp_poi_df_final['lat'] = yp_poi_df['point_lat']
    yp_poi_df_final['lng'] = yp_poi_df['point_lng']

    return yp_poi_df_final


#top poi category

def get_top_category(df, radius = 2000):
    poi_gen = df.apply(lambda x: getPOI(x.point_name, x.lat, x.lng, df_yp, radius), axis = 1)
    yp_poi_df = pd.concat([pd.concat([y for x in poi_gen for y in x], ignore_index = True)], ignore_index = True)
    #yp_poi_df["name"] = yp_poi_df[["name"]].progress_apply(cleanString, axis = 1)
    yp_poi_df = yp_poi_df[yp_poi_df['Category'] != 'Point of Interest']
    #display(yp_poi_df)
    yp_poi_df = yp_poi_df.drop_duplicates(subset = ["Name", "point_lat", "point_lng"])
    yp_poi_df = yp_poi_df.groupby(["point_lat", "point_lng", "Category"]).size().reset_index(name = "counts")
    _ = yp_poi_df.groupby(['Category'])['counts'].sum().sort_values(ascending=False).index.tolist()
    yp_poi_df = pd.pivot_table(yp_poi_df, index = ["point_lat", "point_lng"], aggfunc = np.sum, columns = ["Category"], values = "counts", fill_value=0)
    yp_poi_df = yp_poi_df.reset_index().rename_axis(None, axis = 1)
    yp_poi_df_final = pd.DataFrame()
    yp_poi_df_final = yp_poi_df[_]
    yp_poi_df_final['lat'] = ''
    yp_poi_df_final['lng'] = ''
    yp_poi_df_final['lat'] = yp_poi_df['point_lat']
    yp_poi_df_final['lng'] = yp_poi_df['point_lng']

    return yp_poi_df_final


#get top neighbourhood

def returnMode(x):
    temp = x.value_counts()
    return temp.index[0]

def aggTmn(x):
    d = {}
    d["type_mode"] = returnMode(x["type"])
    d["tenure_mode"] = returnMode(x["tenure"])
    d["price_median"] = round(x["price"].median(), 2)
    return pd.Series(d, index = ["type_mode", "tenure_mode", "price_median"])

def getBrickz(point_name, lat, lng, radius):
    # Define col list
    col_list = ["point_name", "lat", "lng", \
                "tmn_name_1", "tmn_lat_1", "tmn_lng_1", "tmn_majority_tenure_1", "tmn_majority_type_1", "tmn_median_price_1", \
                "tmn_name_2", "tmn_lat_2", "tmn_lng_2", "tmn_majority_tenure_2", "tmn_majority_type_2", "tmn_median_price_2", \
                "tmn_name_3", "tmn_lat_3", "tmn_lng_3", "tmn_majority_tenure_3", "tmn_majority_type_3", "tmn_median_price_3"]
    
    # Lat lng matching
    brickz_temp = df_brickz.assign(point_name = point_name, point_lat = lat, point_lng = lng)
    brickz_temp["distance"] = haversine_np(brickz_temp["point_lng"], brickz_temp["point_lat"], brickz_temp["lng"], brickz_temp["lat"]) * 1000
    subset_df = brickz_temp.query('distance <= @radius')
    
    # Check is empty
    if len(subset_df.index) == 0:
        tmn_data_list = [point_name, lat, lng, "", "", "", "", "", np.NaN, "", "", "", "", "", np.NaN, "", "", "", "", "", np.NaN]
    else:
        subset_df = subset_df.sort_values(by = "distance", ascending = True)
        tmn_list = subset_df.drop_duplicates(subset="taman").taman.values.tolist()
        tmn_list = tmn_list[:3]
        
        tmn_data_list = []
        for taman in enumerate(tmn_list):
            tmn_ind = taman[0]
            tmn_name = taman[1]

            temp_df = subset_df[subset_df["taman"] == tmn_name]

            tmn_lat, tmn_lng = temp_df["lat"].values[0], temp_df["lng"].values[0]

            # Find out latest year in the taman transaction
            #year_list = temp_df.groupby(["taman"]).year.max().to_frame().reset_index()

            # Subset out dataframe with latest year
            #temp_df = pd.merge(year_list, temp_df, on = ["taman", "year"], how = "left")

            temp_df = temp_df.groupby(["taman", "lat", "lng"]).apply(aggTmn).reset_index()

            # Calculate the majority tenure and building type, median price
            major_tenure = temp_df.tenure_mode.values[0]
            major_type = temp_df.type_mode.values[0]
            median_price = temp_df.price_median.values[0]

            tmn_data_list.append([tmn_name, tmn_lat, tmn_lng, major_tenure, major_type, median_price])
            
        for i in range(3 - len(tmn_data_list)):
            tmn_data_list.append(["", "", "", "", "", np.NaN])
        tmn_data_list = list(np.concatenate(tmn_data_list))
        tmn_data_list = [point_name, lat, lng] + tmn_data_list
        
    df = pd.DataFrame([tmn_data_list], columns = col_list)
    yield df
    
def obtainBrickzDF(df, radius = 2000):
    #start_time = time.time()
    brickz_gen = df.apply(lambda x: getBrickz(x.point_name, x.lat, x.lng, radius), axis = 1)
    df = pd.concat([y for x in brickz_gen for y in x], ignore_index = True)
    df = df.replace("", np.NaN)
    #time_taken = time.time() - start_time
    #print("Time taken for obtaining Brickz: ", round(time_taken, 2), "seconds")
    
    return df

def getEachFBPop(point_name, lat, lng, radius):
    # Check is NA of not
    if point_name == "":
        return [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
    else:
        # Calculate the ratio of population
        ratio = radius / 15

        # Speed up by limitting the population lat lng near to the specified taman lat lng
        offset = 0.01 * (math.floor(radius / 1000) + 1)
        lat, lng = float(lat), float(lng)

        # Subset out the total_pop dataset
        total_pop_pruned = df_total_pop.query('latitude < @lat+@offset & latitude > @lat-@offset & longitude < @lng+@offset & longitude > @lng-@offset')
        detailed_pop_pruned = df_detailed_pop.query('latitude < @lat+@offset & latitude > @lat-@offset & longitude < @lng+@offset & longitude > @lng-@offset')

        # Check if after pruned is empty (Assume if one pop df is empty, another also assume empty)
        if (len(total_pop_pruned.index) == 0) | (len(detailed_pop_pruned.index) == 0):
            total_pop, male_pop, female_pop, female_reproductive_pop, youth_pop, elder_pop, child_pop = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        else:
            # Assign the lat lng into pruned pop dataset
            total_pop_pruned = total_pop_pruned.assign(ori_lat = lat, ori_lng = lng)
            detailed_pop_pruned = detailed_pop_pruned.assign(ori_lat = lat, ori_lng = lng)

            # Lat lng matching and record down the distance
            total_pop_pruned["dist"] = haversine_np(total_pop_pruned["ori_lng"], total_pop_pruned["ori_lat"], \
                                                    total_pop_pruned["longitude"], total_pop_pruned["latitude"])
            detailed_pop_pruned["dist"] = haversine_np(detailed_pop_pruned["ori_lng"], detailed_pop_pruned["ori_lat"], \
                                                     detailed_pop_pruned["longitude"], detailed_pop_pruned["latitude"])

            # Need to sort ascendingly
            total_pop_pruned = total_pop_pruned.sort_values(by = "dist")
            detailed_pop_pruned = detailed_pop_pruned.sort_values(by = "dist")

            # Get the first distance and multiply the population and round up
            total_pop = round(total_pop_pruned.iloc[0]["population"] * ratio)
            male_pop = round(detailed_pop_pruned.iloc[0]["population_male"] * ratio)
            female_pop = round(detailed_pop_pruned.iloc[0]["population_female"] * ratio)
            female_reproductive_pop = round(detailed_pop_pruned.iloc[0]["population_reproductive_female"] * ratio)
            youth_pop = round(detailed_pop_pruned.iloc[0]["population_youth"] * ratio)
            elder_pop = round(detailed_pop_pruned.iloc[0]["population_elderly"] * ratio)
            child_pop = round(detailed_pop_pruned.iloc[0]["population_child"] * ratio)
            
        return [total_pop, male_pop, female_pop, female_reproductive_pop, youth_pop, elder_pop, child_pop]
    
def getBrickzFBPop(tmn_name_1, tmn_lat_1, tmn_lng_1, tmn_name_2, tmn_lat_2, tmn_lng_2, tmn_name_3, tmn_lat_3, tmn_lng_3, radius):
    fb_pop_list = []
    fb_pop_list.append(getEachFBPop(tmn_name_1, tmn_lat_1, tmn_lng_1, radius))
    fb_pop_list.append(getEachFBPop(tmn_name_2, tmn_lat_2, tmn_lng_2, radius))
    fb_pop_list.append(getEachFBPop(tmn_name_3, tmn_lat_3, tmn_lng_3, radius))
    fb_pop_list = list(np.concatenate(fb_pop_list))
    
    return fb_pop_list

def obtainBrickzPopDF(df, radius = 2000):
    df[["tmn_fb_total_pop_1", "tmn_fb_male_pop_1", "tmn_fb_female_pop_1", "tmn_fb_female_reproductive_pop_1", "tmn_fb_youth_pop_1", "tmn_fb_elder_pop_1", "tmn_fb_child_pop_1", "tmn_fb_total_pop_2", "tmn_fb_male_pop_2", "tmn_fb_female_pop_2", "tmn_fb_female_reproductive_pop_2", "tmn_fb_youth_pop_2", "tmn_fb_elder_pop_2", "tmn_fb_child_pop_2", "tmn_fb_total_pop_3", "tmn_fb_male_pop_3", "tmn_fb_female_pop_3", "tmn_fb_female_reproductive_pop_3", "tmn_fb_youth_pop_3", "tmn_fb_elder_pop_3", "tmn_fb_child_pop_3"]] = df.apply(lambda x: getBrickzFBPop(x.tmn_name_1, x.tmn_lat_1, x.tmn_lng_1, x.tmn_name_2, x.tmn_lat_2, x.tmn_lng_2, x.tmn_name_3, x.tmn_lat_3, x.tmn_lng_3, radius), axis = 1, result_type = "expand")
    return df

def extract_location_feature(data):
    df_poi = get_top_poi(data)
    df_category = get_top_category(data)
    df_brickz_top = obtainBrickzDF(data)
    df_brickz_top = obtainBrickzPopDF(df_brickz_top)
    
    # Merge data
    df_top_poi = data.merge(df_poi, how='left', left_on=(['lat','lng']), right_on=(['lat','lng']))
    df_poi2 = df_top_poi.merge(df_category, how='left', left_on=(['lat','lng']), right_on=(['lat','lng']))

    # Analytical Dataset
    df_merge_poi = pd.merge(df_poi2, df_brickz_top.drop
                            (columns = ['point_name']), how = 'left', on = ['lat', 'lng'])
    
    return df_merge_poi

#streamlit value
def user_data(lat,lng):
    data = {'point_name':'new_location','lat':[lat],'lng':[lng]}
    user_data=pd.DataFrame(data)
    return user_data



def intersection_of_dfs(dfs_array):
    if len(dfs_array) <= 1:
        # if we only have 1 or 0 elements simply return the origial array
        return dfs_array
    # does a shallow copy.
    dfs = copy.copy(dfs_array)
    length = len(dfs) 
    while length > 1:
        df1 = dfs.pop()
        df2 = dfs.pop()
        col = np.intersect1d(df1.columns.tolist(), df2.columns.tolist())
        df_all = pd.merge(df1[col], df2[col],how="outer")
        dfs.insert(0,df_all)
        length = len(dfs)
    return dfs[0].columns


#pass user input function as a parameter
def generate_dataset(dic):
    
    #get list of all datasets if user choose 1 dataset
    cat = df_yp[df_yp['Category']=='food and beverages']
    cat_li=cat['Name'].value_counts().index[0:20]
    
    
    
    #dic=user_input()
    #dic = ({'key0':value1,'key1':value2,'key2':value3})

    print(dic)
    list_1=[]
    if (len(dic)>1):
        for i in dic.items():
                key=pd.read_csv('./FoodDataset/{}/{}_top_featured_bi.csv'.format(i[1],i[1]))
                list_1.append(key)
            #except:
            #    st.write("Dataset not found")
    else:
        for i in cat_li:
            key=pd.read_csv('./FoodDatast/{}/{}_top_featured_bi.csv'.format(i,i))
            list_1.append(key)   
        
        #intersect columns
    if(len(list_1)==3):
        a=np.intersect1d(list_1[0].columns,list_1[1].columns)
        b=np.intersect1d(a,list_1[2].columns)
    elif(len(list_1)==2):
        b=np.intersect1d(list_1[0].columns,list_1[1].columns)
    else:
        b=intersection_of_dfs(list_1)
               
    
    for data in list_1:
    #drop columns
        data.drop(columns=['lat','lng','tmn_name_1','tmn_name_2','tmn_name_3','tmn_lat_1', 'tmn_lng_1', 'tmn_lat_2', 'tmn_lng_2', 'tmn_lat_3', 'tmn_lng_3'], inplace=True)
       # data.dropna(inplace=True)        
        for i in data.columns:
            if i not in b:
                data.drop(columns=[i], inplace=True)
                data.dropna(inplace=True)
    # concatenating df1 and df2 along rows
    final_data = pd.concat(list_1, axis=0).reset_index(drop=True)
    return final_data,dic



def proceesing_data(gfinal,test_data):
       
    #intersect columns with unseen data point
    for i in gfinal.columns:
        if i not in test_data.columns:
            gfinal.drop(columns=[i],inplace=True)

    #encode the categorical data
    le = LabelEncoder()
    col1 = ['tmn_majority_tenure_1', 'tmn_majority_type_1', 'tmn_majority_tenure_2', 'tmn_majority_type_2',
               'tmn_majority_tenure_3', 'tmn_majority_type_3']


    for feature in col1:
        gfinal[feature] = le.fit_transform(gfinal[feature])

        gfinal.dropna(inplace=True)


    #take the average of median price(to reduce the data dimensionality withoiut compromising the data)
    avg_price=[]
    for i in zip(gfinal['tmn_median_price_1'],gfinal['tmn_median_price_2'],gfinal['tmn_median_price_3']):
        avg_price.append(round(statistics.mean([i[0],i[1],i[2]])))

    gfinal['avg_price']=avg_price

    
    #drop less relevant or not too helpful data
    col2 = ["point_name", "tmn_median_price_1","tmn_median_price_2","tmn_median_price_3","tmn_fb_total_pop_2","tmn_fb_total_pop_3","tmn_fb_male_pop_2","tmn_fb_male_pop_3","tmn_fb_female_pop_2",\
          "tmn_fb_female_pop_3","tmn_fb_female_reproductive_pop_2","tmn_fb_female_reproductive_pop_3","tmn_fb_youth_pop_2","tmn_fb_youth_pop_3", "tmn_fb_elder_pop_2","tmn_fb_elder_pop_3",\
           "tmn_fb_child_pop_2","tmn_fb_child_pop_3"]

    for i in col2:
        gfinal.drop(columns=[i],inplace=True) 

    #normalize the data    
    scaler = MinMaxScaler()
    scaler.fit(gfinal)
    scaled = scaler.fit_transform(gfinal)
    scaled_train = pd.DataFrame(scaled, columns=gfinal.columns)


    #round up the scaled value
    scaled_train = scaled_train.round(decimals = 2)   
    
    
    #process new unseen data
    for feature in col1:
        test_data[feature] = le.fit_transform(test_data[feature])

        test_data.dropna(inplace=True)

    #take the average of median price(to reduce the data dimensionality withoiut compromising the data)
    avg_price2=[]
    for i in zip(test_data['tmn_median_price_1'],test_data['tmn_median_price_2'],test_data['tmn_median_price_3']):
        avg_price2.append(round(statistics.mean([float(i[0]),float(i[1]),float(i[2])])))

    test_data['avg_price']=avg_price2

    for i in col2:
        test_data.drop(columns=i,inplace=True)

    #align the columns
    test_data=test_data[gfinal.columns]

    #normalize by scaling the new data
    scaled_test = scaler.transform(test_data)
    scaled_test = pd.DataFrame(scaled_test, columns=test_data.columns)

    scaled_test = scaled_test.round(decimals = 2)
    
    return scaled_train, scaled_test  



def extract_data(dic,lat,lng):
    gfinal,dic = generate_dataset(dic)
    check_ind= copy.copy(gfinal)
    new_location = extract_location_feature(user_data(lat,lng))
    scaled_train,scaled_test=proceesing_data(gfinal,new_location)
    return check_ind,dic,scaled_train,scaled_test



def get_similarity(dic,lat,lng):
    
    data_org,dic,scaled_df,scaled_test,= extract_data(dic,lat,lng)
    
    #display(data_org.head())
    store_train=[]
    store_new=[]
    store_dist=[]
    point_name = []

    for idx,row in scaled_df.iterrows():
        my_list=row[0:]
        store_train.append(my_list)
        
    for idx,row in scaled_test.iterrows():
        my_list=row[0:]
        store_new.append(my_list) 
        

    for i in store_train:
        dist=cityblock(i.values,store_new[0]).round(decimals=4)
        dist=100-dist
        store_dist.append(dist)   
        
    for index,val in enumerate(store_dist):
        point_name.append(data_org.point_name[index])   
        
    similarity = pd.DataFrame({'business_name':point_name,'dist':store_dist})
    return similarity,dic



def plot(dic,lat,lng):
    
    
    read_data()
    
    res,dic = get_similarity(dic,lat,lng)
    g = res.groupby('business_name')['dist'].mean().round(2).sort_values(ascending=False)
    new_df=pd.DataFrame({"business_name":g.index,"mean":g.values}).reset_index().head()
    #dic1 = {"business_name":new_df['business_name'],"mean":new_df['mean']}
    
    
    #plt.figure(figsize=(12,8))
    fig= plt.figure(figsize=(18,15))

    
    splot=sns.barplot( x='business_name',y='mean',data=new_df)
    
    for p in splot.patches:
        splot.annotate(format(p.get_height()),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(10, 20),
                       textcoords='offset points')
    
    
       # plt.ylim([0,30])
    plt.xticks(rotation = 0, fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel('Business Name', fontsize = 20)
    plt.ylabel('Similarity in percentage', fontsize = 20)
    if len(new_df)>3:
        if dic['key0'] not in new_df['business_name']:
            plt.title('Selected business is not recommended for this location. You are recommended with these following businesses', fontsize = 15)
    else:
        plt.title('Recommended Businesses for given Location', fontsize = 20)
        #plt.show()
    return fig




#st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: grey;marginTop: -85px'>Location Analytics</h1>", unsafe_allow_html=True)

a=st.number_input("How many business?",min_value=0, max_value=3)

if (a>0 and a<=3):
    col=st.columns(2)
        
    with col[0]:
            
    
       if(a==3):
                    bus1=st.text_input("1st")
                    bus2=st.text_input("2nd")
                    bus3=st.text_input("3rd")
                    dic = ({'key0':bus1.lower(),'key1':bus2.lower(),'key2':bus3.lower()})
       elif(a== 2):
                    bus1=st.text_input("1st")
                    bus2=st.text_input("2nd")
                    dic = ({'key0':bus1.lower(),'key1':bus2.lower()})
                    
       elif(a==1):
                    bus1=st.text_input("1st")
                    dic = ({'key0':bus1.lower()})
    
                  
    with col[1]:
          loc_button = Button(label="Allow access to your current location")
          loc_button.js_on_event("button_click", CustomJS(code="""
                        navigator.geolocation.getCurrentPosition(
                            (loc) => {
                                document.dispatchEvent(new CustomEvent("GET_LOCATION", {detail: {lat: loc.coords.latitude, lon: loc.coords.longitude}}))
                            }
                        )
                        """))
          result = streamlit_bokeh_events(
                        loc_button,
                        events="GET_LOCATION",
                        key="get_location",
                        refresh_on_update=False,
                        override_height=75,
                        debounce_time=0)
          #st.write(result) 
          st.markdown(f'<p style="color:green;font-weight: bold;font-size:14px;border-radius:2%;">{result}</p>', unsafe_allow_html=True)
          
    def gigi(result):
        lat=  result['GET_LOCATION']['lat']
        lng = result['GET_LOCATION']['lon']
        return lat,lng      
                #st.markdown(f'<p style="color:BLACK;font-weight: bold;font-size:14px;border-radius:2%;">{result}</p>', unsafe_allow_html=True)
           
    button = st.button("Recommend")
    
    if(button):
        if(len(dic)>0):
            try:
                lat,lng=gigi(result)
                fin= plot(dic,lat,lng)
                #msg1= predict(age,mood)
                st.write(fin)
            #st.markdown(f'<p style="color:black;font-weight: bold;font-size:18px;">Here’s what we suggest: {}</p>', unsafe_allow_html=True)
            except:
                msg2='Required Data are missing, Please key in all the data.'
                st.markdown(f'<p style="color:red;font-weight: bold;font-size:18px; border-radius:2%;">{msg2}</p>', unsafe_allow_html=True)

