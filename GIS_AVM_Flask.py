import GIS_AVM
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import joblib
import pickle
import xgboost as xgb
import pymongo
from pymongo import MongoClient

# client = MongoClient("mongodb+srv://AurumUser:latestPlacid@cluster0.icdds.mongodb.net/predictSample?retryWrites=true&w=majority")
# db = client['predictSample']
# col = db['predictSample']



app = Flask(__name__)
try:
    client1 = MongoClient("mongodb://localhost:27017")
    db = client1['predictSample']
    col=db['predictSample']
except:
    pass

regressor = joblib.load(r"D:\python folder\PycharmProjects\pythonProject2\xgb_reg_gis.pkl")
dd=pd.read_csv(r"D:\python folder\PycharmProjects\pythonProject2\Ref_pune.csv")
# @app.route('/')
# def home():
#     return render_template('index1.html')
#

@app.route('/predict',methods=['POST'])
def predict():

    hello = [x for x in request.form.values()]
    features = hello[0:7]
    ### db insert
    feat_dict = {}
    ff = ['Area','BHK','Covered_Parking','Open_Parking','project_name','Balcony','Location']
    for elind,elem in enumerate(features):
        feat_dict[ff[elind]] = elem
    print(features)
    pr=GIS_AVM.get_proj(features)
    features[4]=pr
    print(features)
    added_feat=['Restaurant_count',
         'Hospital_count',
         'School_count',
         'Shopping_mall_count',
         'Gas_station_count',
         'ATM_count',
         'Best_Hospital_count',
         'Best_ATM_count',
         'Best_Restaurant_count',
         'Best_school_count',
         'Best_Shopping_mall_count',
         'Best_Gas_station_count','Location_Kharadi',
         'Location_Baner',
         'Location_Hinjewadi',
         'Location_Pimpri Chinchwad',
         'Location_Balewadi',
         'Location_Bavdhan',
         'Location_Wakad',
         'Location_Ravet',
         'Location_Dhayri',
         'Location_Mundhwa',
         'Location_Punawale',
         'Location_Wagholi',
         'Location_Tathawade',
         'Location_Kothrud',
         'Location_Dhanori',
         'Location_Moshi',
         'Location_Hadapsar',
         'Location_Mahalunge',
         'Location_Pashan',
         'Location_Aundh',
         'Location_Ambegaon Budruk',
         'Location_Gahunje',
         'Location_Mohammadwadi',
         'Location_Chikhali',
         'Location_Charholi Budruk',
         'Location_Viman Nagar',
         'Location_Lohegaon',
         'Location_Narhe',
         'Location_Talegaon Dabhade',
         'Location_Sus',
         'Location_Undri',
         'Location_Manjari Khurd',
         'Location_Wadgaon Sheri',
         'Location_Pimple Saudagar',
         'Location_Chokhi Dhani',
         'Location_Sus Road-Pashan',
         'Location_Anand Nagar',
         'Location_Alandi',
         'Location_Nanded',
         'Location_Magarpatta City',
         'Location_Maan',
         'Location_Kondhwa',
         'Location_Thergaon',
         'Location_Warje',
         'Location_Mamurdi',
         'Location_Paud Road',
         'Location_Pune',
         'Location_Bhugaon',
         'Location_Pimple Gurav',
         'Location_Pimple Nilakh',
         'Location_Chinchwad',
         'Location_Koregaon Park',
         'Location_Kesnand',
         'Location_Kondhwa Budruk',
         'Location_Hingne Khurd',
         'Location_Bibwewadi',
         'Location_Vadgaon Budruk',
         'Location_Erandwane',
         'Location_Handewadi',
         'Location_Ambegaon',
         'Location_Kalyani Nagar',
         'Location_Fursungi',
         'Location_Pashan Road',
         'Location_Parvati Darshan',
         'Location_Pirangut',
         'Location_Katraj',
         'Location_Ghorpadi',
         'Location_Sinhagad Road',
         'Location_Handewadi Road-Hadapsar',
         'Location_Manjari Budruk']
    city = features[-1]
    ccit=list(dd['Location'])
    cac=[]
    for ind,val in enumerate(ccit):
      if city.lower() in val.lower():
        print(val,ind)
        break
    for j in added_feat:
      cac.append(dd[f'{j}'].iloc[ind])
    features = features[:-1]
    features.extend(cac)
    lp =[]
    for ko in features:
      lp.append(float(ko))
    ss =['Area',
     'BHK',
     'Covered_Parking',
     'Open_Parking',
     'encoded_project_name',
     'Balcony',
     'Restaurant_count',
     'Hospital_count',
     'School_count',
     'Shopping_mall_count',
     'Gas_station_count',
     'ATM_count',
     'Best_Hospital_count',
     'Best_ATM_count',
     'Best_Restaurant_count',
     'Best_school_count',
     'Best_Shopping_mall_count',
     'Best_Gas_station_count',
     'Location_Kharadi',
     'Location_Baner',
     'Location_Hinjewadi',
     'Location_Pimpri Chinchwad',
     'Location_Balewadi',
     'Location_Bavdhan',
     'Location_Wakad',
     'Location_Ravet',
     'Location_Dhayri',
     'Location_Mundhwa',
     'Location_Punawale',
     'Location_Wagholi',
     'Location_Tathawade',
     'Location_Kothrud',
     'Location_Dhanori',
     'Location_Moshi',
     'Location_Hadapsar',
     'Location_Mahalunge',
     'Location_Pashan',
     'Location_Aundh',
     'Location_Ambegaon Budruk',
     'Location_Gahunje',
     'Location_Mohammadwadi',
     'Location_Chikhali',
     'Location_Charholi Budruk',
     'Location_Viman Nagar',
     'Location_Lohegaon',
     'Location_Narhe',
     'Location_Talegaon Dabhade',
     'Location_Sus',
     'Location_Undri',
     'Location_Manjari Khurd',
     'Location_Wadgaon Sheri',
     'Location_Pimple Saudagar',
     'Location_Chokhi Dhani',
     'Location_Sus Road-Pashan',
     'Location_Anand Nagar',
     'Location_Alandi',
     'Location_Nanded',
     'Location_Magarpatta City',
     'Location_Maan',
     'Location_Kondhwa',
     'Location_Thergaon',
     'Location_Warje',
     'Location_Mamurdi',
     'Location_Paud Road',
     'Location_Pune',
     'Location_Bhugaon',
     'Location_Pimple Gurav',
     'Location_Pimple Nilakh',
     'Location_Chinchwad',
     'Location_Koregaon Park',
     'Location_Kesnand',
     'Location_Kondhwa Budruk',
     'Location_Hingne Khurd',
     'Location_Bibwewadi',
     'Location_Vadgaon Budruk',
     'Location_Erandwane',
     'Location_Handewadi',
     'Location_Ambegaon',
     'Location_Kalyani Nagar',
     'Location_Fursungi',
     'Location_Pashan Road',
     'Location_Parvati Darshan',
     'Location_Pirangut',
     'Location_Katraj',
     'Location_Ghorpadi',
     'Location_Sinhagad Road',
     'Location_Handewadi Road-Hadapsar',
     'Location_Manjari Budruk']
    print("lp*****",lp)
    test2= pd.DataFrame([lp],columns= ss,dtype=float)
    y = regressor.predict(test2)
    print(y)
    try:
        feat_dict['Prediction']= f"{y[0]}"
        col.insert_one(feat_dict)
    except:
        pass
    range_val=f"{round(float(y)*.85)} - {round(float(y)*1.15)}"
    print("Outputtttt",range_val)
    # return render_template('index1.html', prediction_text='{}'.format(range_val))
    return(range_val)

if __name__ == "__main__":
    app.run(debug=True)
