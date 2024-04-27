# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 00:03:08 2024

@author: prachet
"""
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models

diabetes_model = pickle.load(open("diabetes_trained_model.sav",'rb'))

heart_disease_model = pickle.load(open("heart_disease_trained_model.sav",'rb'))

parkinson_disease_model = pickle.load(open("parkinsons_disease_trained_model.sav",'rb'))

breast_cancer_model = pickle.load(open("breast_cancer_trained_model.sav",'rb'))



def diabetes_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = diabetes_model.predict(input_data_reshaped)
    
    return prediction

def heart_disease_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data,dtype=np.float64)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = heart_disease_model.predict(input_data_reshaped)
    
    return prediction

def parkinson_disease_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = parkinson_disease_model.predict(input_data_reshaped)
    
    return prediction

def breast_cancer_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data,dtype=np.float64)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = breast_cancer_model.predict(input_data_reshaped)
    
    return prediction

def main():
    # sidebar for navigate

    with st.sidebar:
    
        selected = option_menu('Multiple Disease Prediction System using ML',
                           
                            ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinson Disease Prediction',
                            'Breast Cancer Prediction'],
                           
                           icons = ['capsule','activity','person','virus'],
                           
                           default_index = 0)
    #default index means default page

    # Diabetes Prediction Page
    if( selected == 'Diabetes Prediction'):
       #page title
        st.title('Diabetes Prediction using ML')
    
        #getting input data from user
        #columns for input fields
    
        col1 , col2 , col3 = st.columns(3)
    
        with col1:
            Pregnancies = st.text_input("Number of Pregnancies")
        with col2:
            Glucose = st.text_input("Glucose Level")
        with col3:
            BloodPressure = st.text_input("BloodPressure volume")
        with col1:
            SkinThickness = st.text_input("SkinThickness value")
        with col2:
            Insulin = st.text_input("Insulin level")
        with col3:
            BMI = st.text_input("BMI value")
        with col1:
            DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction value")
        with col2:
            Age = st.text_input("Age of the person")
    
        # code for prediction
        diabetes_diagnosis = ''
    
        #creating a button for Prediction
        if st.button('Diabetes Test Result'):
            diabetes_diagnosis=diabetes_prediction([[Pregnancies,Glucose,BloodPressure,
                                                 SkinThickness,Insulin,BMI,
                                                 DiabetesPedigreeFunction,Age]])
            if(diabetes_diagnosis[0]==0):
                diabetes_diagnosis = 'The person is not diabetic' 
            else:
                diabetes_diagnosis = 'The person is diabetic'
            st.success(diabetes_diagnosis)
    
 
    # Heart Disease Prediction Page
    if( selected == 'Heart Disease Prediction'):
        #page title
        st.title('Heart Disease Prediction using ML')
    
        #getting input data from user
        #columns for input fields
    
        col1 , col2 , col3 = st.columns(3)
    
        with col1:
            age = st.text_input("Age in years")
        with col2:
            sex = st.text_input("Sex (1 = male; 0 = female)")
        with col3:
            chest_pain = st.text_input("Chest Pain type (4 values)")
        with col1:
            resting_bp = st.text_input("Resting Blood Pressure (in mm Hg)")
        with col2:
            serum_cholestoral = st.text_input("Serum Cholestoral in mg/dl")
        with col3:
            fasting_blood_sugar = st.text_input("Fasting Blood Sugar > 120 mg/dl")
        with col1:
            resting_ecg = st.text_input("Resting ECG Results (values 0,1,2)")
        with col2:
            max_heart_achieved = st.text_input("Maximum Heart Rate Achieved")
        with col3:
            exercise_induced_angina = st.text_input("Exercise Induced Angina")
        with col1:
            oldpeak = st.text_input("Oldpeak (ST depression induced by exercise relative to rest)")
        with col2:
            slope_of_peak_exercise = st.text_input("The slope of the peak exercise ST segment")
        with col3:
            number_of_major_vessels = st.text_input("Number of major vessels (0-3) colored by flourosopy")
        with col1:
            thal = st.text_input("Thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")
    
        # code for prediction
        heart_disease_diagnosis = ''
    
        #creating a button for Prediction
        if st.button('Heart Disease Test Result'):
            heart_disease_diagnosis=heart_disease_prediction([[age,sex,chest_pain,
                                                 resting_bp,serum_cholestoral,fasting_blood_sugar,
                                                 resting_ecg,max_heart_achieved,exercise_induced_angina,
                                                 oldpeak,slope_of_peak_exercise,
                                                 number_of_major_vessels,thal]])
            if(heart_disease_diagnosis[0]==0):
                heart_disease_diagnosis = 'The Person does not have Heart Disease' 
            else:
                heart_disease_diagnosis = 'The Person have Heart Disease'
            st.success(heart_disease_diagnosis)

    # Parkinson Disease Prediction Page
    if( selected == 'Parkinson Disease Prediction'):
        #page title
        st.title('Parkinson Disease Prediction using ML')
        
        #getting input data from user
        #columns for input fields
    
        col1 , col2 , col3 , col4 = st.columns(4)
    
        with col1:
            Fo = st.text_input("MDVP_Fo(Hz)")
        with col2:
            Fhi = st.text_input("MDVP_Fhi(Hz)")
        with col3:
            Flo = st.text_input("MDVP_Flo(Hz)")
        with col4:
            Jitter_per = st.text_input("MDVP_Jitter(%)")
        with col1:
            Jitter_Abs = st.text_input("MDVP_Jitter(Abs)")
        with col2:
            RAP = st.text_input("MDVP_RAP")
        with col3:
            PPQ = st.text_input("MDVP_PPQ")
        with col4:
            Jitter_DDP = st.text_input("Jitter_DDP")
        with col1:
            Shimmer = st.text_input("MDVP_Shimmer")
        with col2:
            Shimmer_dB = st.text_input("MDVP_Shimmer(dB)")
        with col3:
            Shimmer_APQ3 = st.text_input("Shimmer_APQ3")
        with col4:
            Shimmer_APQ5  = st.text_input("Shimmer_APQ5")
        with col1:
            APQ = st.text_input("MDVP_APQ")
        with col2:
            Shimmer_DDA  = st.text_input("Shimmer_DDA")
        with col3:
            NHR = st.text_input("NHR")
        with col4:
            HNR = st.text_input("HNR")
        with col1:
            RPDE = st.text_input("RPDE")
        with col2:
            DFA = st.text_input("DFA")
        with col3:
            spread1 = st.text_input("spread1")
        with col4:
            spread2 = st.text_input("spread2")
        with col1:
            D2 = st.text_input("D2")
        with col2:
            PPE = st.text_input("PPE")
    
        # code for prediction
        parkinson_isease_diagnosis = ''
    
        #creating a button for Prediction
        if st.button('Parkinson Disease Test Result'):
            parkinson_isease_diagnosis=parkinson_disease_prediction([[Fo,Fhi,Flo,
                                                 Jitter_per,Jitter_Abs,RAP,
                                                 PPQ,Jitter_DDP,Shimmer,
                                                 Shimmer_dB,Shimmer_APQ3,
                                                 Shimmer_APQ5,APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
            if(parkinson_isease_diagnosis[0]==0):
                parkinson_isease_diagnosis = 'The Person does not have Parkinson Disease' 
            else:
                parkinson_isease_diagnosis = 'The Person have Parkinson Disease'
            st.success(parkinson_isease_diagnosis)
    
    # Breast Cancer Prediction Page
    if( selected == 'Breast Cancer Prediction'):
        #page title
        st.title('Breast Cancer Prediction using ML')
        
        #getting input data from user
        #columns for input fields
    
        col1 , col2 , col3 , col4, col5 = st.columns(5)
    
        with col1:
            mean_radius = st.text_input("mean radius")
        with col2:
            mean_texture = st.text_input("mean texture")
        with col3:
            mean_perimeter = st.text_input("mean_perimeter")
        with col4:
            mean_area = st.text_input("mean_area")
        with col5:
            mean_smoothness = st.text_input("mean_smoothness")
        with col1:
            mean_compactness = st.text_input("mean_compactness")
        with col2:
            mean_concavity = st.text_input("mean_concavity")
        with col3:
            mean_concave_points = st.text_input("mean_concavepoints")
        with col4:
            mean_symmetry = st.text_input("mean_symmetry")
        with col5:
            mean_fractal_dimension = st.text_input("mean_fractal_dim")
        with col1:
            radius_error = st.text_input("radius_error")
        with col2:
            texture_error  = st.text_input("texture_error")
        with col3:
            perimeter_error = st.text_input("perimeter_error")
        with col4:
            area_error  = st.text_input("area_error")
        with col5:
            smoothness_error = st.text_input("smoothness_error")
        with col1:
            compactness_error = st.text_input("compactness_error")
        with col2:
            concavity_error = st.text_input("concavity_error")
        with col3:
            concave_points_error  = st.text_input("concave_points_error")
        with col4:
            symmetry_error = st.text_input("symmetry_error")
        with col5:
            fractal_dimension_error = st.text_input("fractal_dim_error")
        with col1:
            worst_radius = st.text_input("worst_radius")
        with col2:
            worst_texture = st.text_input("worst_texture")
        with col3:
            worst_perimeter = st.text_input("worst_perimeter")
        with col4:
            worst_area  = st.text_input("worst_area ")
        with col5:
            worst_smoothness = st.text_input("worst_smoothness")
        with col1:
            worst_compactness = st.text_input("worst_compactness")
        with col2:
            worst_concavity = st.text_input("worst_concavity")
        with col3:
            worst_concave_points = st.text_input("worst_concavepoints")
        with col4:
            worst_symmetry = st.text_input("worst_symmetry")
        with col5:
            worst_fractal_dimension = st.text_input("worst_fractal_dim")
    
        # code for prediction
        breast_cancer_diagnosis = ''
    
        #creating a button for Prediction
        if st.button('Breast Cancer Test Result'):
            breast_cancer_diagnosis=breast_cancer_prediction([[mean_radius,mean_texture,mean_perimeter,
                                                 mean_area,mean_smoothness,mean_compactness,
                                                 mean_concavity,mean_concave_points,mean_symmetry,
                                                 mean_fractal_dimension,radius_error,
                                                 texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]])
            if(breast_cancer_diagnosis[0]==0):
                breast_cancer_diagnosis = 'The Breast Cancer is Malignant' 
            else:
                breast_cancer_diagnosis = 'The Breast Cancer is Benign'
            st.success(breast_cancer_diagnosis)


    
if __name__ == '__main__':
    main()





