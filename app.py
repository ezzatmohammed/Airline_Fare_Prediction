import streamlit as st
import pandas as pd
import pickle



with open("model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open('input_columns.pkl', 'rb') as file:
    input_columns = pickle.load(file)



def predictions(Airline, Source, Destination, Total_Stops , Month , Day, Dep_Time_hours, Dep_Time_min,Arrival_Time_hours,Arrival_Time_min,Duration_hour,Duration_min):

    df = pd.DataFrame(columns=input_columns)
    df.at[0,'Airline'] = Airline
    df.at[0,'Source'] = Source
    df.at[0,'Destination'] = Destination
    df.at[0,'Total_Stops'] =  Total_Stops
    df.at[0,'Month'] = Month
    df.at[0,'Day'] = Day
    df.at[0,'Dep_Time_hours'] = Dep_Time_hours
    df.at[0,'Dep_Time_min'] = Dep_Time_min
    df.at[0,'Arrival_Time_hours'] = Arrival_Time_hours
    df.at[0,'Arrival_Time_min'] = Arrival_Time_min
    df.at[0,'Duration_hour'] = Duration_hour
    df.at[0,'Duration_min'] = Duration_min


    result = model.predict(df)
    return result


def main():
    st.title("Airline App")

    Arline_options = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet','Multiple carriers', 'GoAir', 'Vistara', 'Air Asia','Jet Airways Business', 'Multiple carriers Premium economy']
    Airline = st.selectbox("Select an Airline:", Arline_options)


    Source_options  = ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
    Source = st.selectbox("Select a Source:", Source_options)


    Destination_options = ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad']
    Destination = st.selectbox("Select a Destination:", Destination_options)


    Total_Stops = st.number_input("Total Stops:", step=1)
    Month = st.number_input("Month:", step=1)
    Day = st.number_input("Day:", step=1)
    Dep_Time_hours = st.number_input("Dep_Time_hours:", step=1)
    Dep_Time_min = st.number_input("Dep_Time_min:", step=1)
    Arrival_Time_hours = st.number_input("Arrival Time  hours:", step=1)
    Arrival_Time_min = st.number_input("Arrival Time minuts:", step=1)
    Duration_hour = st.number_input("Duration hour:", step=1)
    Duration_min = st.number_input("Duration min:", step=1)

    if st.button("Predict"):
        prediction = predictions(Airline, Source, Destination, Total_Stops , Month , Day, Dep_Time_hours, Dep_Time_min,Arrival_Time_hours,Arrival_Time_min,Duration_hour,Duration_min)
        st.write("The predicted fare is:", prediction)

if __name__ == '__main__':
    main()

    
