#### Overall setup 
#########################################################

import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm
from PIL import Image

# Defining some general properties of the app
st.set_page_config(
    page_title = "Die Fussball App",
    page_icon = "⚽",
    layout = "wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

# Define Load functions
@st.cache()
def load_data():
    data = pd.read_csv("data_for_plots_original.csv")
    return(data)

@st.cache()
def load_x_data():
    x_train_data = pd.read_csv("X_train_model.csv")
    return(x_train_data)

@st.cache()
def load_y_data():
    y_train_data = pd.read_csv("y_train_model.csv")
    return(y_train_data)
      
# Load Data and Model
data = load_data()
x_train_data = load_x_data()
y_train_data = load_y_data()


# Define Header of app
st.title("Fussball-Team Analysen und Vorhersagen")
st.markdown("⚽Herzlich willkommen! Dieses Tool beschäftigt sich mit der **Vorhersage** von Fussballergebnissen. Darüberhinaus kann man sich spannende **Analysen** zu seinem Verein anzeigen lassen⚽")

image = Image.open("fussball.jpg")
row0_col1, row0_col2, row0_col3 = st.columns([0.02,5,0.02])
row0_col2.image(image,use_column_width = "always")

# First Row
st.header("Auswahl Home-Team")
row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns([3,3,1,1,1])

liga1 = data["league_name"].unique()
liga_choice1 = row1_col1.selectbox("Bitte wählen Sie die Liga des Home-teams aus! ", liga1)
home_team = data["home_team_name"].loc[data["league_name"] == liga_choice1].unique()
home_team_choice = row1_col2.selectbox('', home_team)

# creating filtered data set according to slider inputs
filtered_home_data = data.loc[(data["home_team_name"] == home_team_choice), : ]
                          
# Add checkbox allowing us to display raw data
check1 = st.checkbox("Gefilterte Home-Daten anzeigen", False)
if check1:
    st.subheader("Rohdaten")
    st.write(filtered_home_data)  

#Erstellen der Metriken in der ersten Zeile 
avg_home_goals = filtered_home_data["home_goals"].mean()
rounded_avg_home_goals = round(avg_home_goals, 2)
average_home_goals = row1_col3.metric(label="Ø - Home - Tore", value=rounded_avg_home_goals)   

avg_away_goals = filtered_home_data["away_goals"].mean()
rounded_avg_away_goals = round(avg_away_goals, 2)
average_away_goals = row1_col4.metric(label="Ø - Home - Gegentore", value=rounded_avg_away_goals)   

avg_home_rating = filtered_home_data["home_team_history_rating_1"].mean()
rounded_avg_home_rating = round(avg_home_rating,2)
average_home_rating = row1_col5.metric(label="Ø-Rating", value=rounded_avg_home_rating)   
    


#Second Row##############################################################################################################################################################
row2_col1, row2_col2, row2_col3, row2_col4, row2_col5 = st.columns([5,1,5,1,5])

# Create a standard matplotlib scatter chart 
barplotdata1 = filtered_home_data[["home_team_history_rating_1", "match_date"]]
fig1, ax = plt.subplots(figsize=(12,6.5))
ax.scatter(barplotdata1.index.astype(str),barplotdata1["home_team_history_rating_1"], color = "#62c6fc")
ax.plot(barplotdata1.index.astype(str), barplotdata1["home_team_history_rating_1"], color = "#62c6fc")
for bars in ax.containers:
    ax.bar_label(bars, rotation = "vertical", padding = 10)
ax.set_title("Rating der letzten Heimspiele")
ax.set_xticklabels(filtered_home_data["match_date"], rotation = "vertical") #so werden die Daten richtig angezeigt
ax.set_ylabel("Rating")
# Put matplotlib figure in col 1 
row2_col1.subheader("Ratings in Home-Spielen")
row2_col1.pyplot(fig1, use_container_width=True)

# Create a standard matplotlib barchart 
barplotdata2 = filtered_home_data[["home_goals", "match_date"]]
fig2, ax = plt.subplots(figsize=(12,6.5))
ax.bar(barplotdata2.index.astype(str), barplotdata2["home_goals"], color = "#fc8d62")
for bars in ax.containers:
    ax.bar_label(bars, rotation = "vertical", padding = 10)
ax.set_title("Heimtore der letzten Spiele")
ax.set_xticklabels(filtered_home_data["match_date"], rotation = "vertical")
ax.set_ylabel("Heimtore")
# Put matplotlib figure in col 3 
row2_col3.subheader("Erzielte Tore in Home-Spielen")
row2_col3.pyplot(fig2, use_container_width=True)


# creating filtered data set according to slider inputs
filtered_league_data = data.loc[(data["league_name"] == liga_choice1), : ]
# Create a standard matplotlib barchart 
barplotdata5 = filtered_league_data[["home_team_name", "home_goals"]].groupby("home_team_name").sum()
fig5, ax = plt.subplots(figsize=(12,6.5))
ax.bar(barplotdata5.index.astype(str), barplotdata5["home_goals"], color = "#00FF7F")
for bars in ax.containers:
    ax.bar_label(bars, rotation = "vertical", padding = 10)
ax.set_title("Kummulierte Heimtore - " + liga_choice1)
ax.set_xticklabels(filtered_league_data["home_team_name"], rotation = "vertical")
ax.set_ylabel("home_goals")
# Put matplotlib figure in col 5 
row2_col5.subheader("Heimtore Liga-Vergleich")
row2_col5.pyplot(fig5, use_container_width=True)



# Third Row##############################################################################################################################################################
st.header("Auswahl Away-Team")
row3_col1, row3_col2, row3_col3, row3_col4, row3_col5 = st.columns([3,3,1,1,1])

liga2 = data["league_name"].unique()
liga_choice2 = row3_col1.selectbox("Bitte wählen Sie die Liga des Away-teams aus! ", liga2)
away_team = data["away_team_name"].loc[data["league_name"] == liga_choice2].unique()
away_team_choice = row3_col2.selectbox('', away_team)

# creating filtered data set according to slider inputs
filtered_away_data = data.loc[(data["away_team_name"] == away_team_choice), : ]
                          
# Add checkbox allowing us to display raw data
check2 = st.checkbox("Gefilterte Away-Daten anzeigen", False)
if check2:
    st.subheader("Rohdaten")
    st.write(filtered_away_data)
    
#Erstellen der Metriken in der zweiten Zeile 
avg_away_goals1 = filtered_away_data["away_goals"].mean()
rounded_avg_away_goals1 = round(avg_away_goals1, 2)
average_away_goals1 = row3_col3.metric(label="Ø - Away - Tore", value=rounded_avg_away_goals1)   

avg_home_goals1 = filtered_away_data["home_goals"].mean()
rounded_avg_home_goals1 = round(avg_home_goals1, 2)
average_home_goals1 = row3_col4.metric(label="Ø - Away - Gegentore", value=rounded_avg_home_goals1)   

avg_away_rating = filtered_away_data["away_team_history_rating_1"].mean()
rounded_avg_away_rating = round(avg_away_rating,2)
average_away_rating = row3_col5.metric(label="Ø-Rating", value=rounded_avg_away_rating)   



#Fourth row##############################################################################################################################################################
row4_col1, row4_col2, row4_col3, row4_col4, row4_col5  = st.columns([5,1,5,1,5])

# Create a standard matplotlib barchart 
barplotdata3 = filtered_away_data[["away_team_history_rating_1", "match_date"]]
fig3, ax = plt.subplots(figsize=(12, 6.5))
ax.scatter(barplotdata3.index.astype(str), barplotdata3["away_team_history_rating_1"], color = "#62c6fc")
ax.plot(barplotdata3.index.astype(str), barplotdata3["away_team_history_rating_1"], color = "#62c6fc")
for bars in ax.containers:
    ax.bar_label(bars, rotation = "vertical", padding = 10)
ax.set_title("Rating der letzten Auswärtsspiele")
ax.set_xticklabels(filtered_away_data["match_date"], rotation = "vertical")
ax.set_ylabel("Rating")
# Put matplotlib figure in col 1 
row4_col1.subheader("Ratings in Away-Spielen")
row4_col1.pyplot(fig3, use_container_width=True)

# Create a standard matplotlib barchart 
barplotdata4 = filtered_away_data[["away_goals", "match_date"]]
fig4, ax = plt.subplots(figsize=(12,6.5))
ax.bar(barplotdata4.index.astype(str), barplotdata4["away_goals"], color = "#fc8d62")
for bars in ax.containers:
    ax.bar_label(bars, rotation = "vertical", padding = 10)
ax.set_title("Auswärtstore der letzten Spiele")
ax.set_xticklabels(filtered_away_data["match_date"], rotation = "vertical")
ax.set_ylabel("Auswärtstore")
# Put matplotlib figure in col 3 
row4_col3.subheader("Erzielte Tore in Away-Spielen")
row4_col3.pyplot(fig4, use_container_width=True)


# creating filtered data set according to slider inputs
filtered_league_data1 = data.loc[(data["league_name"] == liga_choice2), : ]
# Create a standard matplotlib barchart 
barplotdata6 = filtered_league_data1[["away_team_name", "away_goals"]].groupby("away_team_name").sum()
fig6, ax = plt.subplots(figsize=(12,6.5))
ax.bar(barplotdata6.index.astype(str), barplotdata6["away_goals"], color = "#00FF7F")
for bars in ax.containers:
    ax.bar_label(bars, rotation = "vertical", padding = 10)
ax.set_title("Kummulierte Auswärtstore - " + liga_choice2)
ax.set_xticklabels(filtered_league_data1["away_team_name"], rotation = "vertical")
ax.set_ylabel("Auswärtstore")
# Put matplotlib figure in col 5
row4_col5.subheader("Auswärststore Liga-Vergleich")
row4_col5.pyplot(fig6, use_container_width=True)


#### Definition of Section 2 for making predictions#######################################################################################################################

st.header("Ergebnis Vorhersage")
uploaded_data = st.file_uploader("Wähle eine Datei mit Spieldaten um Vorhersagen zu erhalten ")

# Add action to be done if file is uploaded
if uploaded_data is not None:
    
    # Getting Data and Making Predictions
    new_predictions = pd.read_csv(uploaded_data)
    model = lightgbm.LGBMClassifier(learning_rate = 0.3, max_depth = 30, random_state = 1)
    model = model.fit(x_train_data, y_train_data, verbose=20, eval_metric='logloss')
    new_predictions["predicted_target"] = model.predict(new_predictions)
    
    # Add User Feedback
    st.success("⚽ 🎉👍 Sie haben soeben %i Spiele vorhersagen lassen! ⚽ 🎉👍" % new_predictions.shape[0])
    
    # Add Download Button
    if st.download_button(label = "Vorhersagen herunterladen",
                       data = new_predictions.to_csv().encode("utf-8"),
                       file_name = "match_predictions.csv"):
            st.balloons()

    # Add Variable Importance
    def plot_variable_importance(model, X_train):
        from pandas import DataFrame
        imp=DataFrame({"imp":model.feature_importances_, "names":X_train.columns}).sort_values("imp", ascending=True)
        fig, ax = plt.subplots(figsize=(imp.shape[0]/6,imp.shape[0]/5), dpi=1000)
        ax.barh(imp["names"],imp["imp"], color="#00FF7F") 
        ax.set_xlabel('\nVariable Importance')
        ax.set_ylabel('Features\n') 
        ax.set_title('Variable Importance Plot\n')  
 
    row5_col1, row5_col2, row5_col3 = st.columns([1,50,1])
    fig7 = plot_variable_importance(model, x_train_data)
    row5_col2.pyplot(fig7, use_container_width=True)