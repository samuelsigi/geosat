import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import random
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Patch
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Function to display the year-wise graph
def show_year_wise_graph():
    # Tkinter GUI setup for year-wise graph
    year_wise_window = tk.Toplevel()
    year_wise_window.title("Year-wise Graph")

    # Function to handle dropdown selection
    def on_select(event):
        selected_year = year_var.get()
        plot_forest_cover(selected_year, year_wise_window)

    # Dropdown menu for selecting year
    year_var = tk.StringVar(year_wise_window)
    years = [str(yr) for yr in range(2002, 2023)]  # From 2002 to 2022
    year_var.set(years[0])  # Set default value
    year_dropdown = ttk.Combobox(year_wise_window, textvariable=year_var, values=years)
    year_dropdown.grid(row=0, column=0, padx=10, pady=10)
    year_dropdown.bind("<<ComboboxSelected>>", on_select)

# Function to plot forest cover change based on year
def plot_forest_cover(year, window):
    # Load the shapefile containing geometries of districts
    shapefile_path = 'district_india/'
    districts_gdf = gpd.read_file(shapefile_path)

    # Read the district ratings from an external Excel file
    ratings_df = pd.read_excel(f'data/{year}.xlsx')

    # Merge the GeoDataFrame with the ratings DataFrame
    districts_gdf = districts_gdf.merge(ratings_df, on='District', how='left')

    # Define custom labels for the legend
    custom_labels = ['Low', 'Medium', 'High']

    # Convert rating values to colors using a colormap
    cmap = plt.cm.get_cmap('Oranges')  # Change colormap to Oranges
    norm = plt.Normalize(vmin=districts_gdf['Rating'].min(), vmax=districts_gdf['Rating'].max())

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the map with colors based on rating values
    districts_gdf.plot(column='Rating', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    # Add district names over the regions
    #for idx, row in districts_gdf.iterrows():
        #plt.annotate(text=row['District'], xy=row.geometry.centroid.coords[0], horizontalalignment='center')

    # Add title
    ax.set_title(f'Map of District Based Forest Cover Change as of {year}', fontdict={'fontsize': '16', 'fontweight' : 'bold'})

    # Add north arrow
    ax.annotate('N', xy=(0.5, 0.99), xycoords='axes fraction', ha='center', va='top', fontsize=12, color='black', rotation=0)

    # Add scale bar
    scalebar = ScaleBar(1, location='lower right')  # 1:1 scale bar
    ax.add_artist(scalebar)

    # Add color scale indicator
    color_patches = [
        Patch(color=cmap(norm(districts_gdf['Rating'].min())), label=custom_labels[0]),
        Patch(color=cmap(norm(districts_gdf['Rating'].mean())), label=custom_labels[1]),
        Patch(color=cmap(norm(districts_gdf['Rating'].max())), label=custom_labels[2])
    ]
    ax.legend(handles=color_patches, loc='lower left', title='Rating Scale', title_fontsize='medium', fontsize='small')

    # Function to handle click event
    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return

        for idx, row in districts_gdf.iterrows():
            if row['geometry'].contains(Point(event.xdata, event.ydata)):
                district_details = row.to_frame().T
                show_details_popup(district_details)

    # Connect click event to the figure
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Get the Tkinter window associated with the plot
    global visualization_window
    plot_window = fig.canvas.manager.window
    visualization_window = plot_window  # Set visualization_window as global

    # Set window title to 'Graph'
    plot_window.title('Graph')

    # Remove axis
    ax.set_axis_off()

    # Display the map
    plt.show()

# Function to show details in a pop-up window
def show_details_popup(district_details):
    popup = tk.Toplevel()
    popup.title("District Details")
    
    # Remove the specified columns
    columns_to_remove = ['DISTRICT_L', 'Shape_Area', 'State_LGD', 'REMARKS']  # Removed 'REMARKS'
    district_details = district_details.drop(columns=columns_to_remove)
    
    # Rename the 'Shape_Leng' column to 'Geo-graphcal Area'
    district_details = district_details.rename(columns={'Shape_Leng': 'Geo-graphcal Area'})
    
    # Calculate 'Change' based on 'Rating' and 'Geo-graphical Area'
    district_details['Change'] = ((district_details['Rating'] / 100) * district_details['Geo-graphcal Area']) / 10  # Divide by 10 and calculate change
    
    # Ensure 'Change' column contains numeric values
    district_details['Change'] = pd.to_numeric(district_details['Change'], errors='coerce')
    
    # Round the values in 'Change' column to 4 decimal points
    district_details['Change'] = district_details['Change'].round(4)
    
    # Rename 'Rating' column to 'Percentage of Change'
    district_details = district_details.rename(columns={'Rating': 'Percentage of Change'})
    
    # Calculate 'Predicted Value for Next Years' by adding a random value
    district_details['Predicted Change Percentage for Upcoming Years'] = district_details['Percentage of Change'] + random.choice([0.5, 0.8, 1, 1.5])
    
    # Rearrange the columns
    district_details = district_details[['District', 'STATE', 'Geo-graphcal Area', 'Change', 'Percentage of Change', 'Predicted Change Percentage for Upcoming Years']]  # Removed 'REMARKS'
    
    # Specify columns to display, excluding 'geometry'
    columns_to_display = [col for col in district_details.columns if col != 'geometry']
    
    # Create Treeview widget with specified columns
    tree = ttk.Treeview(popup, columns=columns_to_display, show="headings")
    
    # Set headings for columns
    for column in columns_to_display:
        tree.heading(column, text=column)
    
    # Insert data into the treeview
    values_to_insert = district_details.iloc[0][columns_to_display].tolist()
    tree.insert("", "end", values=values_to_insert)
    
    # Pack the treeview widget
    tree.pack(expand=True, fill="both")
    
    # Check if 'Predicted Value for Next Years' is greater than 8
    predicted_value = district_details.iloc[0]['Predicted Change Percentage for Upcoming Years']
    if predicted_value > 8:
        warning_label = tk.Label(popup, text="Extensive Deforestation Detected!! SAVE TREES..", font=("Helvetica", 20, "bold"), fg="red")
        warning_label.pack(padx=10, pady=10)

    popup.mainloop()

# Create Tkinter app
root = tk.Tk()
root.title("Forest Cover Change Visualization")

# Function to handle button click for year-wise graph
def show_year_wise_graph_click():
    show_year_wise_graph()

# Button for year-wise graph
year_wise_button = ttk.Button(root, text="Year-wise Graph", command=show_year_wise_graph_click)
year_wise_button.grid(row=0, column=0, padx=10, pady=10)


root.mainloop()

def predfunction():
    pass

# Load the data
years = [str(yr) for yr in range(2002, 2022)]  # From 2002 to 2022
year_var = 0
year = 2022  
ratings_df_current_year = pd.read_excel(f'data/{year}.xlsx')

# Load the data for the next year if available
next_year = year + 1
next_year_file = f'data/{next_year}.xlsx'
if os.path.exists(next_year_file):
    ratings_df_next_year = pd.read_excel(next_year_file)
else:
    print(f"Data for the year {next_year} is not available.")
    # Define an empty DataFrame for ratings_df_next_year
    ratings_df_next_year = pd.DataFrame(columns=ratings_df_current_year.columns)

# Merge the dataframes on district name
merged_df = pd.merge(ratings_df_current_year, ratings_df_next_year, on='District', suffixes=('_current', '_next'))

# Ensure that the columns exist in the merged DataFrame
if 'Geo-graphcal Area_current' in merged_df.columns and 'Change_current' in merged_df.columns:
    # Prepare the features and target variable
    X = merged_df[['Geo-graphcal Area_current', 'Change_current']]  # Features: geographical area and current forest cover change
    y = merged_df['Percentage of Change_next']  # Target variable: forest cover change for the next year
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = rf_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Train RMSE: {train_rmse}")

    y_pred_test = rf_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Test RMSE: {test_rmse}")

    # Predict forest cover change for the next year
    new_data = [[1000, 5]]  # Example: geographical area = 1000 sq. units, current forest cover change = 5%
    predicted_change = rf_model.predict(new_data)
    print(f"Predicted forest cover change for the next year: {predicted_change}%")

predfunction()