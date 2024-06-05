import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

# Global variable for visualization window
visualization_window = None

# Define the compare_years function
def compare_years(year_1, year_2, window):
    # Load the shapefile containing geometries of districts
    shapefile_path = 'district_india/'
    districts_gdf = gpd.read_file(shapefile_path)

    # Read the district ratings for both years from external Excel files
    ratings_df_1 = pd.read_excel(f'data/{year_1}.xlsx')
    ratings_df_2 = pd.read_excel(f'data/{year_2}.xlsx')

    # Merge the GeoDataFrame with the ratings DataFrames for both years
    districts_gdf = districts_gdf.merge(ratings_df_1, on='District', how='left', suffixes=('_1', '_2'))
    districts_gdf = districts_gdf.merge(ratings_df_2, on='District', how='left', suffixes=('_1', '_2'))

    # Calculate deforestation rates for both years
    deforestation_rate_1 = districts_gdf['Rating_1'].mean()
    deforestation_rate_2 = districts_gdf['Rating_2'].mean()

    # Plot the comparison graph
    fig, ax = plt.subplots()
    ax.bar([f'Year {year_1}', f'Year {year_2}'], [deforestation_rate_1, deforestation_rate_2], color=['green', 'blue'])
    ax.set_ylabel('Deforestation Rate (%)')
    ax.set_title('Comparison of Deforestation Rates')
    plt.show()

def show_comparison_ui():
    # Function to handle the comparison process
    def process_comparison():
        selected_year_1 = year_var_1.get()
        selected_year_2 = year_var_2.get()
        compare_years(selected_year_1, selected_year_2, comparison_window)

    # Tkinter GUI setup for comparison UI
    comparison_window = tk.Toplevel()
    comparison_window.title("Select Years for Comparison")

    # Dropdown menu for selecting the first year
    year_var_1 = tk.StringVar(comparison_window)
    years = [str(yr) for yr in range(2002, 2023)]  # From 2002 to 2022
    year_var_1.set(years[0])  # Set default value
    year_dropdown_1 = ttk.Combobox(comparison_window, textvariable=year_var_1, values=years)
    year_dropdown_1.grid(row=0, column=0, padx=10, pady=10)

    # Dropdown menu for selecting the second year
    year_var_2 = tk.StringVar(comparison_window)
    year_var_2.set(years[1])  # Set default value
    year_dropdown_2 = ttk.Combobox(comparison_window, textvariable=year_var_2, values=years)
    year_dropdown_2.grid(row=0, column=1, padx=10, pady=10)

    # Button to initiate the comparison process
    process_button = ttk.Button(comparison_window, text="Process Comparison", command=process_comparison)
    process_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Main application window
root = tk.Tk()
root.title("Forest Cover Change Visualization")

# Hide the main application window
root.attributes('-alpha', 0)

# Call the function to display the comparison UI
show_comparison_ui()

root.mainloop()
