import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Function to read JSON files
def read_gridlock_files(congress_numbers):
    data = []
    for num in congress_numbers:
        filename = f"gridlock_score_congress_{num}.json"
        try:
            with open(filename, 'r') as file:
                json_data = json.load(file)
                # Extract relevant information
                congress_data = {
                    'congress': int(json_data['congress']),
                    'gridlock_rate': float(json_data['gridlock_rate'])
                }
                data.append(congress_data)
        except FileNotFoundError:
            print(f"Warning: File {filename} not found. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: File {filename} contains invalid JSON. Skipping.")
    
    # Sort by congress number
    data.sort(key=lambda x: x['congress'])
    return data

# Function to map congress numbers to years
def map_congress_to_year(congress_number):
    # Congressional sessions typically start in odd-numbered years
    # The 80th Congress began in 1947, and each subsequent Congress is 2 years
    # Reference points from the graph:
    # 80th Congress = 1947
    # 103rd Congress = 1993
    # 104th Congress = 1995
    
    congress_year_map = {
        80: 1947,
        81: 1949,
        82: 1951,
        83: 1953,
        84: 1955,
        85: 1957,
        86: 1959,
        87: 1961,
        88: 1963,
        89: 1965,
        90: 1967,
        91: 1969,
        92: 1971,
        93: 1973,
        94: 1975,
        95: 1977,
        96: 1979,
        97: 1981,
        98: 1983,
        99: 1985,
        100: 1987,
        101: 1989,
        102: 1991,
        103: 1993,
        104: 1995,
        105: 1997,
        106: 1999,
        107: 2001,
        108: 2003,
        109: 2005,
        110: 2007,
        111: 2009,
        112: 2011,
        113: 2013,
        114: 2015,
        115: 2017,
        116: 2019,
        117: 2021,
        118: 2023
    }
    
    if congress_number in congress_year_map:
        return congress_year_map[congress_number]
    
    # Fallback calculation if not in map
    base_congress = 80
    base_year = 1947
    years_difference = (congress_number - base_congress) * 2
    return base_year + years_difference

# Main plotting function
def plot_gridlock_trends(congress_numbers):
    data = read_gridlock_files(congress_numbers)
    
    if not data:
        print("No valid data found. Please check your files.")
        return
    
    # Extract data for plotting
    congresses = [item['congress'] for item in data]
    gridlock_rates = [item['gridlock_rate'] * 100 for item in data]  # Convert to percentage
    years = [map_congress_to_year(congress) for congress in congresses]
    
    # Create the figure similar to the academic journal style
    plt.figure(figsize=(10, 6))
    
    # Plot the data with circles at each data point
    plt.plot(years, gridlock_rates, '-o', color='black', markersize=5, linewidth=1)
    
    # Set axis labels
    plt.xlabel('Start of Congress', fontsize=12)
    plt.ylabel('% Gridlock', fontsize=12, rotation=90, labelpad=10)
    
    # # Create a title for recent congresses
    # plt.figtext(0.5, 0.95, 'FIGURE 1. Level of Policy Gridlock in Congress, 2013-2023', 
    #            ha='center', fontsize=12, fontweight='bold')
    
    # Set y-axis limits similar to the image
    plt.ylim(25, 75)
    
    # Set x-axis with years for recent congresses (113-118)
    # Map the congress numbers to years
    congress_years = {
        113: 2013,
        114: 2015,
        115: 2017,
        116: 2019,
        117: 2021,
        118: 2023
    }
    
    # Use all years for these recent congresses
    plt.xticks(list(congress_years.values()), fontsize=10)
    plt.yticks([25, 35, 45, 55, 65, 75], fontsize=10)
    
    # Clean up the plot - remove top and right spines like in academic journals
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a subtle grid for the y-axis only
    plt.grid(axis='y', linestyle='-', alpha=0.2)
    
    # Add journal reference text
    plt.figtext(0.1, 0.95, 'Congressional Gridlock Analysis', ha='left', fontsize=12)
    plt.figtext(0.9, 0.95, 'Congresses 113-118', ha='right', fontsize=12)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust layout to accommodate the title
    plt.savefig('gridlock_analysis.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print(f"Plot saved as 'gridlock_analysis.png'")

# If this script is run directly
if __name__ == "__main__":
    # Just use the specific Congress numbers 113-118
    congress_numbers = ['113', '114', '115', '116', '117', '118']
    
    print(f"Using gridlock data for the following congresses: {', '.join(congress_numbers)}")
    plot_gridlock_trends(congress_numbers)
    
    print("""
Instructions:
1. Place all your gridlock_score_congress_XXX.json files in the same directory as this script
2. Run the script with: python gridlock_analysis.py
3. The script will plot data for Congresses 113-118
4. The resulting image will be saved as 'gridlock_analysis.png'
""")