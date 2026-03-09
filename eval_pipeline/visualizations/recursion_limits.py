import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_recursion_errors(csv_file_path_base, csv_file_path_eval):
    # Load CSV files
    df_base = pd.read_csv(csv_file_path_base, sep=';')
    df_eval = pd.read_csv(csv_file_path_eval, sep=';')

    # Define the exact error message
    error_msg = "Recursion limit reached without hitting a stop condition."
    
    # Extract the recursion limit values safely and drop NaN
    df_base['de_error'] = df_base['graph_error_msg_de'].astype(str).str.contains(error_msg, na=False)
    df_base['en_error'] = df_base['graph_error_msg_en'].astype(str).str.contains(error_msg, na=False)
    df_eval['de_error'] = df_eval['graph_error_msg_de'].astype(str).str.contains(error_msg, na=False)
    df_eval['en_error'] = df_eval['graph_error_msg_en'].astype(str).str.contains(error_msg, na=False)

    # Run B: Define the variables for the error cases 
    total_de_error_b = df_base['de_error'].sum() # total number of errors
    total_en_error_b = df_base['en_error'].sum() # total number of errors
    de_error_en_ok_b = df_base[(df_base['de_error']) & (~df_base['en_error'])].shape[0]
    en_error_de_ok_b = df_base[(~df_base['de_error']) & (df_base['en_error'])].shape[0]
    de_error_en_error_b = df_base[(df_base['de_error']) & (df_base['en_error'])].shape[0]
    en_error_de_error_b = de_error_en_error_b  # Symmetric
    
    # Run E: Define the variables for the error cases 
    total_de_error_e = df_eval['de_error'].sum() # total number of errors
    total_en_error_e = df_eval['en_error'].sum() # total number of errors
    de_error_en_ok_e = df_eval[(df_eval['de_error']) & (~df_eval['en_error'])].shape[0]
    en_error_de_ok_e = df_eval[(~df_eval['de_error']) & (df_eval['en_error'])].shape[0]
    de_error_en_error_e = df_eval[(df_eval['de_error']) & (df_eval['en_error'])].shape[0]
    en_error_de_error_e = de_error_en_error_e  # Symmetric 

    categories = [
        'All Error\nQueries',
        'Error + Query\nPartner OK',
        'Error + Query\nPartner Error'
    ]

    # Build data arrays: [German values, English values] for each category
    category_counts_b = np.array([
        [total_de_error_b, total_en_error_b],                    # Totals
        [de_error_en_ok_b, en_error_de_ok_b],                    # German error/English OK
        [de_error_en_error_b, en_error_de_error_b]               # Both error
    ])
    category_counts_e = np.array([
        [total_de_error_e, total_en_error_e],                    # Totals
        [de_error_en_ok_e, en_error_de_ok_e],                    # German error/English OK
        [de_error_en_error_e, en_error_de_error_e]               # Both error
    ])

    # Create grouped bar chart
    x = np.arange(len(categories))
    width = 0.4 # of the bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))  # 1 row, 2 columns

    # Left plot: Run B
    bars1_ax1 = ax1.bar(x - width/2, category_counts_b[:, 0], width, label='German', color="#8b549c", alpha=0.9)
    bars2_ax1 = ax1.bar(x + width/2, category_counts_b[:, 1], width, label='English', color="#95a23c", alpha=0.9)
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7) # vertical line
    ax1.set_title('Run B', fontsize=27, pad=20) 

    # Right plot: Run E
    bars1_ax2 = ax2.bar(x - width/2, category_counts_e[:, 0], width, label='German', color="#8b549c", alpha=0.9)
    bars2_ax2 = ax2.bar(x + width/2, category_counts_e[:, 1], width, label='English', color="#95a23c", alpha=0.9)
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7) # vertical line
    ax2.set_title('Run E', fontsize=27, pad=20)

    # Styling and legend
    for ax in [ax1, ax2]:
        ax.set_xlabel('Error Categories', fontsize=28)
        ax.set_ylabel('Total Count', fontsize=28)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, ha='center', fontsize=22)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=2)
        ax.set_ylim(0, 28)
        ax.legend(fontsize=27)
        ax.tick_params(axis='y', which='major', labelsize=22)

    # Run B: Add count labels for each barplot
    for i in range(len(category_counts_b)):
        for j, bar_group in enumerate([bars1_ax1, bars2_ax1]):
            bar = bar_group[i]
            height = bar.get_height()
            proportion_in_percent = (height / 52) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height+0.3,
                    f'{int(height)}\n({proportion_in_percent:.2f}%)', ha='center', va='bottom', 
                    fontweight='bold', fontsize=19, color='black', 
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85))
    # Identical code for Run E
    for i in range(len(category_counts_e)):
        for j, bar_group in enumerate([bars1_ax2, bars2_ax2]):
            bar = bar_group[i]
            height = bar.get_height()
            proportion_in_percent = (height / 52) * 100
            ax2.text(bar.get_x() + bar.get_width()/2., height+0.3,
                    f'{int(height)}\n({proportion_in_percent:.2f}%)', ha='center', va='bottom', 
                    fontweight='bold', fontsize=19, color='black', 
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85))
            
    plt.tight_layout()

    # Save figure 
    plt.savefig('recursion_errors/recursion_final_edit.png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    plot_recursion_errors('data/baseline_trace.csv', 'data/config_1_trace.csv') 

