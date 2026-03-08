import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def plot_latency(csv_file_path_b, csv_file_path_e):
    # Load CSV files
    df_b = pd.read_csv(csv_file_path_b, sep=';')
    df_e = pd.read_csv(csv_file_path_e, sep=';')

    # Extract the latency values safely and drop NaN
    df_b['de_latency_b'] = pd.to_numeric(df_b['latency_total_de'], errors='coerce')
    df_b['en_latency_b'] = pd.to_numeric(df_b['latency_total_en'], errors='coerce')
    df_e['de_latency_e'] = pd.to_numeric(df_e['latency_total_de'], errors='coerce')
    df_e['en_latency_e'] = pd.to_numeric(df_e['latency_total_en'], errors='coerce')
    
    de_latency_clean_b = df_b['de_latency_b'].dropna()
    en_latency_clean_b = df_b['en_latency_b'].dropna()
    de_latency_clean_e = df_e['de_latency_e'].dropna()
    en_latency_clean_e = df_e['en_latency_e'].dropna()
    all_latency_clean_b = pd.concat([de_latency_clean_b, en_latency_clean_b]).dropna()
    all_latency_clean_e = pd.concat([de_latency_clean_e, en_latency_clean_e]).dropna()

    # Prints
    print(f"DE B: {len(de_latency_clean_b)}, EN B: {len(en_latency_clean_b)}, ALL B: {len(all_latency_clean_b)}")
    print(f"DE E: {len(de_latency_clean_e)}, EN E: {len(en_latency_clean_e)}, ALL E: {len(all_latency_clean_e)}")    
    
    data_de = [de_latency_clean_e, de_latency_clean_b]
    data_en = [en_latency_clean_e, en_latency_clean_b]
    data_all = [all_latency_clean_e, all_latency_clean_b]

    # Define positions for boxplots
    centers = np.arange(2) * 2.00 + 2.25
    offset1 = 0.6   # Left: All samples  
    offset2 = 0      # Middle: German samples  
    offset3 = -0.6    # Right: English samples  
    positions_all  = np.array(centers + offset1)
    positions_de   = np.array(centers + offset2)
    positions_en   = np.array(centers + offset3)

    # Define plot measurements
    width = 0.5
    fig_w = 3.0 
    fig_h = 8.0 
    fig, ax = plt.subplots(figsize=(fig_h, fig_w)) 
    
    # Boxplots
    ax.boxplot(data_all, positions=positions_all, widths=width, patch_artist=True,
               boxprops=dict(facecolor='#bb9e7b', alpha=0.9),
               medianprops=dict(color='black'), vert=False) 
    ax.boxplot(data_de, positions=positions_de, widths=width, patch_artist=True, 
               boxprops=dict(facecolor='#8b549c', alpha=0.9),
               medianprops=dict(color='black'), vert=False)
    ax.boxplot(data_en, positions=positions_en, widths=width, patch_artist=True,
               boxprops=dict(facecolor='#95a23c', alpha=0.9),
               medianprops=dict(color='black'), vert=False) 

    # Calculate Means/SD overlays
    means_de = np.array([s.mean() if len(s) > 0 else np.nan for s in data_de])
    stds_de = np.array([s.std() if len(s) > 1 else 0 for s in data_de])
    means_en = np.array([s.mean() if len(s) > 0 else np.nan for s in data_en])
    stds_en = np.array([s.std() if len(s) > 1 else 0 for s in data_en])
    means_all = np.array([s.mean() if len(s) > 0 else np.nan for s in data_all])
    stds_all = np.array([s.std() if len(s) > 1 else 0 for s in data_all])

    # Asymmetric SD: [lower_distances, upper_distances] to avoid negative latencies
    lower_err_de = np.minimum(stds_de, means_de)
    xerr_de = [lower_err_de, stds_de]

    lower_err_en = np.minimum(stds_en, means_en)
    xerr_en = [lower_err_en, stds_en]
    
    lower_err_all = np.minimum(stds_all, means_all)
    xerr_all = [lower_err_all, stds_all]
        
    # Install errorbars
    ax.errorbar(means_de, positions_de, xerr=xerr_de, fmt='o', color='#8b549c', capsize=5,
                label='German SD', markersize=8, linewidth=1)
    ax.errorbar(means_en, positions_en, xerr=xerr_en, fmt='o', color='#95a23c', capsize=5,
                label='English SD', markersize=8, linewidth=1)
    ax.errorbar(means_all, positions_all, xerr=xerr_all, fmt='o', color='#bb9e7b', capsize=5,
                label='All SD', markersize=8, linewidth=1)

    # Run B label (upper right)
    run_b_german = Patch(facecolor='#8b549c', alpha=0.9, label=f'DE: {means_de[1]:.2f}±{stds_de[1]:.2f}')
    run_b_english = Patch(facecolor='#95a23c', alpha=0.9, label=f'EN: {means_en[1]:.2f}±{stds_en[1]:.2f}')
    run_b_all = Patch(facecolor='#bb9e7b', alpha=0.9, label=f'ALL: {means_all[1]:.2f}±{stds_all[1]:.2f}')
    run_b_legend = ax.legend(handles=[run_b_all, run_b_german, run_b_english], loc='upper right', fontsize=9, frameon=True, fancybox=True)

    # Run E label (lower right)
    run_e_german = Patch(facecolor='#8b549c', alpha=0.9, label=f'DE: {means_de[0]:.2f}±{stds_de[0]:.2f}')
    run_e_english = Patch(facecolor='#95a23c', alpha=0.9, label=f'EN: {means_en[0]:.2f}±{stds_en[0]:.2f}')
    run_e_all = Patch(facecolor='#bb9e7b', alpha=0.9, label=f'ALL: {means_all[0]:.2f}±{stds_all[0]:.2f}')
    run_e_legend = ax.legend(handles=[run_e_all, run_e_german, run_e_english], loc='lower right', fontsize=9, frameon=True, fancybox=True)

    # Add legend
    ax.add_artist(run_b_legend)   
    ax.add_artist(run_e_legend)    

    # Styling
    ax.set_ylabel('Runs', fontsize=13)
    ax.set_xlabel('Latency (s)', fontsize=11)
    ax.set_yticks([centers[0], centers[1]])
    ax.set_yticklabels(['Run E', 'Run B'], ha='right', va='center', fontsize=11, rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_xlim(-10, 225)

    plt.tight_layout()

    # Save figure 
    plt.savefig('latency/latency_final_edit.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_latency('data/baseline_trace.csv', 'data/config_1_trace.csv')


