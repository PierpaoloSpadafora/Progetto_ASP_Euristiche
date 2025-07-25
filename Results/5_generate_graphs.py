#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from pathlib import Path

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    
    df['encoding_type'] = df['base_encoding'].apply(lambda x: 
        'original' if 'original_encoding.lp' in x 
        else 'optimized' if 'optimized_encoding.lp' in x
        else 'original_plus_heuristic' if 'original_encoding_plus_heuristic.lp' in x
        else 'optimized_plus_heuristic')
    
    df['days'] = df['input_file'].str.extract(r'days_(\d+)').astype(int)
    
    df['input_num'] = df['input_file'].str.extract(r'input(\d+)\.lp').astype(int)
    
    df['test_case'] = df['days'].astype(str) + '_' + df['input_num'].astype(str)
    
    return df

def create_cost_comparison_charts(df, encoding1, encoding2, title_prefix=""):
    """Crea bar charts per confrontare i costi tra due encoding"""
    df_filtered = df[df['encoding_type'].isin([encoding1, encoding2])]
    
    # Ordina per test_case per avere un ordine consistente
    test_cases = sorted(df_filtered['test_case'].unique())
    n_cases = len(test_cases)
    
    # Dividi in gruppi di 10
    n_groups = (n_cases + 9) // 10  # Ceiling division
    
    fig, axes = plt.subplots(n_groups, 2, figsize=(16, 4 * n_groups))
    if n_groups == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{title_prefix} - Confronto Costi (Lower is Better)', fontsize=16, fontweight='bold')
    
    for group in range(n_groups):
        start_idx = group * 10
        end_idx = min(start_idx + 10, n_cases)
        group_cases = test_cases[start_idx:end_idx]
        
        # Prepara dati per il gruppo corrente
        cost1_data = {encoding1: [], encoding2: []}
        cost2_data = {encoding1: [], encoding2: []}
        case_labels = []
        
        for case in group_cases:
            case_labels.append(case)
            for encoding in [encoding1, encoding2]:
                case_data = df_filtered[(df_filtered['test_case'] == case) & 
                                      (df_filtered['encoding_type'] == encoding)]
                
                if len(case_data) > 0 and not pd.isna(case_data.iloc[0]['cost_1']):
                    cost1_data[encoding].append(case_data.iloc[0]['cost_1'])
                    cost2_data[encoding].append(case_data.iloc[0]['cost_2'])
                else:
                    cost1_data[encoding].append(np.nan)
                    cost2_data[encoding].append(np.nan)
        
        # Grafico Cost_1
        ax1 = axes[group, 0]
        x = np.arange(len(case_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cost1_data[encoding1], width, 
                       label=encoding1, alpha=0.8, color='#1f77b4')
        bars2 = ax1.bar(x + width/2, cost1_data[encoding2], width, 
                       label=encoding2, alpha=0.8, color='#ff7f0e')
        
        ax1.set_xlabel('Test Case')
        ax1.set_ylabel('Cost_1')
        ax1.set_title(f'Cost_1 Comparison - Group {group + 1} (Lower is Better)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(case_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar in bars1:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        # Grafico Cost_2
        ax2 = axes[group, 1]
        
        bars3 = ax2.bar(x - width/2, cost2_data[encoding1], width, 
                       label=encoding1, alpha=0.8, color='#1f77b4')
        bars4 = ax2.bar(x + width/2, cost2_data[encoding2], width, 
                       label=encoding2, alpha=0.8, color='#ff7f0e')
        
        ax2.set_xlabel('Test Case')
        ax2.set_ylabel('Cost_2')
        ax2.set_title(f'Cost_2 Comparison - Group {group + 1} (Lower is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(case_labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar in bars3:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        for bar in bars4:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

def create_time_comparison_chart(df, encoding1, encoding2, title_prefix=""):
    """Crea line chart per confrontare i tempi di esecuzione"""
    df_filtered = df[df['encoding_type'].isin([encoding1, encoding2])]
    test_cases = sorted(df_filtered['test_case'].unique())
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    times1 = []
    times2 = []
    case_labels = []
    
    for case in test_cases:
        case_labels.append(case)
        
        # Dati per encoding1
        case_data1 = df_filtered[(df_filtered['test_case'] == case) & 
                                (df_filtered['encoding_type'] == encoding1)]
        if len(case_data1) > 0 and not pd.isna(case_data1.iloc[0]['best_model_time']):
            times1.append(case_data1.iloc[0]['best_model_time'])
        else:
            times1.append(np.nan)
        
        # Dati per encoding2
        case_data2 = df_filtered[(df_filtered['test_case'] == case) & 
                                (df_filtered['encoding_type'] == encoding2)]
        if len(case_data2) > 0 and not pd.isna(case_data2.iloc[0]['best_model_time']):
            times2.append(case_data2.iloc[0]['best_model_time'])
        else:
            times2.append(np.nan)
    
    x = np.arange(len(case_labels))
    
    # Line plot
    ax.plot(x, times1, marker='o', linewidth=2, markersize=6, 
           label=encoding1, color='#1f77b4')
    ax.plot(x, times2, marker='s', linewidth=2, markersize=6, 
           label=encoding2, color='#ff7f0e')
    
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Best Model Time (seconds)')
    ax.set_title(f'{title_prefix} - Confronto Tempi di Esecuzione')
    ax.set_xticks(x[::2])  # Mostra ogni secondo label per evitare sovrapposizioni
    ax.set_xticklabels([case_labels[i] for i in range(0, len(case_labels), 2)], 
                      rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Evidenzia quale encoding è più veloce per ogni caso
    for i, (t1, t2) in enumerate(zip(times1, times2)):
        if not np.isnan(t1) and not np.isnan(t2):
            if t1 < t2:
                ax.scatter(i, t1, color='green', s=100, marker='*', zorder=5)
            elif t2 < t1:
                ax.scatter(i, t2, color='green', s=100, marker='*', zorder=5)
    
    # Aggiungi legenda per le stelle
    ax.scatter([], [], color='green', s=100, marker='*', label='Faster encoding')
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_performance_summary_table(df):
    encodings = ['original', 'optimized', 'original_plus_heuristic', 'optimized_plus_heuristic']
    
    summary_data = []
    for encoding in encodings:
        data = df[df['encoding_type'] == encoding]
        
        total_tests = len(data)
        successful_tests = data['cost_1'].notna().sum()
        success_rate = (successful_tests / total_tests) * 100
        
        avg_cost1 = data['cost_1'].mean()
        avg_cost2 = data['cost_2'].mean()
        avg_best_time = data['best_model_time'].mean()
        avg_models = data['model_count'].mean()
        
        summary_data.append({
            'Encoding': encoding,
            'Test Totali': total_tests,
            'Successi': successful_tests,
            'Tasso Successo (%)': f"{success_rate:.1f}",
            'Cost_1 Medio': f"{avg_cost1:.2f}" if not pd.isna(avg_cost1) else "N/A",
            'Cost_2 Medio': f"{avg_cost2:.2f}" if not pd.isna(avg_cost2) else "N/A",
            'Tempo Miglior Modello (s)': f"{avg_best_time:.3f}" if not pd.isna(avg_best_time) else "N/A",
            'Modelli Medi': f"{avg_models:.1f}" if not pd.isna(avg_models) else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def create_difficulty_analysis(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Analisi Difficoltà per Numero di Giorni', fontsize=16, fontweight='bold')
    
    days = sorted(df['days'].unique())
    
    # 1. Tasso di successo per giorni
    ax1 = axes[0]
    success_rates_by_days = []
    for day in days:
        day_data = df[df['days'] == day]
        success_rate = (day_data['cost_1'].notna().sum() / len(day_data)) * 100
        success_rates_by_days.append(success_rate)
    
    bars = ax1.bar(days, success_rates_by_days, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Giorni')
    ax1.set_ylabel('Tasso di successo (%)')
    ax1.set_title('Tasso di successo per difficoltà')
    ax1.set_xticks(days)
    
    # Aggiungi valori sulle barre
    for bar, rate in zip(bars, success_rates_by_days):
        height = bar.get_height()
        ax1.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 2. Cost_1 medio per giorni
    ax2 = axes[1]
    avg_cost1_by_days = df.groupby('days')['cost_1'].mean()
    ax2.plot(avg_cost1_by_days.index, avg_cost1_by_days.values, 
            marker='o', linewidth=3, markersize=8, color='red')
    ax2.set_xlabel('Giorni')
    ax2.set_ylabel('Cost_1 medio')
    ax2.set_title('Cost_1 medio per difficoltà')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(days)
    
    # 3. Tempo medio per giorni
    ax3 = axes[2]
    avg_time_by_days = df.groupby('days')['best_model_time'].mean()
    ax3.plot(avg_time_by_days.index, avg_time_by_days.values, 
            marker='s', linewidth=3, markersize=8, color='green')
    ax3.set_xlabel('Giorni')
    ax3.set_ylabel('Tempo medio (s)')
    ax3.set_title('Tempo medio del miglior modello')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(days)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Genera grafici comparativi per gli encoding')
    parser.add_argument('--output-dir', '-o', default='./graphs', 
                       help='Directory di output per i grafici (default: ./graphs)')
    
    args = parser.parse_args()
    
    # Crea la directory di output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    csv_file = './results.csv'
    print(f"Caricamento dati da {csv_file}...")
    df = load_and_preprocess_data(csv_file)
    
    print(f"Dati caricati: {len(df)} record con {df['encoding_type'].nunique()} encoding diversi")
    print(f"Encoding trovati: {df['encoding_type'].unique().tolist()}")
    
    # 1. Confronto costi Original vs Optimized
    print("Generando confronto costi Original vs Optimized...")
    fig1 = create_cost_comparison_charts(df, 'original', 'optimized', "Original vs Optimized")
    fig1.savefig(output_dir / 'cost_comparison_original_vs_optimized.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Confronto costi con euristiche
    print("Generando confronto costi con euristiche...")
    fig2 = create_cost_comparison_charts(df, 'original_plus_heuristic', 'optimized_plus_heuristic', 
                                        "Original+Heuristic vs Optimized+Heuristic")
    fig2.savefig(output_dir / 'cost_comparison_heuristic.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Confronto tempi Original vs Optimized
    print("Generando confronto tempi Original vs Optimized...")
    fig3 = create_time_comparison_chart(df, 'original', 'optimized', "Original vs Optimized")
    fig3.savefig(output_dir / 'time_comparison_original_vs_optimized.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Confronto tempi con euristiche
    print("Generando confronto tempi con euristiche...")
    fig4 = create_time_comparison_chart(df, 'original_plus_heuristic', 'optimized_plus_heuristic', 
                                       "Original+Heuristic vs Optimized+Heuristic")
    fig4.savefig(output_dir / 'time_comparison_heuristic.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Tabella riassuntiva
    print("Generando tabella riassuntiva...")
    summary_table = create_performance_summary_table(df)
    summary_table.to_csv(output_dir / 'performance_summary.csv', index=False)
    
    # Stampa la tabella riassuntiva
    print("\n" + "="*80)
    print("TABELLA RIASSUNTIVA DELLE PERFORMANCE")
    print("="*80)
    print(summary_table.to_string(index=False))
    
    print(f"\nGrafici salvati in: {output_dir.absolute()}")
    print("File generati:")
    print("- cost_comparison_original_vs_optimized.png")
    print("- cost_comparison_heuristic.png")
    print("- time_comparison_original_vs_optimized.png")
    print("- time_comparison_heuristic.png")
    print("- performance_summary.csv")

if __name__ == "__main__":
    main()
    print("\n" + "="*80)
    print("TABELLA RIASSUNTIVA DELLE PERFORMANCE")
    print("="*80)
    print(summary_table.to_string(index=False))
    
    print(f"\nGrafici salvati in: {output_dir.absolute()}")
    print("File generati:")
    print("- comparison_original_vs_optimized.png")
    print("- comparison_heuristic.png") 
    print("- comprehensive_comparison.png")
    print("- difficulty_analysis.png")
    print("- performance_summary.csv")

if __name__ == "__main__":
    main()
