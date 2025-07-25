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
        else 'original_plus_heuristics' if 'original_encoding_plus_heuristics.lp' in x
        else 'optimized_plus_heuristics')
    
    df['days'] = df['input_file'].str.extract(r'days_(\d+)').astype(int)
    
    df['input_num'] = df['input_file'].str.extract(r'input(\d+)\.lp').astype(int)
    
    df['test_case'] = df['days'].astype(str) + '_' + df['input_num'].astype(str)
    
    return df

def calculate_ranking_score(row):
    if pd.isna(row['cost_1']):
        return float('inf') 
    
    cost_1_norm = row['cost_1'] * 1000  # Peso maggiore per cost_1
    cost_2_norm = row['cost_2'] * 1    # Peso medio per cost_2
    time_norm = row['best_model_time'] if not pd.isna(row['best_model_time']) else 3.0
    
    return cost_1_norm + cost_2_norm + time_norm

def compare_encodings_basic(df, encoding1, encoding2, title_suffix=""):
    df_filtered = df[df['encoding_type'].isin([encoding1, encoding2])]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Confronto {encoding1} vs {encoding2} {title_suffix}', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    for encoding in [encoding1, encoding2]:
        data = df_filtered[df_filtered['encoding_type'] == encoding]
        avg_cost1_by_days = data.groupby('days')['cost_1'].mean()
        ax1.plot(avg_cost1_by_days.index, avg_cost1_by_days.values, 
                marker='o', linewidth=2, label=encoding)
    
    ax1.set_xlabel('Giorni')
    ax1.set_ylabel('Cost_1 medio')
    ax1.set_title('Cost_1 medio per numero di giorni')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for encoding in [encoding1, encoding2]:
        data = df_filtered[df_filtered['encoding_type'] == encoding]
        valid_times = data.dropna(subset=['best_model_time'])
        avg_time_by_days = valid_times.groupby('days')['best_model_time'].mean()
        ax2.plot(avg_time_by_days.index, avg_time_by_days.values, 
                marker='s', linewidth=2, label=encoding)
    
    ax2.set_xlabel('Giorni')
    ax2.set_ylabel('Best Model Time medio (s)')
    ax2.set_title('Tempo medio per trovare il miglior modello')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    cost1_data = []
    labels = []
    for encoding in [encoding1, encoding2]:
        data = df_filtered[df_filtered['encoding_type'] == encoding]['cost_1'].dropna()
        cost1_data.append(data)
        labels.append(encoding)
    
    ax3.boxplot(cost1_data, labels=labels)
    ax3.set_ylabel('Cost_1')
    ax3.set_title('Distribuzione Cost_1')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    for encoding in [encoding1, encoding2]:
        data = df_filtered[df_filtered['encoding_type'] == encoding]
        valid_data = data.dropna(subset=['cost_1', 'cost_2'])
        ax4.scatter(valid_data['cost_1'], valid_data['cost_2'], 
                   alpha=0.6, s=50, label=encoding)
    
    ax4.set_xlabel('Cost_1')
    ax4.set_ylabel('Cost_2')
    ax4.set_title('Cost_1 vs Cost_2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def compare_heuristics(df):
    heuristic_encodings = ['original_plus_heuristics', 'optimized_plus_heuristics']
    return compare_encodings_basic(df, heuristic_encodings[0], heuristic_encodings[1], 
                                 "(con Euristiche)")

def create_comprehensive_comparison(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confronto Completo di Tutti gli Encoding', fontsize=16, fontweight='bold')
    
    encodings = ['original', 'optimized', 'original_plus_heuristics', 'optimized_plus_heuristics']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Cost_1 medio per giorni
    ax1 = axes[0, 0]
    for i, encoding in enumerate(encodings):
        data = df[df['encoding_type'] == encoding]
        avg_cost1_by_days = data.groupby('days')['cost_1'].mean()
        ax1.plot(avg_cost1_by_days.index, avg_cost1_by_days.values, 
                marker='o', linewidth=2, label=encoding, color=colors[i])
    
    ax1.set_xlabel('Giorni')
    ax1.set_ylabel('Cost_1 medio')
    ax1.set_title('Cost_1 medio per numero di giorni')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Best Model Time medio
    ax2 = axes[0, 1]
    for i, encoding in enumerate(encodings):
        data = df[df['encoding_type'] == encoding]
        valid_times = data.dropna(subset=['best_model_time'])
        avg_time_by_days = valid_times.groupby('days')['best_model_time'].mean()
        ax2.plot(avg_time_by_days.index, avg_time_by_days.values, 
                marker='s', linewidth=2, label=encoding, color=colors[i])
    
    ax2.set_xlabel('Giorni')
    ax2.set_ylabel('Best Model Time medio (s)')
    ax2.set_title('Tempo medio per trovare il miglior modello')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Percentuale di successo (trovare almeno una soluzione)
    ax3 = axes[0, 2]
    success_rates = []
    for encoding in encodings:
        data = df[df['encoding_type'] == encoding]
        success_rate = (data['cost_1'].notna().sum() / len(data)) * 100
        success_rates.append(success_rate)
    
    bars = ax3.bar(encodings, success_rates, color=colors)
    ax3.set_ylabel('Percentuale di successo (%)')
    ax3.set_title('Percentuale di test con soluzione trovata')
    ax3.set_xticklabels(encodings, rotation=45, ha='right')
    
    # Aggiungi valori sulle barre
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax3.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 4. Heatmap delle performance per difficoltà
    ax4 = axes[1, 0]
    
    ranking_data = []
    for days in sorted(df['days'].unique()):
        day_scores = []
        for encoding in encodings:
            data = df[(df['encoding_type'] == encoding) & (df['days'] == days)]
            scores = data.apply(calculate_ranking_score, axis=1)
            avg_score = scores.mean() if len(scores) > 0 else float('inf')
            day_scores.append(avg_score if avg_score != float('inf') else np.nan)
        ranking_data.append(day_scores)
    
    ranking_matrix = np.array(ranking_data)
    im = ax4.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto')
    ax4.set_xticks(range(len(encodings)))
    ax4.set_xticklabels(encodings, rotation=45, ha='right')
    ax4.set_yticks(range(len(df['days'].unique())))
    ax4.set_yticklabels([f'{d} giorni' for d in sorted(df['days'].unique())])
    ax4.set_title('Heatmap Performance\n(scuro = migliore)')
    
    # 5. Modelli trovati per encoding
    ax5 = axes[1, 1]
    model_counts = []
    for encoding in encodings:
        data = df[df['encoding_type'] == encoding]
        avg_models = data['model_count'].mean()
        model_counts.append(avg_models)
    
    bars = ax5.bar(encodings, model_counts, color=colors)
    ax5.set_ylabel('Numero medio di modelli')
    ax5.set_title('Numero medio di modelli trovati')
    ax5.set_xticklabels(encodings, rotation=45, ha='right')
    
    # 6. Violin plot per Cost_1
    ax6 = axes[1, 2]
    cost1_data = []
    for encoding in encodings:
        data = df[df['encoding_type'] == encoding]['cost_1'].dropna()
        cost1_data.append(data)
    
    parts = ax6.violinplot(cost1_data, positions=range(len(encodings)), showmeans=True)
    ax6.set_xticks(range(len(encodings)))
    ax6.set_xticklabels(encodings, rotation=45, ha='right')
    ax6.set_ylabel('Cost_1')
    ax6.set_title('Distribuzione Cost_1')
    
    plt.tight_layout()
    return fig

def create_performance_summary_table(df):
    encodings = ['original', 'optimized', 'original_plus_heuristics', 'optimized_plus_heuristics']
    
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
    
    # 1. Confronto Original vs Optimized
    print("Generando confronto Original vs Optimized...")
    fig1 = compare_encodings_basic(df, 'original', 'optimized', "(Encoding Base)")
    fig1.savefig(output_dir / 'comparison_original_vs_optimized.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Confronto con euristiche
    print("Generando confronto con euristiche...")
    fig2 = compare_heuristics(df)
    fig2.savefig(output_dir / 'comparison_heuristics.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Confronto completo
    print("Generando confronto completo...")
    fig3 = create_comprehensive_comparison(df)
    fig3.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Analisi difficoltà
    print("Generando analisi difficoltà...")
    fig4 = create_difficulty_analysis(df)
    fig4.savefig(output_dir / 'difficulty_analysis.png', dpi=300, bbox_inches='tight')
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
    print("- comparison_original_vs_optimized.png")
    print("- comparison_heuristics.png") 
    print("- comprehensive_comparison.png")
    print("- difficulty_analysis.png")
    print("- performance_summary.csv")

if __name__ == "__main__":
    main()
