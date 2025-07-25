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

def natural_sort_key(test_case):
    """Funzione per ordinamento naturale dei test case (5_1, 5_2, ..., 5_9, 5_10)"""
    days, input_num = test_case.split('_')
    return (int(days), int(input_num))

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
    # Verifica che entrambi gli encoding esistano nei dati
    available_encodings = df['encoding_type'].unique()
    if encoding1 not in available_encodings:
        print(f"Warning: Encoding '{encoding1}' non trovato nei dati. Disponibili: {available_encodings}")
        return None
    if encoding2 not in available_encodings:
        print(f"Warning: Encoding '{encoding2}' non trovato nei dati. Disponibili: {available_encodings}")
        return None
    
    df_filtered = df[df['encoding_type'].isin([encoding1, encoding2])]
    
    # Verifica che ci siano dati dopo il filtro
    if len(df_filtered) == 0:
        print(f"Warning: Nessun dato trovato per gli encoding '{encoding1}' e '{encoding2}'")
        return None
    
    # Ordina per test_case usando ordinamento naturale
    test_cases = sorted(df_filtered['test_case'].unique(), key=natural_sort_key)
    n_cases = len(test_cases)
    
    # Verifica che ci siano test case
    if n_cases == 0:
        print(f"Warning: Nessun test case trovato per gli encoding '{encoding1}' e '{encoding2}'")
        return None
    
    # Dividi in gruppi di 10
    n_groups = (n_cases + 9) // 10  # Ceiling division
    
    fig, axes = plt.subplots(n_groups, 2, figsize=(16, 4 * n_groups))
    if n_groups == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{title_prefix} - Confronto Costi (Lower is Better)', fontsize=16, fontweight='bold')
    
    # Schema colori basato sui tipi di encoding
    color_map = {
        'original': '#1f77b4',  # Blu
        'optimized': '#ff7f0e',  # Arancione
        'original_plus_heuristic': "#69428d",  # Viola
        'optimized_plus_heuristic': "#ddbd30"  # Giallo
    }
    
    color1 = color_map.get(encoding1, '#1f77b4')
    color2 = color_map.get(encoding2, '#ff7f0e')
    
    for group in range(n_groups):
        start_idx = group * 10
        end_idx = min(start_idx + 10, n_cases)
        group_cases = test_cases[start_idx:end_idx]
        
        # Prepara dati per il gruppo corrente
        cost1_data = {encoding1: [], encoding2: []}
        cost2_data = {encoding1: [], encoding2: []}
        case_labels = []
        
        # Lista per tenere traccia delle differenze
        cost1_differences = []
        cost2_differences = []
        
        for case in group_cases:
            case_labels.append(case)
            
            # Raccogli dati per entrambi gli encoding
            case_data_enc1 = df_filtered[(df_filtered['test_case'] == case) & 
                                        (df_filtered['encoding_type'] == encoding1)]
            case_data_enc2 = df_filtered[(df_filtered['test_case'] == case) & 
                                        (df_filtered['encoding_type'] == encoding2)]
            
            # Cost_1
            cost1_enc1 = case_data_enc1.iloc[0]['cost_1'] if len(case_data_enc1) > 0 and not pd.isna(case_data_enc1.iloc[0]['cost_1']) else np.nan
            cost1_enc2 = case_data_enc2.iloc[0]['cost_1'] if len(case_data_enc2) > 0 and not pd.isna(case_data_enc2.iloc[0]['cost_1']) else np.nan
            
            cost1_data[encoding1].append(cost1_enc1)
            cost1_data[encoding2].append(cost1_enc2)
            
            # Cost_2
            cost2_enc1 = case_data_enc1.iloc[0]['cost_2'] if len(case_data_enc1) > 0 and not pd.isna(case_data_enc1.iloc[0]['cost_2']) else np.nan
            cost2_enc2 = case_data_enc2.iloc[0]['cost_2'] if len(case_data_enc2) > 0 and not pd.isna(case_data_enc2.iloc[0]['cost_2']) else np.nan
            
            cost2_data[encoding1].append(cost2_enc1)
            cost2_data[encoding2].append(cost2_enc2)
            
            # Controlla differenze
            cost1_diff = None
            cost2_diff = None
            
            if not np.isnan(cost1_enc1) and not np.isnan(cost1_enc2):
                if cost1_enc1 != cost1_enc2:
                    cost1_diff = 'better' if cost1_enc2 < cost1_enc1 else 'worse'
            
            if not np.isnan(cost2_enc1) and not np.isnan(cost2_enc2):
                if cost2_enc1 != cost2_enc2:
                    cost2_diff = 'better' if cost2_enc2 < cost2_enc1 else 'worse'
            
            cost1_differences.append(cost1_diff)
            cost2_differences.append(cost2_diff)
        
        # Grafico Cost_1
        ax1 = axes[group, 0]
        x = np.arange(len(case_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cost1_data[encoding1], width, 
                       label=encoding1, alpha=0.8, color=color1)
        bars2 = ax1.bar(x + width/2, cost1_data[encoding2], width, 
                       label=encoding2, alpha=0.8, color=color2)
        
        ax1.set_ylabel('Cost_1')
        if group == 0:
            ax1.set_title('Cost_1')
        ax1.set_xticks(x)
        ax1.set_xticklabels(case_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre e asterischi per differenze
        for i, (bar1, bar2, diff) in enumerate(zip(bars1, bars2, cost1_differences)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            if not np.isnan(height1):
                ax1.annotate(f'{height1:.0f}', xy=(bar1.get_x() + bar1.get_width() / 2, height1),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            if not np.isnan(height2):
                ax1.annotate(f'{height2:.0f}', xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                
                # Aggiungi asterisco per differenze
                if diff is not None:
                    max_height = max([h for h in cost1_data[encoding1] + cost1_data[encoding2] if not np.isnan(h)])
                    ax1.text(bar2.get_x() + bar2.get_width() / 2, height2 + max_height * 0.05,
                           '★', ha='center', va='bottom', fontsize=12, fontweight='bold',
                           color='#2ca02c' if diff == 'better' else '#d62728')
        
        # Grafico Cost_2
        ax2 = axes[group, 1]
        
        bars3 = ax2.bar(x - width/2, cost2_data[encoding1], width, 
                       label=encoding1, alpha=0.8, color=color1)
        bars4 = ax2.bar(x + width/2, cost2_data[encoding2], width, 
                       label=encoding2, alpha=0.8, color=color2)
        
        ax2.set_ylabel('Cost_2')
        if group == 0:
            ax2.set_title('Cost_2')
        ax2.set_xticks(x)
        ax2.set_xticklabels(case_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre e asterischi per differenze
        for i, (bar3, bar4, diff) in enumerate(zip(bars3, bars4, cost2_differences)):
            height3 = bar3.get_height()
            height4 = bar4.get_height()
            
            if not np.isnan(height3):
                ax2.annotate(f'{height3:.0f}', xy=(bar3.get_x() + bar3.get_width() / 2, height3),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            if not np.isnan(height4):
                ax2.annotate(f'{height4:.0f}', xy=(bar4.get_x() + bar4.get_width() / 2, height4),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                
                # Aggiungi asterisco per differenze
                if diff is not None:
                    max_height = max([h for h in cost2_data[encoding1] + cost2_data[encoding2] if not np.isnan(h)])
                    ax2.text(bar4.get_x() + bar4.get_width() / 2, height4 + max_height * 0.05,
                           '★', ha='center', va='bottom', fontsize=12, fontweight='bold',
                           color='#2ca02c' if diff == 'better' else '#d62728')
    
    # Aggiungi una singola legenda per tutta la figura
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color1, alpha=0.8, label=encoding1),
        Patch(facecolor=color2, alpha=0.8, label=encoding2),
        Patch(facecolor='#2ca02c', alpha=0.8, label=f'{encoding2} (better)'),
        Patch(facecolor='#d62728', alpha=0.8, label=f'{encoding2} (worse)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Lascia spazio per la legenda
    return fig

def create_time_comparison_chart(df, encoding1, encoding2, title_prefix=""):
    """Crea bar chart per confrontare i tempi di esecuzione tra due encoding"""
    # Verifica che entrambi gli encoding esistano nei dati
    available_encodings = df['encoding_type'].unique()
    if encoding1 not in available_encodings:
        print(f"Warning: Encoding '{encoding1}' non trovato nei dati. Disponibili: {available_encodings}")
        return None
    if encoding2 not in available_encodings:
        print(f"Warning: Encoding '{encoding2}' non trovato nei dati. Disponibili: {available_encodings}")
        return None
    
    df_filtered = df[df['encoding_type'].isin([encoding1, encoding2])]
    
    # Verifica che ci siano dati dopo il filtro
    if len(df_filtered) == 0:
        print(f"Warning: Nessun dato trovato per gli encoding '{encoding1}' e '{encoding2}'")
        return None
    
    # Ordina per test_case usando ordinamento naturale
    test_cases = sorted(df_filtered['test_case'].unique(), key=natural_sort_key)
    n_cases = len(test_cases)
    
    # Verifica che ci siano test case
    if n_cases == 0:
        print(f"Warning: Nessun test case trovato per gli encoding '{encoding1}' e '{encoding2}'")
        return None
    
    # Dividi in gruppi di 10
    n_groups = (n_cases + 9) // 10  # Ceiling division
    
    fig, axes = plt.subplots(n_groups, 1, figsize=(16, 4 * n_groups))
    if n_groups == 1:
        axes = [axes]
    
    fig.suptitle(f'{title_prefix} - Confronto Tempi di Esecuzione (Lower is Better)', fontsize=16, fontweight='bold')
    
    # Schema colori basato sui tipi di encoding
    color_map = {
        'original': '#1f77b4',  # Blu
        'optimized': '#ff7f0e',  # Arancione
        'original_plus_heuristic': "#69428d",  # Viola
        'optimized_plus_heuristic': "#ddbd30"  # Giallo
    }
    
    color1 = color_map.get(encoding1, '#1f77b4')
    color2 = color_map.get(encoding2, '#ff7f0e')
    
    for group in range(n_groups):
        start_idx = group * 10
        end_idx = min(start_idx + 10, n_cases)
        group_cases = test_cases[start_idx:end_idx]
        
        # Prepara dati per il gruppo corrente
        time_data = {encoding1: [], encoding2: []}
        case_labels = []
        
        # Lista per tenere traccia delle differenze
        time_differences = []
        
        for case in group_cases:
            case_labels.append(case)
            
            # Raccogli dati per entrambi gli encoding
            case_data_enc1 = df_filtered[(df_filtered['test_case'] == case) & 
                                        (df_filtered['encoding_type'] == encoding1)]
            case_data_enc2 = df_filtered[(df_filtered['test_case'] == case) & 
                                        (df_filtered['encoding_type'] == encoding2)]
            
            # Best model time
            time_enc1 = case_data_enc1.iloc[0]['best_model_time'] if len(case_data_enc1) > 0 and not pd.isna(case_data_enc1.iloc[0]['best_model_time']) else np.nan
            time_enc2 = case_data_enc2.iloc[0]['best_model_time'] if len(case_data_enc2) > 0 and not pd.isna(case_data_enc2.iloc[0]['best_model_time']) else np.nan
            
            time_data[encoding1].append(time_enc1)
            time_data[encoding2].append(time_enc2)
            
            # Controlla differenze
            time_diff = None
            
            if not np.isnan(time_enc1) and not np.isnan(time_enc2):
                # Mostra stella solo se la differenza supera i 2 secondi
                if abs(time_enc1 - time_enc2) > 2.0:
                    time_diff = 'better' if time_enc2 < time_enc1 else 'worse'
            
            time_differences.append(time_diff)
        
        # Grafico tempi
        ax = axes[group]
        x = np.arange(len(case_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, time_data[encoding1], width, 
                       label=encoding1, alpha=0.8, color=color1)
        bars2 = ax.bar(x + width/2, time_data[encoding2], width, 
                       label=encoding2, alpha=0.8, color=color2)
        
        ax.set_ylabel('Best Model Time (seconds)')
        if group == 0:
            ax.set_title('Tempi di Esecuzione')
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre e asterischi per differenze
        for i, (bar1, bar2, diff) in enumerate(zip(bars1, bars2, time_differences)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            if not np.isnan(height1):
                ax.annotate(f'{height1:.3f}', xy=(bar1.get_x() + bar1.get_width() / 2, height1),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            if not np.isnan(height2):
                ax.annotate(f'{height2:.3f}', xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                
                # Aggiungi asterisco per differenze
                if diff is not None:
                    max_height = max([h for h in time_data[encoding1] + time_data[encoding2] if not np.isnan(h)])
                    ax.text(bar2.get_x() + bar2.get_width() / 2, height2 + max_height * 0.05,
                           '★', ha='center', va='bottom', fontsize=12, fontweight='bold',
                           color='#2ca02c' if diff == 'better' else '#d62728')
    
    # Aggiungi una singola legenda per tutta la figura
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color1, alpha=0.8, label=encoding1),
        Patch(facecolor=color2, alpha=0.8, label=encoding2),
        Patch(facecolor='#2ca02c', alpha=0.8, label=f'{encoding2} (faster)'),
        Patch(facecolor='#d62728', alpha=0.8, label=f'{encoding2} (slower)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Lascia spazio per la legenda
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
    if fig1 is not None:
        fig1.savefig(output_dir / 'cost_comparison_original_vs_optimized.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
    
    # 2. Confronto costi con euristiche
    print("Generando confronto costi con euristiche...")
    fig2 = create_cost_comparison_charts(df, 'original_plus_heuristic', 'optimized_plus_heuristic', 
                                        "Original+Heuristic vs Optimized+Heuristic")
    if fig2 is not None:
        fig2.savefig(output_dir / 'cost_comparison_heuristic.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # 3. Confronto costi Original vs Original+Heuristic
    print("Generando confronto costi Original vs Original+Heuristic...")
    fig3 = create_cost_comparison_charts(df, 'original', 'original_plus_heuristic', 
                                        "Original vs Original+Heuristic")
    if fig3 is not None:
        fig3.savefig(output_dir / 'cost_comparison_original_vs_original_heuristic.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    # 4. Confronto costi Optimized vs Optimized+Heuristic
    print("Generando confronto costi Optimized vs Optimized+Heuristic...")
    fig4 = create_cost_comparison_charts(df, 'optimized', 'optimized_plus_heuristic', 
                                        "Optimized vs Optimized+Heuristic")
    if fig4 is not None:
        fig4.savefig(output_dir / 'cost_comparison_optimized_vs_optimized_heuristic.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
    
    # 5. Confronto tempi Original vs Optimized
    print("Generando confronto tempi Original vs Optimized...")
    fig5 = create_time_comparison_chart(df, 'original', 'optimized', "Original vs Optimized")
    if fig5 is not None:
        fig5.savefig(output_dir / 'time_comparison_original_vs_optimized.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
    
    # 6. Confronto tempi con euristiche
    print("Generando confronto tempi con euristiche...")
    fig6 = create_time_comparison_chart(df, 'original_plus_heuristic', 'optimized_plus_heuristic', 
                                       "Original+Heuristic vs Optimized+Heuristic")
    if fig6 is not None:
        fig6.savefig(output_dir / 'time_comparison_heuristic.png', dpi=300, bbox_inches='tight')
        plt.close(fig6)
    
    # 7. Confronto tempi Original vs Original+Heuristic
    print("Generando confronto tempi Original vs Original+Heuristic...")
    fig7 = create_time_comparison_chart(df, 'original', 'original_plus_heuristic', 
                                       "Original vs Original+Heuristic")
    if fig7 is not None:
        fig7.savefig(output_dir / 'time_comparison_original_vs_original_heuristic.png', dpi=300, bbox_inches='tight')
        plt.close(fig7)
    
    # 8. Confronto tempi Optimized vs Optimized+Heuristic
    print("Generando confronto tempi Optimized vs Optimized+Heuristic...")
    fig8 = create_time_comparison_chart(df, 'optimized', 'optimized_plus_heuristic', 
                                       "Optimized vs Optimized+Heuristic")
    if fig8 is not None:
        fig8.savefig(output_dir / 'time_comparison_optimized_vs_optimized_heuristic.png', dpi=300, bbox_inches='tight')
        plt.close(fig8)
    
    # 9. Tabella riassuntiva
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
    print("- cost_comparison_original_vs_original_heuristic.png")
    print("- cost_comparison_optimized_vs_optimized_heuristic.png")
    print("- time_comparison_original_vs_optimized.png")
    print("- time_comparison_heuristic.png")
    print("- time_comparison_original_vs_original_heuristic.png")
    print("- time_comparison_optimized_vs_optimized_heuristic.png")
    print("- performance_summary.csv")

if __name__ == "__main__":
    main()