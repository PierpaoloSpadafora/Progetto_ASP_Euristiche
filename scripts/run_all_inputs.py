
import clingo
import time
import statistics
import json
import multiprocessing
from pathlib import Path
import pandas as pd
import re

def load_config():
    with open("settings.json", "r", encoding="utf-8") as f:
        return json.load(f)

def cost_to_comparable(cost):
    if cost is None:
        return [float('inf')]
    if isinstance(cost, list):
        return cost
    return [cost]

def is_cost_better(cost1, cost2):
    if cost1 is None:
        return False
    if cost2 is None:
        return True
    
    comp1 = cost_to_comparable(cost1)
    comp2 = cost_to_comparable(cost2)

    for c1, c2 in zip(comp1, comp2):
        if c1 < c2:
            return True
        elif c1 > c2:
            return False
    
    return len(comp1) < len(comp2)

def run_clingo_worker(input_data, encoding, control_args, timeout_seconds, shared_dict, lock):
    best_cost = None
    model_count = 0
    start_time = time.time()
    
    def on_model(model):
        nonlocal best_cost, model_count
        model_count += 1
        current_cost = model.cost
        
        if best_cost is None or is_cost_better(current_cost, best_cost):
            best_cost = current_cost
        
        with lock:
            shared_dict['best_cost'] = best_cost
            shared_dict['model_count'] = model_count
            shared_dict['elapsed_time'] = time.time() - start_time
        
        return True
    
    try:
        ctl = clingo.Control(control_args)
        ctl.add("base", [], input_data)
        ctl.add("base", [], encoding)
        ctl.ground([("base", [])])
        
        result = ctl.solve(on_model=on_model)
        
        try:
            stats = ctl.statistics['solving']['solvers']
            solver_stats = {
                'choices': stats.get('choices', 0),
                'conflicts': stats.get('conflicts', 0)
            }
        except:
            solver_stats = {'choices': 0, 'conflicts': 0}
        
        elapsed_time = time.time() - start_time
        
        with lock:
            shared_dict['best_cost'] = best_cost  
            shared_dict['elapsed_time'] = elapsed_time
            shared_dict['result_status'] = str(result)
            shared_dict['solver_stats'] = solver_stats
            shared_dict['timed_out'] = False
            shared_dict['completed'] = True
            shared_dict['model_count'] = model_count
        
    except Exception as e:
        print(f"Errore nel worker: {e}")
        with lock:
            shared_dict['best_cost'] = best_cost 
            shared_dict['elapsed_time'] = time.time() - start_time
            shared_dict['result_status'] = "ERROR"
            shared_dict['solver_stats'] = {}
            shared_dict['timed_out'] = False
            shared_dict['completed'] = True
            shared_dict['model_count'] = model_count

def run_clingo_with_timeout(input_data, encoding, timeout_seconds, control_args):

    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    lock = manager.Lock()
    
    shared_dict['best_cost'] = None
    shared_dict['model_count'] = 0
    shared_dict['elapsed_time'] = 0.0
    shared_dict['result_status'] = "UNKNOWN"
    shared_dict['solver_stats'] = {}
    shared_dict['timed_out'] = False
    shared_dict['completed'] = False
    
    start_time = time.time()
    
    worker = multiprocessing.Process(
        target=run_clingo_worker,
        args=(input_data, encoding, control_args, timeout_seconds, shared_dict, lock)
    )
    worker.start()
    
    worker.join(timeout=timeout_seconds)
    
    if worker.is_alive():
        print(f"   *** TIMEOUT RAGGIUNTO DOPO {timeout_seconds} SECONDI - KILLING PROCESS ***")
        worker.terminate()
        worker.join(timeout=1) 
        
        if worker.is_alive():
            worker.kill()
            worker.join()
        
        with lock:
            elapsed_time = time.time() - start_time
            result_data = {
                'elapsed_time': elapsed_time,
                'best_cost': shared_dict.get('best_cost', None),
                'model_count': shared_dict.get('model_count', 0),
                'result_status': "TIMEOUT",
                'solver_stats': shared_dict.get('solver_stats', {}),
                'timed_out': True
            }
        
        return result_data
    
    with lock:
        result_data = {
            'elapsed_time': shared_dict.get('elapsed_time', time.time() - start_time),
            'best_cost': shared_dict.get('best_cost', None),
            'model_count': shared_dict.get('model_count', 0),
            'result_status': shared_dict.get('result_status', "NO_RESULT"),
            'solver_stats': shared_dict.get('solver_stats', {}),
            'timed_out': False
        }
    
    return result_data

def find_all_input_files():
    input_dir = Path("input")
    if not input_dir.exists():
        print("Cartella input/ non trovata!")
        return []
    
    input_files = list(input_dir.rglob("*.lp"))
    
    input_files.sort(key=lambda x: str(x))
    
    return input_files

def run_multiple_times(input_data, encoding, runs, timeout_seconds, control_args):
    results = []
    for i in range(runs):
        print(f"   → Run {i+1}/{runs}:", flush=True)
        
        result = run_clingo_with_timeout(input_data, encoding, timeout_seconds, control_args)
        results.append(result)
        
        print(f"     Tempo: {result['elapsed_time']:.4f}s, Costo: {result['best_cost']}, "
              f"Modelli: {result['model_count']}, Status: {result['result_status']}")
        if result['timed_out']:
            print("     *** TIMEOUT CONFERMATO ***")
    
    times = [r['elapsed_time'] for r in results]
    valid_costs = [r['best_cost'] for r in results if r['best_cost'] is not None]
    model_counts = [r['model_count'] for r in results]
    timeouts = sum(1 for r in results if r['timed_out'])
    
    best_cost = None
    if valid_costs:
        best_cost = valid_costs[0]
        for cost in valid_costs[1:]:
            if is_cost_better(cost, best_cost):
                best_cost = cost
    
    stats = {
        'results': results,
        'avg_time': statistics.mean(times),
        'avg_models': statistics.mean(model_counts),
        'timeouts': timeouts,
        'total_runs': runs,
        'best_cost': best_cost,
        'valid_solutions': len(valid_costs)
    }
    
    print(f"   → Tempo medio: {stats['avg_time']:.4f}s")
    print(f"   → Modelli medi: {stats['avg_models']:.1f}")
    print(f"   → Timeout: {timeouts}/{runs}")
    if stats['best_cost'] is not None:
        print(f"   → Miglior costo: {stats['best_cost']}")
    print()
    
    return stats

def get_cost_components(cost):
    if cost is None:
        return []
    if isinstance(cost, list):
        return cost
    return [cost]

def calculate_max_cost_components(all_results):
    max_components = 0
    for result in all_results:
        if result['stats']['best_cost'] is not None:
            components = get_cost_components(result['stats']['best_cost'])
            max_components = max(max_components, len(components))
    return max_components

def create_results_dataframe(results, max_cost_components):
    total_rows = 8 + len(results)
    total_cols = 5 + max_cost_components
    
    column_names = list(range(total_cols))
    df = pd.DataFrame('', index=range(total_rows), columns=column_names)
    
    df.iloc[2, 1] = "Riassunto tempi e costi - Original Encoding su tutti gli input"
    df.iloc[4, 2] = f"Costi migliori\n(media su runs)"
    df.iloc[4, 2 + max_cost_components] = f"Tempo di esecuzione\n(media su runs)"
    df.iloc[4, 3 + max_cost_components] = "Timeout"
    
    df.iloc[6, 1] = "File Input"
    
    for i in range(max_cost_components):
        df.iloc[6, 2 + i] = f"costo{i+1}"
    
    df.iloc[6, 2 + max_cost_components] = "tempo (s)"
    df.iloc[6, 3 + max_cost_components] = "timeout"
    
    return df

def sort_results_by_input_name(results):
    return sorted(results, key=lambda x: x['input_file'])

def write_results_to_excel(results, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    sorted_results = sort_results_by_input_name(results)
    
    max_cost_components = calculate_max_cost_components(sorted_results)
    if max_cost_components == 0:
        max_cost_components = 1
    
    df = create_results_dataframe(sorted_results, max_cost_components)
    
    current_row = 7 
    
    for result in sorted_results:
        input_name = result['input_file']
        stats = result['stats']
        
        df.iloc[current_row, 1] = input_name
        
        cost_components = get_cost_components(stats['best_cost'])
        for j in range(max_cost_components):
            if j < len(cost_components):
                df.iloc[current_row, 2 + j] = cost_components[j]
            else:
                df.iloc[current_row, 2 + j] = ''
        
        df.iloc[current_row, 2 + max_cost_components] = round(stats['avg_time'], 3)
        
        timeout_ratio = f"{stats['timeouts']}/{stats['total_runs']}"
        df.iloc[current_row, 3 + max_cost_components] = timeout_ratio
        
        current_row += 1
    
    try:
        import openpyxl
        from openpyxl.styles import Alignment
        
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Original_Encoding_Results", index=False, header=False)
            
            ws = writer.sheets['Original_Encoding_Results']
            
            ws['C5'].alignment = Alignment(wrap_text=True, vertical='center', horizontal='center')
            time_col = chr(ord('A') + 2 + max_cost_components)
            ws[f'{time_col}5'].alignment = Alignment(wrap_text=True, vertical='center', horizontal='center')
        
        print(f"Risultati salvati in → {output_file}")
        print(f"Scritte {len(sorted_results)} righe di risultati")
        
    except ImportError:
        print("ERRORE: openpyxl non disponibile! Installa con: pip install openpyxl")
    except Exception as e:
        print(f"Errore nel salvataggio Excel: {e}")

def main():
    config = load_config()
    
    print("=== RUN ORIGINAL ENCODING ON ALL INPUTS ===")
    print(f"Configurazione da settings.json:")
    print(f"  - Timeout: {config['timeout_seconds']} secondi")
    print(f"  - Runs per input: {config['runs']}")
    print(f"  - Encoding: {config['base_encoding']}")
    print(f"  - Control args: {config['control_args']}")
    print()
    
    input_files = find_all_input_files()
    
    if not input_files:
        print("Nessun file .lp trovato nella cartella input/")
        return
    
    print(f"Trovati {len(input_files)} file input:")
    for f in input_files:
        print(f"  - {f}")
    print()
    
    try:
        with open(config["base_encoding"], "r", encoding="utf-8") as f:
            encoding = f.read()
    except FileNotFoundError:
        print(f"Errore: file encoding {config['base_encoding']} non trovato!")
        return
    
    results = []
    
    for i, input_file in enumerate(input_files, 1):
        print(f"=== PROCESSING {i}/{len(input_files)}: {input_file} ===")
        
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                input_data = f.read()
            
            stats = run_multiple_times(
                input_data,
                encoding,
                config["runs"],
                config["timeout_seconds"],
                config["control_args"]
            )
            
            results.append({
                'input_file': str(input_file),
                'stats': stats
            })
            
            print(f"Completato: {input_file}")
            if stats['best_cost'] is not None:
                print(f"Miglior costo: {stats['best_cost']}")
            else:
                print("Nessuna soluzione trovata")
            print(f"Tempo medio: {stats['avg_time']:.4f}s")
            print(f"Timeout: {stats['timeouts']}/{stats['total_runs']}")
            print()
            
        except Exception as e:
            print(f"Errore nell'elaborazione di {input_file}: {e}")
            print()
            continue
    
    if not results:
        print("Nessun risultato da salvare.")
        return
    
    output_file = Path(config["timings_output_dir"]) / "original_encoding_all_inputs.xlsx"
    write_results_to_excel(results, output_file)
    
    print(f"=== RIEPILOGO FINALE ===")
    print(f"File processati: {len(results)}")
    
    best_result = None
    for result in results:
        if result['stats']['best_cost'] is not None:
            if best_result is None or is_cost_better(result['stats']['best_cost'], best_result['stats']['best_cost']):
                best_result = result
    
    if best_result:
        print(f"Miglior risultato complessivo:")
        print(f"  File: {best_result['input_file']}")
        print(f"  Costo: {best_result['stats']['best_cost']}")
        print(f"  Tempo: {best_result['stats']['avg_time']:.4f}s")
    
    total_timeouts = sum(r['stats']['timeouts'] for r in results)
    total_runs = sum(r['stats']['total_runs'] for r in results)
    print(f"Timeout totali: {total_timeouts}/{total_runs} ({100*total_timeouts/total_runs:.1f}%)")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
