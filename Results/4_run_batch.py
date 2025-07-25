import clingo
import time
import statistics
import json
import multiprocessing
import csv
import os
import glob

TIMEOUT = 10
NUMBER_OF_RUNS = 1

def create_encodings():
    encodings = {
        1: "../scripts/original_encoding.lp",
        2: "../scripts/optimized_encoding.lp", 
        3: "../scripts/original_encoding_plus_heuristic.lp",
        4: "../scripts/optimized_encoding_plus_heuristic.lp"
    }
    return encodings

def find_input_files():
    input_files = []
    patterns = [
        "../scripts/input/days_1/input*.lp",
        "../scripts/input/days_2/input*.lp", 
        "../scripts/input/days_3/input*.lp",
        "../scripts/input/days_5/input*.lp"
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        input_files.extend(sorted(files))
    
    return input_files

def create_settings_configs():
    encodings = create_encodings()
    input_files = find_input_files()
    
    settings_configs = {}
    config_id = 1
    
    for input_file in input_files:
        for encoding_id, encoding_file in encodings.items():
            settings_configs[config_id] = {
                "control_args": ["1", "--opt-mode=optN"],
                "timeout_seconds": TIMEOUT,
                "base_encoding": encoding_file,
                "input_file": input_file,
                "runs": NUMBER_OF_RUNS
            }
            config_id += 1
    
    return settings_configs

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
    best_model_time = None
    model_count = 0
    start_time = time.time()
    
    def on_model(model):
        nonlocal best_cost, best_model_time, model_count
        model_count += 1
        current_cost = model.cost
        elapsed = time.time() - start_time
        
        if best_cost is None or is_cost_better(current_cost, best_cost):
            best_cost = current_cost
            best_model_time = elapsed
        
        print(f"Model {model_count}: Cost = {current_cost} (tempo: {elapsed:.2f}s)")
        
        with lock:
            shared_dict['best_cost'] = best_cost
            shared_dict['best_model_time'] = best_model_time
            shared_dict['model_count'] = model_count
            shared_dict['elapsed_time'] = time.time() - start_time
        
        return True
    
    try:
        ctl = clingo.Control(control_args)
        ctl.add("base", [], input_data)
        ctl.add("base", [], encoding)
        ctl.ground([("base", [])])
        
        result = ctl.solve(on_model=on_model)
        
        elapsed_time = time.time() - start_time
        
        with lock:
            shared_dict['best_cost'] = best_cost
            shared_dict['best_model_time'] = best_model_time
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
            shared_dict['best_model_time'] = best_model_time
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
    shared_dict['best_model_time'] = None
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
        print(f"\n*** TIMEOUT RAGGIUNTO DOPO {timeout_seconds} SECONDI - KILLING PROCESS ***")
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
                'best_model_time': shared_dict.get('best_model_time', None),
                'model_count': shared_dict.get('model_count', 0),
                'result_status': "TIMEOUT",
                'solver_stats': shared_dict.get('solver_stats', {}),
                'timed_out': True
            }
        
        print(f"*** TIMEOUT CONFERMATO ***")
        return result_data
    
    with lock:
        result_data = {
            'elapsed_time': shared_dict.get('elapsed_time', time.time() - start_time),
            'best_cost': shared_dict.get('best_cost', None),
            'best_model_time': shared_dict.get('best_model_time', None),
            'model_count': shared_dict.get('model_count', 0),
            'result_status': shared_dict.get('result_status', "NO_RESULT"),
            'solver_stats': shared_dict.get('solver_stats', {}),
            'timed_out': False
        }
    
    return result_data

def run_configuration(config, config_id):
    print(f"=== CONFIGURAZIONE {config_id} ===")
    print(f"Input: {config['input_file']}")
    print(f"Encoding: {config['base_encoding']}")
    print(f"Timeout: {config['timeout_seconds']} secondi")
    print(f"Runs: {config['runs']}")
    print()
    
    try:
        with open(config["input_file"], "r", encoding="utf-8") as f:
            input_data = f.read()
        
        with open(config["base_encoding"], "r", encoding="utf-8") as f:
            encoding = f.read()
    except FileNotFoundError as e:
        print(f"File non trovato: {e}")
        return []
    
    results = []
    
    for i in range(config["runs"]):
        print(f"-- Run {i + 1} --")
        
        result = run_clingo_with_timeout(
            input_data, 
            encoding, 
            config["timeout_seconds"], 
            config["control_args"]
        )
        
        result['config_id'] = config_id
        result['input_file'] = config['input_file']
        result['base_encoding'] = config['base_encoding']
        result['run_number'] = i + 1
        
        results.append(result)
        
        print(f"Tempo: {result['elapsed_time']:.4f}s")
        print(f"Miglior costo: {result['best_cost']}")
        print(f"Tempo miglior modello: {result.get('best_model_time', 'N/A')}")
        print(f"Modelli trovati: {result['model_count']}")
        print(f"Status: {result['result_status']}")
        if result['timed_out']:
            print("*** TIMEOUT CONFERMATO ***")
        print()
    
    return results

def save_results_to_csv(all_results, filename="results.csv"):
    if not all_results:
        print("Nessun risultato da salvare")
        return
    
    output_path = os.path.join(".", filename)
    
    fieldnames = [
        'config_id', 'input_file', 'base_encoding', 'run_number',
        'cost_1', 'cost_2', 'elapsed_time', 'best_model_time', 'model_count',
        'result_status', 'timed_out'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            cost = result['best_cost']
            cost_1 = cost_2 = None
            
            if cost is not None:
                if isinstance(cost, list):
                    if len(cost) >= 1:
                        cost_1 = cost[0]
                    if len(cost) >= 2:
                        cost_2 = cost[1]
                else:
                    cost_1 = cost
            
            row = {
                'config_id': result['config_id'],
                'input_file': result['input_file'],
                'base_encoding': result['base_encoding'],
                'run_number': result['run_number'],
                'cost_1': cost_1,
                'cost_2': cost_2,
                'elapsed_time': round(result['elapsed_time'], 4),
                'best_model_time': round(result['best_model_time'], 4) if result.get('best_model_time') is not None else None,
                'model_count': result['model_count'],
                'result_status': result['result_status'],
                'timed_out': result['timed_out'],
            }
            
            writer.writerow(row)
    
    print(f"Risultati salvati in {output_path}")

def main():
    settings_configs = create_settings_configs()
    
    all_results = []
    
    for config_id in sorted(settings_configs.keys()):
        config = settings_configs[config_id]
        config_results = run_configuration(config, config_id)
        all_results.extend(config_results)
    
    save_results_to_csv(all_results)
    
    print(f"\n=== RIEPILOGO TOTALE ===")
    print(f"Configurazioni eseguite: {len(settings_configs)}")
    print(f"Runs totali: {len(all_results)}")
    
    valid_results = [r for r in all_results if r['best_cost'] is not None]
    timeouts = sum(1 for r in all_results if r['timed_out'])
    
    print(f"Risultati validi: {len(valid_results)}")
    print(f"Timeout: {timeouts}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()