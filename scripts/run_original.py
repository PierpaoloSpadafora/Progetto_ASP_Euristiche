
import clingo
import time
import statistics
import json
import multiprocessing

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
        
        elapsed = time.time() - start_time
        print(f"Model {model_count}: Cost = {current_cost} (tempo: {elapsed:.2f}s)")
        
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
            'model_count': shared_dict.get('model_count', 0),
            'result_status': shared_dict.get('result_status', "NO_RESULT"),
            'solver_stats': shared_dict.get('solver_stats', {}),
            'timed_out': False
        }
    
    return result_data

def main():
    config = load_config()
    
    print("=== RUN ORIGINAL ENCODING WITH HARD TIMEOUT ===")
    print(f"Timeout: {config['timeout_seconds']} secondi")
    print(f"Runs: {config['runs']}")
    print()
    
    with open(config["input_file"], "r", encoding="utf-8") as f:
        input_data = f.read()
    
    with open(config["base_encoding"], "r", encoding="utf-8") as f:
        encoding = f.read()
    
    results = []
    
    for i in range(config["runs"]):
        print(f"-- Run {i + 1} --")
        
        result = run_clingo_with_timeout(
            input_data, 
            encoding, 
            config["timeout_seconds"], 
            config["control_args"]
        )
        
        results.append(result)
        
        print(f"Tempo: {result['elapsed_time']:.4f}s")
        print(f"Miglior costo: {result['best_cost']}")
        print(f"Modelli trovati: {result['model_count']}")
        print(f"Status: {result['result_status']}")
        print(f"Scelte: {result['solver_stats'].get('choices', 'N/A')}")
        print(f"Conflitti: {result['solver_stats'].get('conflicts', 'N/A')}")
        if result['timed_out']:
            print("*** TIMEOUT CONFERMATO ***")
        print()
    
    print(f"=== ANALISI {config['runs']} RUNS ===")
    
    valid_costs = [r['best_cost'] for r in results if r['best_cost'] is not None]
    times = [r['elapsed_time'] for r in results]
    model_counts = [r['model_count'] for r in results]
    timeouts = sum(1 for r in results if r['timed_out'])
    
    if valid_costs:
        best_cost = None
        worst_cost = None
        
        for cost in valid_costs:
            if best_cost is None or is_cost_better(cost, best_cost):
                best_cost = cost
            if worst_cost is None or is_cost_better(worst_cost, cost):
                worst_cost = cost
        
        print(f"Costo migliore: {best_cost}")
        print(f"Costo peggiore: {worst_cost}")
        
        if len(valid_costs) > 1:
            first_components = []
            for cost in valid_costs:
                if isinstance(cost, list) and len(cost) > 0:
                    first_components.append(cost[0])
                elif cost is not None:
                    first_components.append(cost)
            
            if first_components:
                avg_first_component = statistics.mean(first_components)
                print(f"Media prima componente costo: {avg_first_component:.2f}")
                
                if len(first_components) > 1:
                    std_first_component = statistics.stdev(first_components)
                    print(f"Deviazione standard prima componente: {std_first_component:.2f}")
    else:
        print("Nessuna soluzione trovata entro il timeout")
    
    print(f"Tempo medio: {statistics.mean(times):.4f}s")
    
    if config['runs'] == 1:
        print(f"Modelli trovati: {int(statistics.mean(model_counts))}")
    else:
        print(f"Modelli medi per run: {statistics.mean(model_counts):.1f}")
    
    print(f"Timeout raggiunti: {timeouts}/{config['runs']}")
    
    if valid_costs:
        best_run = None
        for result in results:
            if result['best_cost'] is not None:
                if best_run is None or is_cost_better(result['best_cost'], best_run['best_cost']):
                    best_run = result
        
        if best_run:
            print(f"\nMIGLIOR RISULTATO:")
            print(f"Costo: {best_run['best_cost']}")
            print(f"Tempo: {best_run['elapsed_time']:.4f}s")
            print(f"Modelli: {best_run['model_count']}")
            print(f"Timeout: {'Sì' if best_run['timed_out'] else 'No'}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
