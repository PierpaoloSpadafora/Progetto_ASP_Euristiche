
from pathlib import Path
import clingo
import time
import statistics
import json
import multiprocessing
import signal
import os

def load_config():
    settings_file = Path("settings.json")
    if not settings_file.exists():
        raise FileNotFoundError("File settings.json non trovato!")
    
    with settings_file.open(encoding="utf-8") as f:
        config = json.load(f)
    
    return config

CONFIG = load_config()

RUNS = CONFIG["runs"]
BASE_FILE = Path(CONFIG["base_encoding"])
HEURISTICS_FILE = Path(CONFIG["heuristics_to_try_file"])
PROMISING_FILE = Path(CONFIG["heuristics_file"])
INPUT_FILE = Path(CONFIG["input_file"])
CONTROL_ARGS = CONFIG["control_args"]
TIMEOUT_SECONDS = CONFIG.get("timeout_seconds", 30)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

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
        print(f"   Model {model_count}: Cost = {current_cost} (tempo: {elapsed:.2f}s)")
        
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
        print(f"   Errore nel worker: {e}")
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
        
        print(f"     Tempo: {result_data['elapsed_time']:.4f}s, Miglior costo: {result_data['best_cost']}, "
              f"Modelli usati: {result_data['model_count']}, Status: {result_data['result_status']}")
        print("     *** TIMEOUT CONFERMATO ***")
        
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

def run_multiple_times(facts: str, encoding: str, runs: int, description: str):
    print(f"   Esecuzione {runs} run per {description}...", flush=True)
    
    results = []
    for i in range(runs):
        print(f"   → Run {i+1}/{runs}:", flush=True)
        result = run_clingo_with_timeout(facts, encoding, TIMEOUT_SECONDS, CONTROL_ARGS)
        results.append(result)
        
        if not result['timed_out']:
            print(f"     Tempo: {result['elapsed_time']:.4f}s, Miglior costo: {result['best_cost']}, "
                  f"Modelli usati: {result['model_count']}, Status: {result['result_status']}")
    
    times = [r['elapsed_time'] for r in results]
    valid_costs = [r['best_cost'] for r in results if r['best_cost'] is not None]
    model_counts = [r['model_count'] for r in results]
    timeouts = sum(1 for r in results if r['timed_out'])
    
    best_cost_overall = None
    for cost in valid_costs:
        if best_cost_overall is None or is_cost_better(cost, best_cost_overall):
            best_cost_overall = cost
    
    stats = {
        'results': results,
        'avg_time': statistics.mean(times),
        'avg_models': statistics.mean(model_counts),
        'timeouts': timeouts,
        'total_runs': runs,
        'best_cost': best_cost_overall,
        'valid_solutions': len(valid_costs)
    }
    
    print(f"   → Tempo medio: {stats['avg_time']:.4f}s")
    if runs == 1:
        print(f"   → Modelli usati: {int(stats['avg_models'])}")
    else:
        print(f"   → Modelli medi: {stats['avg_models']:.1f}")
    print(f"   → Timeout: {timeouts}/{runs}")
    if stats['best_cost'] is not None:
        print(f"   → Miglior costo: {stats['best_cost']}")
    print()
    
    return stats

def split_heuristics(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    
    blocks = []
    current_block = []
    
    for line in content.splitlines():
        line = line.strip()
        if line:
            current_block.append(line)
        else:
            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
    
    if current_block:
        blocks.append('\n'.join(current_block))
    
    return blocks

def is_promising_heuristic(baseline_stats, heuristic_stats):
    baseline_timeout_rate = baseline_stats['timeouts'] / baseline_stats['total_runs']
    heuristic_timeout_rate = heuristic_stats['timeouts'] / heuristic_stats['total_runs']
    
    if (baseline_stats['best_cost'] is not None and 
        heuristic_stats['best_cost'] is not None and 
        is_cost_better(heuristic_stats['best_cost'], baseline_stats['best_cost'])):
        
        baseline_first = cost_to_comparable(baseline_stats['best_cost'])[0]
        heuristic_first = cost_to_comparable(heuristic_stats['best_cost'])[0]
        
        if heuristic_first > 0:
            improvement = baseline_first / heuristic_first
        else:
            improvement = float('inf')
            
        return True, f"soluzioni migliori (costo {heuristic_stats['best_cost']} vs {baseline_stats['best_cost']})", improvement
    
    if (heuristic_stats['avg_time'] < baseline_stats['avg_time'] and 
        heuristic_stats['valid_solutions'] > 0 and
        baseline_stats['valid_solutions'] > 0):
        
        speedup = baseline_stats['avg_time'] / heuristic_stats['avg_time']
        
        if (baseline_stats['best_cost'] is None or 
            heuristic_stats['best_cost'] is None or
            not is_cost_better(baseline_stats['best_cost'], heuristic_stats['best_cost'])):
            return True, f"più veloce (speedup {speedup:.2f}×)", speedup
    
    if heuristic_stats['valid_solutions'] == 0:
        return False, "non trova soluzioni valide", 0.0
    
    if baseline_stats['best_cost'] is None:
        return False, "baseline non ha soluzioni per confronto", 0.0
        
    if heuristic_stats['best_cost'] is None:
        return False, "non trova soluzioni valide", 0.0
    
    if is_cost_better(baseline_stats['best_cost'], heuristic_stats['best_cost']):
        return False, f"soluzioni peggiori ({heuristic_stats['best_cost']} vs {baseline_stats['best_cost']})", 0.0
    
    if heuristic_stats['avg_time'] >= baseline_stats['avg_time']:
        slowdown = heuristic_stats['avg_time'] / baseline_stats['avg_time']
        return False, f"più lento (slowdown {slowdown:.2f}×)", 1/slowdown
    
    return False, "non produce miglioramenti significativi", 1.0

def main() -> None:
    print("=== BRUTE FORCE HEURISTICS WITH TIMEOUT SUPPORT ===")
    print(f"Configurazione caricata da settings.json:")
    print(f"  - RUNS: {RUNS}")
    print(f"  - TIMEOUT: {TIMEOUT_SECONDS} secondi")
    print(f"  - BASE_FILE: {BASE_FILE}")
    print(f"  - INPUT_FILE: {INPUT_FILE}")
    print(f"  - HEURISTICS_FILE: {HEURISTICS_FILE}")
    print(f"  - PROMISING_FILE: {PROMISING_FILE}")
    print()
    
    facts = read_text(INPUT_FILE)
    base_enc = read_text(BASE_FILE)
    heuristics = split_heuristics(HEURISTICS_FILE)
    total = len(heuristics)

    print(f"[BASELINE] Calcolo statistiche baseline su {RUNS} run...")
    baseline_stats = run_multiple_times(facts, base_enc, RUNS, "baseline")
    
    print(f"[BASELINE] Risultati:")
    print(f"  - Tempo medio: {baseline_stats['avg_time']:.4f}s")
    if RUNS == 1:
        print(f"  - Modelli usati: {int(baseline_stats['avg_models'])}")
    else:
        print(f"  - Modelli medi: {baseline_stats['avg_models']:.1f}")
    print(f"  - Timeout: {baseline_stats['timeouts']}/{RUNS}")
    if baseline_stats['best_cost'] is not None:
        print(f"  - Miglior costo: {baseline_stats['best_cost']}")
    print()

    promising = []
    for idx, h in enumerate(heuristics, start=1):
        print(f"[{idx}/{total}] Test euristica #{idx}")
        
        heuristic_stats = run_multiple_times(
            facts, 
            base_enc + "\n" + h + "\n", 
            RUNS, 
            f"euristica #{idx}"
        )
        
        is_promising, reason, metric = is_promising_heuristic(baseline_stats, heuristic_stats)
        
        if is_promising:
            print(f"   ✓ PROMETTENTE: {reason}")
            print(f"   → Salvata in promising_ones.lp\n")
            promising.append((idx, h, heuristic_stats, reason, metric))
        else:
            print(f"   ✗ NON PROMETTENTE: {reason}\n")

    if promising:
        def sort_key(item):
            idx, h, stats, reason, metric = item
            cost_key = cost_to_comparable(stats['best_cost'])
            time_key = stats['avg_time']
            return (cost_key, time_key)
        
        promising.sort(key=sort_key)
        
        lines = []
        for euristica_num, block, stats, reason, metric in promising:
            lines.append(f"% Euristica #{euristica_num}")
            
            lines.append(block)
            
            lines.append(f"% Tempo medio: {stats['avg_time']:.4f}s")
            if RUNS == 1:
                lines.append(f"% Modelli usati: {int(stats['avg_models'])}")
            else:
                lines.append(f"% Modelli medi: {stats['avg_models']:.1f}")
            lines.append(f"% Timeout: {stats['timeouts']}/{stats['total_runs']}")
            if stats['best_cost'] is not None:
                lines.append(f"% Miglior costo: {stats['best_cost']}")
            lines.append("")
        
        content = "\n".join(lines).rstrip()
        PROMISING_FILE.write_text(content, encoding="utf-8")
        
        print(f"*** RISULTATO FINALE ***")
        print(f"Salvate {len(promising)} euristiche promettenti in '{PROMISING_FILE}'")
        print(f"Euristiche testate: {total}")
        print(f"Percentuale di successo: {len(promising)/total*100:.1f}%")
        
        print("\nEuristiche ordinate per qualità:")
        for i, (euristica_num, block, stats, reason, metric) in enumerate(promising, 1):
            cost_str = f"{stats['best_cost']}" if stats['best_cost'] is not None else "N/A"
            print(f"  {i}. Euristica #{euristica_num}: Costo {cost_str}, Tempo {stats['avg_time']:.4f}s")
        
    else:
        print("*** RISULTATO FINALE ***")
        print("Nessuna euristica promettente trovata.")
        print(f"Tutte le {total} euristiche testate non migliorano il baseline.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
