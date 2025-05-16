import sys
sys.path.append('/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async')
import os
import argparse
import multiprocessing as mp
import time
import logging
import psutil
import ctypes
import asyncio
import numpy as np
from queue import Empty
import pandas as pd
import queue
from dataclasses import asdict, replace
from multiprocessing import Queue
from pathlib import Path
import json

from LLM_TSP.ablation_config import instance_max_nodes, instance_time_budget
from LLM_TSP.solver.solver import subproblem_solver, subproblem_verifier, sample_independent_subproblem, sample_next_subproblem, GlobalObjRecord   # single line import
from LLM_TSP.config import LLMConfig, SolverConfig

from LLM_TSP.tsp import TravelingSalesmenProblem
from helper.parse_instances import FileParser
from LLM_TSP.initial_solution import initialize_solution
from LLM_TSP.llm import GPT, RoundRobinLLMSelector
from helper.plot_solution import SolutionPlot
from LLM_TSP.selector import RandomSelector
from LLM_TSP.llm_selector.llm_selector import _llm_producer

OPENAI_API_1 = 'YOUR API KEY'
OPENAI_API_2 = 'YOUR API KEY'

def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(processName)s] %(message)s",
                        datefmt="%H:%M:%S",
                        force=True)
    
import multiprocessing as mp
import time
import logging
from typing import Dict, List
def _launch_worker(subproblem, config, active_processes):
    proc = mp.Process(
        target=subproblem_solver,
        args=(subproblem, config),
        daemon=True,
        name=f"SubTSP‑{subproblem.id}",
    )
    proc.start()
    active_processes[proc] = subproblem
    return proc

def _reap_finished(active_processes: Dict[mp.Process, "Subproblem"]):
    for proc in list(active_processes.keys()):
        if not proc.is_alive():
            proc.join(timeout=0.1)  # non‑blocking on already‑exited child
            active_processes.pop(proc, None)

def dynamic_worker_manager(config):
    logger = logging.getLogger(__name__)

    capacity_check = lambda: (
            (config.pending_re_subproblem_queue.qsize() > 0 or
            config.pending_ft_subproblem_queue.qsize() > 0) or 
            (config.gain_subproblem_queue.qsize() >0)
            and len(active_processes) < 1
        )
    
    active_processes: Dict[mp.Process, "Subproblem"] = {}
    
    try:
        while time.time() < config.deadline:
            if capacity_check():

                with config.sol_lock, config.obj_lock:
                    current_route =list(config.global_sol)
                    current_obj = config.global_obj.value

                if config.gain_subproblem_queue.qsize() >0:
                    print('Gain subproblem is available!')
                subproblem = sample_independent_subproblem(config=config,
                                            active_subproblems=list(active_processes.values()),
                                            gain_subproblem_queue=config.gain_subproblem_queue,
                                            subproblem_ft_queue=config.pending_ft_subproblem_queue,
                                            subproblem_re_queue=config.pending_re_subproblem_queue,
                                            traj_lock=config.traj_lock,
                                            current_route=current_route
                                            )

                if subproblem is None:
                    time.sleep(1)  # short back‑off when no viable task
                else:
                    subproblem.solution_version = current_obj
                    # _launch_worker(subproblem, config, active_processes)
                    proc = mp.Process(target=subproblem_solver,
                                      args=(subproblem, config),
                                      name=f"SubTSP‑{subproblem.solution_version}",)
                    proc.start()
                    active_processes[proc] = subproblem
                    
                    logger.debug(
                        "Launched %s (alive=%d)",
                        subproblem.solution_version,
                        len(active_processes),
                    )

            _reap_finished(active_processes)
            # print('Active subproblem is ', len(active_processes))
            time.sleep(1)

    finally:
        # ----------------------------------------------------------
        # Final clean‑up: wait for *all* still‑running children
        # ----------------------------------------------------------
        for proc in list(active_processes.keys()):
            proc.join()
        active_processes.clear()
        logger.info("Manager shut‑down complete – final objective = %s", config.global_obj.value)


def verifier_manager(config):
    logger = logging.getLogger(__name__)

    capacity_check = lambda: (
            (config.pending_re_subproblem_queue.qsize() > 0 or
            config.pending_ft_subproblem_queue.qsize() > 0)
            and len(active_processes) < config.args.max_workers
        )
    
    active_processes: Dict[mp.Process, "Subproblem"] = {}
    
    try:
        while time.time() < config.deadline:
            if capacity_check():

                with config.sol_lock, config.obj_lock:
                    current_route =list(config.global_sol)
                    current_obj = config.global_obj.value

                #TODO: modify the samply strategy
                subproblem = sample_next_subproblem(config=config,
                                            active_subproblems=list(active_processes.values()),
                                            subproblem_ft_queue=config.pending_ft_subproblem_queue,
                                            subproblem_re_queue=config.pending_re_subproblem_queue,
                                            traj_lock=config.traj_lock,
                                            current_route=current_route
                                            )

                if subproblem is None:
                    time.sleep(1)  # short back‑off when no viable task
                else:
                    subproblem.solution_version = current_obj
                    # _launch_worker(subproblem, config, active_processes)
                    proc = mp.Process(target=subproblem_verifier,
                                      args=(subproblem, config),
                                      name=f"SubTSP‑{subproblem.solution_version}",)
                    proc.start()
                    active_processes[proc] = subproblem
                    
                    logger.debug(
                        "Launched %s (alive=%d)",
                        subproblem.solution_version,
                        len(active_processes),
                    )

            _reap_finished(active_processes)
            print('Active subproblem is ', len(active_processes))
            time.sleep(1)

    finally:
        for proc in list(active_processes.keys()):
            proc.join()
        active_processes.clear()
        logger.info("Manager shut‑down complete – final objective = %s", config.global_obj.value)
                    
def launch_llm_process(name,config):

    _configure_logging()
    log = logging.getLogger()
    asyncio.run(_llm_producer(
        name,
        config.args,
        config.tsp_instance,
        config.llm_selector,
        config.pending_subproblem_queue,
        config.global_obj,
        config.global_sol,
        config.sol_lock,
        config.obj_lock,
        config.selection_traj,
        config.deadline,
        config.t0,
        config.traj_queue,
        config.traj_lock,
        config.X_MIN,
        config.X_MAX,
        config.Y_MIN,
        config.Y_MAX,
        config.GRID_RES,
        config.backup_selector,
        config.solution_plotter
    ))
    log.info("complete the llm process")


def tsp_instance_initializer(args):

    def determine_instance_boundary(coordinates):
        MARGIN = 0
        x_min = min(coord[0] for coord in coordinates) - MARGIN
        x_max = max(coord[0] for coord in coordinates) + MARGIN
        y_min = min(coord[1] for coord in coordinates) - MARGIN
        y_max = max(coord[1] for coord in coordinates) + MARGIN

        grid_resolution = max((x_max - x_min), (y_max - y_min)) // 10

        return x_min, x_max, y_min, y_max, grid_resolution

    file_parser = FileParser()
    instance_info = file_parser.parse_instance_from_file(args.instance_path)
    coordinates = instance_info['COORDINATES']
    distance_mat = np.array(instance_info['COST_MATRIX'])
    nodes = {i: (x, y) for i, (x, y) in enumerate(coordinates)}
    X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES = determine_instance_boundary(coordinates)
    boundary_info = (X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES)
    tsp_instance = TravelingSalesmenProblem(node_coords_dict=nodes, distance_mat=distance_mat)

    return tsp_instance, boundary_info
def dump_global_obj_queue(q: Queue, csv_path: str | Path) -> pd.DataFrame:
    """
    Drain *q* (containing `GlobalObjRecord`s) into a DataFrame
    and save it to *csv_path*.

    Returns
    -------
    pd.DataFrame
        The dataframe that was written, so you can keep using it.
    """
    records = []
    while True:
        try:
            rec = q.get_nowait()          # type: GlobalObjRecord
            records.append(asdict(rec))
        except queue.Empty:
            break

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return df

def pin(proc: mp.Process, cores: list[int]) -> None:
    """Bind *proc* to the given CPU *cores*."""
    psutil.Process(proc.pid).cpu_affinity(cores)

def main(args):
    _configure_logging()
    log = logging.getLogger()
    tsp_instance, boundary_info = tsp_instance_initializer(args)
    X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES = boundary_info 

    fast_thinking_llm_selector = RoundRobinLLMSelector([GPT(OPENAI_API_1, args.fast_llm_model)])
    reasoning_llm_selector = RoundRobinLLMSelector([GPT(OPENAI_API_2, args.reasoning_llm_model)])

    backup_selector = RandomSelector(model_name='random')
    solution_plotter = SolutionPlot()

    instance_name = Path(args.instance_path).stem
     # Solution initialization
    try:
        with open(f'/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async/experiments/LKH_solutions/{instance_name}_solution.json', 'r') as f:
            data = json.load(f)

            current_route = data['current_route']
            current_obj = data['current_obj']
            warmstart_latency = data['warmstart_latency']
    except:
        current_route, current_obj, warmstart_latency = initialize_solution(args, tsp_instance)
    # -------- solution initialization (warm start)

    # Save to file
        data = {
            'current_route': current_route,
            'current_obj': current_obj,
            'warmstart_latency': warmstart_latency
        }
        

        with open(f'/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async/experiments/LKH_solutions/{instance_name}_solution.json', 'w') as f:
            json.dump(data, f)
    # -------- shared queues and values among parallel processes
    gain_subproblem_queue       = mp.Queue() # used to save subproblems with definite gains
    pending_ft_subproblem_queue = mp.Queue() # save the pending subproblems from fast thinking LLM
    pending_re_subproblem_queue = mp.Queue() # from reasoning LLM
    track_global_obj_queue      = mp.Queue() # save the whole trajectory to track the global solution improvement
    selection_traj              = mp.Queue()
    global_obj                  = mp.Value('i', 0) # creating using manager.Value may cause broken pipe
    global_sol                  = mp.Array(ctypes.c_int, len(current_route), lock=True) # to avoid broken pipe when concorde did not succeed in finding optimal tour
    obj_lock                    = mp.RLock() # use the lock may not be a good idea
    sol_lock                    = mp.RLock()
    traj_lock                   = mp.RLock()
    solver_proc_lock            = mp.RLock()
    
    with obj_lock:
        global_obj.value = current_obj
    with sol_lock:
        global_sol[:] = current_route[:]

    now = round(warmstart_latency, 2)
    record = GlobalObjRecord(latency=now,
                             new_obj=current_obj,
                             coords=None,
                             num_nodes_removed=None,
                             llm_mode=args.initial_solution_model,
                             global_solution_version=None,
                             )
    track_global_obj_queue.put(record)
            

    t0 = time.time()
    deadline = t0 + args.total_time_budget - warmstart_latency 

    solver_config = SolverConfig(args=args,
                                 warmstart_latency=warmstart_latency,
                                 tsp_instance=tsp_instance,
                                 pending_ft_subproblem_queue=pending_ft_subproblem_queue,
                                 pending_re_subproblem_queue=pending_re_subproblem_queue,
                                 gain_subproblem_queue = gain_subproblem_queue,
                                 global_obj=global_obj,
                                 global_sol=global_sol,
                                 obj_lock=obj_lock,
                                 sol_lock=sol_lock,
                                 traj_lock=traj_lock,
                                 solver_proc_lock=solver_proc_lock,
                                 selection_traj=selection_traj,
                                 t0=t0,
                                 deadline=deadline,
                                 track_global_obj_queue=track_global_obj_queue
                                 )
    ft_llm_config = LLMConfig(args=args,
                           tsp_instance=tsp_instance,
                           llm_selector=fast_thinking_llm_selector,
                           pending_subproblem_queue=pending_ft_subproblem_queue,
                           global_obj=global_obj,
                           global_sol=global_sol,
                           sol_lock=sol_lock,
                           obj_lock=obj_lock,
                           selection_traj=selection_traj,
                           deadline=deadline,
                           t0=t0,
                           traj_queue=track_global_obj_queue,
                           traj_lock=traj_lock,
                           X_MIN=X_MIN,
                           X_MAX=X_MAX,
                           Y_MIN=Y_MIN,
                           Y_MAX=Y_MAX,
                           GRID_RES=GRID_RES,
                           backup_selector=backup_selector,
                           solution_plotter=solution_plotter
                           )
    
    re_llm_config = replace(ft_llm_config,
                            llm_selector=reasoning_llm_selector,
                            pending_subproblem_queue=pending_re_subproblem_queue,)

    
    #TODO: simulating putting some subproblems in the queue
    
    dynamic_solver_proc      = mp.Process(name='Concorde', 
                                     target=dynamic_worker_manager,
                                     args=(solver_config,))

    verifier_proc            = mp.Process(name='Concorde', 
                                     target=verifier_manager,
                                     args=(solver_config,))
    
    reasoning_llm_proc       = mp.Process(name="reasoning_LLM-Producer", 
                                     target=launch_llm_process,
                                     args=('reasoning', re_llm_config,))
    
    fast_thinking_llm_proc   = mp.Process(name="fast_thinking_LLM-Producer", 
                                     target=launch_llm_process,
                                     args=('fast_thinking', ft_llm_config,))
    
    
    n_cpus = mp.cpu_count()  # e.g. 48
    llm1_core = [0, 1]  # one core for each LLM producer
    llm2_core = [2, 3]
    # solver_cores = list(range(4, n_cpus))  # the rest for Concorde


    reasoning_llm_proc.start()
    fast_thinking_llm_proc.start()
    dynamic_solver_proc.start()

    
    verifier_proc.start()


    # pin(dynamic_solver_proc, solver_cores)
    pin(fast_thinking_llm_proc, llm1_core)
    pin(reasoning_llm_proc, llm2_core)

    # dynamic_solver_proc.join()
    dynamic_solver_proc.join(timeout=args.total_time_budget + 10)
    if dynamic_solver_proc.is_alive():
        print("dynamic_solver_proc is stuck!")
        dynamic_solver_proc.terminate()
        dynamic_solver_proc.join()

    instance_name = Path(args.instance_path).stem

    df = dump_global_obj_queue(track_global_obj_queue,
                            f'/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async/experiments/LLM_TSP_exp/{instance_name}_max_nodes_{args.max_node_for_solver}_time_budget_{args.total_time_budget}_initial_{args.initial_solution_model}_llm_{args.fast_llm_model}_{args.reasoning_llm_model}_solver_{args.solver_model}_subproblem_{args.llm_subproblem_selection}_parallel_workers.csv')
    print("Saved", len(df), "records")

    
    
    verifier_proc.join(timeout=5)
    if verifier_proc.is_alive():
        print("verifier_proc is stuck!")
        verifier_proc.terminate()
        verifier_proc.join()

    
    reasoning_llm_proc.join(timeout=5)
    if reasoning_llm_proc.is_alive():
        print("reasoning_llm_proc is stuck!")
        reasoning_llm_proc.terminate()
        reasoning_llm_proc.join()

    fast_thinking_llm_proc.join(timeout=5)
    if fast_thinking_llm_proc.is_alive():
        print("fast_thinking_llm_proc is stuck!")
        fast_thinking_llm_proc.terminate()
        fast_thinking_llm_proc.join()

    # instance_name = Path(args.instance_path).stem

    # df = dump_global_obj_queue(track_global_obj_queue,
    #                         f'/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async/experiments/LLM_TSP_exp/{instance_name}_max_nodes_{args.max_node_for_solver}_time_budget_{args.total_time_budget}_initial_{args.initial_solution_model}_llm_{args.fast_llm_model}_{args.reasoning_llm_model}_solver_{args.solver_model}_subproblem_{args.llm_subproblem_selection}_parallel_workers.csv')
    # print("Saved", len(df), "records")

    # fast_llm_proc.join()
    # dynamic_solver_proc.join()
    
    # # reasoning_llm_proc.join()
    # print('complete')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Traveling Salesmen Problem Solver")
    parser.add_argument('--instance_path', type=str,
                        default='/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async/instances/tsplib',
                        help='Path to the instance file')
    parser.add_argument('--max_iterations', type=int, default=5,
                        help='Maximum number of iterations for optimization')
    parser.add_argument('--total_time_budget', type=float, default=2000,
                        help='Wall time in seconds')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of solvers working in parallel')
    # ---------------------------------------------------------------------------
    # Initializer Specification
    # ---------------------------------------------------------------------------
    parser.add_argument('--initial_solution_model', type=str, default='LKH',
                        help='model to generate initial solution')

    # ---------------------------------------------------------------------------
    # Solver Specification
    # ---------------------------------------------------------------------------
    parser.add_argument('--solver_model', type=str, default='concorde',
                        help='solver name for reoptimization')
    parser.add_argument('--SolverTimeLimit', type=float, default=10,
                        help='Time allowed for Concorde solver')
    parser.add_argument('--max_node_for_solver', type=int, default=1000,
                        help='Max number of nodes sent to solver')

    # ---------------------------------------------------------------------------
    # Selector Specification
    # ---------------------------------------------------------------------------
    parser.add_argument('--fast_llm_model', type=str, default='gpt-4.1-2025-04-14', 
                        help='LLM model name for selector, qwen2.5-32b-v, gpt-4o ') 
    parser.add_argument('--reasoning_llm_model', type=str, default='o4-mini-2025-04-16', 
                        help='LLM model name for selector, qwen2.5-32b-v, gpt-4o ') 
    parser.add_argument('--keep_selection_trajectory', action='store_true',
                        help='whether incorporating selection trajectory for llm')
    parser.add_argument('--llm_subproblem_selection', type=int, default=2,
                        help='number of subproblems that LLM should select in its first try')
    parser.add_argument('--select_sequence', action='store_true',
                        help='whether selecting a sequence or rectangle as subproblem')
    parser.add_argument('--random_selection', action='store_true',
                        help='whether selecting a sequence or rectangle as subproblem')
    parser.add_argument('--hard_coded_subrectangle', action='store_true',
                        help='Flag to enable hard-coded subrectangle')
    parser.add_argument('--gridding_resolution', type=int, default=5,
                        help='divide the plot into K if wanting to fix')

    args = parser.parse_args()

    file_path = args.instance_path
    tsp_files = [
        # 'dsj1000.tsp',
        # 'pr1002.tsp',
        # 'u1060.tsp',
        # 'vm1084.tsp',
        # 'pcb1173.tsp',
        # 'd1291.tsp',
        # 'rl1304.tsp',
        # 'rl1323.tsp',
        # 'nrw1379.tsp',
        # 'fl1400.tsp',
        # 'u1432.tsp',
        # 'fl1577.tsp',
        # 'd1655.tsp',
        # 'vm1748.tsp',
        # 'u1817.tsp',
        # 'rl1889.tsp',
        # 'd2103.tsp',
        # 'u2152.tsp',
        # 'u2319.tsp',
        # 'pr2392.tsp',
        # 'pcb3038.tsp',
        # 'fl3795.tsp',
        # 'fnl4461.tsp',
        # 'rl5915.tsp',
        # 'rl5934.tsp',
        # 'pla7397.tsp',
        # 'rl11849.tsp',
        # 'usa13509.tsp',
        # 'brd14051.tsp',
        # 'd15112.tsp',
        # 'd18512.tsp',
        # 'pla33810.tsp',
        # 'pla85900.tsp',
    ]

    for file in tsp_files:
        args.instance_path = f'{file_path}/{file}'
        main(args)
