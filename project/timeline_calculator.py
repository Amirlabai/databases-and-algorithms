import csv
import datetime
import math
import ast # For safely evaluating string representations of dicts
import copy # For deep copying worker pool
import pandas as pd
from graph import Graph, Vertex, Edge # Assuming graph.py is in the same directory or project path

# 1. Define the project start date
PROJECT_START_DATE = datetime.date(2022, 5, 9)

# 2. Implement a function parse_worker_costs(filepath)
def parse_worker_costs(filepath: str) -> tuple[dict[str, int], dict[str, int]]:
    """
    Parses worker costs and availability from a CSV file using pandas.
    """
    worker_availability = {}
    worker_daily_costs = {}
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        for row in df.itertuples(index=False):
            try:
                worker_type = str(row.WorkerType)
                if pd.isna(row.Availability) or pd.isna(row.DailyCost):
                    print(f"Warning: Skipping row in {filepath} due to missing Availability/DailyCost: {row}")
                    continue
                availability = int(row.Availability)
                daily_cost = int(row.DailyCost)
                worker_availability[worker_type] = availability
                worker_daily_costs[worker_type] = daily_cost
            except (AttributeError, ValueError) as e:
                print(f"Error processing row in {filepath}: {row}. Details: {e}")
                continue
            except Exception as e_row:
                print(f"Unexpected error processing row {row} in {filepath}: {e_row}")
                continue
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return {}, {}
    except Exception as e_file:
        print(f"An unexpected error occurred while parsing {filepath} with pandas: {e_file}")
        return {}, {}
    return worker_availability, worker_daily_costs

# 3. Implement a function parse_tasks(filepath)
def parse_tasks(filepath: str) -> list[dict]:
    """
    Parses tasks from a CSV file using pandas.
    """
    tasks_list = []
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df = df[:6]
        for row in df.itertuples(index=False):
            try:
                task_id = int(row.Tid)
                name = str(row._1)
                base_duration_str = str(row.Duration)
                if 'days' not in base_duration_str.lower():
                     print(f"Warning: 'base_duration' for task {task_id} ('{name}') is missing 'days': '{base_duration_str}'. Assuming value is days.")
                base_duration = int(base_duration_str.split()[0])
                
                predecessors_str = str(row.Predecessors)
                predecessors = []
                if pd.notna(row.Predecessors) and predecessors_str.lower() != 'nan' and predecessors_str.strip() != '':
                    predecessors = [int(p.strip()) for p in predecessors_str.split(',') if p.strip()]
                
                required_workers_str = str(row.workers)
                required_workers = {}
                if pd.notna(row.workers) and required_workers_str.lower() != 'nan' and required_workers_str.strip() != '':
                    try:
                        required_workers = ast.literal_eval(required_workers_str)
                        if not isinstance(required_workers, dict):
                            print(f"Warning: 'required_workers' for task {task_id} ('{name}') parsed to non-dict: '{required_workers_str}'. Treating as empty.")
                            required_workers = {}
                    except (ValueError, SyntaxError) as e_ast:
                        print(f"Error parsing 'required_workers' for task {task_id} ('{name}'): '{required_workers_str}'. Details: {e_ast}. Treating as empty.")
                        required_workers = {}
                
                tasks_list.append({
                    'id': task_id, 'name': name, 'base_duration': base_duration,
                    'predecessors': predecessors, 'required_workers': required_workers
                })
            except (AttributeError, ValueError, TypeError) as e_row:
                print(f"Error processing task row: {row}. Details: {e_row}")
                continue
            except Exception as e_row_general:
                print(f"Unexpected error processing task row {row}: {e_row_general}")
                continue
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e_file:
        print(f"An unexpected error occurred while parsing {filepath} with pandas: {e_file}")
        return []
    return tasks_list

# Task Class Definition
class Task:
    def __init__(self, id, name, base_duration, predecessors, required_workers):
        self.id = id
        self.name = name
        self.base_duration = base_duration
        self.predecessors = predecessors
        self.required_workers = required_workers
        self.actual_duration = 0.0
        self.assigned_workers_map = {}
        self.es = None
        self.ef = None
        self.ls = None
        self.lf = None
        self.slack = None
        self.cost = 0.0
        self.status = 'pending'
        self.notes = ""
    def __repr__(self):
        return (f"Task(id={self.id}, name='{self.name}', base_duration={self.base_duration}, "
                f"actual_duration={self.actual_duration}, status='{self.status}', "
                f"assigned_workers_map={self.assigned_workers_map}, notes='{self.notes}')")

# Initial Calculation Function
def calculate_task_initial_values(task_dict_list, worker_availability, worker_daily_costs) -> list[Task]:
    task_objects = []
    for task_data in task_dict_list:
        task = Task(**task_data) 
        if not task.required_workers:
            task.actual_duration = 0.0
            task_objects.append(task)
            continue
        current_task_assigned_workers_map = {}
        duration_contributions = []
        task_is_impossible = False
        for worker_type, num_required in task.required_workers.items():
            if num_required <= 0: continue
            num_available_globally = worker_availability.get(worker_type, 0)
            if num_available_globally == 0:
                task.actual_duration = float('inf')
                task.notes += f"Worker type '{worker_type}' ({num_required} required) has zero global availability. Task impossible. "
                task_is_impossible = True
                break
            num_assigned_for_type = min(num_required, num_available_globally)
            current_task_assigned_workers_map[worker_type] = num_assigned_for_type
            try:
                duration_contribution = math.ceil((float(task.base_duration) * num_required) / num_assigned_for_type)
                duration_contributions.append(duration_contribution)
            except ZeroDivisionError:
                task.actual_duration = float('inf')
                task.notes += f"Error calculating duration for worker type '{worker_type}'. Division by zero. "
                task_is_impossible = True
                break
        if task_is_impossible:
            task.assigned_workers_map = {}
        else:
            if not duration_contributions and any(n > 0 for n in task.required_workers.values()):
                task.actual_duration = float('inf') if task.base_duration > 0 else 0.0
                if task.actual_duration == float('inf'): task.notes += "Failed to calculate duration contribution. "
            elif not duration_contributions:
                task.actual_duration = 0.0
            else:
                task.actual_duration = float(max(duration_contributions, default=0.0))
            task.assigned_workers_map = current_task_assigned_workers_map
        task_objects.append(task)
    return task_objects

# Simulation-based Scheduling Logic (Forward Pass - Graph Aware)
def schedule_tasks_forward_pass(
    tasks_in_graph: list[Task], 
    initial_worker_availability: dict[str, int], 
    project_start_date: datetime.date,
    task_graph: Graph, 
    vertex_map: dict[int, Vertex], 
    topologically_sorted_task_elements: list[Task] | None
) -> list[Task]:
    current_date = project_start_date
    worker_pool_free_count = copy.deepcopy(initial_worker_availability)
    completed_task_ids = set()
    active_tasks_details = {}
    tasks_map = {task.id: task for task in tasks_in_graph}

    if topologically_sorted_task_elements is None:
        print("Warning: Forward pass received no valid topological sort. Scheduling may be incorrect if cycles exist.")

    max_project_days = 5 * 365 
    project_end_horizon = project_start_date + datetime.timedelta(days=max_project_days)

    while len(completed_task_ids) < len(tasks_in_graph):
        if current_date > project_end_horizon:
            print("Error: Simulation exceeded maximum project horizon.")
            for task_id_active in list(active_tasks_details.keys()):
                 active_task_obj = active_tasks_details[task_id_active]['task_obj']
                 active_task_obj.notes += "Scheduling aborted (timeout); "
                 active_task_obj.status = "error_timeout"
            for task in tasks_in_graph:
                if task.id not in completed_task_ids and task.status == 'pending':
                    task.notes += "Scheduling aborted (timeout); "
                    task.status = "error_timeout"
            break
        tasks_processed_this_iteration = False
        for task_id_active in list(active_tasks_details.keys()):
            details = active_tasks_details[task_id_active]
            if details['ef_date'] <= current_date:
                completed_task_obj = details['task_obj']
                completed_task_obj.ef = details['ef_date']
                completed_task_obj.status = 'completed'
                completed_task_ids.add(completed_task_obj.id)
                for worker_type, count in details['workers_used'].items():
                    worker_pool_free_count[worker_type] = worker_pool_free_count.get(worker_type, 0) + count
                del active_tasks_details[task_id_active]
                tasks_processed_this_iteration = True
        
        task_candidates_for_start = []
        iteration_source = topologically_sorted_task_elements if topologically_sorted_task_elements is not None else tasks_in_graph
        for t_obj in iteration_source:
            if t_obj.id in tasks_map and tasks_map[t_obj.id].status == 'pending':
                 task_candidates_for_start.append(tasks_map[t_obj.id])
        if topologically_sorted_task_elements is None: 
            task_candidates_for_start.sort(key=lambda t: t.id)

        for task_to_schedule in task_candidates_for_start:
            task_vertex = vertex_map.get(task_to_schedule.id)
            if not task_vertex: continue
            all_graph_preds_completed_and_finished = True
            graph_predecessor_task_elements = [edge.opposite(task_vertex).element() for edge in task_graph.incident_edges(task_vertex, outgoing=False)]
            if graph_predecessor_task_elements:
                for pred_task_obj in graph_predecessor_task_elements:
                    processed_pred_task = tasks_map.get(pred_task_obj.id)
                    if not processed_pred_task or processed_pred_task.status != 'completed' or not (processed_pred_task.ef < current_date):
                        all_graph_preds_completed_and_finished = False; break
            if not all_graph_preds_completed_and_finished: continue
            can_allocate_resources = True
            if not task_to_schedule.assigned_workers_map: pass 
            else:
                for worker_type, needed_count in task_to_schedule.assigned_workers_map.items():
                    if worker_pool_free_count.get(worker_type, 0) < needed_count:
                        can_allocate_resources = False; break
            if can_allocate_resources:
                task_to_schedule.es = current_date
                duration_in_days = int(math.ceil(task_to_schedule.actual_duration))
                ef_date = current_date if duration_in_days == 0 else current_date + datetime.timedelta(days=duration_in_days)
                active_tasks_details[task_to_schedule.id] = {
                    'task_obj': task_to_schedule, 'ef_date': ef_date,
                    'workers_used': copy.deepcopy(task_to_schedule.assigned_workers_map)}
                if task_to_schedule.assigned_workers_map:
                    for worker_type, count in task_to_schedule.assigned_workers_map.items():
                        worker_pool_free_count[worker_type] -= count
                task_to_schedule.status = 'active'
                tasks_processed_this_iteration = True
        if not tasks_processed_this_iteration and active_tasks_details:
            min_next_ef_date = min(details['ef_date'] for details in active_tasks_details.values())
            current_date = max(current_date + datetime.timedelta(days=1), min_next_ef_date)
        elif not tasks_processed_this_iteration and not active_tasks_details and len(completed_task_ids) < len(tasks_in_graph):
            print(f"Warning: Deadlock detected at {current_date}. No tasks active, but {len(tasks_in_graph) - len(completed_task_ids)} tasks pending from graph.")
            for task in tasks_in_graph:
                if task.id not in completed_task_ids and tasks_map[task.id].status == 'pending':
                    tasks_map[task.id].notes += "Scheduling deadlock detected; "
                    tasks_map[task.id].status = "error_deadlock"
            break 
        else: current_date += datetime.timedelta(days=1)
    return tasks_in_graph

# Backward Pass Scheduling Logic (Graph Aware)
def schedule_tasks_backward_pass(
    task_objects_with_ef: list[Task],  # Should be tasks from the graph, processed by forward pass
    task_graph: 'Graph',              # The task dependency graph
    vertex_map: dict[int, 'Vertex']   # Maps task.id to Vertex object for tasks in the graph
) -> list[Task]:
    """
    Calculates Late Start (LS), Late Finish (LF), and Slack for tasks using a backward pass.

    This function operates on tasks that have already had their Early Start (ES) and
    Early Finish (EF) calculated. It iteratively processes tasks backwards from the
    project end date, propagating timing constraints through the dependency graph
    until all LS and LF values have converged.

    Args:
        task_objects_with_ef: A list of Task objects, updated with ES and EF.
        task_graph: The dependency graph of the tasks.
        vertex_map: A dictionary mapping task IDs to their corresponding graph Vertex.

    Returns:
        The same list of Task objects, now updated in-place with LS, LF, and Slack.
    """
    scheduled_tasks = [t for t in task_objects_with_ef if t.es is not None and t.ef is not None]

    if not scheduled_tasks:
        return task_objects_with_ef

    # The project's late finish is the latest of all early finish times.
    project_finish_date = max(task.ef for task in scheduled_tasks)

    # Initialize LS, LF, and Slack to None for a clean calculation.
    for task in scheduled_tasks:
        task.lf = None
        task.ls = None
        task.slack = None

    # Iteratively calculate LF and LS until the values stabilize.
    # This handles complex graphs where a simple reverse-sort isn't sufficient.
    MAX_ITERATIONS = len(scheduled_tasks) + 1
    for i in range(MAX_ITERATIONS):
        changed_in_iteration = False

        # Sort by Early Finish (desc) as a heuristic to process tasks closer to the end first.
        for task in sorted(scheduled_tasks, key=lambda t: t.ef, reverse=True):
            task_vertex = vertex_map.get(task.id)
            #task_vertex = task_graph.vertices(task_vertex)

            if not task_vertex:
                print(f"Warning: Task {task.id} ('{task.name}') was scheduled but not in vertex_map. Skipping.")
                continue

            # Find all successor tasks in the dependency graph.
            graph_successors = []
            for edge in task_graph.incident_edges(task_vertex, outgoing=True):
                successor_vertex = edge.opposite(task_vertex)
                successor_task_object = successor_vertex.element()
                # Successor must be part of the scheduled set to be considered.
                if successor_task_object.ls is not None or successor_task_object.ef is not None:
                     graph_successors.append(successor_task_object)


            # Determine the new Late Finish (LF) for the current task.
            new_lf = None
            if not graph_successors:
                # If a task has no successors, its Late Finish is the project finish date.
                new_lf = project_finish_date
            else:
                # A task's LF is the minimum of its successors' Late Starts (LS).
                # This can only be calculated if *all* successors have a calculated LS.
                if all(s.ls is not None for s in graph_successors):
                    new_lf = min(s.ls for s in graph_successors)
                # else: we must wait for a future iteration when successors' LS are known.

            # If a new, valid LF was calculated, update the task.
            if new_lf is not None and task.lf != new_lf:
                task.lf = new_lf
                changed_in_iteration = True

            # If the task's LF is set, we can now calculate its LS and Slack.
            if task.lf is not None:
                # Calculate LS
                # The duration requires ceiling to handle partial days.
                duration_days = int(math.ceil(task.actual_duration))
                
                # BUG FIX: Corrected date arithmetic. A 1-day task ending today must start today.
                # LS = LF - (Duration - 1 day).
                new_ls = task.lf - datetime.timedelta(days=max(0, duration_days - 1))

                if task.ls != new_ls:
                    task.ls = new_ls
                    changed_in_iteration = True

                # Calculate Slack (Total Float) if possible
                if task.ls is not None and task.es is not None:
                    new_slack = task.ls - task.es
                    if task.slack != new_slack:
                        task.slack = new_slack
                        # Note: A change in slack doesn't mean the core LF/LS values are unstable,
                        # so we don't set changed_in_iteration = True here.

        # If a full pass over all tasks results in no changes, the values have converged.
        if not changed_in_iteration and i > 0:
            break
            
    # Final safety check for any tasks that couldn't be processed (e.g., disconnected graph components)
    for task in scheduled_tasks:
        if task.lf is None:
            # Fallback: A task that could not be processed (e.g. part of a cycle or disconnected)
            # is treated as a leaf node. This is a common strategy.
            task.lf = project_finish_date
            duration_days = int(math.ceil(task.actual_duration))
            task.ls = task.lf - datetime.timedelta(days=max(0, duration_days - 1))
            if task.es:
                task.slack = task.ls - task.es


    return task_objects_with_ef

# Task Cost Calculation Logic
def calculate_task_costs(task_objects, worker_daily_costs) -> list[Task]:
    for task in task_objects:
        if task.es is not None and task.actual_duration != float('inf') and task.actual_duration > 0:
            task.cost = 0.0
            if task.assigned_workers_map:
                for worker_type, count in task.assigned_workers_map.items():
                    task.cost += count * task.actual_duration * worker_daily_costs.get(worker_type, 0)
        else: task.cost = 0.0
    return task_objects

# Graph-based functions
def build_task_graph(task_objects: list[Task]) -> tuple[Graph | None, dict[int, Vertex]]:
    task_dependency_graph = Graph(directed=True)
    vertex_map = {} 
    for task in task_objects:
        if task.actual_duration == float('inf'): 
            print(f"Skipping impossible task {task.id} ('{task.name}') from graph.")
            continue
        v = task_dependency_graph.insert_vertex(task)
        vertex_map[task.id] = v
    for task in task_objects:
        if task.id not in vertex_map: continue
        v_successor = vertex_map[task.id]
        for pred_id in task.predecessors:
            if pred_id in vertex_map:
                v_predecessor = vertex_map[pred_id]
                if not task_dependency_graph.get_edge(v_predecessor, v_successor):
                     task_dependency_graph.insert_edge(v_predecessor, v_successor, None)
            else:
                original_pred_task_exists = any(t.id == pred_id for t in task_objects)
                if original_pred_task_exists:
                     print(f"Note: Predecessor ID {pred_id} for task {task.id} ('{task.name}') refers to an impossible task. This dependency link is ignored in the graph.")
                else:
                     print(f"Warning: Predecessor ID {pred_id} for task {task.id} ('{task.name}') not found. Edge not created.")
    return task_dependency_graph, vertex_map

def topological_sort_kahn(graph: Graph, vertex_map: dict[int, Vertex]) -> list[Task] | None:
    if not graph or not graph.vertices(): return []
    in_degree = {v: 0 for v in graph.vertices()}
    for edge in graph.edges():
        _origin, dest = edge.endpoints()
        if dest in in_degree: in_degree[dest] += 1
        else: print(f"Warning: Edge destination {dest} not in in_degree map during topological sort.")
    queue = [v for v in graph.vertices() if v in in_degree and in_degree[v] == 0]
    sorted_tasks_list = []
    while queue:
        u = queue.pop(0)
        sorted_tasks_list.append(u.element())
        for edge in graph.incident_edges(u, outgoing=True):
            v = edge.opposite(u)
            if v in in_degree:
                in_degree[v] -= 1
                if in_degree[v] == 0: queue.append(v)
            else: print(f"Warning: Successor vertex {v} not found in in_degree map during sort.")
    if len(sorted_tasks_list) != len(in_degree):
        print(f"Error: Cycle detected. Sorted: {len(sorted_tasks_list)}, Expected: {len(in_degree)}")
        stuck_tasks = [v.element().name for v in in_degree if in_degree[v] > 0]
        if stuck_tasks: print(f"Tasks affected by cycle: {', '.join(stuck_tasks)}")
        return None
    return sorted_tasks_list

# Function to export timeline to CSV
def export_timeline_to_csv(task_objects: list[Task], filepath: str):
    report_data = []
    report_columns = ['Tid', 'Task Name', 'Actual Duration (days)', 'Start Date (ES)', 'End Date (EF)', 
                    'Slack (days)', 'Cost']
    sorted_tasks_for_export = sorted(task_objects, key=lambda t: t.id)
    for task in sorted_tasks_for_export:
        cost_val = task.cost if task.cost is not None else 0.0
        report_data.append({
            'Tid': task.id, 'Task Name': task.name,
            'Actual Duration (days)': format_duration_days(task.actual_duration),
            'Start Date (ES)': format_date(task.es), 'End Date (EF)': format_date(task.ef),
            'Slack (days)': format_duration_days(task.slack),
            'Cost': f"{cost_val:.2f}"
        })
    try:
        pd.DataFrame(report_data, columns=report_columns).to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\nTimeline report successfully exported to {filepath}")
    except Exception as e: print(f"\nError exporting timeline to CSV at {filepath}: {e}")

def auto_connect_parallel_leaves_with_map(
    task_graph: 'Graph', 
    task_vertex_map: dict[int, 'Vertex']
) -> tuple['Graph', dict[int, 'Vertex']]:
    """
    Finds leaf nodes and connects them to their parallel sibling's successor.
    This version is fully compatible with your provided Graph class.
    """
    all_vertices = list(task_vertex_map.values())
    
    # 1. Find all leaf nodes first
    # *** CORRECTION HERE ***
    # Changed task_graph.out_degree(v) to task_graph.degree(v, outgoing=True)
    # to match the method name in your Graph class.
    leaf_vertices = [v for v in all_vertices if task_graph.degree(v, outgoing=True) == 0]
    
    print(f"DEBUG: Found {len(leaf_vertices)} initial leaf vertices.")

    for leaf in leaf_vertices:
        leaf_task = leaf.element()
        
        predecessors = [edge.opposite(leaf) for edge in task_graph.incident_edges(leaf, outgoing=False)]
        
        if not predecessors:
            continue
            
        primary_predecessor = predecessors[0]

        siblings_and_self = [
            edge.opposite(primary_predecessor) 
            for edge in task_graph.incident_edges(primary_predecessor, outgoing=True)
        ]

        target_successor = None
        sibling_donor = None
        for sibling in siblings_and_self:
            if sibling == leaf:
                continue

            # *** CORRECTION HERE *** (and here as well)
            if task_graph.degree(sibling, outgoing=True) > 0:
                sibling_successors = [
                    edge.opposite(sibling) 
                    for edge in task_graph.incident_edges(sibling, outgoing=True)
                ]
                target_successor = sibling_successors[0]
                sibling_donor = sibling
                break

        if target_successor:
            print(f"INFO: Auto-connecting leaf '{leaf_task.name}' "
                  f"to '{target_successor.element().name}' "
                  f"(borrowed from sibling '{sibling_donor.element().name}').")
            task_graph.insert_edge(leaf, target_successor)

    return task_graph, task_vertex_map

# Helper functions for formatting output
def format_date(date_obj):
    if date_obj is None: return "N/A"
    return date_obj.strftime("%d/%m/%Y")
def format_workers(workers_map):
    if not workers_map: return "None"
    return ', '.join([f"{k}:{v}" for k, v in workers_map.items()])
def format_duration_days(duration_val, unit="days"):
    if duration_val is None: return "N/A"
    if isinstance(duration_val, datetime.timedelta): return f"{duration_val.days} {unit}"
    if isinstance(duration_val, (int, float)):
        if duration_val == float('inf'): return "Infinite"
        return f"{int(round(duration_val))} {unit}"
    return "N/A"

# Main execution block
if __name__ == "__main__":
    print(f"Project Start Date: {format_date(PROJECT_START_DATE)}")
    worker_costs_filepath = 'project/worker_costs.csv'
    open(worker_costs_filepath, 'r').close() 
    tasks_filepath = 'project/tasks.csv'
    open(tasks_filepath, 'r').close()

    print("\n--- Parsing Worker Costs ---")
    availability, daily_costs = parse_worker_costs(worker_costs_filepath)
    if availability and daily_costs: print("Worker Availability:", availability)
    print("\n--- Parsing Tasks (as Dictionaries) ---")
    tasks_list_dicts = parse_tasks(tasks_filepath)
    if tasks_list_dicts: print(f"Total tasks parsed from CSV: {len(tasks_list_dicts)}")
    else: tasks_list_dicts = []
    final_tasks_to_report = [] # Initialize to ensure it's defined

    if tasks_list_dicts and availability:
        print("\n--- Calculating Initial Task Values (Creating Task Objects) ---")
        initial_task_objects = calculate_task_initial_values(tasks_list_dicts, availability, daily_costs)
        if initial_task_objects: print(f"Total Task objects created: {len(initial_task_objects)}")
        else: initial_task_objects = [] # Ensure it's a list
        
        final_tasks_to_report = copy.deepcopy(initial_task_objects) 

        print("\n--- Building Task Dependency Graph ---")
        task_graph, task_vertex_map = build_task_graph(initial_task_objects) 
        
        task_graph, task_vertex_map = auto_connect_parallel_leaves_with_map(task_graph, task_vertex_map)

        topologically_sorted_tasks_elements = None # Ensure defined
        if task_graph and task_vertex_map:
            print(f"Task graph built. Vertices: {task_graph.vertex_count()}, Edges: {task_graph.edge_count()}")
            print("\n--- Performing Topological Sort (Kahn's Algorithm) ---")
            topologically_sorted_tasks_elements = topological_sort_kahn(task_graph, task_vertex_map)
            if topologically_sorted_tasks_elements is not None:
                print("Topological Sort Order (Task ID, Name):")
                for task_obj in topologically_sorted_tasks_elements: print(f"  - {task_obj.id}, {task_obj.name}")
            else: print("Could not perform topological sort (cycle detected or other graph issue).")

            if task_graph.vertex_count() > 0:
                schedulable_tasks_for_sim_dict = {tid: vtx.element() for tid, vtx in task_vertex_map.items()}
                schedulable_tasks_for_sim_list = list(schedulable_tasks_for_sim_dict.values())

                print("\n--- Scheduling Tasks (Forward Pass) ---")
                schedule_tasks_forward_pass(
                    schedulable_tasks_for_sim_list, copy.deepcopy(availability), PROJECT_START_DATE,
                    task_graph, task_vertex_map, topologically_sorted_tasks_elements)
                
                # --- START DEBUGGING BLOCK ---
                # Pick one sample task that you know is in both the list and the graph
                sample_task_from_list = schedulable_tasks_for_sim_list[0]
                sample_vertex_from_map = task_vertex_map[sample_task_from_list.id]
                sample_task_from_graph = sample_vertex_from_map.element()

                print(f"--- OBJECT IDENTITY CHECK ---")
                print(f"Task from list: '{sample_task_from_list.name}', ES: {sample_task_from_list.es}")
                print(f"Task from graph: '{sample_task_from_graph.name}', ES: {sample_task_from_graph.es}")

                # The id() function gives the memory address of an object.
                # If they are the same object, the IDs will be identical.
                are_they_the_same_object = (id(sample_task_from_list) == id(sample_task_from_graph))
                
                print(f"\nAre they the same object in memory? -> {are_they_the_same_object}")
                if not are_they_the_same_object:
                    print("\n>>> BUG CONFIRMED: The forward pass updated copies, not the original tasks in the graph!")
                # --- END DEBUGGING BLOCK ---
                
                print("\n--- Scheduling Tasks (Backward Pass) ---")
                schedule_tasks_backward_pass(
                    schedulable_tasks_for_sim_list, 
                    task_graph,      # Added task_graph
                    task_vertex_map  # Added task_vertex_map
                ) 
                
                print("\n--- Calculating Task Costs ---")
                calculate_task_costs(schedulable_tasks_for_sim_list, daily_costs)

                for i, task_in_report in enumerate(final_tasks_to_report):
                    if task_in_report.id in schedulable_tasks_for_sim_dict: 
                        final_tasks_to_report[i] = schedulable_tasks_for_sim_dict[task_in_report.id]
            else: print("No schedulable tasks in the graph to simulate.")
        else: print("Task graph could not be built or is empty. Skipping simulation.")
    else: print("\nSkipping calculations and scheduling due to parsing errors or missing data.")

    print("\n--- Project Timeline & Task Details ---")
    header = (f"{'ID':<4} {'Task Name':<40} {'Duration':<10} {'Start (ES)':<12} {'End (EF)':<12} "
              f"{'Late Start (LS)':<15} {'Late Finish (LF)':<15} {'Assigned Workers':<30} "
              f"{'Slack':<10} {'Cost':<10} {'Status':<15} {'Notes'}")
    print(header); print("-" * (len(header) + 10))
    if final_tasks_to_report:
        clean_report = []
        total_project_cost = 0.0
        actual_project_finish_date = PROJECT_START_DATE 
        sorted_tasks_for_report = sorted(final_tasks_to_report, key=lambda t: t.es)
        for i,task in enumerate(sorted_tasks_for_report): 
            
            if task.status not in ['pending', 'error_cycle_dependency'] and task.ef is not None and task.ef > actual_project_finish_date :
                actual_project_finish_date = task.ef
            if task.cost is not None: total_project_cost += task.cost
            print(f"{task.id:<4} {task.name:<40} {format_duration_days(task.actual_duration):<10} {format_date(task.es):<12} {format_date(task.ef):<12} "
                  f"{format_date(task.ls):<15} {format_date(task.lf):<15} {format_workers(task.assigned_workers_map):<30} "
                  f"{format_duration_days(task.slack):<10} {f'{task.cost:.2f}':<10} {task.status:<15} {task.notes if task.notes else ''}")
        finish_date_str = format_date(actual_project_finish_date) if actual_project_finish_date != PROJECT_START_DATE or any(t.ef for t in sorted_tasks_for_report if t.ef is not None) else "N/A (No tasks scheduled)"
        print("\n--- Project Summary ---")
        print(f"Initial Project Start Date: {format_date(PROJECT_START_DATE)}")
        print(f"Calculated Project Finish Date: {finish_date_str}")
        print(f"Total Estimated Project Cost: {total_project_cost:.2f}")
    else: print("No tasks to display.")
    if final_tasks_to_report : export_timeline_to_csv(final_tasks_to_report, "project/timeline_report.csv") 
    print("\n--- Script Execution Finished ---")
