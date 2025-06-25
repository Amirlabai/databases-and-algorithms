import os
import datetime
import math
import ast
import copy
import pandas as pd
from graph import Graph, Vertex, Edge

# 1. Define the project start date
PROJECT_START_DATE = None
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

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
        #df = df[:15]
        for row in df.itertuples(index=False):
            try:
                task_id = int(row.Tid)
                name = str(row._1)
                base_duration_str = str(row.Duration)
                if 'days' not in base_duration_str.lower():
                    pass
                     #print(f"Warning: 'base_duration' for task {task_id} ('{name}') is missing 'days': '{base_duration_str}'. Assuming value is days.")
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
    return df,tasks_list

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
        return (f"Task(id={self.id}, name='{self.name}',Early start={self.es}, Early finish={self.ef}, base_duration={self.base_duration}, "
                f"actual_duration={self.actual_duration}, predeccsor='{self.predecessors}', "
                f"assigned_workers_map={format_workers(self.assigned_workers_map)})")

# Initial Calculation Function
def calculate_task_initial_values(task_dict_list, worker_availability) -> list[Task]:
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
                    if not processed_pred_task or processed_pred_task.status != 'completed' or not (processed_pred_task.ef <= current_date):
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
        else: pass#current_date += datetime.timedelta(days=1)
    return tasks_in_graph

def schedule_tasks_custom_backward_pass(
    task_objects_with_ef: list[Task],
    task_graph: 'Graph',
    vertex_map: dict[int, 'Vertex']
) -> list[Task]:
    """
    Calculates LF, LS, and Slack using a custom heuristic.

    - Rule 1: If a task has predecessors, successors, and no siblings, its LF is its EF.
    - Rule 2: Otherwise, its LF is the latest EF among its siblings and cousins.
    
    This function should be called after the forward pass is complete.
    """
    if not task_objects_with_ef:
        return []

    # Determine the project's overall finish date as a fallback.
    project_finish_date = max(
        (task.ef for task in task_objects_with_ef if task.ef is not None),
        default=PROJECT_START_DATE
    )

    # First, calculate Late Finish (LF) for all tasks based on the custom rules.
    for task in task_objects_with_ef:
        # Skip tasks that couldn't be scheduled in the forward pass.
        if task.status != 'completed' or task.ef is None:
            continue

        task_vertex = vertex_map.get(task.id)
        if not task_vertex:
            continue

        # Check conditions for Rule 1.
        has_predecessors = task_graph.degree(task_vertex, outgoing=False) > 0
        has_successors = task_graph.degree(task_vertex, outgoing=True) > 0
        
        parents = {edge.opposite(task_vertex) for edge in task_graph.incident_edges(task_vertex, outgoing=False)}
        has_siblings = any(task_graph.degree(p, outgoing=True) > 1 for p in parents)
        base = task.base_duration
        actual = task.actual_duration
        # Apply Rule 1: A task in a simple chain is considered "critical."
        if (has_predecessors and has_successors) or base < actual or base == 0:
            task.lf = task.ef
        else:
            # Apply Rule 2: Find LF from relatives' Early Finish times.
            siblings, cousins = _find_relatives(task_vertex, task_graph)
            relatives = siblings.union(cousins)
            
            if relatives:
                latest_relative_ef = max(
                    (r.ef for r in relatives if r.ef is not None),
                    default=None
                )
                # Set LF to the latest EF of relatives, or use project finish as fallback.
                task.lf = latest_relative_ef if latest_relative_ef is not None else project_finish_date
            else:
                # If no relatives, the task's LF is the project finish date.
                task.lf = project_finish_date

    # Second, calculate Late Start (LS) and Slack for all tasks based on their new LF.
    for task in task_objects_with_ef:
        if task.lf is not None:
            duration_days = int(math.ceil(task.actual_duration))
            task.ls = task.lf - datetime.timedelta(days=max(0, duration_days))
            
            if task.es is not None:
                task.slack = (task.ls - task.es).days

    return task_objects_with_ef

def _find_relatives(task_vertex: 'Vertex', task_graph: 'Graph') -> tuple[set, set]:
    """
    Finds all sibling and cousin tasks for a given task vertex in the graph.

    - Siblings are other tasks that share the same parent (predecessor).
    - Cousins are tasks whose parents are siblings of the given task's parent.

    Args:
        task_vertex: The vertex of the task to find relatives for.
        task_graph: The project's dependency graph.

    Returns:
        A tuple containing a set of sibling Task objects and a set of cousin Task objects.
    """
    siblings = set()
    cousins = set()
    
    # 1. Find Siblings
    # A sibling shares the same parent.
    parents = {edge.opposite(task_vertex) for edge in task_graph.incident_edges(task_vertex, outgoing=False)}
    for parent_v in parents:
        for sibling_edge in task_graph.incident_edges(parent_v, outgoing=True):
            sibling_v = sibling_edge.opposite(parent_v)
            if sibling_v != task_vertex:
                siblings.add(sibling_v.element())

    # 2. Find Cousins
    # A cousin's parent is a sibling of this task's parent.
    grandparents = {edge.opposite(p) for p in parents for edge in task_graph.incident_edges(p, outgoing=False)}
    
    aunts_uncles = set()
    for gp_v in grandparents:
        for uncle_edge in task_graph.incident_edges(gp_v, outgoing=True):
            uncle_v = uncle_edge.opposite(gp_v)
            if uncle_v not in parents:
                aunts_uncles.add(uncle_v)

    for uncle_v in aunts_uncles:
        for cousin_edge in task_graph.incident_edges(uncle_v, outgoing=True):
            cousin_v = cousin_edge.opposite(uncle_v)
            cousins.add(cousin_v.element())
            
    return siblings, cousins

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
                     task_dependency_graph.insert_edge(v_predecessor, v_successor, v_predecessor._element.base_duration)
            else:
                original_pred_task_exists = any(t.id == pred_id for t in task_objects)
                if original_pred_task_exists:
                     print(f"Note: Predecessor ID {pred_id} for task {task.id} ('{task.name}') refers to an impossible task. This dependency link is ignored in the graph.")
                else:
                     print(f"Warning: Predecessor ID {pred_id} for task {task.id} ('{task.name}') not found. Edge not created.")
    return task_dependency_graph, vertex_map

def topological_sort_kahn(graph: Graph) -> list[Task] | None:
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

# This helper function is still needed for the fallback strategy
def _find_ancestry_edges(
    leaf: 'Vertex', 
    task_graph: 'Graph'
) -> set['Edge']:
    """
    Performs a backward traversal (reverse BFS) from a leaf to find all
    edges that are part of its ancestral path.
    """
    ancestry_edges = set()
    queue = [leaf]
    visited = {leaf}
    
    while queue:
        current_vertex = queue.pop(0)
        
        # Find all edges coming into the current vertex
        for edge in task_graph.incident_edges(current_vertex, outgoing=False):
            ancestry_edges.add(edge)
            predecessor = edge.opposite(current_vertex)
            if predecessor not in visited:
                visited.add(predecessor)
                queue.append(predecessor)
                
    return ancestry_edges

def auto_connect_parallel_leaves_by_max_weight(
    task_graph: 'Graph', 
    task_vertex_map: dict[int, 'Vertex']
) -> tuple['Graph', dict[int, 'Vertex']]:
    """
    Finds leaf nodes and connects them to a successor using a hybrid strategy.

    1.  **Primary Strategy (Local Search):** If a leaf has immediate siblings
        (other children of the same parent), it will only look at the outgoing
        edges from those siblings to find the highest-weight connection.

    2.  **Fallback Strategy (Global Search):** If a leaf is a "single child"
        (has no immediate siblings), it will look for the highest-weight edge
        anywhere in the graph, excluding its own ancestral path.
        
    This version assumes the 'element' of an edge is its weight.
    """
    all_vertices = list(task_vertex_map.values())
    leaf_vertices = [v for v in all_vertices if task_graph.degree(v, outgoing=True) == 0]
    
    if not leaf_vertices:
        return task_graph, task_vertex_map
        
    print(f"DEBUG: Found {len(leaf_vertices)} initial leaf vertices.")

    all_graph_edges = list(task_graph.edges())

    for leaf in leaf_vertices:
        leaf_task = leaf.element()
        max_weight_edge = None

        # --- HYBRID LOGIC: First, check for immediate siblings ---
        predecessors = list(task_graph.incident_edges(leaf, outgoing=False))
        has_siblings = False

        if predecessors:
            # For simplicity, we check siblings from the first parent found.
            # In a tree structure, there will only be one.
            parent = predecessors[0].opposite(leaf)
            parent_successors = list(task_graph.incident_edges(parent, outgoing=True))
            
            if len(parent_successors) > 1:
                has_siblings = True
                siblings = [edge.opposite(parent) for edge in parent_successors if edge.opposite(parent) != leaf]
                
                # --- PRIMARY STRATEGY (Local Sibling Search) ---
                print(f"INFO: Leaf '{leaf_task.name}' has siblings. Using local search.")
                for sibling in siblings:
                    # Don't learn from other leaves
                    if task_graph.degree(sibling, outgoing=True) == 0:
                        continue
                    
                    # Examine the sibling's outgoing edges
                    for edge in task_graph.incident_edges(sibling, outgoing=True):
                        if edge.element() is not None:
                            if max_weight_edge is None or edge.element() > max_weight_edge.element():
                                max_weight_edge = edge

        if not has_siblings:
            # --- FALLBACK STRATEGY (Global Ancestry-Based Search) ---
            print(f"INFO: Leaf '{leaf_task.name}' has no siblings. Using graph-wide search.")
            ancestry_edges = _find_ancestry_edges(leaf, task_graph)

            for edge in all_graph_edges:
                if edge in ancestry_edges:
                    continue  # Skip edges from the leaf's own history

                if edge.element() is not None:
                    if max_weight_edge is None or edge.element() > max_weight_edge.element():
                        max_weight_edge = edge
        
        # --- CONNECTION LOGIC (Applies to the result of either strategy) ---
        if max_weight_edge:
            target_successor = max_weight_edge.endpoints()[1]
            donor_source = max_weight_edge.endpoints()[0]
            
            print(f"INFO: Auto-connecting leaf '{leaf_task.name}' "
                  f"to '{target_successor.element().name}' "
                  f"(from donor edge '{donor_source.element().name}'->"
                  f"'{target_successor.element().name}' with "
                  f"highest-weight: {max_weight_edge.element()}).")
            
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
    user_input = ""
    execute = False
    while not execute:
        user_input = input("Project Start Date (d/m/yyyy|Press Enter for Today's Date): ") or datetime.datetime.today().strftime('%d/%m/%Y')
        try:
        # 1. Parse the string using datetime.strptime()
        # %d = day, %m = month, %Y = 4-digit year
        # This returns a datetime object.
            date_obj = datetime.datetime.strptime(user_input, "%d/%m/%Y")
        
        # 2. Extract just the date part (if you don't need time)
            PROJECT_START_DATE = date_obj.date()
        
        # 3. Set execute to True to exit the loop on success
            execute = True

        except ValueError:
            # Catches errors if the input doesn't match the format
            print("Invalid date format. Please use d/m/yyyy and try again.")
    
    worker_costs_filepath = os.path.join(PROJECT_PATH,'worker_costs.csv')
    open(worker_costs_filepath, 'r').close() 
    tasks_filepath = os.path.join(PROJECT_PATH,'tasks.csv')
    open(tasks_filepath, 'r').close()

    availability, daily_costs = parse_worker_costs(worker_costs_filepath)
    
    df,tasks_list_dicts = parse_tasks(tasks_filepath)
    if not tasks_list_dicts: tasks_list_dicts = []
    final_tasks_to_report = [] # Initialize to ensure it's defined

    if tasks_list_dicts and availability:
        all_task_objects = calculate_task_initial_values(tasks_list_dicts, availability)
        
        # Pass the original list to build the graph. It will filter out impossible tasks.
        task_graph, task_vertex_map = build_task_graph(all_task_objects) 

        topologically_sorted_tasks_elements = None # Ensure defined
        if task_graph and task_vertex_map:
            topologically_sorted_tasks_elements = topological_sort_kahn(task_graph)
            if topologically_sorted_tasks_elements is None:
                print("Could not perform topological sort (cycle detected or other graph issue).")

            if task_graph.vertex_count() > 0:
                schedulable_tasks_list = [v.element() for v in task_graph.vertices()]

                schedule_tasks_forward_pass(
                    schedulable_tasks_list, copy.deepcopy(availability), PROJECT_START_DATE,
                    task_graph, task_vertex_map, topologically_sorted_tasks_elements)
                
                schedule_tasks_custom_backward_pass(
                    schedulable_tasks_list, 
                    task_graph,      # Added task_graph
                    task_vertex_map  # Added task_vertex_map
                ) 
                
                calculate_task_costs(schedulable_tasks_list, daily_costs)

                # --- CONSOLIDATE FINAL REPORT ---
                # Create a dictionary of the fully processed, schedulable tasks for easy lookup.
                processed_tasks_dict = {task.id: task for task in schedulable_tasks_list}

                final_tasks_to_report = [
                                        processed_tasks_dict.get(task.id, task) for task in all_task_objects
                                        ]
            else: print("No schedulable tasks in the graph to simulate.")
        else: print("Task graph could not be built or is empty. Skipping simulation.")
    else: print("\nSkipping calculations and scheduling due to parsing errors or missing data.")

    if final_tasks_to_report:
        clean_report = []
        total_project_cost = 0.0
        actual_project_finish_date = PROJECT_START_DATE 
        sorted_tasks_for_report = sorted(final_tasks_to_report, key=lambda t: t.es)
        for i,task in enumerate(sorted_tasks_for_report): 
            if task.status not in ['pending', 'error_cycle_dependency'] and task.ef is not None and task.ef > actual_project_finish_date :
                actual_project_finish_date = task.ef
            if task.cost is not None: total_project_cost += task.cost
        finish_date_str = format_date(actual_project_finish_date) if actual_project_finish_date != PROJECT_START_DATE or any(t.ef for t in sorted_tasks_for_report if t.ef is not None) else "N/A (No tasks scheduled)"
        print("\n--- Project Summary ---")
        print(f"Initial Project Start Date: {format_date(PROJECT_START_DATE)}")
        print(f"Calculated Project Finish Date: {finish_date_str}")
        print(f"Total Estimated Project Cost: {total_project_cost:.2f}")
    else: print("No tasks to display.")
    if final_tasks_to_report : export_timeline_to_csv(final_tasks_to_report, os.path.join(PROJECT_PATH,'timeline_report.csv')) 
    print("\n--- Script Execution Finished ---")
