import csv
import datetime
import math
import ast  # For safely evaluating string representations of dicts
import copy # For deep copying worker pool
import pandas as pd
from graph import Graph, Vertex, Edge # Assuming graph.py is in the same directory or project path

# 1. Define the project start date
PROJECT_START_DATE = datetime.date(2025, 6, 30)

# 2. Implement a function parse_worker_costs(filepath) - No changes needed
def parse_worker_costs(filepath: str) -> tuple[dict[str, int], dict[str, int]]:
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
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return {}, {}
    except Exception as e_file:
        print(f"An unexpected error occurred while parsing {filepath} with pandas: {e_file}")
        return {}, {}
    return worker_availability, worker_daily_costs

# 3. Implement a function parse_tasks(filepath) - REFINED
def parse_tasks(filepath: str) -> list[dict]:
    """
    Parses tasks from a CSV file using pandas.
    """
    tasks_list = []
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        # Robustness: Ensure column names are clean for itertuples
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]

        for row in df.itertuples(index=False):
            try:
                # --- REFINEMENT ---
                # Changed from row._1 to row.Task_Name for clarity and robustness.
                # Assumes the column header is 'Task Name'.
                task_id = int(row.Tid)
                name = str(row.Task_Name) 
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
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e_file:
        print(f"An unexpected error occurred while parsing {filepath} with pandas: {e_file}")
        return []
    return tasks_list

# Task Class Definition - No changes needed
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
        return f"Task(id={self.id}, name='{self.name}', ...)"

# Initial Calculation Function - No changes needed
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
                task.notes += f"Worker type '{worker_type}' has zero global availability. "
                task_is_impossible = True
                break
            num_assigned_for_type = min(num_required, num_available_globally)
            current_task_assigned_workers_map[worker_type] = num_assigned_for_type
            duration_contributions.append(math.ceil(task.base_duration * num_required / num_assigned_for_type))

        if task_is_impossible:
            task.assigned_workers_map = {}
        else:
            task.actual_duration = float(max(duration_contributions, default=0.0))
            task.assigned_workers_map = current_task_assigned_workers_map
        task_objects.append(task)
    return task_objects

# Simulation-based Scheduling Logic (Forward Pass) - REFINED
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

    max_project_days = 10 * 365 
    project_end_horizon = project_start_date + datetime.timedelta(days=max_project_days)

    while len(completed_task_ids) < len(tasks_in_graph):
        if current_date > project_end_horizon:
            print("Error: Simulation exceeded maximum project horizon.")
            for task_id_active in list(active_tasks_details.keys()):
                 active_tasks_details[task_id_active]['task_obj'].status = "error_timeout"
            for task in tasks_in_graph:
                if task.status == 'pending': task.status = "error_timeout"
            break
        
        tasks_processed_this_iteration = False
        for task_id_active in list(active_tasks_details.keys()):
            details = active_tasks_details[task_id_active]
            if details['ef_date'] < current_date:
                completed_task_obj = details['task_obj']
                completed_task_obj.status = 'completed'
                completed_task_ids.add(completed_task_obj.id)
                for worker_type, count in details['workers_used'].items():
                    worker_pool_free_count[worker_type] += count
                del active_tasks_details[task_id_active]
                tasks_processed_this_iteration = True
        
        iteration_source = topologically_sorted_task_elements if topologically_sorted_task_elements is not None else tasks_in_graph
        for t_obj in iteration_source:
            if t_obj.id in tasks_map and tasks_map[t_obj.id].status == 'pending':
                task_to_schedule = tasks_map[t_obj.id]
                task_vertex = vertex_map.get(task_to_schedule.id)
                if not task_vertex: continue

                graph_predecessor_tasks = [edge.opposite(task_vertex).element() for edge in task_graph.incident_edges(task_vertex, outgoing=False)]
                if not all(tasks_map[p.id].status == 'completed' for p in graph_predecessor_tasks):
                    continue
                
                can_allocate_resources = True
                for worker_type, needed_count in task_to_schedule.assigned_workers_map.items():
                    if worker_pool_free_count.get(worker_type, 0) < needed_count:
                        can_allocate_resources = False; break
                
                if can_allocate_resources:
                    task_to_schedule.es = current_date
                    duration_in_days = int(math.ceil(task_to_schedule.actual_duration))
                    task_to_schedule.ef = current_date + datetime.timedelta(days=duration_in_days)
                    active_tasks_details[task_to_schedule.id] = {
                        'task_obj': task_to_schedule, 'ef_date': task_to_schedule.ef,
                        'workers_used': copy.deepcopy(task_to_schedule.assigned_workers_map)}
                    
                    for worker_type, count in task_to_schedule.assigned_workers_map.items():
                        worker_pool_free_count[worker_type] -= count
                    
                    task_to_schedule.status = 'active'
                    tasks_processed_this_iteration = True

        if not tasks_processed_this_iteration and active_tasks_details:
            # --- REFINEMENT ---
            # Optimized the time-jump logic to be more direct.
            min_next_ef_date = min(details['ef_date'] for details in active_tasks_details.values())
            current_date = min_next_ef_date
        elif not tasks_processed_this_iteration and not active_tasks_details and len(completed_task_ids) < len(tasks_in_graph):
            print(f"Warning: Deadlock detected at {current_date}.")
            for task in tasks_in_graph:
                if task.status == 'pending':
                    task.status = "error_deadlock"
            break 
        else: 
            current_date += datetime.timedelta(days=1)
    return tasks_in_graph

# Backward Pass Scheduling Logic - REWRITTEN
def schedule_tasks_backward_pass(
    task_objects: list[Task],
    task_graph: Graph,
    vertex_map: dict[int, Vertex],
    topologically_sorted_tasks_elements: list[Task] | None
) -> list[Task]:
    """
    Calculates LS, LF, and Slack using a reversed topological sort. This is more
    efficient and reliable than iterative relaxation.
    """
    if topologically_sorted_tasks_elements is None:
        print("Warning: Cannot perform backward pass without a valid topological sort (cycle likely).")
        return task_objects

    tasks_map = {t.id: t for t in task_objects}
    scheduled_tasks = [t for t in task_objects if t.es is not None and t.ef is not None]

    if not scheduled_tasks:
        return task_objects

    project_finish_date = max(task.ef for task in scheduled_tasks)

    # Initialize LF and LS for all tasks
    for task in task_objects:
        task.lf = None
        task.ls = None
        task.slack = None

    # Iterate backwards through the topologically sorted list.
    # This guarantees that when we process a task, all its successors
    # have already had their LS/LF calculated.
    for task_element in reversed(topologically_sorted_tasks_elements):
        task = tasks_map.get(task_element.id)
        
        # Skip tasks that were not scheduled in the forward pass
        if not task or task.es is None:
            continue

        task_vertex = vertex_map[task.id]
        graph_successors = [edge.opposite(task_vertex).element() for edge in task_graph.incident_edges(task_vertex, outgoing=True)]
        
        if not graph_successors:
            # If a task has no successors, its Late Finish is the project finish date.
            task.lf = project_finish_date
        else:
            # LF is the minimum of the Late Starts of all its successors.
            successor_ls_dates = [tasks_map[s.id].ls for s in graph_successors if s.id in tasks_map and tasks_map[s.id].ls is not None]
            if successor_ls_dates:
                task.lf = min(successor_ls_dates)
            else:
                # This case can happen if successors were impossible tasks
                task.lf = project_finish_date

        # Calculate Late Start and Slack
        duration_days = int(math.ceil(task.actual_duration))
        task.ls = task.lf - datetime.timedelta(days=duration_days)
        task.slack = (task.ls - task.es).days

    return task_objects

# Task Cost Calculation Logic - No changes needed
def calculate_task_costs(task_objects, worker_daily_costs) -> list[Task]:
    for task in task_objects:
        if task.es is not None and task.actual_duration != float('inf') and task.actual_duration > 0:
            task.cost = 0.0
            if task.assigned_workers_map:
                for worker_type, count in task.assigned_workers_map.items():
                    task.cost += count * task.actual_duration * worker_daily_costs.get(worker_type, 0)
        else: task.cost = 0.0
    return task_objects

# Graph-based functions - No changes needed
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
                     print(f"Note: Predecessor ID {pred_id} for task {task.id} refers to an impossible task and is ignored.")
                else:
                     print(f"Warning: Predecessor ID {pred_id} for task {task.id} not found. Edge not created.")
    return task_dependency_graph, vertex_map

def topological_sort_kahn(graph: Graph, vertex_map: dict[int, Vertex]) -> list[Task] | None:
    if not graph or not graph.vertices(): return []
    in_degree = {v: 0 for v in graph.vertices()}
    for edge in graph.edges():
        _origin, dest = edge.endpoints()
        in_degree[dest] += 1
    queue = [v for v in graph.vertices() if in_degree[v] == 0]
    sorted_tasks_list = []
    while queue:
        u = queue.pop(0)
        sorted_tasks_list.append(u.element())
        for edge in graph.incident_edges(u, outgoing=True):
            v = edge.opposite(u)
            in_degree[v] -= 1
            if in_degree[v] == 0: queue.append(v)
    if len(sorted_tasks_list) != len(in_degree):
        print(f"Error: Cycle detected in task graph. Cannot create a valid schedule.")
        return None
    return sorted_tasks_list

# Helper/Export functions - Refined formatting
def export_timeline_to_csv(task_objects: list[Task], filepath: str):
    report_data = []
    report_columns = ['Tid', 'Task Name', 'Actual Duration (days)', 'Start Date (ES)', 'End Date (EF)', 
                      'Late Start (LS)', 'Late Finish (LF)', 'Assigned Workers', 'Slack (days)', 
                      'Cost', 'Status', 'Notes']
    sorted_tasks_for_export = sorted(task_objects, key=lambda t: t.id)
    for task in sorted_tasks_for_export:
        report_data.append({
            'Tid': task.id, 'Task Name': task.name,
            'Actual Duration (days)': format_value(task.actual_duration, is_duration=True),
            'Start Date (ES)': format_date(task.es), 'End Date (EF)': format_date(task.ef),
            'Late Start (LS)': format_date(task.ls), 'Late Finish (LF)': format_date(task.lf),
            'Assigned Workers': format_workers(task.assigned_workers_map),
            'Slack (days)': format_value(task.slack),
            'Cost': f"{task.cost:.2f}" if task.cost is not None else "0.00",
            'Status': task.status, 'Notes': task.notes
        })
    try:
        pd.DataFrame(report_data, columns=report_columns).to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\nTimeline report successfully exported to {filepath}")
    except Exception as e: print(f"\nError exporting timeline to CSV: {e}")

def format_date(date_obj):
    if date_obj is None: return "N/A"
    return date_obj.strftime("%d/%m/%Y")

def format_workers(workers_map):
    if not workers_map: return "None"
    return ', '.join([f"{k}:{v}" for k, v in workers_map.items()])

def format_value(value, is_duration=False):
    if value is None: return "N/A"
    if isinstance(value, float) and value == float('inf'): return "Infinite"
    if is_duration:
        return int(round(value))
    return value

# Main execution block - UPDATED
if __name__ == "__main__":
    worker_costs_filepath = 'project\\worker_costs.csv'
    tasks_filepath = 'project\\tasks.csv'

    # --- For Demonstration: Create dummy CSV files ---
    # In a real scenario, these files would already exist.
    import os
    if not os.path.exists('project'): os.makedirs('project')
    # --- End of dummy file creation ---

    print(f"Project Start Date: {format_date(PROJECT_START_DATE)}")
    print("\n--- 1. Parsing Data ---")
    availability, daily_costs = parse_worker_costs(worker_costs_filepath)
    tasks_list_dicts = parse_tasks(tasks_filepath)
    
    final_tasks_to_report = []

    if tasks_list_dicts and availability:
        print("\n--- 2. Initializing Tasks & Building Graph ---")
        initial_task_objects = calculate_task_initial_values(tasks_list_dicts, availability, daily_costs)
        final_tasks_to_report = copy.deepcopy(initial_task_objects) 
        
        task_graph, task_vertex_map = build_task_graph(initial_task_objects)
        
        if task_graph and task_vertex_map:
            print(f"Task graph built. Vertices: {task_graph.vertex_count()}, Edges: {task_graph.edge_count()}")
            print("\n--- 3. Performing Topological Sort ---")
            topologically_sorted_tasks = topological_sort_kahn(task_graph, task_vertex_map)
            
            if topologically_sorted_tasks:
                print("Topological Sort Order (IDs):", [t.id for t in topologically_sorted_tasks])
                
                schedulable_tasks_list = [v.element() for v in task_graph.vertices()]

                print("\n--- 4. Scheduling (Forward Pass for ES, EF) ---")
                scheduled_forward = schedule_tasks_forward_pass(
                    schedulable_tasks_list, copy.deepcopy(availability), PROJECT_START_DATE,
                    task_graph, task_vertex_map, topologically_sorted_tasks)
                
                print("\n--- 5. Scheduling (Backward Pass for LS, LF, Slack) ---")
                # UPDATED function call with new parameters
                scheduled_backward = schedule_tasks_backward_pass(
                    scheduled_forward, task_graph, task_vertex_map, topologically_sorted_tasks)
                
                print("\n--- 6. Calculating Task Costs ---")
                final_scheduled_tasks = calculate_task_costs(scheduled_backward, daily_costs)
                
                # Update the final report list with the new, fully-calculated task objects
                final_tasks_map = {t.id: t for t in final_scheduled_tasks}
                for i, task in enumerate(final_tasks_to_report):
                    if task.id in final_tasks_map:
                        final_tasks_to_report[i] = final_tasks_map[task.id]
            else:
                print("Could not schedule due to graph cycle.")
        else:
            print("Task graph is empty. Skipping simulation.")
    else:
        print("\nSkipping scheduling due to parsing errors or missing data.")

    print("\n--- 7. Project Timeline & Task Details ---")
    header = (f"{'ID':<4} {'Task Name':<30} {'Duration':<10} {'Start':<12} {'End':<12} "
              f"{'Late Start':<12} {'Late Finish':<12} {'Slack':<7} {'Cost':<10} {'Status'}")
    print(header); print("-" * len(header))
    
    if final_tasks_to_report:
        total_project_cost = 0.0
        actual_project_finish_date = PROJECT_START_DATE
        for task in sorted(final_tasks_to_report, key=lambda t: t.id):
            if task.ef is not None and task.ef > actual_project_finish_date:
                actual_project_finish_date = task.ef
            total_project_cost += task.cost
            print(f"{task.id:<4} {task.name:<30} {format_value(task.actual_duration, is_duration=True):<10} {format_date(task.es):<12} {format_date(task.ef):<12} "
                  f"{format_date(task.ls):<12} {format_date(task.lf):<12} {format_value(task.slack):<7} {f'{task.cost:.2f}':<10} {task.status}")
        
        print("\n--- Project Summary ---")
        print(f"Calculated Project Finish Date: {format_date(actual_project_finish_date)}")
        print(f"Total Estimated Project Cost: {total_project_cost:.2f}")
        export_timeline_to_csv(final_tasks_to_report, "project/timeline_report.csv")
    else:
        print("No tasks to display.")

    print("\n--- Script Execution Finished ---")