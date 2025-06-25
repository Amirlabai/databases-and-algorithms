import unittest
import datetime
from project.submition.timeline_calculator import (
    Task, PROJECT_START_DATE,
    calculate_task_initial_values, schedule_tasks_forward_pass,
    schedule_tasks_backward_pass, calculate_task_costs,
    build_task_graph, topological_sort_kahn
)

class TestTaskScheduling(unittest.TestCase):
    def setUp(self):
        self.sample_tasks_dict = [
            {"id": 1, "name": "Start", "base_duration": 2, "predecessors": [], "required_workers": {"A": 1}},
            {"id": 2, "name": "Middle", "base_duration": 3, "predecessors": [1], "required_workers": {"A": 1}},
            {"id": 3, "name": "End", "base_duration": 1, "predecessors": [2], "required_workers": {"A": 1}},
        ]
        self.worker_availability = {"A": 1}
        self.worker_costs = {"A": 100}

    def test_initial_duration_calc(self):
        tasks = calculate_task_initial_values(self.sample_tasks_dict, self.worker_availability)
        self.assertEqual(tasks[0].actual_duration, 2.0)
        self.assertEqual(tasks[1].actual_duration, 3.0)
        self.assertEqual(tasks[2].actual_duration, 1.0)

    def test_forward_pass_ef(self):
        tasks = calculate_task_initial_values(self.sample_tasks_dict, self.worker_availability)
        graph, vertex_map = build_task_graph(tasks)
        sorted_tasks = topological_sort_kahn(graph)
        scheduled = schedule_tasks_forward_pass(tasks, self.worker_availability, PROJECT_START_DATE, graph, vertex_map, sorted_tasks)
        task_dict = {t.id: t for t in scheduled}
        self.assertEqual(task_dict[1].es, PROJECT_START_DATE)
        self.assertEqual(task_dict[1].ef, PROJECT_START_DATE + datetime.timedelta(days=1))
        self.assertEqual(task_dict[3].status, 'completed')

    def test_backward_pass_slack(self):
        tasks = calculate_task_initial_values(self.sample_tasks_dict, self.worker_availability)
        graph, vertex_map = build_task_graph(tasks)
        sorted_tasks = topological_sort_kahn(graph)
        schedule_tasks_forward_pass(tasks, self.worker_availability, PROJECT_START_DATE, graph, vertex_map, sorted_tasks)
        schedule_tasks_backward_pass(tasks, graph, vertex_map)
        for t in tasks:
            self.assertIsNotNone(t.slack)

    def test_cost_calculation(self):
        tasks = calculate_task_initial_values(self.sample_tasks_dict, self.worker_availability)
        graph, vertex_map = build_task_graph(tasks)
        sorted_tasks = topological_sort_kahn(graph)
        schedule_tasks_forward_pass(tasks, self.worker_availability, PROJECT_START_DATE, graph, vertex_map, sorted_tasks)
        calculate_task_costs(tasks, self.worker_costs)
        self.assertGreater(tasks[0].cost, 0)

if __name__ == '__main__':
    unittest.main()
