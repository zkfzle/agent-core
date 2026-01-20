"""Unit tests for TaskManager

Tests all core functionality of the TaskManager class including:
- State management
- CRUD operations
- Task hierarchy management
- Status management
- Priority management
"""

# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.


import unittest

from openjiuwen.core.controller.modules.task_manager import TaskManager, TaskManagerState, TaskFilter
from openjiuwen.core.controller.schema.task import Task, TaskStatus


class TestTaskManager(unittest.TestCase):
    """Test suite for TaskManager class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        config = {"default_task_priority": 1}
        self.task_manager = TaskManager(config=config)
        
        self.sample_task = Task(
            session_id="session1",
            task_id="task1",
            task_type="test_task",
            description="Test task",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        
        self.sample_tasks = [
            Task(
                session_id="session1",
                task_id="task1",
                task_type="test_task",
                description="Task 1",
                priority=1,
                Status=TaskStatus.SUBMITTED
            ),
            Task(
                session_id="session1",
                task_id="task2",
                task_type="test_task",
                description="Task 2",
                priority=2,
                Status=TaskStatus.WORKING
            ),
            Task(
                session_id="session2",
                task_id="task3",
                task_type="test_task",
                description="Task 3",
                priority=1,
                Status=TaskStatus.COMPLETED
            ),
        ]

    # ==================== Add Task Tests ====================
    def test_add_single_task(self):
        """Test adding a single task"""
        self.task_manager.add_task(self.sample_task)

        self.assertIn(self.sample_task.task_id, self.task_manager.tasks)
        self.assertEqual(self.task_manager.tasks[self.sample_task.task_id], self.sample_task)

    def test_add_multiple_tasks(self):
        """Test adding multiple tasks at once"""
        self.task_manager.add_task(self.sample_tasks)

        self.assertEqual(len(self.task_manager.tasks), 3)
        for task in self.sample_tasks:
            self.assertIn(task.task_id, self.task_manager.tasks)

    def test_add_task_with_parent(self):
        """Test adding a task with a parent task"""
        parent_task = Task(
            session_id="session1",
            task_id="parent_task",
            task_type="test_task",
            description="Parent task",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )

        self.task_manager.add_task(parent_task)
        self.sample_task.parent_task_id = "parent_task"
        self.task_manager.add_task(self.sample_task)

    # ==================== Get Task Tests ====================
    def test_get_task_by_id(self):
        """Test getting a task by ID"""
        self.task_manager.add_task(self.sample_task)

        result = self.task_manager.get_task(task_filter=TaskFilter(task_id="task1"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].task_id, "task1")

    def test_get_task_by_id_list(self):
        """Test getting tasks by ID list"""
        self.task_manager.add_task(self.sample_tasks)

        result = self.task_manager.get_task(task_filter=TaskFilter(task_id=["task1", "task2"]))
        self.assertEqual(len(result), 2)
        self.assertEqual({t.task_id for t in result}, {"task1", "task2"})

    def test_get_task_by_session_id(self):
        """Test getting tasks by session ID"""
        self.task_manager.add_task(self.sample_tasks)

        result = self.task_manager.get_task(task_filter=TaskFilter(session_id="session1"))
        self.assertEqual(len(result), 2)
        self.assertTrue(all(t.session_id == "session1" for t in result))

    def test_get_task_by_priority(self):
        """Test getting tasks by priority"""
        self.task_manager.add_task(self.sample_tasks)

        result = self.task_manager.get_task(task_filter=TaskFilter(priority=1))
        self.assertEqual(len(result), 2)
        self.assertTrue(all(t.priority == 1 for t in result))

    def test_get_task_by_status(self):
        """Test getting tasks by status"""
        self.task_manager.add_task(self.sample_tasks)

        result = self.task_manager.get_task(task_filter=TaskFilter(status=TaskStatus.SUBMITTED))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].status, TaskStatus.SUBMITTED)

    def test_get_task_by_user_id(self):
        """Test getting tasks by user_id in metadata"""
        self.sample_task.metadata = {"user_id": "user1"}
        self.task_manager.add_task(self.sample_task)

        result = self.task_manager.get_task(task_filter=TaskFilter(user_id="user1"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].task_id, "task1")

    def test_get_root_tasks(self):
        """Test getting root tasks"""
        self.task_manager.add_task(self.sample_tasks)

        # Add a child task
        child_task = Task(
            session_id="session1",
            task_id="child_task",
            task_type="test_task",
            description="Child task",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="task1"
        )
        self.task_manager.add_task(child_task)

        result = self.task_manager.get_task(task_filter=TaskFilter(is_root=True))
        self.assertEqual(len(result), 3)  # task1, task2, task3 (not child_task)
        self.assertNotIn("child_task", {t.task_id for t in result})

    def test_get_task_with_children(self):
        """Test getting tasks with children"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child1 = Task(
            session_id="session1",
            task_id="child1",
            task_type="test_task",
            description="Child 1",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )
        child2 = Task(
            session_id="session1",
            task_id="child2",
            task_type="test_task",
            description="Child 2",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )

        self.task_manager.add_task([parent, child1, child2])

        result = self.task_manager.get_task(task_filter=TaskFilter(task_id="parent", with_children=True))
        self.assertEqual(len(result), 3)  # parent + 2 children
        self.assertEqual({t.task_id for t in result}, {"parent", "child1", "child2"})

    def test_get_task_with_recursive_children(self):
        """Test getting tasks with recursive children"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child = Task(
            session_id="session1",
            task_id="child",
            task_type="test_task",
            description="Child",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )
        grandchild = Task(
            session_id="session1",
            task_id="grandchild",
            task_type="test_task",
            description="Grandchild",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="child"
        )

        self.task_manager.add_task([parent, child, grandchild])

        result = self.task_manager.get_task(task_filter=TaskFilter(task_id="parent", with_children=True))
        self.assertEqual(len(result), 3)  # parent + child + grandchild (with_children is recursive)
        self.assertEqual({t.task_id for t in result}, {"parent", "child", "grandchild"})

    def test_get_all_tasks(self):
        """Test getting all tasks when no filter is provided"""
        self.task_manager.add_task(self.sample_tasks)

        # When task_filter is None, get_task returns all tasks
        result = self.task_manager.get_task(task_filter=None)
        self.assertEqual(len(result), 3)

    def test_get_task_highest_priority_error(self):
        """Test that get_task raises error for 'highest' priority"""
        self.task_manager.add_task(self.sample_tasks)

        with self.assertRaises(Exception):  # Should raise JiuWenBaseException
            self.task_manager.get_task(task_filter=TaskFilter(priority="highest"))

    # ==================== Pop Task Tests ====================
    def test_pop_task_by_id(self):
        """Test popping a task by ID"""
        self.task_manager.add_task(self.sample_task)

        result = self.task_manager.pop_task(task_filter=TaskFilter(task_id="task1"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].task_id, "task1")
        self.assertNotIn("task1", self.task_manager.tasks)

    def test_pop_task_highest_priority(self):
        """Test popping task with highest priority"""
        self.task_manager.add_task(self.sample_tasks)

        result = self.task_manager.pop_task(task_filter=TaskFilter(priority="highest"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].priority, 2)  # Highest priority in sample_tasks

    def test_pop_task_empty(self):
        """Test popping from empty task manager"""
        result = self.task_manager.pop_task(task_filter=TaskFilter(priority="highest"))
        self.assertEqual(result, [])

    # ==================== Update Task Tests ====================
    def test_update_task(self):
        """Test updating a task"""
        self.task_manager.add_task(self.sample_task)

        self.sample_task.description = "Updated description"
        self.sample_task.status = TaskStatus.WORKING

        success = self.task_manager.update_task(self.sample_task)
        self.assertTrue(success)
        self.assertEqual(self.task_manager.tasks["task1"].description, "Updated description")
        self.assertEqual(self.task_manager.tasks["task1"].status, TaskStatus.WORKING)

    def test_update_nonexistent_task(self):
        """Test updating a non-existent task"""
        success = self.task_manager.update_task(self.sample_task)
        self.assertFalse(success)
        self.assertNotIn("task1", self.task_manager.tasks)

    # ==================== Remove Task Tests ====================
    def test_remove_task_by_id(self):
        """Test removing a task by ID"""
        self.task_manager.add_task(self.sample_task)

        self.task_manager.remove_task(task_filter=TaskFilter(task_id="task1"))
        self.assertNotIn("task1", self.task_manager.tasks)

    def test_remove_task_with_children(self):
        """Test removing a task with children"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child = Task(
            session_id="session1",
            task_id="child",
            task_type="test_task",
            description="Child",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )

        self.task_manager.add_task([parent, child])
        self.task_manager.remove_task(task_filter=TaskFilter(task_id="parent", with_children=True))

        self.assertNotIn("parent", self.task_manager.tasks)
        self.assertNotIn("child", self.task_manager.tasks)

    def test_remove_task_promotes_children_to_root(self):
        """Test that removing a parent promotes children to root"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child = Task(
            session_id="session1",
            task_id="child",
            task_type="test_task",
            description="Child",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )

        self.task_manager.add_task([parent, child])
        self.task_manager.remove_task(task_filter=TaskFilter(task_id="parent"))

        # Child should be promoted to root
        self.assertIsNone(self.task_manager.tasks["child"].parent_task_id)

    def test_remove_task_by_session_id(self):
        """Test removing tasks by session ID"""
        self.task_manager.add_task(self.sample_tasks)

        self.task_manager.remove_task(task_filter=TaskFilter(session_id="session1"))

        self.assertEqual(len(self.task_manager.tasks), 1)
        self.assertIn("task3", self.task_manager.tasks)  # Only session2 task remains

    def test_remove_task_by_status(self):
        """Test removing tasks by status"""
        self.task_manager.add_task(self.sample_tasks)

        self.task_manager.remove_task(task_filter=TaskFilter(status=TaskStatus.COMPLETED))

        self.assertEqual(len(self.task_manager.tasks), 2)
        self.assertNotIn("task3", self.task_manager.tasks)

    def test_remove_task_no_filter_error(self):
        """Test that remove_task raises error when no filter criteria provided"""
        with self.assertRaises(Exception):  # Should raise JiuWenBaseException
            self.task_manager.remove_task(task_filter=TaskFilter(task_id=None, session_id=None, 
                                                                  user_id=None, priority=None, 
                                                                  status=None, is_root=False))

    def test_remove_task_highest_priority_error(self):
        """Test that remove_task raises error for 'highest' priority"""
        self.task_manager.add_task(self.sample_tasks)

        with self.assertRaises(Exception):  # Should raise JiuWenBaseException
            self.task_manager.remove_task(task_filter=TaskFilter(priority="highest"))

    def test_pop_task_none_filter_error(self):
        """Test that pop_task raises error when task_filter is None"""
        with self.assertRaises(Exception):  # Should raise JiuWenBaseException
            self.task_manager.pop_task(task_filter=None)

    def test_remove_task_none_filter_error(self):
        """Test that remove_task raises error when task_filter is None"""
        with self.assertRaises(Exception):  # Should raise JiuWenBaseException
            self.task_manager.remove_task(task_filter=None)

    # ==================== Get Child Task Tests ====================
    def test_get_child_task(self):
        """Test getting child tasks"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child1 = Task(
            session_id="session1",
            task_id="child1",
            task_type="test_task",
            description="Child 1",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )
        child2 = Task(
            session_id="session1",
            task_id="child2",
            task_type="test_task",
            description="Child 2",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )

        self.task_manager.add_task([parent, child1, child2])

        result = self.task_manager.get_child_task("parent")
        self.assertEqual(len(result), 2)
        self.assertEqual({t.task_id for t in result}, {"child1", "child2"})

    def test_get_child_task_recursive(self):
        """Test getting child tasks recursively"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child = Task(
            session_id="session1",
            task_id="child",
            task_type="test_task",
            description="Child",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )
        grandchild = Task(
            session_id="session1",
            task_id="grandchild",
            task_type="test_task",
            description="Grandchild",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="child"
        )

        self.task_manager.add_task([parent, child, grandchild])

        result = self.task_manager.get_child_task("parent", is_recursive=True)
        self.assertEqual(len(result), 2)  # child + grandchild
        self.assertEqual({t.task_id for t in result}, {"child", "grandchild"})

    # ==================== Update Task Status Tests ====================
    def test_update_task_status(self):
        """Test updating task status"""
        self.task_manager.add_task(self.sample_task)

        self.task_manager.update_task_status("task1", TaskStatus.WORKING)

        self.assertEqual(self.task_manager.tasks["task1"].status, TaskStatus.WORKING)

    def test_update_task_status_with_children(self):
        """Test updating task status with children"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child = Task(
            session_id="session1",
            task_id="child",
            task_type="test_task",
            description="Child",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )

        self.task_manager.add_task([parent, child])
        self.task_manager.update_task_status("parent", TaskStatus.WORKING, with_children=True)

        self.assertEqual(self.task_manager.tasks["parent"].status, TaskStatus.WORKING)
        self.assertEqual(self.task_manager.tasks["child"].status, TaskStatus.WORKING)

    def test_update_task_status_recursive(self):
        """Test updating task status recursively"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child = Task(
            session_id="session1",
            task_id="child",
            task_type="test_task",
            description="Child",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )
        grandchild = Task(
            session_id="session1",
            task_id="grandchild",
            task_type="test_task",
            description="Grandchild",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="child"
        )

        self.task_manager.add_task([parent, child, grandchild])
        self.task_manager.update_task_status("parent", TaskStatus.WORKING, with_children=True, is_recursive=True)

        self.assertEqual(self.task_manager.tasks["parent"].status, TaskStatus.WORKING)
        self.assertEqual(self.task_manager.tasks["child"].status, TaskStatus.WORKING)
        self.assertEqual(self.task_manager.tasks["grandchild"].status, TaskStatus.WORKING)

    # ==================== Set Priority Tests ====================
    def test_set_priority(self):
        """Test setting task priority"""
        self.task_manager.add_task(self.sample_task)

        self.task_manager.set_priority("task1", 5)

        self.assertEqual(self.task_manager.tasks["task1"].priority, 5)

    def test_set_priority_string(self):
        """Test setting priority with string value"""
        self.task_manager.add_task(self.sample_task)

        self.task_manager.set_priority("task1", "3")

        self.assertEqual(self.task_manager.tasks["task1"].priority, 3)

    def test_set_priority_with_children(self):
        """Test setting priority with children"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child = Task(
            session_id="session1",
            task_id="child",
            task_type="test_task",
            description="Child",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )

        self.task_manager.add_task([parent, child])
        self.task_manager.set_priority("parent", 5, with_children=True)

        self.assertEqual(self.task_manager.tasks["parent"].priority, 5)
        self.assertEqual(self.task_manager.tasks["child"].priority, 5)

    def test_set_priority_recursive(self):
        """Test setting priority recursively"""
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child = Task(
            session_id="session1",
            task_id="child",
            task_type="test_task",
            description="Child",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="parent"
        )
        grandchild = Task(
            session_id="session1",
            task_id="grandchild",
            task_type="test_task",
            description="Grandchild",
            priority=1,
            Status=TaskStatus.SUBMITTED,
            parent_task_id="child"
        )

        self.task_manager.add_task([parent, child, grandchild])
        self.task_manager.set_priority("parent", 5, with_children=True, is_recursive=True)

        self.assertEqual(self.task_manager.tasks["parent"].priority, 5)
        self.assertEqual(self.task_manager.tasks["child"].priority, 5)
        self.assertEqual(self.task_manager.tasks["grandchild"].priority, 5)

    # ==================== State Management Tests ====================
    def test_get_state(self):
        """Test getting task manager state"""
        self.task_manager.add_task(self.sample_tasks)

        state = self.task_manager.get_state()

        self.assertIsInstance(state, TaskManagerState)
        self.assertEqual(len(state.tasks), 3)
        self.assertGreater(len(state.priority_index), 0)
        self.assertGreater(len(state.root_tasks), 0)

    def test_load_state(self):
        """Test loading task manager state"""
        self.task_manager.add_task(self.sample_tasks)
        state = self.task_manager.get_state()

        # Create new manager and load state
        new_manager = TaskManager(config={})
        new_manager.load_state(state)

        self.assertEqual(len(new_manager.tasks), 3)
        self.assertEqual(new_manager.tasks["task1"].task_id, "task1")
        self.assertEqual(new_manager.tasks["task2"].task_id, "task2")
        self.assertEqual(new_manager.tasks["task3"].task_id, "task3")

    def test_state_persistence(self):
        """Test that state can be saved and restored correctly"""
        # Create hierarchical tasks
        parent = Task(
            session_id="session1",
            task_id="parent",
            task_type="test_task",
            description="Parent",
            priority=1,
            Status=TaskStatus.SUBMITTED
        )
        child = Task(
            session_id="session1",
            task_id="child",
            task_type="test_task",
            description="Child",
            priority=2,
            Status=TaskStatus.WORKING,
            parent_task_id="parent"
        )

        self.task_manager.add_task([parent, child])
        state = self.task_manager.get_state()

        # Create new manager and verify state
        new_manager = TaskManager(config={})
        new_manager.load_state(state)

        self.assertIn("parent", new_manager.tasks)
        self.assertIn("child", new_manager.tasks)

