# -*- coding: utf-8 -*-
"""
Task Scheduler for ATLAS Agent
Manages task scheduling and execution with priority queues and resource management
"""

import heapq
import time
import threading
import asyncio
import queue
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import defaultdict

logger = logging.getLogger(__name__)

class TaskPriority(IntEnum):
    """Task priority levels (lower number = higher priority)"""
    CRITICAL = 1    # Critical tasks (system recovery)
    HIGH = 2        # High priority (real-time inference)
    NORMAL = 3      # Normal priority (regular training)
    LOW = 4         # Low priority (data cleanup)
    BACKGROUND = 5  # Background tasks (log compression)

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class ResourceRequirement:
    """Resource requirement definition"""
    cpu_cores: float = 1.0
    memory_mb: int = 1024
    gpu_memory_mb: int = 0
    disk_io_mb_per_sec: int = 100
    network_mb_per_sec: int = 10
    estimated_duration_seconds: int = 300

@dataclass
class TaskMetadata:
    """Task metadata"""
    task_id: str
    task_type: str
    priority: TaskPriority
    resource_requirements: ResourceRequirement
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: int = 3600
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: Optional[TaskStatus] = None

@dataclass
class ScheduledTask:
    """Scheduled task wrapper"""
    metadata: TaskMetadata
    payload: Any
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Priority comparison for heap sorting"""
        if self.metadata.priority != other.metadata.priority:
            return self.metadata.priority < other.metadata.priority
        
        # Same priority, sort by time
        self_time = self.metadata.scheduled_at or self.metadata.created_at
        other_time = other.metadata.scheduled_at or other.metadata.created_at
        return self_time < other_time

@dataclass
class TaskResult:
    """Task result object"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    completed_at: Optional[datetime] = None

class ResourceMonitor:
    """System resource monitor"""
    
    def __init__(self):
        self.available_resources = {
            'cpu': 8.0,           # 8 CPU cores
            'memory': 16384,      # 16GB memory
            'gpu': 8192,          # 8GB GPU memory
            'disk_io': 1000,      # 1GB/s disk IO
            'network': 1000       # 1GB/s network
        }
        
        self.allocated_resources = defaultdict(float)
        self.resource_lock = threading.Lock()
    
    def can_allocate(self, requirements: ResourceRequirement) -> bool:
        """Check if resources can be allocated"""
        with self.resource_lock:
            checks = [
                self.allocated_resources['cpu'] + requirements.cpu_cores <= self.available_resources['cpu'],
                self.allocated_resources['memory'] + requirements.memory_mb <= self.available_resources['memory'],
                self.allocated_resources['gpu'] + requirements.gpu_memory_mb <= self.available_resources['gpu'],
                self.allocated_resources['disk_io'] + requirements.disk_io_mb_per_sec <= self.available_resources['disk_io'],
                self.allocated_resources['network'] + requirements.network_mb_per_sec <= self.available_resources['network']
            ]
            return all(checks)
    
    def allocate_resources(self, task_id: str, requirements: ResourceRequirement) -> bool:
        """Allocate resources for task"""
        if not self.can_allocate(requirements):
            return False
        
        with self.resource_lock:
            self.allocated_resources['cpu'] += requirements.cpu_cores
            self.allocated_resources['memory'] += requirements.memory_mb
            self.allocated_resources['gpu'] += requirements.gpu_memory_mb
            self.allocated_resources['disk_io'] += requirements.disk_io_mb_per_sec
            self.allocated_resources['network'] += requirements.network_mb_per_sec
            
            logger.info(f"Resources allocated for task {task_id}")
            return True
    
    def release_resources(self, task_id: str, requirements: ResourceRequirement):
        """Release task resources"""
        with self.resource_lock:
            self.allocated_resources['cpu'] -= requirements.cpu_cores
            self.allocated_resources['memory'] -= requirements.memory_mb
            self.allocated_resources['gpu'] -= requirements.gpu_memory_mb
            self.allocated_resources['disk_io'] -= requirements.disk_io_mb_per_sec
            self.allocated_resources['network'] -= requirements.network_mb_per_sec
            
            logger.info(f"Resources released for task {task_id}")
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization percentages"""
        with self.resource_lock:
            return {
                "cpu_utilization": (self.allocated_resources['cpu'] / self.available_resources['cpu']) * 100,
                "memory_utilization": (self.allocated_resources['memory'] / self.available_resources['memory']) * 100,
                "gpu_utilization": (self.allocated_resources['gpu'] / self.available_resources['gpu']) * 100 if self.available_resources['gpu'] > 0 else 0,
                "disk_io_utilization": (self.allocated_resources['disk_io'] / self.available_resources['disk_io']) * 100,
                "network_utilization": (self.allocated_resources['network'] / self.available_resources['network']) * 100
            }

class TaskScheduler:
    """Main task scheduler"""
    
    def __init__(self, max_concurrent_tasks: int = 10, enable_resource_management: bool = True):
        # Core data structures
        self.pending_queue = []  # Priority heap
        self.scheduled_tasks = {}  # task_id -> ScheduledTask
        self.running_tasks = {}    # task_id -> (ScheduledTask, Future)
        self.completed_tasks = {}  # task_id -> TaskResult
        self.failed_tasks = {}     # task_id -> TaskResult
        
        # Configuration
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_resource_management = enable_resource_management
        
        # Resource management
        self.resource_monitor = ResourceMonitor() if enable_resource_management else None
        
        # Control variables
        self.is_running = False
        self.scheduler_thread = None
        
        # Thread safety
        self.queue_lock = threading.Lock()
        self.running_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_scheduled': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_cancelled': 0,
            'average_execution_time': 0.0
        }
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Task Scheduler started successfully")
    
    def stop(self, wait_for_completion: bool = True, timeout: int = 30):
        """Stop the scheduler"""
        if not self.is_running:
            return
        
        logger.info("Stopping Task Scheduler...")
        self.is_running = False
        
        if wait_for_completion:
            # Wait for running tasks to complete
            start_time = time.time()
            while self.running_tasks and (time.time() - start_time) < timeout:
                time.sleep(1)
                logger.info(f"Waiting for {len(self.running_tasks)} tasks to complete...")
            
            # Cancel remaining tasks
            if self.running_tasks:
                logger.warning(f"Force cancelling {len(self.running_tasks)} remaining tasks")
                self._cancel_running_tasks()
        
        logger.info("Task Scheduler stopped")
    
    def schedule_task(self, 
                     task_payload: Any,
                     task_type: str,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     resource_requirements: Optional[ResourceRequirement] = None,
                     scheduled_at: Optional[datetime] = None,
                     dependencies: List[str] = None,
                     max_retries: int = 3,
                     timeout_seconds: int = 3600,
                     tags: Dict[str, str] = None) -> str:
        """Schedule a new task"""
        
        task_id = str(uuid.uuid4())
        
        # Create task metadata
        metadata = TaskMetadata(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            resource_requirements=resource_requirements or ResourceRequirement(),
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            dependencies=dependencies or [],
            tags=tags or {},
            scheduled_at=scheduled_at,
            status=TaskStatus.PENDING
        )
        
        # Create scheduled task
        scheduled_task = ScheduledTask(
            metadata=metadata,
            payload=task_payload
        )
        
        # Add to scheduling queue
        with self.queue_lock:
            heapq.heappush(self.pending_queue, scheduled_task)
            self.scheduled_tasks[task_id] = scheduled_task
        
        self.stats['total_scheduled'] += 1
        
        logger.info(f"Task {task_id} scheduled with priority {priority.name}")
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        # Check if in pending queue
        with self.queue_lock:
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.metadata.status = TaskStatus.CANCELLED
                # Remove from queue (rebuild heap)
                self.pending_queue = [t for t in self.pending_queue if t.metadata.task_id != task_id]
                heapq.heapify(self.pending_queue)
                del self.scheduled_tasks[task_id]
                self.stats['total_cancelled'] += 1
                logger.info(f"Task {task_id} cancelled from pending queue")
                return True
        
        # Check if running
        with self.running_lock:
            if task_id in self.running_tasks:
                task, future = self.running_tasks[task_id]
                future.cancel()
                task.metadata.status = TaskStatus.CANCELLED
                self.stats['total_cancelled'] += 1
                logger.info(f"Task {task_id} cancelled from running tasks")
                return True
        
        logger.warning(f"Task {task_id} not found for cancellation")
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task status"""
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        elif task_id in self.failed_tasks:
            return self.failed_tasks[task_id]
        
        # Check running tasks
        with self.running_lock:
            if task_id in self.running_tasks:
                task, _ = self.running_tasks[task_id]
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    result=None
                )
        
        # Check pending tasks
        with self.queue_lock:
            if task_id in self.scheduled_tasks:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.PENDING,
                    result=None
                )
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status"""
        with self.queue_lock, self.running_lock:
            return {
                'pending_tasks': len(self.pending_queue),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'resource_utilization': self.resource_monitor.get_resource_utilization() if self.resource_monitor else {},
                'statistics': self.stats.copy()
            }
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Scheduler loop started")
        
        while self.is_running:
            try:
                self._process_pending_tasks()
                self._monitor_running_tasks()
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _process_pending_tasks(self):
        """Process pending tasks"""
        with self.queue_lock:
            current_time = datetime.now()
            ready_tasks = []
            temp_queue = []
            
            while self.pending_queue:
                task = heapq.heappop(self.pending_queue)
                
                # Check execution time
                scheduled_time = task.metadata.scheduled_at or task.metadata.created_at
                if scheduled_time > current_time:
                    temp_queue.append(task)
                    continue
                
                # Check concurrency limit
                with self.running_lock:
                    if len(self.running_tasks) >= self.max_concurrent_tasks:
                        temp_queue.append(task)
                        continue
                
                # Check resources
                if self.resource_monitor and not self.resource_monitor.can_allocate(task.metadata.resource_requirements):
                    temp_queue.append(task)
                    continue
                
                ready_tasks.append(task)
            
            # Rebuild queue
            for task in temp_queue:
                heapq.heappush(self.pending_queue, task)
        
        # Execute ready tasks
        for task in ready_tasks:
            self._start_task_execution(task)
    
    def _start_task_execution(self, task: ScheduledTask):
        """Start task execution"""
        task_id = task.metadata.task_id
        
        # Allocate resources
        if self.resource_monitor:
            if not self.resource_monitor.allocate_resources(task_id, task.metadata.resource_requirements):
                logger.warning(f"Failed to allocate resources for task {task_id}")
                return
        
        # Update status
        task.metadata.status = TaskStatus.RUNNING
        task.metadata.started_at = datetime.now()
        
        # Create async task
        loop = asyncio.new_event_loop()
        future = loop.run_in_executor(None, self._execute_task_sync, task)
        
        # Add to running queue
        with self.running_lock:
            self.running_tasks[task_id] = (task, future)
        
        # Remove from scheduled queue
        with self.queue_lock:
            if task_id in self.scheduled_tasks:
                del self.scheduled_tasks[task_id]
        
        logger.info(f"Task {task_id} started execution")
    
    def _execute_task_sync(self, task: ScheduledTask):
        """Execute task synchronously"""
        task_id = task.metadata.task_id
        start_time = time.time()
        
        try:
            # Simulate task execution
            # In real implementation, this would call the appropriate engine
            if task.metadata.task_type == "inference":
                # Simulate inference
                time.sleep(2)
                result = {"status": "completed", "type": "inference"}
            elif task.metadata.task_type == "training":
                # Simulate training
                time.sleep(5)
                result = {"status": "completed", "type": "training"}
            else:
                time.sleep(1)
                result = {"status": "completed", "type": "unknown"}
            
            # Task completed
            task.metadata.status = TaskStatus.COMPLETED
            task.metadata.completed_at = datetime.now()
            
            execution_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_completed'] += 1
            self.stats['average_execution_time'] = (
                (self.stats['average_execution_time'] * (self.stats['total_completed'] - 1) + execution_time) /
                self.stats['total_completed']
            )
            
            # Create result
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                completed_at=task.metadata.completed_at
            )
            
            self.completed_tasks[task_id] = task_result
            
            logger.info(f"Task {task_id} completed successfully in {execution_time:.2f} seconds")
            
        except Exception as e:
            # Task failed
            task.metadata.status = TaskStatus.FAILED
            task.metadata.completed_at = datetime.now()
            
            self.stats['total_failed'] += 1
            
            # Create failure result
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - start_time,
                completed_at=task.metadata.completed_at
            )
            
            self.failed_tasks[task_id] = task_result
            
            logger.error(f"Task {task_id} failed: {e}")
        
        finally:
            # Cleanup
            with self.running_lock:
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
            
            # Release resources
            if self.resource_monitor:
                self.resource_monitor.release_resources(task_id, task.metadata.resource_requirements)
    
    def _monitor_running_tasks(self):
        """Monitor running tasks"""
        current_time = datetime.now()
        timeout_tasks = []
        
        with self.running_lock:
            for task_id, (task, future) in self.running_tasks.items():
                # Check for timeouts
                if task.metadata.started_at:
                    running_time = (current_time - task.metadata.started_at).total_seconds()
                    if running_time > task.metadata.timeout_seconds:
                        timeout_tasks.append(task_id)
        
        # Handle timeouts
        for task_id in timeout_tasks:
            logger.warning(f"Task {task_id} timed out")
            self.cancel_task(task_id)
    
    def _cancel_running_tasks(self):
        """Cancel all running tasks"""
        with self.running_lock:
            for task_id, (task, future) in self.running_tasks.items():
                future.cancel()
                task.metadata.status = TaskStatus.CANCELLED
            self.running_tasks.clear()
