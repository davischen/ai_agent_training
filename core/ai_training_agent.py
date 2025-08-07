# -*- coding: utf-8 -*-
"""
ATLAS - Adaptive Training and Learning Automation System
Main AI Training Agent that orchestrates all components
"""

import os
import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from threading import Thread
import core.task_scheduler as task_scheduler
from pii_trainer import create_pii_trainer, quick_train_pii
# 修正：改為同一層目錄的導入
from core.model_manager import ModelManager
from core.inference_engine import InferenceEngine, InferenceRequest
from core.training_engine import TrainingEngine, TrainingRequest, DataGenerator
from core.task_scheduler import TaskScheduler, TaskPriority, ResourceRequirement

logger = logging.getLogger(__name__)

class AITrainingAgent:
    """Main AI Training Agent that orchestrates all components"""
    
    def __init__(self, model_dir: str = "models", max_concurrent_tasks: int = 10):
        # Initialize core components
        self.model_manager = ModelManager(model_dir)
        self.data_generator = DataGenerator()
        self.inference_engine = InferenceEngine(self.model_manager)
        self.training_engine = TrainingEngine(self.model_manager, self.data_generator)
        self.scheduler = TaskScheduler(max_concurrent_tasks=max_concurrent_tasks)
        
        # Start scheduler
        self.scheduler.start()
        
        # Setup periodic tasks
        self._setup_periodic_tasks()
        
        logger.info("ATLAS AI Training Agent initialized successfully")
    
    def _setup_periodic_tasks(self):
        """Setup periodic maintenance and monitoring tasks"""
        # Schedule daily maintenance at 2:00 AM
        task_scheduler.every().day.at("02:00").do(self._daily_maintenance)
        
        # Schedule hourly health checks
        task_scheduler.every().hour.do(self._hourly_health_check)
        
        # Start schedule runner thread
        schedule_thread = Thread(target=self._run_schedule, daemon=True)
        schedule_thread.start()
    
    def _run_schedule(self):
        """Run periodic scheduled tasks"""
        while True:
            try:
                task_scheduler.run_pending()
                import time
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in schedule runner: {e}")
                import time
                time.sleep(300)  # Wait 5 minutes on error
    
    def _daily_maintenance(self):
        """Daily maintenance tasks"""
        logger.info("Running daily maintenance...")
        try:
            # Cleanup old completed tasks
            cutoff_time = datetime.now() - timedelta(days=7)
            old_tasks = [
                task_id for task_id, result in self.scheduler.completed_tasks.items()
                if hasattr(result, 'completed_at') and result.completed_at and result.completed_at < cutoff_time
            ]
            for task_id in old_tasks:
                del self.scheduler.completed_tasks[task_id]
            
            logger.info(f"Cleaned up {len(old_tasks)} old tasks")
            
            # Model performance check
            model_info = self.model_manager.list_models()
            logger.info(f"Current models: {list(model_info.keys())}")
            
        except Exception as e:
            logger.error(f"Error in daily maintenance: {e}")
    
    def _hourly_health_check(self):
        """Hourly system health check"""
        logger.info("Running health check...")
        try:
            # Check scheduler status
            queue_status = self.scheduler.get_queue_status()
            logger.info(f"Queue status: {queue_status}")
            
            # Check resource utilization
            if self.scheduler.resource_monitor:
                resources = self.scheduler.resource_monitor.get_resource_utilization()
                logger.info(f"Resource utilization: {resources}")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    async def submit_inference_request(self, request: InferenceRequest) -> str:
        """Submit an inference request"""
        task_id = self.scheduler.schedule_task(
            task_payload=request,
            task_type="inference",
            priority=TaskPriority.HIGH,
            resource_requirements=ResourceRequirement(
                cpu_cores=1.0,
                memory_mb=2048,
                estimated_duration_seconds=30
            )
        )
        logger.info(f"Submitted inference request: {task_id}")
        return task_id
    
    async def submit_training_request(self, request: TrainingRequest) -> str:
        """Submit a training request"""
        task_id = self.scheduler.schedule_task(
            task_payload=request,
            task_type="training",
            priority=TaskPriority.NORMAL,
            resource_requirements=ResourceRequirement(
                cpu_cores=4.0,
                memory_mb=8192,
                gpu_memory_mb=4096,
                estimated_duration_seconds=3600
            ),
            scheduled_at=request.scheduled_at
        )
        logger.info(f"Submitted training request: {task_id}")
        return task_id
    
    def get_task_status(self, task_id: str):
        """Get the status of a task"""
        return self.scheduler.get_task_status(task_id)
    
    def generate_inference_request(self, text_data: List[str], **kwargs) -> InferenceRequest:
        """Generate an inference request with default parameters"""
        return InferenceRequest(
            request_id=str(uuid.uuid4()),
            text_data=text_data,
            model_name=kwargs.get('model_name', 'all-MiniLM-L6-v2'),
            batch_size=kwargs.get('batch_size', 32),
            return_embeddings=kwargs.get('return_embeddings', False),
            threshold=kwargs.get('threshold', 0.5)
        )
    
    def generate_training_request(self, 
                                task_type: str = "training",
                                **kwargs) -> TrainingRequest:
        """Generate a training request with default parameters"""
        return TrainingRequest(
            request_id=str(uuid.uuid4()),
            task_type=task_type,
            priority=kwargs.get('priority', 1),
            data_path=kwargs.get('data_path'),
            model_config=kwargs.get('model_config', {'base_model': 'all-MiniLM-L6-v2'}),
            training_params=kwargs.get('training_params', {'epochs': 3, 'batch_size': 16}),
            scheduled_at=kwargs.get('scheduled_at')
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            queue_status = self.scheduler.get_queue_status()
            model_info = self.model_manager.list_models()
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'scheduler': queue_status,
                'models': model_info,
                'system': {
                    'max_concurrent_tasks': self.scheduler.max_concurrent_tasks,
                    'resource_management_enabled': self.scheduler.enable_resource_management
                }
            }
            
            if self.scheduler.resource_monitor:
                status['resources'] = self.scheduler.resource_monitor.get_resource_utilization()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def load_model(self, model_name: str):
        """Load a specific model"""
        return self.model_manager.load_model(model_name)
    
    def list_available_models(self) -> Dict[str, str]:
        """List all available models"""
        return self.model_manager.list_models()
    
    async def quick_inference(self, texts: List[str], 
                            model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
        """Perform quick inference without going through scheduler"""
        try:
            request = self.generate_inference_request(
                text_data=texts,
                model_name=model_name
            )
            
            result = await self.inference_engine.process_inference(request)
            return result
            
        except Exception as e:
            logger.error(f"Quick inference failed: {e}")
            raise
    
    def create_sample_training_data(self, num_samples: int = 100) -> str:
        """Create sample training data and save to file"""
        try:
            df = self.data_generator.generate_synthetic_emails(num_samples)
            
            # Save to data directory
            data_dir = os.path.join(os.path.dirname(self.model_manager.model_dir), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            file_path = os.path.join(data_dir, f'synthetic_training_data_{int(datetime.now().timestamp())}.csv')
            df.to_csv(file_path, index=False)
            
            logger.info(f"Created sample training data: {file_path} with {len(df)} samples")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to create sample training data: {e}")
            raise
        
    async def train_pii_model(self, **training_params):
        """訓練PII識別模型"""
        trainer = create_pii_trainer()
        return trainer.train(**training_params)
    
    async def train_model_from_data(self, data_path: str, 
                                  model_config: Dict[str, Any] = None,
                                  training_params: Dict[str, Any] = None) -> str:
        """Train a model from data file"""
        try:
            # Create training request
            request = self.generate_training_request(
                data_path=data_path,
                model_config=model_config or {'base_model': 'all-MiniLM-L6-v2'},
                training_params=training_params or {'epochs': 3, 'batch_size': 16}
            )
            
            # Submit training request
            task_id = await self.submit_training_request(request)
            
            logger.info(f"Started model training from {data_path}, task ID: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to start model training: {e}")
            raise
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            result = self.scheduler.cancel_task(task_id)
            if result:
                logger.info(f"Task {task_id} cancelled successfully")
            else:
                logger.warning(f"Failed to cancel task {task_id}")
            return result
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False
    
    def get_training_statistics(self, data_path: str) -> Dict[str, Any]:
        """Get statistics about training data"""
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            return self.training_engine.get_training_statistics(df)
        except Exception as e:
            logger.error(f"Error getting training statistics: {e}")
            return {'error': str(e)}
    
    def export_model(self, model_name: str, export_path: str) -> bool:
        """Export a trained model"""
        try:
            # Load the model
            model = self.model_manager.load_model(model_name)
            
            # Save to export path
            model.save(export_path)
            
            logger.info(f"Model {model_name} exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model {model_name}: {e}")
            return False
    
    def import_model(self, model_path: str, model_name: str) -> bool:
        """Import a model from file"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model from path
            model = SentenceTransformer(model_path)
            
            # Save to model directory
            save_path = os.path.join(self.model_manager.model_dir, model_name)
            model.save(save_path)
            
            logger.info(f"Model imported from {model_path} as {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import model from {model_path}: {e}")
            return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        try:
            system_status = self.get_system_status()
            
            # Get scheduler statistics
            scheduler_stats = getattr(self.scheduler, 'stats', {})
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'system_status': system_status,
                'scheduler_statistics': scheduler_stats,
                'model_count': len(self.model_manager.list_models()),
                'uptime_info': {
                    'scheduler_running': self.scheduler.is_running,
                    'total_tasks_processed': scheduler_stats.get('total_completed', 0) + scheduler_stats.get('total_failed', 0)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def backup_system_state(self, backup_path: str) -> bool:
        """Backup system state"""
        try:
            import json
            
            # Collect system state
            state = {
                'timestamp': datetime.now().isoformat(),
                'system_status': self.get_system_status(),
                'models': self.model_manager.list_models(),
                'scheduler_config': {
                    'max_concurrent_tasks': self.scheduler.max_concurrent_tasks,
                    'enable_resource_management': self.scheduler.enable_resource_management
                }
            }
            
            # Save to file
            with open(backup_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"System state backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup system state: {e}")
            return False
    
    def shutdown(self, wait_for_completion: bool = True, timeout: int = 30):
        """Gracefully shutdown the agent"""
        logger.info("Shutting down ATLAS AI Training Agent...")
        
        try:
            # Stop the scheduler
            self.scheduler.stop(wait_for_completion=wait_for_completion, timeout=timeout)
            
            # Clear scheduled tasks
            task_scheduler.clear()
            
            logger.info("ATLAS AI Training Agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# 便利函數和測試
def create_ai_training_agent(model_dir: str = "models", max_concurrent_tasks: int = 10) -> AITrainingAgent:
    """Create and return an AITrainingAgent instance"""
    return AITrainingAgent(model_dir=model_dir, max_concurrent_tasks=max_concurrent_tasks)

async def test_ai_training_agent():
    """Test function for AITrainingAgent"""
    print("🧪 Testing AITrainingAgent...")
    
    try:
        # Initialize agent
        agent = AITrainingAgent(model_dir="./test_models")
        print("✅ AITrainingAgent initialized")
        
        # Test quick inference
        sample_texts = [
            "Urgent: Verify your account now!",
            "Meeting scheduled for tomorrow"
        ]
        
        result = await agent.quick_inference(sample_texts)
        print(f"✅ Quick inference completed: {result['predictions']}")
        
        # Test system status
        status = agent.get_system_status()
        print(f"✅ System status: {len(status['models'])} models available")
        
        # Test sample data creation
        data_path = agent.create_sample_training_data(10)
        print(f"✅ Sample data created: {data_path}")
        
        # Test training statistics
        stats = agent.get_training_statistics(data_path)
        print(f"✅ Training statistics: {stats['total_samples']} samples")
        
        # Shutdown
        agent.shutdown()
        print("✅ Agent shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Usage example and helper functions
async def example_usage():
    """Example usage of the AI Training Agent"""
    
    # Initialize the agent
    agent = AITrainingAgent(model_dir="./models")
    
    try:
        # Example 1: Quick inference
        print("🔍 Testing quick inference...")
        sample_emails = [
            "Urgent: Your account will be suspended unless you verify immediately!",
            "Hi team, please find the quarterly report attached for your review.",
            "WINNER! You've won $1,000,000! Click here to claim your prize now!"
        ]
        
        result = await agent.quick_inference(sample_emails)
        print(f"Inference results: {result['predictions']}")
        
        # Example 2: Create and use training data
        print("\n📊 Creating sample training data...")
        data_path = agent.create_sample_training_data(num_samples=50)
        
        # Get training statistics
        stats = agent.get_training_statistics(data_path)
        print(f"Training data stats: {stats}")
        
        # Example 3: Schedule training
        print("\n🎓 Scheduling model training...")
        training_task_id = await agent.train_model_from_data(
            data_path=data_path,
            training_params={'epochs': 2, 'batch_size': 8}  # Small for demo
        )
        print(f"Training task ID: {training_task_id}")
        
        # Example 4: Monitor system
        print("\n📋 System status:")
        status = agent.get_system_status()
        print(f"Pending tasks: {status['scheduler']['pending_tasks']}")
        print(f"Running tasks: {status['scheduler']['running_tasks']}")
        print(f"Available models: {list(status['models'].keys())}")
        
        # Wait a bit for tasks to process
        await asyncio.sleep(10)
        
        # Check task status
        task_status = agent.get_task_status(training_task_id)
        print(f"\n📈 Training task status: {task_status}")
        
        # Example 5: Generate performance report
        print("\n📊 Performance report:")
        report = agent.get_performance_report()
        print(f"Total models: {report['model_count']}")
        print(f"Tasks processed: {report['uptime_info']['total_tasks_processed']}")
        
    finally:
        # Shutdown gracefully
        agent.shutdown()
        print("\n🏁 Agent shutdown complete")

if __name__ == "__main__":
    # Run example or test
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_ai_training_agent())
    else:
        asyncio.run(example_usage())
