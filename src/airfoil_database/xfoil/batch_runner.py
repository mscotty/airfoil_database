# batch_runner.py
import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import traceback

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    reynolds_list: List[float]
    mach_list: List[float]
    alpha_list: List[float]
    ncrit_list: List[float]
    max_workers: Optional[int] = None
    skip_existing: bool = True
    save_config: bool = True
    config_name: Optional[str] = None
    timeout_per_airfoil: Optional[float] = None  # seconds
    retry_failed: bool = True
    retry_count: int = 2

class BatchAirfoilRunner:
    """
    Batch runner for processing multiple airfoils with the same conditions.
    """
    
    def __init__(self, aero_analyzer, xfoil_runner, config_dir="batch_configs"):
        """
        Initialize the batch runner.
        
        Args:
            aero_analyzer: AeroAnalyzer instance
            xfoil_runner: XFoilRunner instance
            config_dir: Directory to save/load batch configurations
        """
        self.analyzer = aero_analyzer
        self.xfoil_runner = xfoil_runner
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Statistics tracking
        self.stats = {
            'total_airfoils': 0,
            'completed_airfoils': 0,
            'failed_airfoils': 0,
            'skipped_airfoils': 0,
            'total_alphas': 0,
            'successful_alphas': 0,
            'failed_alphas': 0,
            'start_time': None,
            'end_time': None,
            'failed_airfoil_names': [],
            'processing_times': {},
            'retry_attempts': {}
        }

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("batch_logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"batch_run_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Batch runner initialized. Log file: {log_file}")

    def get_all_airfoil_names(self) -> List[str]:
        """Get all available airfoil names from the database."""
        try:
            # Try different methods to get airfoil names
            if hasattr(self.analyzer, 'db') and hasattr(self.analyzer.db, 'get_all_airfoil_names'):
                return self.analyzer.db.get_all_airfoil_names()
            elif hasattr(self.analyzer, 'get_all_airfoil_names'):
                return self.analyzer.get_all_airfoil_names()
            else:
                return self._get_airfoil_names_fallback()
        except Exception as e:
            self.logger.error(f"Error getting airfoil names: {e}")
            return self._get_airfoil_names_fallback()

    def _get_airfoil_names_fallback(self) -> List[str]:
        """Fallback method to get airfoil names."""
        try:
            from sqlmodel import Session, select
            from airfoil_database.core.models import Airfoil
            
            with Session(self.analyzer.engine) as session:
                statement = select(Airfoil.name).distinct()
                results = session.exec(statement).all()
                return list(results)
        except Exception as e:
            self.logger.error(f"Error in fallback method: {e}")
            try:
                # Another fallback - try to get from analyzer directly
                if hasattr(self.analyzer, 'engine'):
                    with self.analyzer.engine.connect() as conn:
                        result = conn.execute("SELECT DISTINCT name FROM airfoils")
                        return [row[0] for row in result]
            except:
                pass
            return []

    def filter_airfoils(self, airfoil_names: List[str], 
                       include_patterns: List[str] = None,
                       exclude_patterns: List[str] = None) -> List[str]:
        """
        Filter airfoil names based on patterns.
        
        Args:
            airfoil_names: List of airfoil names to filter
            include_patterns: List of patterns to include (e.g., ['NACA', 'Clark'])
            exclude_patterns: List of patterns to exclude (e.g., ['test', 'experimental'])
            
        Returns:
            Filtered list of airfoil names
        """
        filtered = airfoil_names.copy()
        
        # Apply include filters
        if include_patterns:
            filtered = [name for name in filtered 
                       if any(pattern.lower() in name.lower() for pattern in include_patterns)]
        
        # Apply exclude filters
        if exclude_patterns:
            filtered = [name for name in filtered 
                       if not any(pattern.lower() in name.lower() for pattern in exclude_patterns)]
        
        self.logger.info(f"Filtered {len(airfoil_names)} airfoils to {len(filtered)} airfoils")
        return filtered

    def check_existing_results(self, airfoil_name: str, config: BatchConfig) -> Dict[str, Any]:
        """
        Check what results already exist for an airfoil.
        
        Returns:
            dict: Information about existing results
        """
        existing_count = 0
        missing_conditions = []
        
        try:
            for reynolds in config.reynolds_list:
                for mach in config.mach_list:
                    for ncrit in config.ncrit_list:
                        try:
                            existing_results = self.analyzer.get_aero_coeffs(
                                airfoil_name, Re=reynolds, Mach=mach
                            )
                            
                            # Check which alphas are missing
                            existing_alphas = {result.alpha for result in existing_results 
                                             if result.ncrit == ncrit}
                            missing_alphas = [alpha for alpha in config.alpha_list 
                                            if alpha not in existing_alphas]
                            
                            if missing_alphas:
                                missing_conditions.append({
                                    'reynolds': reynolds,
                                    'mach': mach,
                                    'ncrit': ncrit,
                                    'missing_alphas': missing_alphas
                                })
                            
                            existing_count += len(existing_alphas)
                        except Exception as e:
                            self.logger.warning(f"Error checking condition Re={reynolds}, M={mach}, Ncrit={ncrit} for {airfoil_name}: {e}")
                            # Assume all alphas are missing for this condition
                            missing_conditions.append({
                                'reynolds': reynolds,
                                'mach': mach,
                                'ncrit': ncrit,
                                'missing_alphas': config.alpha_list
                            })
        
        except Exception as e:
            self.logger.error(f"Error checking existing results for {airfoil_name}: {e}")
            # Assume all conditions are missing
            for reynolds in config.reynolds_list:
                for mach in config.mach_list:
                    for ncrit in config.ncrit_list:
                        missing_conditions.append({
                            'reynolds': reynolds,
                            'mach': mach,
                            'ncrit': ncrit,
                            'missing_alphas': config.alpha_list
                        })
        
        return {
            'existing_count': existing_count,
            'missing_conditions': missing_conditions
        }

    def save_config(self, config: BatchConfig, name: str = None) -> str:
        """Save batch configuration to file."""
        if not name:
            name = f"batch_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config_path = self.config_dir / f"{name}.json"
        
        try:
            config_dict = asdict(config)
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Configuration saved to: {config_path}")
            return str(config_path)
        
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return ""

    def load_config(self, name: str) -> Optional[BatchConfig]:
        """Load batch configuration from file."""
        config_path = self.config_dir / f"{name}.json"
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            return BatchConfig(**config_dict)
        
        except Exception as e:
            self.logger.error(f"Error loading configuration {name}: {e}")
            return None

    def list_configs(self) -> List[str]:
        """List all saved configurations."""
        try:
            return [f.stem for f in self.config_dir.glob("*.json")]
        except Exception as e:
            self.logger.error(f"Error listing configurations: {e}")
            return []

    def save_progress(self, stats: Dict[str, Any], filename: str = None):
        """Save current progress to file."""
        if not filename:
            filename = f"batch_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        progress_path = self.config_dir / filename
        
        try:
            with open(progress_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            self.logger.info(f"Progress saved to: {progress_path}")
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")

    def process_single_airfoil(self, airfoil_name: str, config: BatchConfig) -> Dict[str, Any]:
        """
        Process a single airfoil with retry logic.
        
        Returns:
            dict: Processing results
        """
        result = {
            'airfoil_name': airfoil_name,
            'success': False,
            'error': None,
            'duration': 0,
            'alphas_processed': 0,
            'retry_count': 0
        }
        
        for attempt in range(config.retry_count + 1):
            try:
                start_time = time.time()
                
                # Run XFoil analysis
                self.xfoil_runner.run_analysis_parallel(
                    airfoil_name=airfoil_name,
                    reynolds_list=config.reynolds_list,
                    mach_list=config.mach_list,
                    alpha_list=config.alpha_list,
                    ncrit_list=config.ncrit_list,
                    max_workers=config.max_workers
                )
                
                end_time = time.time()
                result['duration'] = end_time - start_time
                result['success'] = True
                result['alphas_processed'] = len(config.alpha_list) * len(config.reynolds_list) * len(config.mach_list) * len(config.ncrit_list)
                
                self.logger.info(f"✓ Completed {airfoil_name} in {result['duration']:.1f}s")
                break
                
            except Exception as e:
                result['error'] = str(e)
                result['retry_count'] = attempt + 1
                
                if attempt < config.retry_count:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {airfoil_name}: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"✗ Failed {airfoil_name} after {config.retry_count + 1} attempts: {e}")
        
        return result

    def run_batch_analysis(self, 
                          airfoil_names: List[str] = None, 
                          config: BatchConfig = None,
                          config_name: str = None,
                          include_patterns: List[str] = None,
                          exclude_patterns: List[str] = None,
                          max_airfoils: int = None) -> Dict[str, Any]:
        """
        Run batch analysis on multiple airfoils.
        
        Args:
            airfoil_names: List of airfoil names to process (None for all)
            config: BatchConfig object with analysis parameters
            config_name: Name of saved configuration to load
            include_patterns: List of patterns to include in airfoil names
            exclude_patterns: List of patterns to exclude from airfoil names
            max_airfoils: Maximum number of airfoils to process (for testing)
            
        Returns:
            dict: Summary of batch processing results
        """
        # Load configuration if name provided
        if config_name:
            config = self.load_config(config_name)
            if not config:
                raise ValueError(f"Could not load configuration: {config_name}")
        
        if not config:
            raise ValueError("Either config or config_name must be provided")
        
        # Get airfoil names
        if airfoil_names is None:
            airfoil_names = self.get_all_airfoil_names()
        
        # Filter airfoils
        if include_patterns or exclude_patterns:
            airfoil_names = self.filter_airfoils(airfoil_names, include_patterns, exclude_patterns)
        
        # Limit number of airfoils for testing
        if max_airfoils and len(airfoil_names) > max_airfoils:
            airfoil_names = airfoil_names[:max_airfoils]
            self.logger.info(f"Limited to first {max_airfoils} airfoils for testing")
        
        if not airfoil_names:
            self.logger.error("No airfoils found to process")
            return self.stats
        
        # Save configuration if requested
        if config.save_config:
            save_name = config.config_name or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.save_config(config, save_name)
        
        # Initialize statistics
        self.stats['total_airfoils'] = len(airfoil_names)
        self.stats['start_time'] = time.time()
        
        # Calculate total expected alphas
        total_conditions = (len(config.reynolds_list) * len(config.mach_list) * 
                          len(config.ncrit_list))
        self.stats['total_alphas'] = len(airfoil_names) * total_conditions * len(config.alpha_list)
        
        self.logger.info(f"Starting batch analysis:")
        self.logger.info(f"  Airfoils: {len(airfoil_names)}")
        self.logger.info(f"  Reynolds: {config.reynolds_list}")
        self.logger.info(f"  Mach: {config.mach_list}")
        self.logger.info(f"  Alpha range: {min(config.alpha_list)} to {max(config.alpha_list)} ({len(config.alpha_list)} points)")
        self.logger.info(f"  Ncrit: {config.ncrit_list}")
        self.logger.info(f"  Conditions: {total_conditions}")
        self.logger.info(f"  Total alpha points: {self.stats['total_alphas']}")
        self.logger.info(f"  Max workers: {config.max_workers}")
        self.logger.info(f"  Skip existing: {config.skip_existing}")
        
        # Process each airfoil
        for i, airfoil_name in enumerate(airfoil_names):
            try:
                self.logger.info(f"\n--- Processing airfoil {i+1}/{len(airfoil_names)}: {airfoil_name} ---")
                
                # Check existing results if skip_existing is enabled
                if config.skip_existing:
                    existing_info = self.check_existing_results(airfoil_name, config)
                    if existing_info['existing_count'] > 0:
                        self.logger.info(f"Found {existing_info['existing_count']} existing results for {airfoil_name}")
                    
                    if not existing_info['missing_conditions']:
                        self.logger.info(f"All results exist for {airfoil_name}. Skipping.")
                        self.stats['skipped_airfoils'] += 1
                        continue
                
                # Process the airfoil
                result = self.process_single_airfoil(airfoil_name, config)
                
                # Update statistics
                self.stats['processing_times'][airfoil_name] = result['duration']
                self.stats['retry_attempts'][airfoil_name] = result['retry_count']
                
                if result['success']:
                    self.stats['completed_airfoils'] += 1
                    self.stats['successful_alphas'] += result['alphas_processed']
                else:
                    self.stats['failed_airfoils'] += 1
                    self.stats['failed_airfoil_names'].append(airfoil_name)
                
                # Progress update
                progress = (i + 1) / len(airfoil_names) * 100
                elapsed = time.time() - self.stats['start_time']
                eta = (elapsed / (i + 1)) * (len(airfoil_names) - i - 1)
                
                self.logger.info(f"Progress: {i+1}/{len(airfoil_names)} ({progress:.1f}%) | "
                               f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
                
                # Save progress periodically
                if (i + 1) % 10 == 0:
                    self.save_progress(self.stats)
                
            except Exception as e:
                self.logger.error(f"Unexpected error processing {airfoil_name}: {e}")
                self.logger.error(traceback.format_exc())
                self.stats['failed_airfoils'] += 1
                self.stats['failed_airfoil_names'].append(airfoil_name)
        
        # Final statistics
        self.stats['end_time'] = time.time()
        total_duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BATCH ANALYSIS COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        self.logger.info(f"Completed airfoils: {self.stats['completed_airfoils']}/{self.stats['total_airfoils']}")
        self.logger.info(f"Skipped airfoils: {self.stats['skipped_airfoils']}")
        self.logger.info(f"Failed airfoils: {self.stats['failed_airfoils']}")
        self.logger.info(f"Success rate: {self.stats['completed_airfoils']/self.stats['total_airfoils']*100:.1f}%")
        
        if self.stats['failed_airfoil_names']:
            self.logger.info(f"Failed airfoil names: {self.stats['failed_airfoil_names'][:10]}...")
        
        # Save final results
        self.save_progress(self.stats, f"batch_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        return self.stats

    def run_all_airfoils(self, 
                        reynolds: float = 1e6,
                        mach: float = 0.0,
                        ncrit: float = 9.0,
                        alpha_range: tuple = (-10, 20, 1),
                        max_workers: int = None,
                        include_patterns: List[str] = None,
                        exclude_patterns: List[str] = None,
                        max_airfoils: int = None) -> Dict[str, Any]:
        """
        Run XFoil analysis on ALL airfoils in the database.
        
        Args:
            reynolds: Reynolds number (default: 1e6)
            mach: Mach number (default: 0.0)
            ncrit: Ncrit value (default: 9.0)
            alpha_range: (start, stop, step) for alpha values (default: -10 to 20 by 1)
            max_workers: Number of parallel workers
            include_patterns: Only process airfoils containing these patterns
            exclude_patterns: Skip airfoils containing these patterns
            max_airfoils: Limit number of airfoils (for testing)
            
        Returns:
            dict: Summary of batch processing results
        """
        # Generate alpha list
        alpha_start, alpha_stop, alpha_step = alpha_range
        alpha_list = []
        current = alpha_start
        while current <= alpha_stop:
            alpha_list.append(float(current))
            current += alpha_step
        
        # Create configuration
        config = BatchConfig(
            reynolds_list=[reynolds],
            mach_list=[mach],
            alpha_list=alpha_list,
            ncrit_list=[ncrit],
            max_workers=max_workers,
            skip_existing=True,
            save_config=True,
            config_name=f"all_airfoils_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            retry_failed=True,
            retry_count=2
        )
        
        self.logger.info(f"Running ALL airfoils with: Re={reynolds}, M={mach}, Ncrit={ncrit}")
        self.logger.info(f"Alpha range: {alpha_start} to {alpha_stop} by {alpha_step} ({len(alpha_list)} points)")
        if include_patterns:
            self.logger.info(f"Including patterns: {include_patterns}")
        if exclude_patterns:
            self.logger.info(f"Excluding patterns: {exclude_patterns}")
        
        return self.run_batch_analysis(
            airfoil_names=None,  # Process all airfoils
            config=config,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            max_airfoils=max_airfoils
        )

    def run_quick_batch(self, 
                       reynolds: float = 1e6,
                       mach: float = 0.0,
                       ncrit: float = 9.0,
                       alpha_range: tuple = (-10, 20, 1),
                       airfoil_names: List[str] = None,
                       max_workers: int = None) -> Dict[str, Any]:
        """
        Quick batch runner with common default parameters.
        
        Args:
            reynolds: Reynolds number (default: 1e6)
            mach: Mach number (default: 0.0)
            ncrit: Ncrit value (default: 9.0)
            alpha_range: (start, stop, step) for alpha values (default: -10 to 20 by 1)
            airfoil_names: List of airfoil names (None for all)
            max_workers: Number of parallel workers
            
        Returns:
            dict: Summary of batch processing results
        """
        # Generate alpha list
        alpha_start, alpha_stop, alpha_step = alpha_range
        alpha_list = []
        current = alpha_start
        while current <= alpha_stop:
            alpha_list.append(float(current))
            current += alpha_step
        
        # Create configuration
        config = BatchConfig(
            reynolds_list=[reynolds],
            mach_list=[mach],
            alpha_list=alpha_list,
            ncrit_list=[ncrit],
            max_workers=max_workers,
            skip_existing=True,
            save_config=True,
            config_name=f"quick_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        self.logger.info(f"Quick batch with: Re={reynolds}, M={mach}, Ncrit={ncrit}")
        self.logger.info(f"Alpha range: {alpha_start} to {alpha_stop} by {alpha_step} ({len(alpha_list)} points)")
        
        return self.run_batch_analysis(airfoil_names=airfoil_names, config=config)

    def resume_failed_batch(self, stats: Dict[str, Any] = None, progress_file: str = None) -> Dict[str, Any]:
        """
        Resume processing of failed airfoils from a previous batch.
        
        Args:
            stats: Statistics dictionary from previous batch run
            progress_file: Path to progress file to resume from
            
        Returns:
            dict: Summary of resumed processing
        """
        if progress_file:
            try:
                with open(progress_file, 'r') as f:
                    stats = json.load(f)
                self.logger.info(f"Loaded progress from: {progress_file}")
            except Exception as e:
                self.logger.error(f"Error loading progress file: {e}")
                return {}
        
        if not stats:
            self.logger.error("No stats provided for resume")
            return {}
        
        failed_names = stats.get('failed_airfoil_names', [])
        if not failed_names:
            self.logger.info("No failed airfoils to resume")
            return stats
        
        self.logger.info(f"Resuming processing of {len(failed_names)} failed airfoils")
        
        # Use the last saved configuration
        configs = self.list_configs()
        if not configs:
            self.logger.error("No saved configurations found for resume")
            return stats
        
        latest_config = sorted(configs)[-1]
        self.logger.info(f"Using configuration: {latest_config}")
        
        return self.run_batch_analysis(
            airfoil_names=failed_names,
            config_name=latest_config
        )


# Example usage and utility functions
def create_standard_configs():
    """Create some standard batch configurations."""
    configs = {
        'low_reynolds': BatchConfig(
            reynolds_list=[1e5, 5e5],
            mach_list=[0.0],
            alpha_list=list(range(-10, 21)),
            ncrit_list=[9.0],
            max_workers=4,
            skip_existing=True,
            save_config=True
        ),
        'high_reynolds': BatchConfig(
            reynolds_list=[1e6, 3e6, 6e6],
            mach_list=[0.0],
            alpha_list=list(range(-10, 21)),
            ncrit_list=[9.0],
            max_workers=6,
            skip_existing=True,
            save_config=True
        ),
        'transonic': BatchConfig(
            reynolds_list=[1e6],
            mach_list=[0.3, 0.5, 0.7],
            alpha_list=list(range(-5, 11)),
            ncrit_list=[9.0],
            max_workers=4,
            skip_existing=True,
            save_config=True
        ),
        'detailed_analysis': BatchConfig(
            reynolds_list=[1e6],
            mach_list=[0.0],
            alpha_list=[i * 0.5 for i in range(-20, 41)],  # -10 to 20 by 0.5
            ncrit_list=[9.0],
            max_workers=8,
            skip_existing=True,
            save_config=True
        ),
        'all_airfoils_standard': BatchConfig(
            reynolds_list=[1e6],
            mach_list=[0.0],
            alpha_list=list(range(-10, 21)),
            ncrit_list=[9.0],
            max_workers=12,
            skip_existing=True,
            save_config=True,
            retry_failed=True,
            retry_count=2
        )
    }
    return configs


if __name__ == "__main__":
    # Example usage
    print("Enhanced Batch Airfoil Runner - Example Usage")
    print("="*60)
    
    # This would be your actual initialization
    # aero_analyzer = AeroAnalyzer(database)
    # xfoil_runner = XFoilRunner(aero_analyzer)
    # batch_runner = BatchAirfoilRunner(aero_analyzer, xfoil_runner)
    
    print("\n1. Run ALL airfoils in database:")
    print("stats = batch_runner.run_all_airfoils()")
    
    print("\n2. Run all NACA airfoils:")
    print("stats = batch_runner.run_all_airfoils(include_patterns=['NACA'])")
    
    print("\n3. Run all airfoils except test ones:")
    print("stats = batch_runner.run_all_airfoils(exclude_patterns=['test', 'experimental'])")
    
    print("\n4. Test run with limited airfoils:")
    print("stats = batch_runner.run_all_airfoils(max_airfoils=10)")
    
    print("\n5. Multiple Reynolds numbers:")
    print("config = BatchConfig(")
    print("    reynolds_list=[1e5, 5e5, 1e6],")
    print("    mach_list=[0.0],")
    print("    alpha_list=list(range(-10, 21)),")
    print("    ncrit_list=[9.0],")
    print("    max_workers=8")
    print(")")
    print("stats = batch_runner.run_batch_analysis(config=config)")
    
    print("\n6. Resume failed batch:")
    print("stats = batch_runner.resume_failed_batch(progress_file='batch_configs/batch_progress_20240101_120000.json')")