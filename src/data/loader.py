"""Data loading utilities for multi-class classification."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from loguru import logger


class DataLoader:
    """Handles data loading from CSV and TXT files."""
    
    def load_from_config(self, config: Dict[str, Any], project_root: Path = Path('.')) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load data from configuration.
        
        Config formats:
        
        1. Multiple class files (each file = one class):
           {0: 'file1.txt', 1: 'file2.txt', skip_rows: 1}
           {tissue: 'tissue.csv', cross: 'cross.csv', skip_rows: 1}
        
        2. Single file with label column:
           {file: 'data.csv', label_column: 'class', skip_rows: 1}
           {data_file: 'data.csv', label_column: 2, skip_rows: 1}
        
        3. Inference files (with or without labels):
           {files: ['test.csv'], label_column: 'class', skip_rows: 1}
           {files: 'test.csv', skip_rows: 1}  # No labels
        
        Args:
            config: train_files or inference_files dictionary
            project_root: Root directory for relative paths
            
        Returns:
            X: Feature matrix
            y: Label vector (None if no labels)
        """

        skip_rows = config.get('skip_rows', 0)
        label_column = config.get('label_column')
        files = config.get('files')
        
        X_list = []
        y_list = []
        
        if files:
            # Inference mode with file list
            file_list = files if isinstance(files, list) else [files]
            
            for file_path in file_list:
                X_file, y_file = self._load_file(project_root / file_path, skip_rows, label_column)
                X_list.append(X_file)
                if y_file is not None:
                    y_list.extend(y_file)
        else:
            # Training mode
            if label_column is not None:
                # Single file with labels in column
                file_path = config.get('file') or config.get('data_file')
                if not file_path:
                    raise ValueError("When using label_column, provide 'file' or 'data_file' key")
                
                X, y = self._load_file(project_root / file_path, skip_rows, label_column)
                X_list.append(X)
                y_list = y.tolist()
            # Multiple files, each representing a class
            else:
                file_entries = {k: v for k, v in config.items() if k not in ['skip_rows', 'label_column', 'file', 'data_file']}
                if not file_entries:
                    raise ValueError("No file entries found in config. Check your configuration structure.")
                
                for class_label, file_path in sorted(file_entries.items()):
                    data = self._read_data(project_root / file_path, skip_rows)
                    X_list.append(data)
                    y_list.extend([class_label] * len(data))
                    logger.info(f"Class {class_label}: {len(data)} samples from {file_path}")
        
        X = np.vstack(X_list)
        y = np.array(y_list) if y_list else None
        
        logger.info(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _load_file(self, file_path: Path, skip_rows: int, label_column: Optional[Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load file and optionally extract labels."""
        if label_column is None:
            return self._read_data(file_path, skip_rows), None
        
        if file_path.suffix == '.csv' and isinstance(label_column, str):
            df = pd.read_csv(file_path, skiprows=skip_rows)
            X = df.drop(columns=[label_column]).values
            y = df[label_column].values
        else:
            data = self._read_data(file_path, skip_rows)
            X = np.delete(data, label_column, axis=1)
            y = data[:, label_column]
        
        return X, y
    
    def _read_data(self, file_path: Path, skip_rows: int = 0) -> np.ndarray:
        """Read data from file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path, skiprows=skip_rows).values
        
        # TXT - auto-detect delimiter
        with open(file_path, 'r') as f:
            for _ in range(skip_rows):
                f.readline()
            first_line = f.readline().strip()
        
        for delimiter in [',', '\t', ' ', '|', ';']:
            if delimiter in first_line:
                data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip_rows)
                break
        else:
            data = np.loadtxt(file_path, skiprows=skip_rows)
        
        return data.reshape(-1, 1) if data.ndim == 1 else data