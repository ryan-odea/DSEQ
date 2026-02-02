from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import polars as pl


class Offloader:
    """Manages disk-based storage for models and intermediate data"""

    def __init__(self, enabled: bool, dir: str, compression: int = 3):
        self.enabled = enabled
        self.dir = Path(dir)
        self.compression = compression
        # Create a cached loader bound to this instance
        self._init_cache()

    def _init_cache(self):
        """Initialize the LRU cache for model loading."""
        self._cached_load = lru_cache(maxsize=32)(self._load_from_disk)

    def __getstate__(self):
        """Prepare state for pickling - exclude the unpicklable cache."""
        state = self.__dict__.copy()
        # Remove the cache wrapper which can't be pickled
        del state['_cached_load']
        return state

    def __setstate__(self, state):
        """Restore state after unpickling - recreate the cache."""
        self.__dict__.update(state)
        # Recreate the cache after unpickling
        self._init_cache()

    def save_model(
        self, model: Any, name: str, boot_idx: Optional[int] = None
    ) -> Union[Any, str]:
        """Save a fitted model to disk and return a reference"""
        if not self.enabled:
            return model

        filename = (
            f"{name}_boot{boot_idx}.pkl" if boot_idx is not None else f"{name}.pkl"
        )
        filepath = self.dir / filename

        joblib.dump(model, filepath, compress=self.compression)

        return str(filepath)

    def _load_from_disk(self, filepath: str) -> Any:
        """Internal method to load a model from disk (cached)."""
        return joblib.load(filepath)

    def load_model(self, ref: Union[Any, str]) -> Any:
        """Load a model, using cache for repeated loads of the same file."""
        if not self.enabled or not isinstance(ref, str):
            return ref

        return self._cached_load(ref)

    def clear_cache(self) -> None:
        """Clear the model loading cache. Call between bootstrap iterations if needed."""
        self._cached_load.cache_clear()

    def save_dataframe(self, df: pl.DataFrame, name: str) -> Union[pl.DataFrame, str]:
        if not self.enabled:
            return df

        filename = f"{name}.parquet"
        filepath = self.dir / filename

        df.write_parquet(filepath, compression="zstd")

        return str(filepath)

    def load_dataframe(self, ref: Union[pl.DataFrame, str]) -> pl.DataFrame:
        if not self.enabled or not isinstance(ref, str):
            return ref

        return pl.read_parquet(ref)
