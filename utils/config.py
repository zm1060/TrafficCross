import yaml
from pathlib import Path
from typing import Dict, Any, List, Union
import re
import logging
from dataclasses import dataclass


@dataclass
class ConfigPaths:
    """Configuration paths."""
    log_dir: Path
    output_dir: Path
    model_dir: Path
    cache_dir: Path


class ConfigValidator:
    """Validate configuration values."""

    @staticmethod
    def validate_split_ratio(split_ratio: List[float]) -> None:
        """Validate dataset split ratios.

        Args:
            split_ratio: List of train, validation, and test ratios.

        Raises:
            ValueError: If the ratios are invalid.
        """
        if len(split_ratio) != 3:
            raise ValueError("split_ratio must contain three values")
        if not all(0 <= x <= 1 for x in split_ratio):
            raise ValueError("split_ratio values must be between 0 and 1")
        if abs(sum(split_ratio) - 1.0) > 1e-6:
            raise ValueError("split_ratio values must sum to 1")

    @staticmethod
    def validate_paths(paths: ConfigPaths) -> None:
        """Validate path settings.

        Args:
            paths: Configuration path object.

        Raises:
            ValueError: If a path setting is invalid.
        """
        for path_name, path in paths.__dict__.items():
            if not isinstance(path, Path):
                raise ValueError(f"{path_name} must be a Path object")


def substitute_variables(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute variables in the configuration.

    Args:
        config: Configuration dictionary.
        context: Variable lookup context.

    Returns:
        The configuration with variables substituted.

    Raises:
        ValueError: If substitution fails.
    """
    try:
        if isinstance(config, dict):
            return {k: substitute_variables(v, context) for k, v in config.items()}
        elif isinstance(config, list):
            return [substitute_variables(v, context) for v in config]
        elif isinstance(config, str):
            # Replace variables of the form ${var}.
            def replace_var(match: re.Match) -> str:
                var_name = match.group(1)
                if var_name not in context:
                    logging.warning(f"Variable not found: {var_name}")
                return str(context.get(var_name, match.group(0)))

            return re.sub(r'\${(\w+)}', replace_var, config)
        return config
    except Exception as e:
        raise ValueError(f"Variable substitution failed: {str(e)}")


def convert_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert configuration value types.

    Args:
        config: Configuration dictionary.

    Returns:
        The configuration with converted values.

    Raises:
        ValueError: If type conversion fails.
    """
    try:
        if isinstance(config, dict):
            return {k: convert_config_values(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [convert_config_values(v) for v in config]
        elif isinstance(config, str):
            # Try converting scientific notation strings.
            try:
                if 'e' in config.lower():
                    return float(config)
            except ValueError:
                pass
            return config
        return config
    except Exception as e:
        raise ValueError(f"Failed to convert configuration values: {str(e)}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration completeness and values.

    Args:
        config: Configuration dictionary.

    Raises:
        ValueError: If the configuration is invalid.
    """
    # Validate required top-level sections.
    required_sections = ['data', 'flow', 'model', 'training', 'paths', 'logging']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {', '.join(missing_sections)}")

    # Validate data settings.
    if 'data' in config:
        data_config = config['data']
        # Validate dataset split ratios.
        if 'split_ratio' in data_config:
            ConfigValidator.validate_split_ratio(data_config['split_ratio'])

    # Validate path settings.
    if 'paths' in config:
        paths = ConfigPaths(
            log_dir=Path(config['paths']['log_dir']),
            output_dir=Path(config['paths']['output_dir']),
            model_dir=Path(config['paths']['model_dir']),
            cache_dir=Path(config['paths']['cache_dir'])
        )
        ConfigValidator.validate_paths(paths)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and process a configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        The loaded and processed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If YAML parsing fails.
        ValueError: If configuration validation fails.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    try:
        with config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML parsing failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to read configuration file: {str(e)}")

    try:
        # Convert configuration value types.
        config = convert_config_values(config)

        # Substitute variables.
        # Collect available variable values first.
        context = {}
        if 'data' in config:
            context.update(config['data'])

        config = substitute_variables(config, context)

        # Validate the configuration.
        validate_config(config)

        return config

    except Exception as e:
        raise ValueError(f"Configuration processing failed: {str(e)}")
