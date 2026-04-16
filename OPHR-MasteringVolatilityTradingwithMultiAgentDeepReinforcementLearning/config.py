import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from decimal import Decimal


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(path: str, env_vars: Optional[Dict[str, str]] = None) -> str:
    if env_vars is None:
        env_vars = os.environ
    
    # Replace ${VAR} with environment variable values
    import re
    pattern = r'\$\{([^}]+)\}'
    
    def replacer(match):
        var_name = match.group(1)
        return env_vars.get(var_name, match.group(0))
    
    return re.sub(pattern, replacer, path)


@dataclass
class PMConfig:
    price_range: Decimal
    min_expiry_delta_shock: Decimal
    annualized_move_risk: Decimal
    extended_dampener: Decimal
    volatility_range_up: Decimal
    volatility_range_down: Decimal
    short_term_vega_power: Decimal
    long_term_vega_power: Decimal
    delta_total_liquidity_shock_threshold: Decimal
    max_delta_shock: Decimal
    min_volatility_for_shock_up: Decimal
    extended_table_factor: Decimal
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, float]) -> 'PMConfig':
        return cls(**{k: Decimal(str(v)) for k, v in config_dict.items()})


@dataclass
class FeeConfig:
    futures_perpetual: Decimal
    options_per_contract: Decimal
    options_capped_at: Decimal
    combo_second_leg_reduction: Decimal
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeeConfig':
        return cls(
            futures_perpetual=Decimal(str(config_dict['futures_perpetual'])),
            options_per_contract=Decimal(str(config_dict['options']['per_contract'])),
            options_capped_at=Decimal(str(config_dict['options']['capped_at'])),
            combo_second_leg_reduction=Decimal(str(config_dict['combo_fees']['second_leg_reduction']))
        )


@dataclass
class EnvConfig:
    episode_length: int
    option_interval: int
    hedge_interval: int
    initial_capital: Decimal
    pm_config: Dict[str, PMConfig]
    fee_config: FeeConfig
    data_paths: Dict[str, str]
    
    @classmethod
    def from_yaml(cls, yaml_path: str, env_vars: Optional[Dict[str, str]] = None) -> 'EnvConfig':
        config = load_yaml(yaml_path)
        
        # Parse PM configs
        pm_configs = {}
        for crypto, pm_dict in config['pm_config'].items():
            pm_configs[crypto] = PMConfig.from_dict(pm_dict)
        
        # Parse fee config
        fee_config = FeeConfig.from_dict(config['fee_config'])
        
        # Resolve data paths
        data_paths = {}
        for key, path in config['data_paths'].items():
            data_paths[key] = resolve_path(path, env_vars)
        
        return cls(
            episode_length=config['episode_length'],
            option_interval=config['option_interval'],
            hedge_interval=config['hedge_interval'],
            initial_capital=Decimal(str(config['initial_capital'])),
            pm_config=pm_configs,
            fee_config=fee_config,
            data_paths=data_paths
        )
    
    def get_pm_config(self, crypto: str) -> PMConfig:
        if crypto not in self.pm_config:
            raise ValueError(f"PM config for {crypto} not found")
        return self.pm_config[crypto]


@dataclass
class TrainingConfig:
    # OP-Agent config
    op_hidden_dims: list
    op_activation: str
    op_learning_rate: float
    op_batch_size: int
    op_replay_buffer_size: int
    op_n_step: int
    op_gamma: float
    op_epsilon_start: float
    op_epsilon_end: float
    op_epsilon_decay: float
    op_update_frequency: int
    op_target_update_frequency: int
    oracle_episodes: int
    op_train_episodes: int
    op_eval_frequency: int
    
    # HR-Agent config
    hr_hidden_dims: list
    hr_activation: str
    hr_learning_rate: float
    hr_batch_size: int
    hr_n_hr: int
    hr_train_episodes: int
    
    # Iterative training
    num_iterations: int
    op_episodes_per_iter: int
    hr_episodes_per_iter: int
    
    # Oracle parameters
    oracle_beta: float
    oracle_lookforward_window: int
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        config = load_yaml(yaml_path)
        
        return cls(
            # OP-Agent
            op_hidden_dims=config['op_agent']['hidden_dims'],
            op_activation=config['op_agent']['activation'],
            op_learning_rate=config['op_agent']['learning_rate'],
            op_batch_size=config['op_agent']['batch_size'],
            op_replay_buffer_size=config['op_agent']['replay_buffer_size'],
            op_n_step=config['op_agent']['n_step'],
            op_gamma=config['op_agent']['gamma'],
            op_epsilon_start=config['op_agent']['epsilon_start'],
            op_epsilon_end=config['op_agent']['epsilon_end'],
            op_epsilon_decay=config['op_agent']['epsilon_decay'],
            op_update_frequency=config['op_agent']['update_frequency'],
            op_target_update_frequency=config['op_agent']['target_update_frequency'],
            oracle_episodes=config['op_agent']['oracle_episodes'],
            op_train_episodes=config['op_agent']['train_episodes'],
            op_eval_frequency=config['op_agent']['eval_frequency'],
            
            # HR-Agent
            hr_hidden_dims=config['hr_agent']['hidden_dims'],
            hr_activation=config['hr_agent']['activation'],
            hr_learning_rate=config['hr_agent']['learning_rate'],
            hr_batch_size=config['hr_agent']['batch_size'],
            hr_n_hr=config['hr_agent']['n_hr'],
            hr_train_episodes=config['hr_agent']['train_episodes'],
            
            # Iterative
            num_iterations=config['iterative_training']['num_iterations'],
            op_episodes_per_iter=config['iterative_training']['op_episodes_per_iter'],
            hr_episodes_per_iter=config['iterative_training']['hr_episodes_per_iter'],
            
            # Oracle
            oracle_beta=config['oracle']['beta'],
            oracle_lookforward_window=config['oracle']['lookforward_window']
        )


@dataclass
class HedgerPoolConfig:
    # Pool settings
    hedger_pool_size: int
    
    # Delta-based hedgers
    delta_hedger_thresholds: list
    delta_hedger_ratio: float
    
    # Price-based hedgers  
    price_hedger_thresholds: list
    price_hedger_ratio: float
    
    # Deep hedgers
    deep_hedger_models: list  # List of dicts with model_path, risk_aversion, device
    deep_hedger_feature_config: dict
    
    # Baseline hedger
    baseline_hedger: dict
    
    # Selection criteria
    selection_criteria: dict
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'HedgerPoolConfig':
        config = load_yaml(yaml_path)
        
        return cls(
            hedger_pool_size=config.get('hedger_pool_size', 30),
            delta_hedger_thresholds=config.get('delta_hedger_thresholds', []),
            delta_hedger_ratio=config.get('delta_hedger_ratio', 1.0),
            price_hedger_thresholds=config.get('price_hedger_thresholds', []),
            price_hedger_ratio=config.get('price_hedger_ratio', 1.0),
            deep_hedger_models=config.get('deep_hedger_models', []),
            deep_hedger_feature_config=config.get('deep_hedger_feature_config', {}),
            baseline_hedger=config.get('baseline_hedger', {}),
            selection_criteria=config.get('selection_criteria', {})
        )


@dataclass
class HedgerConfig:
    hedger_type: str
    delta_threshold: float
    hedge_ratio: float
    
    @classmethod
    def from_yaml(cls, yaml_path: str, hedger_name: str = 'baseline_hedger') -> 'HedgerConfig':
        config = load_yaml(yaml_path)
        hedger_config = config[hedger_name]
        
        return cls(
            hedger_type=hedger_config['type'],
            delta_threshold=hedger_config['delta_threshold'],
            hedge_ratio=hedger_config['hedge_ratio']
        )
    
    @classmethod
    def load_all_hedgers(cls, yaml_path: str) -> Dict[str, 'HedgerConfig']:
        config = load_yaml(yaml_path)
        available = config.get('available_hedgers', [])
        
        hedgers = {}
        for name in available:
            if name in config:
                hedger_dict = config[name]
                hedgers[name] = cls(
                    hedger_type=hedger_dict['type'],
                    delta_threshold=hedger_dict['delta_threshold'],
                    hedge_ratio=hedger_dict['hedge_ratio']
                )
        
        return hedgers


@dataclass
class EvaluationConfig:
    annual_trading_days: int
    risk_free_rate: float
    compute_metrics: list
    
    # Visualization
    figure_size: list
    save_format: str
    dpi: int
    
    # Report
    save_csv: bool
    save_metrics_txt: bool
    output_dir: str
    
    # Trade analysis
    separate_directions: bool
    min_holding_period: int
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EvaluationConfig':
        config = load_yaml(yaml_path)
        
        return cls(
            annual_trading_days=config['metrics']['annual_trading_days'],
            risk_free_rate=config['metrics']['risk_free_rate'],
            compute_metrics=config['metrics']['compute_metrics'],
            figure_size=config['visualization']['figure_size'],
            save_format=config['visualization']['save_format'],
            dpi=config['visualization']['dpi'],
            save_csv=config['report']['save_csv'],
            save_metrics_txt=config['report']['save_metrics_txt'],
            output_dir=config['report']['output_dir'],
            separate_directions=config['trade_analysis']['separate_directions'],
            min_holding_period=config['trade_analysis']['min_holding_period']
        )


class ConfigManager:
    
    def __init__(self, config_dir: str = 'configs', data_root: str = None, crypto: str = None):
        self.config_dir = config_dir
        self._env_config = None
        self._training_config = None
        self._hedger_config = None
        self._hedger_pool_config = None
        self._evaluation_config = None

        self._env_vars = {
            'DATA_ROOT': data_root or os.environ.get('DATA_ROOT', 'sample_data'),
            'CRYPTO': crypto or os.environ.get('CRYPTO', 'BTC')
        }
    
    @property
    def env_config(self) -> EnvConfig:
        if self._env_config is None:
            path = os.path.join(self.config_dir, 'env_config.yaml')
            self._env_config = EnvConfig.from_yaml(path, env_vars=self._env_vars)
        return self._env_config
    
    @property
    def training_config(self) -> TrainingConfig:
        if self._training_config is None:
            path = os.path.join(self.config_dir, 'training_config.yaml')
            self._training_config = TrainingConfig.from_yaml(path)
        return self._training_config
    
    @property
    def hedger_config(self) -> Dict[str, HedgerConfig]:
        if self._hedger_config is None:
            path = os.path.join(self.config_dir, 'hedger_config.yaml')
            self._hedger_config = HedgerConfig.load_all_hedgers(path)
        return self._hedger_config
    
    @property
    def hedger_pool_config(self) -> HedgerPoolConfig:
        if self._hedger_pool_config is None:
            path = os.path.join(self.config_dir, 'hedger_config.yaml')
            self._hedger_pool_config = HedgerPoolConfig.from_yaml(path)
        return self._hedger_pool_config
    
    @property
    def evaluation_config(self) -> EvaluationConfig:
        if self._evaluation_config is None:
            path = os.path.join(self.config_dir, 'evaluation_config.yaml')
            self._evaluation_config = EvaluationConfig.from_yaml(path)
        return self._evaluation_config


if __name__ == '__main__':
    # Test configuration loading
    print("=" * 80)
    print("Configuration Manager Test")
    print("=" * 80)
    
    manager = ConfigManager()
    
    print("\nEnvironment Config:")
    print(f"  Episode length: {manager.env_config.episode_length} days")
    print(f"  Option interval: {manager.env_config.option_interval} minutes")
    print(f"  Initial capital: {manager.env_config.initial_capital} BTC")
    
    print("\nData Paths:")
    for key, path in manager.env_config.data_paths.items():
        print(f"  {key}: {path}")
    
    print("\nTraining Config:")
    print(f"  OP-Agent learning rate: {manager.training_config.op_learning_rate}")
    print(f"  n-step: {manager.training_config.op_n_step}")
    
    print("\nHedger Config:")
    for name, config in manager.hedger_config.items():
        print(f"  {name}: threshold={config.delta_threshold}")
    
    print("\nEvaluation Config:")
    print(f"  Metrics: {manager.evaluation_config.compute_metrics}")
    
    print("\n" + "=" * 80)
    print("Configuration loaded successfully!")
    print("=" * 80)



