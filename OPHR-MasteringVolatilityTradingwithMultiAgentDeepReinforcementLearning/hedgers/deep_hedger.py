from decimal import Decimal
from typing import Dict, Optional, Any
import numpy as np
from .base_hedger import BaseHedger


class DeepHedger(BaseHedger):
    def __init__(
        self,
        model_path: str,   
        feature_config: Optional[Dict[str, bool]] = None,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__()
        
        self.model_path = model_path
        self.device = device
        self.feature_config = feature_config or {
            'extra_feature_dim': 0,
            'position_feature_dim': 0,  
        }

        self.selected_feature_path = self.feature_config.get('selected_feature_path', None)
        self._selected_indices = self._load_selected_indices(self.selected_feature_path) if self.selected_feature_path else None
        
        self.position_feature_path = self.feature_config.get('position_feature_path', None)
        self._position_indices = self._load_selected_indices(self.position_feature_path) if self.position_feature_path else None
        
        self.model = self._load_model()
        self.feature_dim = self._calculate_feature_dim()
        
    def _load_model(self):
        try:
            import torch
            model = torch.load(self.model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Warning: Failed to load PyTorch model: {e}")
            return None
    
    def _calculate_feature_dim(self) -> int:
        greeks_dim = 4
        
        if self._selected_indices is not None and len(self._selected_indices) > 0:
            extra_dim = len(self._selected_indices)
        else:
            extra_dim = int(self.feature_config.get('extra_feature_dim', 0))
        
        if self._position_indices is not None and len(self._position_indices) > 0:
            position_dim = len(self._position_indices)
        else:
            position_dim = int(self.feature_config.get('position_feature_dim', 0))
        
        return greeks_dim + max(0, extra_dim) + max(0, position_dim)

    def _load_selected_indices(self, path: str):
        try:
            import pickle
            with open(path, 'rb') as f: indices = pickle.load(f)
            if isinstance(indices, list) and all(isinstance(x, int) for x in indices):
                return indices if len(indices) > 0 else None
            return None
        except Exception:
            return None
    
    def _prepare_features(
        self,
        delta: Decimal,
        gamma: Decimal,
        theta: Decimal,
        vega: Decimal,
        position_info: Dict,
        market_info: Dict
    ) -> np.ndarray:
        features = [
            float(delta),
            float(gamma),
            float(theta),
            float(vega),
        ]

        market_vec = self._extract_market_features(market_info, position_info)
        features.extend(market_vec.tolist())
        

        position_vec = self._extract_position_features(position_info)
        features.extend(position_vec.tolist())

        return np.array(features, dtype=np.float32)
    
    def _extract_market_features(self, market_info: Dict, position_info: Dict) -> np.ndarray:
        
        env_features = None
        if isinstance(market_info, dict):
            env_features = market_info.get('env_features', None)
        if env_features is None and isinstance(position_info, dict):
            env_features = position_info.get('env_features', None)

        selected_indices = self._selected_indices
        if selected_indices is not None and env_features is not None:
            env_arr = np.asarray(env_features, dtype=np.float32).flatten()
            extra_arr = []
            for idx in selected_indices:
                if 0 <= idx < env_arr.shape[0]:
                    extra_arr.append(env_arr[idx])
                else:
                    extra_arr.append(0.0)
            return np.asarray(extra_arr, dtype=np.float32)
        else:
            extra_dim_cfg = None
            if self._selected_indices is not None and len(self._selected_indices) > 0:
                extra_dim_cfg = len(self._selected_indices)
            else:
                extra_dim_cfg = int(self.feature_config.get('extra_feature_dim', 0))

            extra_features = None
            if isinstance(market_info, dict):
                extra_features = market_info.get('extra_features', None)
            if extra_features is None and isinstance(position_info, dict):
                extra_features = position_info.get('extra_features', None)

            if extra_features is None:
                return np.zeros(extra_dim_cfg, dtype=np.float32)
            else:
                extra_arr = np.asarray(extra_features, dtype=np.float32).flatten()
                if extra_arr.shape[0] < extra_dim_cfg:
                    pad_len = extra_dim_cfg - extra_arr.shape[0]
                    return np.concatenate([extra_arr, np.zeros(pad_len, dtype=np.float32)], axis=0)
                else:
                    return extra_arr[:extra_dim_cfg]
    
    def _extract_position_features(self, position_info: Dict) -> np.ndarray:
        if not isinstance(position_info, dict):
            position_dim = int(self.feature_config.get('position_feature_dim', 0))
            return np.zeros(position_dim, dtype=np.float32)
        
        position_features = position_info.get('position_features', None)
        
        if self._position_indices is not None and len(self._position_indices) > 0:
            if position_features is None:
                return np.zeros(len(self._position_indices), dtype=np.float32)
            
            pos_arr = np.asarray(position_features, dtype=np.float32).flatten()
            result = []
            for idx in self._position_indices:
                if 0 <= idx < pos_arr.shape[0]:
                    result.append(pos_arr[idx])
                else:
                    result.append(0.0)
            return np.asarray(result, dtype=np.float32)

        position_dim = int(self.feature_config.get('position_feature_dim', 0))
        
        if position_features is None or position_dim == 0:
            return np.zeros(position_dim, dtype=np.float32)
        
        pos_arr = np.asarray(position_features, dtype=np.float32).flatten()
        if pos_arr.shape[0] < position_dim:
            pad_len = position_dim - pos_arr.shape[0]
            return np.concatenate([pos_arr, np.zeros(pad_len, dtype=np.float32)], axis=0)
        else:
            return pos_arr[:position_dim]
    
    def _predict(self, features: np.ndarray) -> float:
        if self.model is None:
            return -1.0
        
        try:
            import torch
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                output = self.model(x)
                hedge_ratio = output.item()
            return hedge_ratio
        except Exception as e:
            print(f"Warning: Model prediction failed: {e}")
            return -1.0
    
    def compute_hedge(
        self,
        delta: Decimal,
        gamma: Decimal,
        theta: Decimal,
        vega: Decimal,
        position_info: Optional[Dict] = None,
        market_info: Optional[Dict] = None
    ) -> Decimal:

        if position_info is None:
            position_info = {}
        if market_info is None:
            market_info = {}
        
        features = self._prepare_features(
            delta, gamma, theta, vega,
            position_info, market_info
        )
        hedge_ratio = self._predict(features)
        hedge_amount = delta * Decimal(str(hedge_ratio))
        
        return hedge_amount
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_path': self.model_path,
            'feature_config': self.feature_config,
            'feature_dim': self.feature_dim,
            'device': self.device,
            'model_loaded': self.model is not None,
            'has_position_features': (self._position_indices is not None) or 
                                     (self.feature_config.get('position_feature_dim', 0) > 0)
        }

