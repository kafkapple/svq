import logging
import os
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.train import Trainer
from src.utils.visualization import Visualizer
from src.utils.metrics import compute_metrics

# 현재 파일의 디렉토리 경로
CURRENT_DIR = Path(__file__).parent

def setup_logging(debug: bool = False):
    """로깅 설정"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def train(cfg: DictConfig):
    """학습 실행"""
    # 설정 디버그 출력
    logging.info("Current configuration:")
    logging.info(OmegaConf.to_yaml(cfg))
    
    # 설정 파일 경로 확인
    logging.info("Config file locations:")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Config file path: {CURRENT_DIR / 'conf' / 'config.yaml'}")
    
    # training 섹션 구조 확인
    logging.info("\nTraining section structure:")
    if hasattr(cfg, 'training'):
        logging.info(OmegaConf.to_yaml(cfg.training))
        if hasattr(cfg.training, 'debug'):
            logging.info("\nDebug settings:")
            logging.info(OmegaConf.to_yaml(cfg.training.debug))
        else:
            logging.error("Debug settings not found in training section!")
    else:
        logging.error("Training section not found in config!")
        logging.error("Available config keys: " + str(list(cfg.keys())))
    
    # data 섹션 구조 확인
    logging.info("\nData section structure:")
    logging.info(OmegaConf.to_yaml(cfg.data))
    
    # model 섹션 구조 확인
    logging.info("\nModel section structure:")
    logging.info(OmegaConf.to_yaml(cfg.model))
    
    logging.info("\nStarting training...")
    trainer = Trainer(cfg)
    trainer.train()

def evaluate(cfg: DictConfig, checkpoint_path: Optional[str] = None):
    """평가 실행"""
    logging.info("Starting evaluation...")
    trainer = Trainer(cfg)
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)
    metrics = trainer.evaluate()
    logging.info(f"Evaluation metrics: {metrics}")

def visualize(cfg: DictConfig, checkpoint_path: Optional[str] = None):
    """시각화 실행"""
    logging.info("Starting visualization...")
    trainer = Trainer(cfg)
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)
    
    visualizer = Visualizer(cfg)
    visualizer.visualize_final(trainer.model, trainer.val_loader)

def ablation(cfg: DictConfig):
    """Ablation study 실행"""
    if not cfg.visualization.ablation.enabled:
        logging.error("Ablation study is not enabled in config")
        return

    logging.info("Starting ablation study...")
    results = {}
    
    # 각 실험 실행
    for experiment in cfg.visualization.ablation.experiments:
        logging.info(f"Running experiment: {experiment.name}")
        exp_cfg = OmegaConf.merge(cfg, experiment.config)
        exp_results = []
        
        # 여러 번 실행
        for seed in cfg.visualization.ablation.execution.seed_range:
            logging.info(f"Running with seed: {seed}")
            exp_cfg.experiment.seed = seed
            trainer = Trainer(exp_cfg)
            metrics = trainer.train()
            exp_results.append(metrics)
        
        # 결과 저장
        results[experiment.name] = {
            'metrics': exp_results,
            'mean': {k: sum(r[k] for r in exp_results) / len(exp_results) 
                    for k in exp_results[0].keys()},
            'std': {k: (sum((r[k] - results[experiment.name]['mean'][k])**2 
                           for r in exp_results) / len(exp_results))**0.5
                   for k in exp_results[0].keys()}
        }
    
    # 결과 시각화
    visualizer = Visualizer(cfg)
    visualizer.visualize_ablation_results(results, Path(cfg.visualization.ablation.save_dir))
    
    # 통계적 분석
    if cfg.visualization.ablation.execution.statistical_test:
        from scipy import stats
        baseline_results = results['baseline']['metrics']
        
        for exp_name, exp_data in results.items():
            if exp_name == 'baseline':
                continue
                
            for metric in cfg.visualization.ablation.experiments[0].metrics:
                baseline_metric = [r[metric] for r in baseline_results]
                exp_metric = [r[metric] for r in exp_data['metrics']]
                
                # Wilcoxon 검정
                if cfg.visualization.ablation.execution.test_method == 'wilcoxon':
                    stat, pval = stats.wilcoxon(baseline_metric, exp_metric)
                    logging.info(f"{exp_name} vs baseline - {metric}:")
                    logging.info(f"  p-value: {pval:.4f}")
                    logging.info(f"  significant: {pval < cfg.visualization.ablation.execution.significance_level}")

@hydra.main(config_path=str(CURRENT_DIR / "conf"), config_name="config", version_base=None)
def main(cfg: DictConfig):
    """메인 실행 함수"""
    # 설정 파일 존재 여부 확인
    config_path = CURRENT_DIR / "conf" / "config.yaml"
    logging.info(f"Looking for config file at: {config_path}")
    if not config_path.exists():
        logging.error(f"Config file not found at {config_path}")
        return
    
    # 설정 파일 내용 확인 (UTF-8 인코딩 사용)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            logging.info("\nRaw config file contents:")
            logging.info(f.read())
    except UnicodeDecodeError as e:
        logging.error(f"Error reading config file: {str(e)}")
        logging.error("Trying to read with different encodings...")
        # 다른 인코딩으로 시도
        encodings = ['utf-8-sig', 'cp949', 'euc-kr']
        for encoding in encodings:
            try:
                with open(config_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    logging.info(f"Successfully read with {encoding} encoding")
                    logging.info("\nRaw config file contents:")
                    logging.info(content)
                    break
            except UnicodeDecodeError:
                continue
    
    # 디버그 모드 설정 (최상위 레벨에서 접근)
    debug_enabled = cfg.get('debug', {}).get('enabled', False)
    setup_logging(debug=debug_enabled)
    if debug_enabled:
        logging.info("Debug mode enabled")
    
    # 실행 모드에 따른 함수 호출
    mode = cfg.get('mode', 'train')
    if mode == 'train':
        train(cfg)
    elif mode == 'evaluate':
        evaluate(cfg)
    elif mode == 'visualize':
        visualize(cfg)
    elif mode == 'ablation':
        ablation(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()
