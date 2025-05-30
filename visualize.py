import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data.toy_datasets import get_data_loaders
# 모델 로드
from src.models.svq_model import SVQ

@hydra.main(config_path="src/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SVQ(
        image_size=cfg.experiment.dataset.image_size,
        in_channels=3,
        num_slots=cfg.experiment.model.num_slots,
        num_iterations=cfg.experiment.model.num_iterations,
        slot_size=cfg.experiment.model.slot_size,
        num_codebooks=cfg.experiment.model.num_codebooks,
        codebook_size=cfg.experiment.model.codebook_size,
        code_dim=cfg.experiment.model.code_dim,
        hidden_dim=cfg.experiment.model.hidden_dim,
        commitment_cost=cfg.experiment.model.commitment_cost
    ).to(device)

    # prior 초기화
    model.init_prior(
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    )

    model.load_state_dict(torch.load("checkpoint_epoch_90.pt")["model_state_dict"])

    train_loader, val_loader = get_data_loaders(
        dataset_name=cfg.experiment.dataset.name,
        batch_size=cfg.experiment.dataset.batch_size,
        num_workers=cfg.experiment.dataset.num_workers,
        num_samples=cfg.experiment.dataset.num_samples,
        image_size=cfg.experiment.dataset.image_size,
        max_shapes=cfg.experiment.dataset.max_shapes
    )
    # 샘플 데이터
    data = next(iter(val_loader))

    # 모델 순전파
    outputs = model(data)

    # 결과 시각화
    plt.figure(figsize=(15, 5))

    # 원본 이미지
    plt.subplot(1, 3, 1)
    input_img = data[0].permute(1, 2, 0).numpy()
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())  # 정규화
    plt.imshow(input_img)
    plt.title("Input")
    plt.axis("off")

    # 재구성 이미지
    plt.subplot(1, 3, 2)
    recon_img = outputs["recon"][0].detach().permute(1, 2, 0).numpy()
    recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())  # 정규화
    plt.imshow(recon_img)
    plt.title("Reconstruction")
    plt.axis("off")

    # 첫 번째 슬롯 마스크
    plt.subplot(1, 3, 3)
    mask = outputs["masks"][0, 0, 0].detach().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())  # 정규화
    plt.imshow(mask, cmap="viridis")
    plt.title("Slot 1 Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("visualization.png")

if __name__ == "__main__":
    main()