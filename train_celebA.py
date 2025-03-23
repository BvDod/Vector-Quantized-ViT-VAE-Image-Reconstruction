import train

if __name__ == "__main__":
    settings = {
        "dataset": "celebA",
        "save_model": True,

        "print_debug": False,
        "example_image_amount": 4,
        "save_reconstructions_first_epoch": True,
        "batch_size": 32,
        "learning_rate": 3e-4, # for x-ray
        "max_epochs": 100000,
        "early_stopping_epochs": 3,

        "model_settings" : {
            "encoder_architecture": "VIT",
            "num_hidden": 128,
            "num_residual_hidden": 32,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "commitment_cost": 0.5,
            "transformer_layers": 6,
            "attention_heads": 4,
            "patch_size": 8,
        }
    }
    train.train_vq_vae(settings)