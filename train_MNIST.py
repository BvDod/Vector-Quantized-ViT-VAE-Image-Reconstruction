import train

if __name__ == "__main__":
    settings = {
        "dataset": "MNIST",

        "print_debug": False,
        "example_image_amount": 8,
        "save_reconstructions_first_epoch": True,
        "batch_size": 32,
        "learning_rate": 1e-4, # for Mnsist
        "max_epochs": 100,
        "early_stopping_epochs": 5,

        "model_settings" : {
            "num_hidden": 64,
            "num_residual_hidden": 32,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "commitment_cost": 1 # for mnist
        }
    }
    train.train_vq_vae(settings)