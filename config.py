# configurations for training
configurations = {
    1: dict(
        MODEL="resnet34",
        DATA_ROOT="input",
        CHECKPOINT_ROOT="output/models/checkpoint",
        RESULT_ROOT="output/result",
        BATCH_SIZE=64
    ),
}


# configurations for deployment
THRESH = 0.6
NUM_CLASS = 9