# PX4AI

PX4AI is a transformer-based autoencoder designed to efficiently annotate anomalies in PX4 log files.

Autoencoders learn to compress regular data from log files into a compact latent space. This process reduces
the data to its essential features, helping to filter out unnecessary details and noise. When an autoencoder
encounters new data that includes anomalies or unusual patterns, it struggles to reconstruct these accurately
because they don't match the typical patterns it has learned. As a result, the error between the input and its
reconstruction---often measured by how much the output differs from the input---increases significantly. This higher
error signals the presence of an anomaly, as the model fails to replicate the input accurately when it differs from
the norm.

PX4AI was trained on mission mode flight data from approximately 500 log files of hexacopters and quadcopters.
Only a selected subset of attributes from the Ulog files was used for training.

![px4ai](https://github.com/AkashKarnatak/PX4AI/assets/54985621/bc3dc53d-3714-4b70-98c4-944f1e7e3e8a)

## Setup and Installation

### 1. Clone the repository:

   ```bash
   git clone https://github.com/AkashKarnatak/PX4AI.git
   ```

### 2. Navigate to the project directory:

   ```bash
   cd PX4AI
   ```

### 3. Setup environment and install dependencies:

   ```bash
   conda env create -f environment.yml
   ```

then activate the environment using,

   ```bash
   conda activate px4ai
   ```

### 4. Prepare dataset:

Preprocess ulog files in csv format as described [here](https://github.com/AkashKarnatak/annotate_px4_logs?tab=readme-ov-file#6-create-database).
Store the csv files in the `./data/csv_files` directory. You can also download the dataset that
was used for training the model, [here](https://drive.google.com/file/d/1l1SKKseJ2SdpKUMd_jcAM9TWEdwvtqLU/view?usp=drive_link).

Once you have prepared your data in the `./data/csv_files` directory, you can proceed towards
model training or testing.

### 5. Model training:

You can start training the model by running the following command,

   ```bash
   python3 train.py
   ```

You can also view training metrics and graphs on tensorboard,

   ```bash
   tensorboard --logdir=runs
   ```

### 6. Model testing:

Model checkpoints are available in the `./checkpoints` directory. You can test your model on the
test dataset by running the following command,

   ```bash
   python3 inference.py
   ```

## Contributing

Contributions are welcome! If you find a bug, have an idea for an enhancement, or want to contribute in any way, feel free to open an issue or submit a pull request.

## License

This project is licensed under the AGPL3 License. For details, see the [LICENSE](LICENSE) file.
