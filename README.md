# SASRec (Self-Attention Sequential Recommendation)

SASRec is a recommendation model that utilizes self-attention mechanisms to capture sequential user-item interactions for personalized recommendations. It is based on the Transformer architecture, commonly used in natural language processing tasks, but adapted for sequential recommendation systems.

This repository contains the implementation of the SASRec model, a state-of-the-art method for sequential recommendation tasks, which can learn from the order of user-item interactions to predict the next item a user will interact with.

## Features:
- **Self-Attention Mechanism**: Leverages the Transformer architecture to model sequential dependencies in user behavior.
- **Efficient Sampling**: Uses negative sampling and data partitioning for training and validation.
- **Flexible Configuration**: Easily customizable parameters such as batch size, learning rate, and sequence length.
- **Evaluation**: Implements standard evaluation metrics such as NDCG (Normalized Discounted Cumulative Gain) and Hit Rate for recommendation accuracy.

## Requirements:
- TensorFlow (2.x)
- TensorFlow Addons
- numpy
- random
- Other dependencies (listed in `requirements.txt`)

## Usage:
1. **Prepare the dataset**: The dataset should consist of user-item interaction data, with each line representing a user-item interaction (user_id, item_id).
2. **Train the model**: After preparing your dataset, you can train the model by running the following command:
   ```bash
   python main.py --dataset your_dataset_name --train_dir /path/to/train_dir --batch_size 64 --lr 0.001 --maxlen 10
