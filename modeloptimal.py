import numpy as np
import tensorflow as tf
from modules import *  
import random

class RCLModel(tf.keras.Model):
    def __init__(self, usernum, itemnum, args):
        super(RCLModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(usernum + 1, args.hidden_units)
        self.item_embedding = tf.keras.layers.Embedding(itemnum + 1, args.hidden_units)
        self.args = args
        self.reward_threshold = 0.5  # Start with a threshold for reward-based difficulty selection

        # Define your transformer layers or SASRec components here
        self.transformer_layer = tf.keras.layers.Transformer(
            num_layers=args.num_blocks,
            d_model=args.hidden_units,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
        )

    def call(self, inputs):
        user_id, item_seq, item_idx = inputs
        user_embed = self.user_embedding(user_id)
        seq_embed = self.item_embedding(item_seq)
        item_embed = self.item_embedding(item_idx)

        # Pass through Transformer layers
        output = self.transformer_layer([user_embed, seq_embed])

        logits = tf.reduce_sum(output * tf.expand_dims(item_embed, axis=1), axis=-1)
        return logits

    def sample_curriculum_batch(self, train_data, difficulty='easy'):
        """
        Sample batch with controlled difficulty based on RCL principle.
        """
        batch = []
        for user, items in train_data.items():
            # Easy samples: shorter sequences, hard samples: longer sequences
            if difficulty == 'easy' and len(items) <= self.args.maxlen // 2:
                batch.append((user, items))
            elif difficulty == 'hard' and len(items) > self.args.maxlen // 2:
                batch.append((user, items))
        return random.sample(batch, min(len(batch), self.args.batch_size))

    def train_step(self, batch):
        user_ids, sequences, labels = batch
        with tf.GradientTape() as tape:
            logits = self([user_ids, sequences, labels])
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

def train_with_rcl(model, train_data, args):
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    for epoch in range(args.num_epochs):
        if epoch < args.num_epochs * 0.5:
            difficulty = 'easy'  # Start with easier samples
        else:
            difficulty = 'hard'  # Progress to harder samples

        batch_data = model.sample_curriculum_batch(train_data, difficulty)
        
        user_ids = np.array([data[0] for data in batch_data])
        sequences = np.array([data[1][:-1] for data in batch_data])
        labels = np.array([data[1][-1] for data in batch_data])

        loss = model.train_step((user_ids, sequences, labels))
        print(f"Epoch {epoch + 1}, Difficulty: {difficulty}, Loss: {loss.numpy().mean()}")

        # Optional: Adapt reward threshold based on loss
        if loss.numpy().mean() < model.reward_threshold:
            model.reward_threshold *= 0.9  # Reduce threshold to increase difficulty in future batches
