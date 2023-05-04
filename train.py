import tensorflow as tf
import pickle
import numpy as np
import os
import time
import argparse

from caption_generator import CaptionGenerator
from data_preprocessing import load_captions_data, load_image_features


def train(args):
    # Load data
    train_captions, img_name_vector = load_captions_data(args.captions_path, args.images_dir)
    img_name_vector, train_captions = load_image_features(img_name_vector, train_captions, args.features_dir)

    # Initialize tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=args.top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create padded sequences
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_name_vector, cap_vector))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda img_name, cap: (tf.numpy_function(load_image, [img_name], tf.float32), cap))

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=args.buffer_size).batch(args.batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Initialize model
    caption_generator = CaptionGenerator(args.embedding_dim, args.units, args.top_k, tokenizer, args.dropout_rate)

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Initialize loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Define loss function
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # Initialize checkpoint
    checkpoint_path = args.checkpoint_path
    ckpt = tf.train.Checkpoint(caption_generator=caption_generator, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=args.max_checkpoints)

    # Start training
    loss_plot = []
    for epoch in range(args.epochs):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(caption_generator, optimizer, loss_function, img_tensor, target)

            total_loss += t_loss

            if batch % args.print_every == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_manager.save()

        loss_plot.append(total_loss / args.num_steps)

        print(f'Epoch {epoch+1} Loss {total_loss/args.num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

    # Save final model
    ckpt_manager.save()

    # Save loss plot
    plt.plot(loss_history, label='train_loss')
    plt.plot(val_loss_history, label='val_loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.checkpoint_dir, 'loss_plot.png'))
    plt.show()
    
