import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time

from PIL import Image
from caption_generator import CaptionGenerator


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def generate_caption(model, tokenizer, image_path, max_length):
    img, image_path = load_image(image_path)
    image_tensor = tf.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

    features = model.encoder(image_tensor)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    hidden = tf.zeros((1, model.units))

    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = model.decoder(dec_input, features, hidden)

        # get the word with the highest probability
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            break

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    # remove start and end tokens from the final output
    result = result[1:-1]
    caption = ' '.join(result)
    return caption


def plot_attention(image, caption, attention):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    # plot the image
    plt.imshow(temp_image)

    # plot the attention weights as an image map
    attention = np.resize(attention, (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(attention, cmap='gray', alpha=0.6)

    # set the caption as the title of the plot
    plt.title(caption, fontsize=20)
    plt.axis('off')
    plt.show()


def main(args):
    # Load the trained model
    model = load_model(args.model_path)

    # Load the tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Load the image
    img = load_image(args.image_path)

    # Generate the caption
    caption = generate_caption(model, tokenizer, img)

    # Print the caption
    print(caption)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    main(args)
