import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def evaluate_model(model, tokenizer, data_generator, max_length):
    """
    Function to evaluate the trained model on validation set and compute BLEU scores
    :param model: Trained model to be evaluated
    :param tokenizer: Tokenizer object to tokenize the captions
    :param data_generator: Data generator object for the validation set
    :param max_length: Maximum length of the captions
    :return: Tuple of BLEU scores for 1 to 4 n-grams
    """
    actual, predicted = [], []
    for batch in data_generator:
        image_batch, caption_batch = batch[0], batch[1]
        for idx in range(len(image_batch)):
            # Generate caption using the model
            predicted_caption = generate_caption(model, image_batch[idx], tokenizer, max_length)
            actual_caption = " ".join([tokenizer.index_word[i] for i in caption_batch[idx] if i not in [0]])
            # Store the actual and predicted captions for the current batch
            actual.append([actual_caption.split()])
            predicted.append(predicted_caption.split())

    # Compute BLEU scores for 1 to 4 n-grams
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4


def generate_caption(model, image, tokenizer, max_length):
    """
    Function to generate caption for a given image using the trained model
    :param model: Trained model to be used for caption generation
    :param image: Image for which the caption is to be generated
    :param tokenizer: Tokenizer object to tokenize the captions
    :param max_length: Maximum length of the captions
    :return: Generated caption for the given image
    """
    image_tensor = np.expand_dims(image, axis=0)
    # Generate feature vectors for the image
    features = model.encoder(image_tensor)
    # Initialize the decoder input with the start token
    dec_input = tf.expand_dims([tokenizer.word_index['startseq']], 0)
    # Initialize the hidden state of the decoder
    hidden = tf.zeros((1, model.units))

    result = []

    for i in range(max_length):
        # Generate the next word in the caption using the decoder
        predictions, hidden, _ = model.decoder(dec_input, features, hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()
        # If the end token is predicted, stop caption generation
        if tokenizer.index_word[predicted_id] == 'endseq':
            break
        # Append the predicted word to the caption
        result.append(tokenizer.index_word[predicted_id])
        # Update the decoder input with the predicted word
        dec_input = tf.expand_dims([predicted_id], 0)

    # Join the predicted words to form the caption
    caption = ' '.join(result)
    return caption
