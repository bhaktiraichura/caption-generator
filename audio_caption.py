from gtts import gTTS
import os
from caption_generator import evaluate

# Set the maximum number of words in the generated caption
max_caption_words = 20

# Get the image path from the user
image_path = input("Enter the image path: ")

# Generate the caption
result, attention_plot = evaluate(image_path, max_caption_words)
caption = ' '.join(result)

# Convert the caption to an audio file
tts = gTTS(caption)
tts.save('caption.mp3')

# Play the audio file
os.system('mpg321 caption.mp3')
