import nltk
from nltk.corpus import gutenberg
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download the required datasets
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load the text of Moby Dick from the Gutenberg dataset
moby_dick_text = gutenberg.raw('melville-moby_dick.txt')

# Tokenization
tokens = word_tokenize(moby_dick_text.lower())

# Stop filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

tagged_tokens = nltk.pos_tag(filtered_tokens)

# POS frequency
pos_freq = FreqDist(tag[1] for tag in tagged_tokens)
top_pos = pos_freq.most_common(5)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word, pos=pos[0].lower() if pos[0].lower() in ['a', 'r', 'n', 'v'] else 'n') for word, pos in tagged_tokens[:20]]

#frequency distribution
pos_freq.plot()

# Display the 5 most common parts of speech and their frequency
for pos, freq in top_pos:
    print(f"{pos}: {freq}")

print(lemmatized_words)