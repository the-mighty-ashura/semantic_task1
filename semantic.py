
import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

'''Similarities between "cat," "monkey," and "banana":

    The code calculates and prints the similarity scores between "cat" and "monkey," "banana" and "monkey," and "banana" and "cat."
    These scores represent the semantic similarity between the pairs of words. You can observe how the similarity scores vary,
    indicating the degree of similarity between the concepts represented by the words.
    For example, if "cat" and "monkey" have a low similarity score,
    it suggests that these two words represent different concepts and are not closely related in meaning.
    On the other hand, if "banana" and "monkey" have a higher similarity score,
    it indicates that these two words are more semantically similar, possibly due to their association in certain contexts (e.g., monkeys eating bananas).'''


'''Comparison between 'en_core_web_sm' and 'en_core_web_md' models:

    In this code, I am using the 'en_core_web_md' model, which is a larger and more comprehensive version of the English language model provided by spaCy.
    It includes word vectors that enable more accurate similarity calculations.
    There is also the 'en_core_web_sm' model, which is a smaller and lighter version of the English language model.
    The 'sm' model doesn't include word vectors, so you won't be able to calculate word or sentence similarities using the similarity method.
    However, the 'sm' model can still perform other natural language processing tasks, such as tokenization, part-of-speech tagging, and named entity recognition.'''
