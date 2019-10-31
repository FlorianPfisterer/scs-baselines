"""
TODO: 
- load JSON recipes, extract sentences
- build vocabulary from sentences
- build action vocabulary

- transform JSON recipes to indices for words (input), entities & actions & attributes (labels)
- shuffle / batches / split stuff
"""
import json
import glob

class Recipe:
    def __init__(self, steps):
        self.steps = steps


class RecipeDataLoader:
    def __init__(self):
        pass

    @staticmethod
    def __create_vocabulary(sentences):
        words = set()
        for sentence in sentences:
            for word in sentence:
                words.add(word)

        vocabulary = list(words)
        print('vocabulary has {} words'.format(len(vocabulary)))

        word_to_idx = {}
        idx_to_word = {}
        for i, word in enumerate(vocabulary):
            word_to_idx[word] = i
            idx_to_word[i] = word

        return word_to_idx, idx_to_word

    @staticmethod
    def load_recipes(paths):
        # first, create a vocabulary
        raw_recipes = []
        all_sentences = []
        for path in paths:
            with open(path, 'r') as json_file:
                recipe = json.load(json_file)
                raw_recipes.append(recipe)

        recipes = []
        for recipe in raw_recipes:
            steps = [[]] * len(recipe['text'].keys())
            for key, value in recipe['text'].iteritems():
                steps[int(key)] = value
                all_sentences.append(value)
            recipes.append(Recipe(steps))
        word_to_idx, idx_to_word = RecipeDataLoader.__create_vocabulary(all_sentences)

        # now tokenize the sentences in each recipe
        for recipe in recipes:
            steps = recipe.steps    # list of lists
            steps_tokenized = [
                RecipeDataLoader.__tokenize(sentence, word_to_idx) for sentence in steps
            ]
            recipe.steps = steps_tokenized

        return recipes

    @staticmethod
    def __tokenize(sentence, word_to_idx):
        return list(map(lambda word: word_to_idx[word], sentence))


RECIPES_DIR = '/Users/fp/Downloads/cooking_dataset/recipes'
SAMPLE_SIZE = -1
if __name__ == '__main__':
    # test
    paths = glob.glob(RECIPES_DIR + '/*.json')
    recipes = RecipeDataLoader.load_recipes(paths=paths[:-1])
    print(recipes[0].steps)
