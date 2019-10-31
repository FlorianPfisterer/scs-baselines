"""
TODO: 
- load JSON recipes, extract sentences
- build vocabulary from sentences
- build action vocabulary

- transform JSON recipes to indices for words (input), entities & actions & attributes (labels)
- shuffle / batches / split stuff
"""
import json

class RecipeExample:
    def __init__(self, sentences, sentence_lengths, num_ingredients, ingredients_steps, actions):
        """
        :param sentences: List[Tensor[long]]: num_sentences x max_sentence_length
            list of tokenized sentences with padding (word indices)
        :param sentence_lengths: Tensor[long]: num_sentences
            list of un-padded sentence lengths
        :param num_ingredients: int
            number of ingredients participating anywhere in the recipe
        :param ingredients_steps: List[List[long]]: num_sentences x <variable>
            for each recipe step, the list of the ingredient indices that participate
        :param actions: List[List[long]]: num_sentences x <variable>
            for each recipe step, the list of action indices that take place in this step
        """
        self.sentences = sentences
        self.sentence_lengths = sentence_lengths
        self.num_ingredients = num_ingredients
        self.ingredients_steps = ingredients_steps
        self.actions = actions


class RecipeDataLoader:
    def __init__(self):
        pass

    def __load_recipe(self, path):
        with open(path, 'r') as json_file:
            pass

