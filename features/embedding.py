# from torch.nn import Embedding
#
# from data import dataset
# from features.tokenizer import Vocabulary
#
#
# class CustomEmbedding:
#     def __init__(self, embedding_dim=128):
#         self.vocabulary = Vocabulary()
#         self.vocabulary.build(dataset.get_corpus()[0])
#         self.embedding = Embedding(len(self.vocabulary), embedding_dim, 0)
#
