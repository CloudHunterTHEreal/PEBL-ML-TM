import pandas as pd
import numpy as np
import time
import re
from sklearn.feature_extraction.text import CountVectorizer


class peblMl(object):
    def __init__(self, factor=1.0,  min_df=1, stop_words=['и'], lowercase=True, analyzer="word", binary=False,
                 ngram_range=(1, 1), min_df_category=1):
        self.factor = factor
        self.min_df = min_df
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.analyzer = analyzer
        self.binary = binary
        self.ngram_range = ngram_range
        self.min_df_category = min_df_category
        self.vectorizerContent = CountVectorizer(min_df=self.min_df, lowercase=self.lowercase, analyzer=self.analyzer,
                                                 stop_words=self.stop_words)
        self.vectorizerContent_2 = CountVectorizer(min_df=self.min_df, lowercase=self.lowercase, analyzer=self.analyzer,
                                                   binary=self.binary, ngram_range=self.ngram_range)
        self.vectorizerCategory = CountVectorizer(min_df=self.min_df_category)

    def fit(self, content, category):
        X = self.vectorizerContent.fit_transform(content)
        X_2 = self.vectorizerContent_2.fit_transform(content)
        Xcat = self.vectorizerCategory.fit_transform(category)
        xx = X.toarray().transpose()
        self.doc_lengths=X.toarray()
        self.selected_feature_names_Cont = np.asarray(self.vectorizerContent.get_feature_names())
        self.selected_feature_names_Cat = np.asarray(self.vectorizerCategory.get_feature_names())
        self.wordTopic = X.toarray().transpose() @ Xcat
        ProbTopicWord = self.wordTopic.transpose() / (np.sum(self.wordTopic, axis=1) + 0.01)
        ProbTopicWord = ProbTopicWord.transpose() / ProbTopicWord.transpose().sum(axis=1)[:, None]
        self.ProbTopicWord = ProbTopicWord.transpose()
        ProbTopicDoc = X @ self.ProbTopicWord.transpose()
        ProbTopicDoc = ProbTopicDoc.transpose() / (np.sum(ProbTopicDoc, axis=1) + 0.01)
        ProbTopicDoc = ProbTopicDoc.transpose() / ProbTopicDoc.transpose().sum(axis=1)[:, None]
        self.ProbTopicDoc = ProbTopicDoc
        return self

    def predict(self, query, newClassName='newTopic'):
        predictMatrixW = []
        el_data = {}
        new_text_1 = re.findall('\w+', query)
        new_text_2 = self.vectorizerContent_2.transform([query]).toarray()
        penalty = (len(new_text_1) - sum(sum(new_text_2))) * self.factor
        self.penalty = penalty
        self.topicNewTest = np.dot(self.vectorizerContent.transform([query]).toarray(), self.ProbTopicWord.transpose())
        c = 0
        for cat in self.selected_feature_names_Cat:
            self.topicNew = np.dot(self.ProbTopicWord[c],
                                   self.vectorizerContent.transform([query]).toarray().transpose())
            el_data[self.selected_feature_names_Cat[c]] = self.topicNew.sum() / (self.topicNewTest.sum() + penalty)
            c += 1
        el_data[newClassName] = penalty / (self.topicNewTest.sum() + penalty)
        self.el_data = el_data
        predictMatrixW.append(dict(el_data))
        dfMW = pd.DataFrame(predictMatrixW)
        dfSumMW = dfMW.sum()
        dfSumMW.sort_values( ascending=False, kind='quicksort', na_position='last', inplace=True)
        self.dfSumMW = dfSumMW
        return dfSumMW
