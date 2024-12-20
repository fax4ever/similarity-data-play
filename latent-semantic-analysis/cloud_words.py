from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt

class WordCloudImage:
    def __init__(self, series: pd.Series):
        self.frequences = {}
        words = list(series.axes[0])
        n = len(words)
        for i in range(n):
            word = words[i]
            freq = series.iloc[i]
            self.frequences[word] = freq
        pass

    def show(self):
        wc = WordCloud(background_color="white", max_words=20)
        wc.generate_from_frequencies(self.frequences)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show() 