# -*- coding: utf-8 -*-

#coding: utf-8
import multiprocessing
import gensim
import sys
import cPickle as pickle
from random import shuffle

def parse_playlist_get_sequence(in_line, playlist_sequence):
	song_sequence = []
	contents = in_line.strip().split("\t")
	# 解析歌单序列
	for song in contents[1:]:
		try:
			song_id, song_name, artist, popularity = song.split(":::")
			song_sequence.append(song_id)
		except:
			print("song format error")
			print(song+"\n")
	for i in range(len(song_sequence)):
		shuffle(song_sequence)
        #打乱顺序
		playlist_sequence.append(song_sequence)


def train_song2vec(in_file, out_file):
	#所有歌单序列
	playlist_sequence = []
	#遍历所有歌单
	for line in open(in_file):
		parse_playlist_get_sequence(line, playlist_sequence)
	#使用word2vec训练
	cores = multiprocessing.cpu_count()
	print("using all "+str(cores)+" cores")
	print("Training word2vec model...")
	model = gensim.models.Word2Vec(sentences=playlist_sequence, size=150, min_count=3, window=7, workers=cores)

    #min_count (int) – Ignores all words with total frequency lower than this.
    #size (int) – Dimensionality of the feature vectors.

	print("Saving model...")
	model.save(out_file)



if __name__ == '__main__':
    song_sequence_file = "./popular.playlist"
    model_file = "./song2vec.model"
    train_song2vec(song_sequence_file, model_file)

    song_dic = pickle.load(open("popular_song.pkl","rb"))
    model_str = "./song2vec.model"
    model = gensim.models.Word2Vec.load(model_str)

    song_id_list = song_dic.keys()[1000:1500:50]
    for song_id in song_id_list:
        result_song_list = model.most_similar(song_id)

        print(song_id, song_dic[song_id])
        print("\n相似歌曲 和 相似度 分别为:")
        for song in result_song_list:
            print("\t", song_dic[song[0]], song[1])
        print("\n")
        #进一步：求得每首歌曲的相似度后，根据item-based协同过滤方法实现基于物品的协同过滤

