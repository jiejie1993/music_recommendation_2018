# -*- coding:utf-8 -*-
__author__ = 'neuclil'


import _pickle as pickle
import sys
#保存歌单信息和歌曲信息
def parse_playlist_get_info(in_line, playlist_dict, song_dict):
    
    #歌单到内部id，歌曲到内部id
    contents = in_line.strip().split('\t')
    name, tags, playlist_id, subscribed_count = contents[0].split('##')
    playlist_dict[playlist_id] = name
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split(':::')
            song_dict[song_id] = song_name + '\t' + artist
        except Exception as e:
            print('song format error')
            print(song+'\n')


def parse_file(in_file, out_playlist, out_song):
    #从歌单id到歌单名称的映射字典
    playlist_dict = {}
    #从歌曲id到歌曲名称的映射字典
    song_dict = {}
    for line in open(in_file):
        parse_playlist_get_info(line, playlist_dict, song_dict)
    ##把映射字典保存在二进制文件中
    pickle.dump(playlist_dict, open(out_playlist, 'wb'))
    #可以通过 playlist_dic = pickle.load(open("playlist.pkl","rb"))重新载入
    pickle.dump(song_dict, open(out_song, 'wb'))


if __name__ == '__main__':
    parse_file('../data/netease_music_playlist.txt', '../data/playlist.pkl', '../data/song.pkl')