__all__ = ['names', 'play']


import itunes


def names():
    return itunes.tell('get name of playlists').split(", ")


def play(playlist_name):
    return itunes.tell('play playlist named "%s"' % playlist_name)
