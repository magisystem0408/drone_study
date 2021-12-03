__all__ = ['play']


import itunes


def play(track, playlist):
    itunes.tell('play track "%s" of playlist "%s"' % (track, playlist))
