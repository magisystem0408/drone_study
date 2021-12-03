__all__ = ['tell', 'pause', 'play', 'mute',
           'muted', 'unmute', 'stop', 'next', 'prev', 'state', 'isplaying', 'activate', 'isfrontmost', 'kill', 'getpid', 'quit']


import os
import applescript
import itunes.playlists
import itunes.tracks
import itunes.volume


def tell(code):
    """execute applescript `tell application "iTunes" ...`"""
    return applescript.tell.app("iTunes", code)


def pause():
    """pause iTunes"""
    tell("pause")


def play():
    tell("play")


def mute():
    """mute iTunes"""
    tell("set mute to true")


def muted():
    """return True if iTunes muted, else False"""
    return "true" in tell("get mute").out


def unmute():
    """unmute iTunes"""
    tell("set mute to false")


def stop():
    """stop"""
    tell("stop")


def next():
    """play next track"""
    tell("play next track")


def prev():
    """play previous track"""
    tell("play previous track")


def isplaying():
    """return True if iTunes is playing, else False"""
    return getpid() and "playing" in state()


def state():
    """return player state string"""
    return tell("(get player state as text)").out


"""process functions"""


def activate():
    """open iTunes and make it frontmost"""
    tell('activate')


def isfrontmost():
    """return True if `iTunes.app` is frontmost app, else False"""
    out = os.popen("lsappinfo info -only name `lsappinfo front`").read()
    return getpid() and "iTunes" in out.split('"')


def kill():
    os.popen("kill %s &> /dev/null" % getpid())


def getpid():
    """return iTunes.app pid"""
    for l in os.popen("ps -ax").read().splitlines():
        if "/Applications/iTunes.app" in l and "iTunesHelper" not in l:
            return int(list(filter(None, l.split(" ")))[0])


def quit():
    """Quit iTunes"""
    getpid() and tell("quit")
