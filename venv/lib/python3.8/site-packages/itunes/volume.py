__all__ = ['get', 'change']


import itunes


def get():
    if itunes.pid():
        return int(itunes.tell('sound volume').out)


def change(value):
    itunes.tell('set sound volume to %s' % value)
