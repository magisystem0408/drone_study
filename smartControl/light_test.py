from yeelight import Bulb
from yeelight import discover_bulbs
import time

if __name__ == '__main__':
    # ipアドレスを特定する。
    bulb_info = discover_bulbs()
    bulb_ip =bulb_info[0]["ip"]

    print("yeelightのIPアドレス："+bulb_ip)

    # yeelightインスタンス化
    blob =Bulb(str(bulb_ip))
    # 電源をつける
    blob.turn_on()
    time.sleep(1)
    blob.set_rgb(255,255,0)
    time.sleep(2)
    blob.set_rgb(255,0,255)
