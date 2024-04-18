from datetime import datetime
import os


def get_title_46():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(46):
        if k == 0:
            label = 'omg'
            i = 0
        elif k == 3:
            label = 'eul'
            i = 0
        elif k == 6:
            label = 'cmd'
            i = 0
        elif k == 9:
            label = 'n_pos'
            i = 0
        elif k == 21:
            label = 'n_vel'
            i = 0
        elif k == 33:
            label = 'n_act'
            i = 0
        elif k == 45:
            label = 'sin'
            i = 0

        title += f'{label}_{i},'
        i += 1
    return title


def get_title_82():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(82):
        if k == 0:
            label = 'omg'
            i = 0
        elif k == 3:
            label = 'eul'
            i = 0
        elif k == 6:
            label = 'cmd'
            i = 0
        elif k == 9:
            label = 'n_pos'
            i = 0
        elif k == 21:
            label = 'n_vel'
            i = 0
        elif k == 33:
            label = 'n_act'
            i = 0
        elif k == 45:
            label = 'r_pos'
            i = 0
        elif k == 57:
            label = 'r_vel'
            i = 0
        elif k == 69:
            label = 'r_act'
            i = 0
        elif k == 81:
            label = 'sin'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title


class SimpleLogger:
    def __init__(self, path, title):
        now = datetime.now()
        # 将当前时间格式化为字符串
        formatted_time = now.strftime('%Y-%m-%d_%H:%M:%S')
        filename = f"{path}/log_{formatted_time}.csv"
        if not os.path.exists(path):
            os.mkdir(path)
        self.file = open(filename, "a+")
        print(f"Saving log! Path: {filename}")
        self.file.write(f'{title}\n')

    def save(self, obs, step, time):
        for row in obs:
            k = 0
            self.file.write('%d,%d,' % (step, int(time * 10 ** 6)))
            for index, item in enumerate(row):  # 39
                self.file.write(' %.4f,' % item)
                k += 1

            self.file.write('\n')
            break

    def close(self):
        self.file.close()
