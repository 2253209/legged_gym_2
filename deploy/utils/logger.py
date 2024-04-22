from datetime import datetime
import os


def get_title_short():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(59):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 23:
            label = 'n_vel'
            i = 0
        elif k == 35:
            label = 'n_act'
            i = 0
        elif k == 47:
            label = 'sc_act'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title


def get_title_long():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(95):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 23:
            label = 'n_vel'
            i = 0
        elif k == 35:
            label = 'n_act'
            i = 0
        elif k == 47:
            label = 'r_pos'
            i = 0
        elif k == 59:
            label = 'r_vel'
            i = 0
        elif k == 71:
            label = 'r_act'
            i = 0
        elif k == 83:
            label = 'uf_act'
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
            self.file.write('%d,%d,' % (step, int(time * 10 ** 3)))  # us
            for index, item in enumerate(row):
                self.file.write(' %.4f,' % item)
                k += 1

            self.file.write('\n')
            break

    def close(self):
        self.file.close()
