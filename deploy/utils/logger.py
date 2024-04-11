from datetime import datetime
import os

class SimpleLogger:
    def __init__(self, path):

        now = datetime.now()
        # 将当前时间格式化为字符串
        formatted_time = now.strftime('%Y-%m-%d_%H:%M:%S')
        filename = f"{path}/log_{formatted_time}.csv"
        if not os.path.exists(path):
            os.mkdir(path)
        self.file = open(filename, "a+")
        print(f"Saving log! Path: {filename}")
        i = 0
        label = 'No'
        for k in range(40):
            if k == 1:
                label = 'omg'
                i = 0
            elif k == 4:
                label = 'eul'
                i = 0
            elif k == 7:
                label = 'cmd'
                i = 0
            elif k == 10:
                label = 'pos'
                i = 0
            elif k == 20:
                label = 'vel'
                i = 0
            elif k == 30:
                label = 'act'
                i = 0

            self.file.write(f'{label}_{i},')
            i += 1

        self.file.write('time,')
        self.file.write('\n')

    def save(self, obs, step, time):
        for row in obs:
            k = 0
            self.file.write('%d, ' % step)
            for index, item in enumerate(row):
                if 29 <= index < 40:
                    self.file.write(' %.4f,' % (item / 4))
                else:
                    self.file.write(' %.4f,' % item)
                k += 1
            self.file.write(' %d,' % int(time * 10 ** 6))
            self.file.write('\n')

    def close(self):
        self.file.close()


