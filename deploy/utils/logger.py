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
        label = 'omg'
        self.file.write('No,Time,')
        for k in range(75):
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
            elif k == 19:
                label = 'n_vel'
                i = 0
            elif k == 29:
                label = 'n_act'
                i = 0
            elif k == 39:
                label = 'r_pos'
                i = 0
            elif k == 51:
                label = 'r_vel'
                i = 0
            elif k == 63:
                label = 'r_act'
                i = 0
            self.file.write(f'{label}_{i},')
            i += 1

        self.file.write('\n')

    def save_39(self, obs, step, time):
        for row in obs:
            k = 0
            self.file.write('%d,%d,' % (step,int(time * 10 ** 6)))
            for index, item in enumerate(row):  # 39
                if 29 <= index < 39:
                    self.file.write(' %.4f,' % (item / 4))
                else:
                    self.file.write(' %.4f,' % item)
                k += 1

            self.file.write('\n')

    def save_75(self, obs, step, time):
        for row in obs:
            k = 0
            self.file.write('%d,%d,' % (step,int(time * 10 ** 6)))
            for index, item in enumerate(row):  # 75
                if 29 <= index < 39:
                    self.file.write(' %.4f,' % (item / 4))
                elif 63 <= index < 75:
                    self.file.write(' %.4f,' % (item / 4))
                else:
                    self.file.write(' %.4f,' % item)
                k += 1

            self.file.write('\n')

    def close(self):
        self.file.close()


