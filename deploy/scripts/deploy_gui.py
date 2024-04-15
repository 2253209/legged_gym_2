import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class CSVPlotter:
    def __init__(self):
        self.master = tk.Tk()
        self.master.title("CSV数据折线图")
        self.style = ttk.Style()

        self.style.configure("TLabel", font=("Helvetica", 14))
        self.style.configure("TButton", font=("Helvetica", 14), padding=10)

        self.title_frame = pd.DataFrame()
        self.data_frame = pd.DataFrame()
        self.selected_columns = []
        self.status_text = '停止模式'

        self.create_widgets()
        self.create_menu()
        self.create_listbox()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()


        self.toolbar_frame = ttk.Frame(self.graph_frame)  # 创建一个Frame来放置canvas
        self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.create_toolbar()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.mainloop()

    def create_toolbar(self):
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X, in_=self.toolbar_frame)

    def create_widgets(self):
        # 创建按钮显示区域
        self.button_frame = ttk.LabelFrame(self.master, text='操作')
        self.button_frame.pack(side=tk.TOP, fill=tk.X, expand=False, padx=5, pady=5)
        button1 = ttk.Button(self.button_frame, text='站立模式', command=self.switch_stand)
        button1.pack(side=tk.LEFT,fill=tk.Y, expand=False, padx=10, pady=10)
        button2 = ttk.Button(self.button_frame, text='网络模式', command=self.switch_net)
        button2.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=10, pady=10)
        button3 = ttk.Button(self.button_frame, text='不发送', command=self.switch_stop)
        button3.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=10, pady=10)


        # 创建列表显示区域
        self.list_frame = ttk.LabelFrame(self.master, text='列表')
        self.list_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=5, pady=5)

        # 创建图形显示区域
        self.graph_frame = ttk.LabelFrame(self.master, text='绘图')
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建表头（列）显示区域/滚动条
        scrollbar = ttk.Scrollbar(self.list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 创建表头（列）显示区域
        self.b1 = tk.Listbox(self.list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        self.b1.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

    def create_menu(self):
        menu_bar = tk.Menu(self.master)
        self.master.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="文件", menu=file_menu)

        # 在 "File" 菜单下添加一个看起来像按钮的菜单项
        file_menu.add_command(label="打开", command=lambda: print("打开文件"))
        file_menu.add_command(label="保存", command=lambda: print("保存文件"))
        file_menu.add_separator()  # 添加分隔线
        file_menu.add_command(label="退出", command=self.master.quit)

        # menu_bar.(label="站立姿态", command=self.switch_stand)
        # menu_bar.add_cascade(label="动作生成", command=self.switch_actgen)
        # menu_bar.add_cascade(label="神经网络", command=self.switch_net)
        # menu_bar.add_cascade(label="停止输出", command=self.switch_stop)

    def create_listbox(self):
        def callback(event):
            selection = self.b1.curselection()
            self.selected_columns = [self.b1.get(i) for i in selection]
            self.plot_graph()

        self.b1.bind("<<ListboxSelect>>", callback)

    def populate_options(self):
        self.b1.delete(0, tk.END)
        for column in self.title_frame.columns:
            self.b1.insert(tk.END, column)

    def plot_graph(self):
        if self.selected_columns:
            try:
                self.ax.clear()  # 清除子图
                self.ax.text(2, 2, self.status_text, transform=ax.transData, fontsize=12, family='sans-serif')
                for column in self.selected_columns:
                    self.ax.plot(self.data_frame[column], marker='.', linestyle='-', label=column)
                self.ax.legend(loc='best')
                self.canvas.draw()
            except Exception as e:
                messagebox.showerror("错误", f"绘图失败: {e}")

    def on_closing(self):
        response = messagebox.askyesno("确认退出", "您确定要退出程序吗？")
        if response:
            self.master.quit()


    def switch_stand(self):
        self.status_text = '站立姿态'
        print(self.status_text)

    def switch_actgen(self):
        self.status_text = '动作生成'
        print(self.status_text)

    def switch_net(self):
        self.status_text = '神经网络'
        print(self.status_text)

    def switch_stop(self):
        self.status_text = '停止模式'
        print(self.status_text)

    def add_data(self, obs):
        self.data_frame.insert(obs[0])
        # self.data_frame = pd.concat([self.data_frame, new_data_frame], ignore_index=True)
        self.plot_graph()


# 使用方法示例
if __name__ == "__main__":
    CSVPlotter()

