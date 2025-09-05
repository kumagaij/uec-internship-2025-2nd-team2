import tkinter
import tkinter.ttk

root = tkinter.Tk()
root.title("conbobox")
root.geometry("400x200")
la01 = tkinter.Label(root, text="フィルタレベル",font=("System",12))
la01.place(x=20,y=50)

#設定
data_list1 = ["3","5","7"]
combo1 = tkinter.ttk.Combobox(state = "readonly",
values = data_list1,font = ("System",12))

combo1.set(data_list1[0])
combo1.bind("<<ComboboxSelected>>")
combo1.place(x=150,y=50)

root.mainloop()
