import os
import sys

if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.platform == "win32":
    import ctypes
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)





import tkinter as tk
import single
import together1



def main():
    root = tk.Tk()
    root.title("TA Coreference System")
    root.geometry("900x700")

    top_frame = tk.Frame(root)
    top_frame.pack(side="top", fill="both", expand=True)

    bottom_frame = tk.Frame(root)
    bottom_frame.pack(side="bottom", fill="both", expand=True)

    resolver = single.TaCorefResolver()
    resolver.create_gui(top_frame)

  
    resolver.create_gui(top_frame)

  
    batch_app = together1.BatchProcessorApp(bottom_frame)

    root.mainloop()

if __name__ == "__main__":
    main()
