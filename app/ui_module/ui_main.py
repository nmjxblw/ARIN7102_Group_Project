import tkinter as tk
from tkinter import ttk


def create_main_window():
    root = tk.Tk()
    root.title("Main Window")
    root.geometry("400x300")

    # Create a label
    label = ttk.Label(root, text="Welcome to the Main Window!")
    label.pack(pady=20)

    # Create a button
    button = ttk.Button(root, text="Click Me", command=lambda: print("Button Clicked!"))
    button.pack(pady=10)

    return root


if __name__ == "__main__":
    root = create_main_window()
    root.mainloop()
