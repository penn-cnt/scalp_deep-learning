import tkinter as tk
from tkinter import ttk

class pipegui:

    def __init__(self):
        pass

    def on_dropdown_change(self,event):
        selected_item = dropdown_var.get()
        update_text_widget(selected_item)

    def update_text_widget(self,selected_item):
        text_widget.delete("1.0", tk.END)  # Clear previous text
        # Add explanation for the selected item
        explanation = explanations.get(selected_item, "No explanation available.")
        text_widget.insert(tk.END, explanation)

    def main(self):
        # Sample explanations for each dropdown item (modify as needed)
        explanations = {
            "Option 1": "Explanation for Option 1",
            "Option 2": "Explanation for Option 2",
            "Option 3": "Explanation for Option 3",
        }

        # Create the main window
        root = tk.Tk()
        root.title("GUI Template")

        # Create a frame for the left side (containing widgets)
        left_frame = ttk.Frame(root, padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew")

        # Create a canvas for the right side (containing text widget)
        canvas = tk.Canvas(root, borderwidth=0, background="white")
        canvas.grid(row=0, column=1, sticky="nsew")

        # Configure the grid layout
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=2)

        # Create a vertical separator line
        separator = ttk.Separator(root, orient="vertical")
        separator.grid(row=0, column=2, sticky="ns", padx=5)

        # Widgets on the left side
        dropdown_var = tk.StringVar()
        dropdown_var.set("Option 1")  # Set default selection
        dropdown = ttk.Combobox(left_frame, textvariable=dropdown_var, values=list(explanations.keys()))
        dropdown.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_change)

        # Add more widgets as needed

        # Text widget on the right side
        text_widget = tk.Text(canvas, wrap="word", width=40, height=20)
        text_widget.pack(padx=10, pady=10)

        # Run the GUI
        root.mainloop()

if __name__ == '__main__':

    # Get the arguments
    pass

    # Create the GUI
    PG = pipegui()
    PG.main()