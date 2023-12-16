import tkinter as tk
from tkinter import filedialog
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from keras.models import load_model

class TextEditor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Analizador de Sentimientos")

        self.text_area = tk.Text(self.window, wrap=tk.WORD)
        self.text_area.pack(expand=tk.YES, fill=tk.BOTH)

        self.create_menu()

        self.window.mainloop()

    def create_menu(self):
        menu = tk.Menu(self.window)
        self.window.config(menu=menu)

        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_command(label="Analyze", command=self.analyze_sentiment)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.window.quit)

    def new_file(self):
        self.text_area.delete(1.0, tk.END)

    def open_file(self):
        file = filedialog.askopenfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file:
            self.window.title(f"Analizador de Sentimientos - {file}")
            self.text_area.delete(1.0, tk.END)
            with open(file, "r") as file_handler:
                self.text_area.insert(tk.INSERT, file_handler.read())

    def save_file(self):
        file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file:
            with open(file, "w") as file_handler:
                file_handler.write(self.text_area.get(1.0, tk.END))
            self.window.title(f"Analizador de Sentimientos - {file}")
    
    def analyze_sentiment(self):
        # Obtener el texto del área de texto
        text_to_analyze = self.text_area.get(1.0, tk.END)

        if not text_to_analyze.strip():
            tk.messagebox.showinfo("Info", "El área de texto está vacía. Escribe algo para analizar.")
            return
        
        try:
            loaded_model_embedding = SentenceTransformer(r'C:\Users\mitoo\Documents\Proyecto 2\Desktop App\modelEmbedding')
            embedding = loaded_model_embedding.encode(text_to_analyze)
            embedding = np.array([embedding])

            model = load_model(r'C:\Users\mitoo\Documents\Proyecto 2\Desktop App\analizador_SentimientosUNITEC.h5')
            prediction = np.argmax(model.predict(embedding), axis=1)

            mapeo = {0: 'positivo', 1: 'neutral', 2: 'negativo'}
            array_mapeado = list(map(lambda x: mapeo[x], prediction))
            tk.messagebox.showinfo("Resultado del Análisis de Sentimientos", f"Predicción: {array_mapeado}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error al analizar el sentimiento: {str(e)}")
            
if __name__ == "__main__":
    text_editor = TextEditor()