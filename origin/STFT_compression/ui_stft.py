import tkinter as tk
from tkinter import filedialog, messagebox
import compress_stft as compress
import decompress_stft as decompress
import error

class ErrorMenu:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.title("Error Plotting")

        self.wav1_path = None
        self.wav2_path = None

        # Select first WAV file
        tk.Label(self.window, text="Selecteaza primul fisier WAV:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.btn_wav1 = tk.Button(self.window, text="Alege fisier 1", command=self.select_wav1)
        self.btn_wav1.grid(row=0, column=1, padx=5, pady=5)

        self.label_wav1 = tk.Label(self.window, text="Nici un fisier selectat")
        self.label_wav1.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

        # Select second WAV file
        tk.Label(self.window, text="Selecteaza al doilea fisier WAV:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.btn_wav2 = tk.Button(self.window, text="Alege fisier 2", command=self.select_wav2)
        self.btn_wav2.grid(row=2, column=1, padx=5, pady=5)

        self.label_wav2 = tk.Label(self.window, text="Nici un fisier selectat")
        self.label_wav2.grid(row=3, column=0, columnspan=2, sticky="w", padx=10)

        # Process button
        self.btn_compare = tk.Button(self.window, text="Ploteaza Eroarea", command=self.run_error_script)
        self.btn_compare.grid(row=4, column=0, columnspan=2, pady=20)

    def select_wav1(self):
        filepath = filedialog.askopenfilename(
            title="Selecteaza primul fisier WAV",
            filetypes=[("Fisiere WAV", "*.wav")],
            initialdir="."
        )
        if filepath:
            self.wav1_path = filepath
            self.label_wav1.config(text=f"Selectat: {filepath.split('/')[-1]}")

    def select_wav2(self):
        filepath = filedialog.askopenfilename(
            title="Selecteaza al doilea fisier WAV",
            filetypes=[("Fisiere WAV", "*.wav")],
            initialdir="."
        )
        if filepath:
            self.wav2_path = filepath
            self.label_wav2.config(text=f"Selectat: {filepath.split('/')[-1]}")

    def run_error_script(self):
        if not self.wav1_path or not self.wav2_path:
            messagebox.showerror("Eroare", "Va rugam selectati ambele fisiere WAV.")
            return

        try:
            error.error_calc(self.wav1_path, self.wav2_path)
            messagebox.showinfo("Succes", "Plotarea erorii s-a realizat cu succes.")
        except Exception as e:
            messagebox.showerror("Eroare", f"A aparut o eroare:\n{e}")



# === Compression Menu ===
class CompressionMenu:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.title("Compresie WAV")
        self.wav_file = None

        # Upload WAV
        self.btn_upload = tk.Button(self.window, text="Incarca fisier WAV", command=self.upload_wav)
        self.btn_upload.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

        self.label_file = tk.Label(self.window, text="Nici un fisier selectat")
        self.label_file.grid(row=1, column=0, columnspan=2, sticky="w")

        # Force Mono checkbox
        self.mono_var = tk.BooleanVar(value=True)
        self.chk_mono = tk.Checkbutton(self.window, text="Forteaza Mono (2D)", variable=self.mono_var)
        self.chk_mono.grid(row=2, column=0, columnspan=2, pady=10, sticky="w")

        # Step
        tk.Label(self.window, text="Step (int):").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        self.entry_step = tk.Entry(self.window)
        self.entry_step.insert(0, "50")  # Default value
        self.entry_step.grid(row=3, column=1, sticky="w", padx=5)

        # Cutoff
        tk.Label(self.window, text="Cutoff (int):").grid(row=4, column=0, sticky="e", padx=5, pady=2)
        self.entry_cutoff = tk.Entry(self.window)
        self.entry_cutoff.insert(0, "14000")  # Default value
        self.entry_cutoff.grid(row=4, column=1, sticky="w", padx=5)

        # Threshold
        tk.Label(self.window, text="Threshold (int):").grid(row=5, column=0, sticky="e", padx=5, pady=2)
        self.entry_threshold = tk.Entry(self.window)
        self.entry_threshold.insert(0, "1000")  # Default value
        self.entry_threshold.grid(row=5, column=1, sticky="w", padx=5)

        # Process Button
        self.btn_process = tk.Button(self.window, text="Proceseaza Fisierul", command=self.process_audio)
        self.btn_process.grid(row=6, column=0, columnspan=2, pady=20, sticky="ew")

    def upload_wav(self):
        filepath = filedialog.askopenfilename(
            title="Selecteaza un fisier WAV",
            filetypes=[("Fisiere WAV", "*.wav")],
            initialdir="."
        )
        if filepath:
            self.wav_file = filepath
            self.label_file.config(text=f"Selectat: {filepath.split('/')[-1]}")
        else:
            self.label_file.config(text="Nu s-a selectat un fisier")

    def process_audio(self):
        if not self.wav_file:
            messagebox.showerror("Eroare", "Va rugam selectati un fisier WAV.")
            return

        try:
            step = int(self.entry_step.get())
            cutoff = int(self.entry_cutoff.get())
            threshold = int(self.entry_threshold.get())
        except ValueError:
            messagebox.showerror("Eroare", "Toate campurile trebuie sa fie numere intregi.")
            return

        force_mono = self.mono_var.get()
        base_path = self.wav_file[:-4]

        try:
            compress.Compression(base_path, force_mono, frequency_step=step, frequency_cutoff=cutoff, noise_threshold=threshold)
            messagebox.showinfo("Succes", "Fisierul a fost procesat cu succes.")
        except Exception as e:
            messagebox.showerror("Eroare", f"A aparut o eroare:\n{e}")



# === Decompression Menu ===
class DecompressionMenu:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.title("Decompresie STFT")

        self.stft_file = None

        # File selection
        self.btn_upload = tk.Button(self.window, text="Incarca fisier STFT", command=self.upload_stft)
        self.btn_upload.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

        self.label_file = tk.Label(self.window, text="Nici un fisier selectat")
        self.label_file.grid(row=1, column=0, columnspan=2, sticky="w")

        # Metoda selection
        tk.Label(self.window, text="Metoda:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.metoda_var = tk.StringVar(value="spline3")
        self.dropdown = tk.OptionMenu(self.window, self.metoda_var, "spline1", "spline2", "spline3", "np_spline", command=self.toggle_k_input)
        self.dropdown.grid(row=2, column=1, sticky="w")

        # K value (only used if np_spline is selected)
        tk.Label(self.window, text="Valoare k:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.entry_k = tk.Entry(self.window)
        self.entry_k.grid(row=3, column=1, sticky="w", padx=5)
        self.entry_k.insert(0, "3")  # Default value
        self.entry_k.configure(state='disabled')  # Initially disabled

        # Frequency to plot input
        tk.Label(self.window, text="Frecventa de plotat:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.entry_freq = tk.Entry(self.window)
        self.entry_freq.grid(row=4, column=1, sticky="w", padx=5)
        self.entry_freq.insert(0, "150")  # Example default frequency

        # Checkboxes for plotting options
        self.plot_singular_var = tk.BooleanVar(value=False)
        self.chk_singular = tk.Checkbutton(self.window, text="Plot singular wave", variable=self.plot_singular_var)
        self.chk_singular.grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        self.plot_entire_var = tk.BooleanVar(value=False)
        self.chk_entire = tk.Checkbutton(self.window, text="Plot entire waveform", variable=self.plot_entire_var)
        self.chk_entire.grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Decompress button
        self.btn_decompress = tk.Button(self.window, text="Decompreseaza", command=self.decompress_file)
        self.btn_decompress.grid(row=7, column=0, columnspan=2, pady=20, sticky="ew")

    def upload_stft(self):
        filepath = filedialog.askopenfilename(
            title="Selecteaza un fisier STFT",
            filetypes=[("Fisiere STFT", "*.stft")],
            initialdir="."
        )
        if filepath:
            self.stft_file = filepath
            self.label_file.config(text=f"Selectat: {filepath.split('/')[-1]}")
        else:
            self.label_file.config(text="Nu s-a selectat un fisier")

    def toggle_k_input(self, method_selected):
        if method_selected == "np_spline":
            self.entry_k.configure(state='normal')
        else:
            self.entry_k.configure(state='disabled')

    def decompress_file(self):
        if not self.stft_file:
            messagebox.showerror("Eroare", "Va rugam selectati un fisier STFT.")
            return

        method = self.metoda_var.get()
        k_value = None

        if method == "np_spline":
            try:
                k_value = int(self.entry_k.get())
            except ValueError:
                messagebox.showerror("Eroare", "Valoarea lui k trebuie sa fie un numar intreg.")
                return

        try:
            freq = int(self.entry_freq.get())
        except ValueError:
            messagebox.showerror("Eroare", "Frecventa trebuie sa fie un numar intreg.")
            return

        plot_singular = self.plot_singular_var.get()
        plot_entire = self.plot_entire_var.get()

        base_path = self.stft_file[:-5]

        try:
            decompress.Decompress(base_path, method, k_value, freq, plot_singular, plot_entire)

            messagebox.showinfo("Succes", f"Fisierul a fost decompresat cu metoda: {method}")
        except Exception as e:
            messagebox.showerror("Eroare", f"A aparut o eroare:\n{e}")

# === Main Menu ===
class MainApp:
    def __init__(self, root):
        self.root = root
        root.title("Meniu Principal")

        tk.Label(root, text="Alegeti Modul de Operare", font=("Arial", 14)).pack(pady=20)

        btn1 = tk.Button(root, text="Compresie WAV", width=25, command=self.open_compression_menu)
        btn1.pack(pady=10)

        btn2 = tk.Button(root, text="Decompresie STFT", width=25, command=self.open_decompression_menu)
        btn2.pack(pady=10)

        btn3 = tk.Button(root, text="Graph Eroare", width=25, command=self.open_error_menu)
        btn3.pack(pady=10)

    def open_compression_menu(self):
        CompressionMenu(self.root)

    def open_decompression_menu(self):
        DecompressionMenu(self.root)

    def open_error_menu(self):
        ErrorMenu(self.root)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.geometry("300x250")
    root.mainloop()
