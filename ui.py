import tkinter as tk
from tkinter import filedialog, messagebox

class AudioApp:
    def __init__(self, root):
        self.root = root
        root.title("Song Approximation")

        # --- Upload WAV File ---
        self.wav_file = None
        self.btn_upload = tk.Button(root, text="Incarca fisier WAV", command=self.upload_wav)
        self.btn_upload.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")

        self.label_file = tk.Label(root, text="Nici un fisier selectat")
        self.label_file.grid(row=1, column=0, columnspan=3, sticky="w")

        # --- Force Mono checkbox ---
        self.mono_var = tk.BooleanVar(value=True)
        self.chk_mono = tk.Checkbutton(root, text="Forteaza Mono (2D)", variable=self.mono_var)
        self.chk_mono.grid(row=2, column=0, columnspan=3, pady=10, sticky="w")

        # --- Method selection ---
        self.method_var = tk.StringVar(value="linear_1")

        # Column 1: Regression polynomials
        col1_label = tk.Label(root, text="Polinoame de Regresie")
        col1_label.grid(row=3, column=0, pady=(10, 5))
        reg_options = [
            ("Regresie Liniara (Grad 1)", "pol_1"),
            ("Polinom de regresie Grad 2", "pol_2"),
            ("Polinom de regresie Grad 3", "poly_3"),
            ("Polinom de regresie Grad 10", "poly_10"),
        ]
        for i, (text, val) in enumerate(reg_options, start=4):
            rb = tk.Radiobutton(root, text=text, variable=self.method_var, value=val)
            rb.grid(row=i, column=0, sticky="w", padx=5)

        # Column 2: Spline interpolation
        col2_label = tk.Label(root, text="Spline")
        col2_label.grid(row=3, column=1, pady=(10, 5))
        spline_options = [
            ("Spline Liniar", "spline_1"),
            ("Spline Patratic", "spline_2"),
            ("Spline Cubic", "spline_3"),
        ]
        for i, (text, val) in enumerate(spline_options, start=4):
            rb = tk.Radiobutton(root, text=text, variable=self.method_var, value=val)
            rb.grid(row=i, column=1, sticky="w", padx=5)

        # Column 3: Interpolation methods + Chebyshev checkbox
        col3_label = tk.Label(root, text="Interpolare")
        col3_label.grid(row=3, column=2, pady=(10, 5))
        interp_options = [
            ("Polinom Lagrange", "lagrange"),
            ("Interpolarea Naiva", "naiva"),
        ]
        for i, (text, val) in enumerate(interp_options, start=4):
            rb = tk.Radiobutton(root, text=text, variable=self.method_var, value=val)
            rb.grid(row=i, column=2, sticky="w", padx=5)

        # Chebyshev usage checkbox (independent)
        self.chebyshev_var = tk.BooleanVar(value=False)
        chk_cheb = tk.Checkbutton(root, text="Foloseste Cebasev", variable=self.chebyshev_var)
        chk_cheb.grid(row=7, column=2, sticky="w", padx=5, pady=(10,0))

        # --- Process button ---
        self.btn_process = tk.Button(root, text="Proceseaza Fisierul", command=self.process_audio)
        self.btn_process.grid(row=8, column=0, columnspan=3, pady=20, sticky="ew")

    def upload_wav(self):
        filepath = filedialog.askopenfilename(
            title="Selecteaza un fisier WAV",
            filetypes=[("Fisiere WAV", "*.wav")]
        )
        if filepath:
            self.wav_file = filepath
            self.label_file.config(text=f"Selectat: {filepath.split('/')[-1]}")
        else:
            self.label_file.config(text="Nu s-a selectat un fisier")

    def process_audio(self):
        if not self.wav_file:
            messagebox.showerror("Eroare:", "Va rugam selectati un fisier")
            return

        selected_method = self.method_var.get()
        force_mono = self.mono_var.get()
        use_chebyshev = self.chebyshev_var.get()

        info = (
            f"Fisier: {self.wav_file}\n"
            f"Forteaza Mono: {'da' if force_mono else 'nu'}\n"
            f"Metoda Selectata: {selected_method}\n"
            f"Cebasev: {'da' if use_chebyshev else 'nu'}"
        )
        messagebox.showinfo("Se Proceseaza", info)

        

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
