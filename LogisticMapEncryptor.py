import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

IV = 0xA5
MAX_PIXELS = 1_500_000    
PREVIEW = 360



def orbit(x0, r, n, remove=1000):
    x = x0
    for _ in range(remove):
        x = r * x * (1 - x)
    out = np.empty(n)
    for i in range(n):
        x = r * x * (1 - x)
        out[i] = x
    return out


def quantize(x, mode):
    if mode == "top":      # leading bits: non-uniform histogram, parabola visible
        return np.minimum((x * 256).astype(np.int64), 255).astype(np.uint8)
    return ((x * 1e8).astype(np.int64) % 256).astype(np.uint8)   # low digits


def make_streams(n, key, quant, shuffle):
    seq = orbit(key[0], key[1], 2 * n)
    ks = quantize(seq[:n], quant)
    perm = np.argsort(seq[n:], kind="stable") if shuffle else np.arange(n)
    return ks, perm


def encrypt_flat(flat, key, quant, shuffle, feedback):
    n = flat.size
    ks, perm = make_streams(n, key, quant, shuffle)
    s = flat[perm]
    if not feedback:
        return s ^ ks
    out = [0] * n
    prev = IV
    for i, (a, b) in enumerate(zip(s.tolist(), ks.tolist())):
        prev = ((a + prev) & 255) ^ b
        out[i] = prev
    return np.array(out, dtype=np.uint8)


def decrypt_flat(cipher, key, quant, shuffle, feedback):
    n = cipher.size
    ks, perm = make_streams(n, key, quant, shuffle)
    if feedback:
        prev = np.concatenate(([IV], cipher[:-1])).astype(np.uint8)
        s = (((cipher ^ ks).astype(np.int16) - prev.astype(np.int16)) % 256).astype(np.uint8)
    else:
        s = cipher ^ ks
    flat = np.empty_like(s)
    flat[perm] = s
    return flat


def entropy(arr):
    p = np.bincount(arr.ravel(), minlength=256) / arr.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def adjacent_corr(arr):
    a = arr if arr.ndim == 2 else arr[..., 0]
    return float(np.corrcoef(a[:, :-1].ravel().astype(float),
                             a[:, 1:].ravel().astype(float))[0, 1])


def analyze(arr, key, quant, shuffle, feedback):
    """Returns (lines_of_text, encrypted_array)."""
    enc = encrypt_flat(arr.ravel(), key, quant, shuffle, feedback).reshape(arr.shape)
    lines = [
        f"Image shape: {arr.shape}",
        f"Entropy        plain {entropy(arr):.4f}   cipher {entropy(enc):.4f}   (ideal 8)",
        f"Adjacent corr  plain {adjacent_corr(arr):+.4f}   cipher {adjacent_corr(enc):+.4f}   (ideal ~0)",
    ]
    arr2 = arr.copy()
    arr2.ravel()[0] ^= 1
    enc2 = encrypt_flat(arr2.ravel(), key, quant, shuffle, feedback).reshape(arr.shape)
    npcr = (enc != enc2).mean() * 100
    uaci = (np.abs(enc.astype(int) - enc2.astype(int)) / 255).mean() * 100
    lines.append(f"NPCR {npcr:.2f}% (ideal ~99.6)   UACI {uaci:.2f}% (ideal ~33.5)")
    if not feedback:
        lines.append("  NPCR/UACI ~0 is expected without feedback (1 pixel in -> 1 pixel out).")
    elif npcr < 90:
        lines.append("  NPCR < 99%: feedback only runs forward, so it depends on where the "
                     "changed pixel lands after shuffling.")
    wrong = decrypt_flat(enc.ravel(), (key[0] + 1e-15, key[1]), quant, shuffle,
                         feedback).reshape(arr.shape)
    lines.append(f"Key sensitivity (x0 + 1e-15): {(wrong == arr).mean() * 100:.2f}% pixels "
                 f"correct (ideal ~0.4)")
    ok = np.array_equal(decrypt_flat(enc.ravel(), key, quant, shuffle, feedback).reshape(arr.shape), arr)
    lines.append(f"Round-trip with correct key exact: {ok}")
    return lines, enc


# =====================================================================
# GUI
# =====================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Logistic Map Image Cipher")
        self.geometry("1100x760")
        self.input_arr = None
        self.result_arr = None
        self.q = queue.Queue()
        self.busy = False
        self._photos = {}
        self._build()

    # ---------- layout ----------
    def _build(self):
        side = ttk.Frame(self, padding=10)
        side.pack(side="left", fill="y")

        ttk.Label(side, text="Image", font=("", 11, "bold")).pack(anchor="w")
        ttk.Button(side, text="Open Image...", command=self.open_image).pack(fill="x", pady=2)
        self.file_lbl = ttk.Label(side, text="No image loaded", wraplength=200)
        self.file_lbl.pack(anchor="w", pady=(0, 10))

        ttk.Label(side, text="Secret key", font=("", 11, "bold")).pack(anchor="w")
        self.x0_var = tk.StringVar(value="0.3141592")
        self.r_var = tk.StringVar(value="3.99")
        for label, var in (("x0  (0 to 1)", self.x0_var), ("r  (3.57 to 4)", self.r_var)):
            ttk.Label(side, text=label).pack(anchor="w")
            ttk.Entry(side, textvariable=var).pack(fill="x", pady=(0, 4))

        ttk.Label(side, text="Options", font=("", 11, "bold")).pack(anchor="w", pady=(8, 0))
        ttk.Label(side, text="Byte quantization").pack(anchor="w")
        self.quant_var = tk.StringVar(value="low")
        ttk.Combobox(side, textvariable=self.quant_var, values=["low", "top"],
                     state="readonly").pack(fill="x")
        self.shuffle_var = tk.BooleanVar(value=True)
        self.feedback_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="Shuffle pixel positions", variable=self.shuffle_var).pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(side, text="Ciphertext feedback", variable=self.feedback_var).pack(anchor="w")

        ttk.Label(side, text="Actions", font=("", 11, "bold")).pack(anchor="w", pady=(12, 2))
        self.btns = [
            ttk.Button(side, text="Encrypt", command=lambda: self.start("encrypt")),
            ttk.Button(side, text="Decrypt", command=lambda: self.start("decrypt")),
            ttk.Button(side, text="Analyze (stats)", command=lambda: self.start("analyze")),
            ttk.Button(side, text="Result  ->  Input", command=self.result_to_input),
            ttk.Button(side, text="Save Result as PNG...", command=self.save_result),
        ]
        for b in self.btns:
            b.pack(fill="x", pady=2)

        self.status = ttk.Label(side, text="Ready", foreground="gray")
        self.status.pack(anchor="w", pady=(10, 0))

        main = ttk.Frame(self, padding=10)
        main.pack(side="left", fill="both", expand=True)

        panels = ttk.Frame(main)
        panels.pack(fill="x")
        self.in_img, self.in_hist = self._panel(panels, "Input", 0)
        self.out_img, self.out_hist = self._panel(panels, "Result", 1)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)

        ttk.Label(main, text="Log / statistics", font=("", 11, "bold")).pack(anchor="w", pady=(10, 0))
        self.log = tk.Text(main, height=11, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)
        self.say("Open an image to begin.")

    def _panel(self, parent, title, col):
        f = ttk.Frame(parent)
        f.grid(row=0, column=col, padx=6, sticky="n")
        ttk.Label(f, text=title, font=("", 11, "bold")).pack()
        img = tk.Label(f, width=PREVIEW // 8, height=PREVIEW // 16, relief="sunken",
                       text="(empty)")
        img.pack()
        hist = tk.Canvas(f, width=PREVIEW, height=90, bg="white", highlightthickness=1,
                         highlightbackground="#aaa")
        hist.pack(pady=(6, 0))
        return img, hist

    # ---------- helpers ----------
    def say(self, text, clear=False):
        if clear:
            self.log.delete("1.0", "end")
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def show(self, arr, label, canvas):
        im = Image.fromarray(arr)
        im.thumbnail((PREVIEW, PREVIEW))
        ph = ImageTk.PhotoImage(im)
        self._photos[label] = ph      # keep a reference or Tk drops it
        label.config(image=ph, text="", width=ph.width(), height=ph.height())
        self.draw_hist(arr, canvas)

    def draw_hist(self, arr, canvas):
        canvas.delete("all")
        counts = np.bincount(arr.ravel(), minlength=256).astype(float)
        counts /= max(counts.max(), 1)
        w, h = int(canvas["width"]), int(canvas["height"])
        bw = w / 256
        for i, c in enumerate(counts):
            canvas.create_rectangle(i * bw, h - c * (h - 6), (i + 1) * bw, h,
                                    fill="#3b6fb6", outline="")

    def params(self):
        try:
            x0, r = float(self.x0_var.get()), float(self.r_var.get())
        except ValueError:
            raise ValueError("x0 and r must be numbers.")
        if not 0 < x0 < 1:
            raise ValueError("x0 must be strictly between 0 and 1.")
        if not 3.57 <= r <= 4:
            raise ValueError("r must be in [3.57, 4] (below that the map is not chaotic).")
        return (x0, r), self.quant_var.get(), self.shuffle_var.get(), self.feedback_var.get()

    # ---------- actions ----------
    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path)
            img = img.convert("L") if img.mode in ("L", "1") else img.convert("RGB")
        except Exception as e:
            messagebox.showerror("Cannot open image", str(e))
            return
        note = ""
        n = img.width * img.height * (1 if img.mode == "L" else 3)
        if n > MAX_PIXELS:
            scale = (MAX_PIXELS / n) ** 0.5
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                             Image.LANCZOS)
            note = f" (downscaled to {img.width}x{img.height} for speed)"
        self.input_arr = np.array(img, dtype=np.uint8)
        self.file_lbl.config(text=path.split("/")[-1].split("\\")[-1])
        self.show(self.input_arr, self.in_img, self.in_hist)
        self.say(f"Loaded {path}{note}", clear=True)

    def result_to_input(self):
        if self.result_arr is None:
            messagebox.showinfo("Nothing to move", "Run Encrypt or Decrypt first.")
            return
        self.input_arr = self.result_arr.copy()
        self.show(self.input_arr, self.in_img, self.in_hist)
        self.say("Result moved to Input. Now press Decrypt (same key and options) "
                 "or Encrypt again.")

    def save_result(self):
        if self.result_arr is None:
            messagebox.showinfo("Nothing to save", "Run Encrypt or Decrypt first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG (lossless)", "*.png")])
        if path:
            Image.fromarray(self.result_arr).save(path, format="PNG")
            self.say(f"Saved {path}")

    def start(self, action):
        if self.busy:
            return
        if self.input_arr is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        try:
            key, quant, shuffle, feedback = self.params()
        except ValueError as e:
            messagebox.showerror("Invalid key", str(e))
            return
        self.busy = True
        for b in self.btns:
            b.state(["disabled"])
        self.status.config(text=f"Working ({action})...")
        arr = self.input_arr.copy()

        def work():
            try:
                if action == "encrypt":
                    out = encrypt_flat(arr.ravel(), key, quant, shuffle, feedback).reshape(arr.shape)
                    self.q.put(("ok", action, out, None))
                elif action == "decrypt":
                    out = decrypt_flat(arr.ravel(), key, quant, shuffle, feedback).reshape(arr.shape)
                    self.q.put(("ok", action, out, None))
                else:
                    lines, enc = analyze(arr, key, quant, shuffle, feedback)
                    self.q.put(("ok", action, enc, lines))
            except Exception as e:  # report instead of dying silently in the thread
                self.q.put(("err", action, None, str(e)))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self.poll)

    def poll(self):
        try:
            status, action, out, extra = self.q.get_nowait()
        except queue.Empty:
            self.after(100, self.poll)
            return
        self.busy = False
        for b in self.btns:
            b.state(["!disabled"])
        if status == "err":
            self.status.config(text="Error")
            messagebox.showerror("Error", extra)
            return
        self.result_arr = out
        self.show(out, self.out_img, self.out_hist)
        self.status.config(text="Done")
        if action == "analyze":
            self.say("--- Analysis (current input, current key/options) ---", clear=True)
            for line in extra:
                self.say(line)
        elif action == "encrypt":
            self.say("Encrypted. Check the histogram: flat means good statistics, "
                     "and 'top' quantization will look less flat.")
        else:
            self.say("Decrypted. If the key or options differ from encryption you get noise.")


if __name__ == "__main__":
    App().mainloop()