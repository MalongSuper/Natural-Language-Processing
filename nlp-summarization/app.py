import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading

from summarization import summarize_text


class SummarizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Summarization")
        self.root.geometry("900x700")

        # ==========================
        # TITLE
        # ==========================
        title = tk.Label(
            root,
            text="Text Summarization",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=10)

        # ==========================
        # INPUT TEXT
        # ==========================
        input_label = tk.Label(
            root,
            text="Input Text:"
        )
        input_label.pack(anchor="w", padx=10)

        self.input_text = scrolledtext.ScrolledText(
            root,
            height=15,
            wrap=tk.WORD
        )
        self.input_text.pack(
            fill="both",
            expand=False,
            padx=10,
            pady=5
        )

        # ==========================
        # UPLOAD BUTTON
        # ==========================
        upload_btn = tk.Button(
            root,
            text="Upload TXT File",
            command=self.upload_file
        )
        upload_btn.pack(pady=5)

        # ==========================
        # MODE SELECTION
        # ==========================
        mode_frame = tk.LabelFrame(
            root,
            text="Summary Style"
        )
        mode_frame.pack(
            fill="x",
            padx=10,
            pady=10
        )

        self.mode_var = tk.StringVar(value="paragraph")

        modes = [
            ("Casual", "casual"),
            ("Formal", "formal"),
            ("Bullet", "bullet"),
            ("Paragraph", "paragraph")
        ]

        for text, value in modes:
            tk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value
            ).pack(
                side="left",
                padx=10,
                pady=5
            )

        # ==========================
        # GENERATE BUTTON
        # ==========================
        self.generate_btn = tk.Button(
            root,
            text="Generate Summary",
            command=self.start_summarization
        )
        self.generate_btn.pack(pady=10)

        # ==========================
        # OUTPUT TEXT
        # ==========================
        output_label = tk.Label(
            root,
            text="Summary Output:"
        )
        output_label.pack(anchor="w", padx=10)

        self.output_text = scrolledtext.ScrolledText(
            root,
            height=12,
            wrap=tk.WORD,
            state="disabled"
        )
        self.output_text.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        # ==========================
        # SAVE BUTTON
        # ==========================
        save_btn = tk.Button(
            root,
            text="Save Summary",
            command=self.save_summary
        )
        save_btn.pack(pady=10)

    # =====================================
    # LOAD TXT FILE
    # =====================================
    def upload_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt")]
        )

        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()

            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, content)

        except Exception as error:
            messagebox.showerror(
                "Error",
                str(error)
            )

    # =====================================
    # START THREAD
    # =====================================
    def start_summarization(self):
        thread = threading.Thread(
            target=self.generate_summary,
            daemon=True
        )
        thread.start()

    # =====================================
    # GENERATE SUMMARY
    # =====================================
    def generate_summary(self):
        text = self.input_text.get(
            "1.0",
            tk.END
        ).strip()

        if not text:
            messagebox.showwarning(
                "Warning",
                "Please enter text first."
            )
            return

        self.generate_btn.config(
            state="disabled",
            text="Generating..."
        )

        try:
            mode = self.mode_var.get()

            summary = summarize_text(
                text,
                mode=mode
            )

            self.output_text.config(
                state="normal"
            )
            self.output_text.delete(
                "1.0",
                tk.END
            )
            self.output_text.insert(
                tk.END,
                summary
            )
            self.output_text.config(
                state="disabled"
            )

        except Exception as error:
            messagebox.showerror(
                "Error",
                str(error)
            )

        finally:
            self.generate_btn.config(
                state="normal",
                text="Generate Summary"
            )

    # =====================================
    # SAVE SUMMARY
    # =====================================
    def save_summary(self):
        summary = self.output_text.get(
            "1.0",
            tk.END
        ).strip()

        if not summary:
            messagebox.showwarning(
                "Warning",
                "No summary to save."
            )
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )

        if not filepath:
            return

        try:
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(summary)

            messagebox.showinfo(
                "Success",
                "Summary saved successfully."
            )

        except Exception as error:
            messagebox.showerror(
                "Error",
                str(error)
            )


def main():
    root = tk.Tk()
    app = SummarizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
