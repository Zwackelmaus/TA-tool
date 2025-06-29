# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 20:11:58 2025

@author: 14242
"""

import os
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import jieba.posseg as pseg
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import csv

class TaCorefResolver:
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def extract_noun_candidates(self, text):
        noun_flags = {"n", "nr", "ns", "nt", "nz", "s", "vn"}
        words = list(pseg.cut(text))
        ta_pos = None
        for i, (word, flag) in enumerate(words):
            if word.lower() == "ta":
                ta_pos = i
                break
        if ta_pos is None:
            return []
        candidates = [word for word, flag in words[:ta_pos] if flag in noun_flags and word.lower() != "ta"]
        return candidates

    def predict(self, input_text):
        if "ta" not in input_text.lower():
            return {'input_text': input_text, 'predicted_entity': "", 'confidence': None}
        candidates = self.extract_noun_candidates(input_text)
        if not candidates:
            return {'input_text': input_text, 'predicted_entity': "", 'confidence': None}

        pairs = [("ta", cand) for cand in candidates]
        inputs = self.tokenizer(
            [p[0] for p in pairs], [p[1] for p in pairs],
            padding=True, truncation=True, max_length=128, return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            scores = torch.softmax(logits, dim=1)[:, 1].tolist()
        best_idx = int(torch.tensor(scores).argmax())
        return {
            'input_text': input_text,
            'predicted_entity': candidates[best_idx],
            'confidence': scores[best_idx]
        }

class BatchProcessorApp:
    def __init__(self, master):
       
        self.master = master

        self.queue = queue.Queue()

     

        tk.Label(master, text="Input CSV file:").grid(row=0, column=0, sticky="e")
        self.input_path_var = tk.StringVar()
        tk.Button(master, text="Select Input CSV", command=self.browse_input).grid(row=0, column=1, sticky="w")
        self.input_label = tk.Label(master, text="", fg="blue")
        self.input_label.grid(row=1, column=1, sticky="w")

        tk.Label(master, text="Model directory:").grid(row=2, column=0, sticky="e")
        self.model_path_var = tk.StringVar()
        tk.Button(master, text="Select Model Directory", command=self.browse_model).grid(row=2, column=1, sticky="w")
        self.model_label = tk.Label(master, text="", fg="blue")
        self.model_label.grid(row=3, column=1, sticky="w")

        tk.Label(master, text="Output CSV file:").grid(row=4, column=0, sticky="e")
        self.output_path_var = tk.StringVar()
        tk.Button(master, text="Select Output CSV", command=self.browse_output).grid(row=4, column=1, sticky="w")
        self.output_label = tk.Label(master, text="", fg="blue")
        self.output_label.grid(row=5, column=1, sticky="w")

        self.status_label = tk.Label(master, text="Waiting to start...", fg="green")
        self.status_label.grid(row=6, column=0, columnspan=2, pady=10)

        self.run_btn = tk.Button(master, text="Run", command=self.start_processing)
        self.run_btn.grid(row=7, column=0, columnspan=2, pady=10)

        self.master.after(100, self.process_queue)

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.input_path_var.set(path)
            self.input_label.config(text=path)

    def browse_model(self):
        path = filedialog.askdirectory()
        if path:
            self.model_path_var.set(path)
            self.model_label.config(text=path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.output_path_var.set(path)
            self.output_label.config(text=path)

    def start_processing(self):
        input_path = self.input_path_var.get()
        model_dir = self.model_path_var.get()
        output_path = self.output_path_var.get()

        if not input_path or not model_dir or not output_path:
            messagebox.showerror("Error", "Please select input file, model directory, and output file.")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Processing...")

        thread = threading.Thread(target=self.process_batch, args=(input_path, output_path, model_dir))
        thread.start()

    def process_batch(self, input_path, output_path, model_dir):
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
            resolver = TaCorefResolver(model_dir)
            results = []
            total = len(df)

            for i, (_, row) in enumerate(df.iterrows(), start=1):
                text = str(row.get("input_text", "")).strip()
                if text:
                    result = resolver.predict(text)
                    conf_str = f"{result['confidence']:.4f}" if result['confidence'] is not None else ""
                    results.append({
                        "input_text": result['input_text'],
                        "predicted_entity": result['predicted_entity'],
                        "confidence": conf_str
                    })
                self.queue.put(('progress', i, total))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, mode='w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["input_text", "predicted_entity", "confidence"])
                writer.writeheader()
                writer.writerows(results)

            self.queue.put(('done', output_path))
        except Exception as e:
            self.queue.put(('error', str(e)))

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == 'progress':
                    _, current, total = msg
                    self.status_label.config(text=f"Processing: {current} / {total}")
                elif msg[0] == 'done':
                    _, output_path = msg
                    self.status_label.config(text="Processing completed!")
                    self.run_btn.config(state=tk.NORMAL)
                    messagebox.showinfo("Done", f"✅ Batch processing completed.\nResults saved to:\n{output_path}")
                elif msg[0] == 'error':
                    _, error_msg = msg
                    self.status_label.config(text="Error occurred.")
                    self.run_btn.config(state=tk.NORMAL)
                    messagebox.showerror("Error", error_msg)
        except queue.Empty:
            pass
        self.master.after(100, self.process_queue)


def launch_ta_coref_gui():
    root = tk.Tk()
    root.title("TA Coreference Batch Processor")
    app = BatchProcessorApp(root)
    root.mainloop()
