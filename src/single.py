# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import jieba.posseg as pseg
import tkinter as tk
from tkinter import scrolledtext, filedialog

class TaCorefResolver:
    def __init__(self, model_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        if model_dir:
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.model = BertForSequenceClassification.from_pretrained(model_dir).to(self.device)
            self.model.eval()
        else:
            self.tokenizer = None
            self.model = None

    def extract_noun_candidates(self, text):
        noun_flags = {"n", "nr", "ns", "nt", "nz", "s", "vn"}
        words = pseg.cut(text)
        return [word for word, flag in words if flag in noun_flags and word.lower() != "ta"]

    def predict(self, input_text):
        if not self.model or not self.tokenizer:
            return {'input_text': input_text, 'predicted_entity': "", 'confidence': None}

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

    def create_gui(self, master):
        # Clear master content if any
        for widget in master.winfo_children():
            widget.destroy()

        def run_prediction():
            sentence = input_text.get().strip()
            output_text.delete("1.0", tk.END)

            if not self.model_dir:
                output_text.insert(tk.END, "Please select a model directory first!")
                return

            if not sentence:
                output_text.insert(tk.END, "Please enter a Chinese sentence containing 'ta'!")
                return

        
            if self.tokenizer is None or self.model is None:
                output_text.insert(tk.END, "⏳ Loading model, please wait...\n")
                master.update_idletasks()
                try:
                    self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
                    self.model = BertForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
                    self.model.eval()
                except Exception as e:
                    output_text.insert(tk.END, f"❌ Failed to load model: {e}")
                    return

         
            result = self.predict(sentence)
            if result['predicted_entity']:
                output_text.insert(tk.END, f"✅ Prediction result: 'ta' refers to → 『{result['predicted_entity']}』\n")
                output_text.insert(tk.END, f"📊 Confidence score: {result['confidence']:.4f}")
            else:
                output_text.insert(tk.END, "❌ Could not identify a confident noun reference for 'ta'.")

        def select_model_path():
            selected_dir = filedialog.askdirectory(title="Select Model Directory")
            if selected_dir:
                self.model_dir = selected_dir
                model_path_label.config(text=f"Model path:\n{self.model_dir}")
                self.tokenizer = None
                self.model = None

        tk.Label(master, text="Please enter a Chinese sentence containing 'ta' (case insensitive):").pack(pady=5)

        input_text = tk.Entry(master, width=80)
        input_text.pack(padx=10, pady=5)
        input_text.focus()

        button_frame = tk.Frame(master)
        button_frame.pack(pady=5)

        predict_button = tk.Button(button_frame, text="Start Prediction", command=run_prediction)
        predict_button.pack(side="left", padx=5)

        select_button = tk.Button(button_frame, text="Select Model", command=select_model_path)
        select_button.pack(side="left", padx=5)

        model_path_label = tk.Label(master, text="No model selected", fg="blue", justify="left")
        model_path_label.pack(padx=10, pady=2)

        output_text = scrolledtext.ScrolledText(master, width=80, height=10, wrap=tk.WORD)
        output_text.pack(padx=10, pady=10)
