from tkinter import filedialog, END

import customtkinter as ctk
import numpy as np
import torch
from torch.nn.functional import softmax

from src.features.delta_spetral_cepsrtal_extractor import extract_delta_spectral_cepstral_features

ctk.set_appearance_mode("System")


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.num_rows = 3
        self.num_cols = 3
        self.label = None
        self.file_name_entry = None
        self.configure_window()
        self.create_sidebar()
        self.create_body()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feat_len = 750
        self.load_model()

    def configure_window(self):
        self.title("Spoof Voice Detection App")
        self.geometry(f"{780}x{480}")

        self.grid_columnconfigure(1, weight=1)
        for row in range(self.num_rows):
            if row != 1:
                self.grid_rowconfigure(row, weight=1)

    def create_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=140, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        sidebar.grid_rowconfigure(1, weight=1)
        logo_label = ctk.CTkLabel(
            sidebar,
            text="ASVSpoof",
            font=ctk.CTkFont(size=20, weight="normal")
        )
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        appearance_mode_option = ctk.CTkOptionMenu(
            sidebar,
            values=["Light", "Dark"],
            command=self.on_change_appearance_mode
        )
        appearance_mode_option.grid(row=2, column=0, padx=20, pady=(10, 10))

    def create_body(self):
        upload_file_button = ctk.CTkButton(self, text="Upload file", command=self.on_click_upload_file_btn)
        upload_file_button.grid(row=1, column=2, padx=20, pady=(10, 10))
        label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=30, weight="normal")
        )
        label.grid(row=0, column=1, columnspan=2, padx=20, pady=20)
        self.label = label
        file_name_entry = ctk.CTkEntry(
            self,
            placeholder_text="File name",
            font=ctk.CTkFont(size=14, weight="normal")
        )
        file_name_entry.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        self.file_name_entry = file_name_entry

    @staticmethod
    def on_change_appearance_mode(new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def on_click_upload_file_btn(self):
        file_path = filedialog.askopenfilename(filetypes=(("Audio Files", ".wav .ogg .flac"), ("All Files", "*.*")))
        if file_path == "":
            return
        file_name = file_path.split('/')[-1]
        self.file_name_entry.delete(0, END)
        self.file_name_entry.insert(0, file_name)
        if self.is_spoof_voice(file_path):
            self.label.configure(text="Spoof")
        else:
            self.label.configure(text="Bonafide")

    def load_model(self):
        self.model = torch.load(
            "src/models/anti-spoofing_lfcc_model.pt",
            map_location=torch.device(self.device)
        )

    def is_spoof_voice(self, file_path):
        features = extract_delta_spectral_cepstral_features(file_path)
        features = torch.from_numpy(features)
        this_feat_len = features.shape[1]
        if this_feat_len > self.feat_len:
            start_p = np.random.randint(this_feat_len - self.feat_len)
            features = features[:, start_p:start_p + self.feat_len]
        if this_feat_len < self.feat_len:
            mul = int(np.ceil(self.feat_len / features.shape[1]))
            features = features.repeat(1, mul)[:, :self.feat_len]
        features = features.unsqueeze(0).unsqueeze(0).float().to(self.device)
        feats, lfcc_outputs = self.model(features)
        lfcc_outputs = lfcc_outputs[0]
        scores = softmax(lfcc_outputs, dim=0)
        real_score, fake_score = scores
        return fake_score > real_score


if __name__ == "__main__":
    app = App()
    app.mainloop()
