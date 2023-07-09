import datetime
import math
import time
from threading import Thread
from tkinter import *
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
import pygame
import torch
from torch.nn.functional import softmax

from src.features.delta_spetral_cepsrtal_extractor import extract_delta_spectral_cepstral_features

ctk.set_appearance_mode("System")


def convert_seconds_to_string(seconds):
    seconds = math.ceil(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    sec = str(sec) if sec > 9 else f"0{str(sec)}"
    minutes = str(minutes) if minutes > 9 else f"0{str(minutes)}"
    if hours > 0:
        hours = str(hours)
        return f"{hours}:{minutes}:{sec}"
    return f"{minutes}:{sec}"


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.num_rows = 2
        self.num_cols = 2

        # ui elements
        self.file_entry = None
        self.progressbar = None
        self.time_label = None
        self.result_label = None

        # data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feat_len = 750
        self.file_path = None
        self.is_playing = False

        self.configure_window()
        self.create_body()
        self.load_model()
        pygame.mixer.init()

    def configure_window(self):
        self.title("Spoof Voice Detection App")
        self.geometry(f"{580}x{280}")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def create_body(self):
        upload_file_button = ctk.CTkButton(self, text="Upload file", command=self.on_click_upload_btn)
        upload_file_button.grid(row=0, column=2, padx=10, pady=(20, 10))
        file_entry = ctk.CTkEntry(
            self,
            placeholder_text="File name",
            font=ctk.CTkFont(size=14, weight="normal")
        )
        file_entry.grid(row=0, column=0, columnspan=2, padx=(10, 0), pady=(20, 10), sticky="ew")
        self.file_entry = file_entry
        result_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=25, weight="normal")
        )
        result_label.grid(row=1, column=1, padx=(10, 0), pady=10, sticky="ew")
        self.result_label = result_label
        progressbar = ctk.CTkProgressBar(self, progress_color='#32a85a')
        progressbar.set(0)
        progressbar.grid(row=2, column=0, columnspan=3, padx=10, pady=0, sticky="ew")
        self.progressbar = progressbar
        play_btn = ctk.CTkButton(self, text="Play", width=100, command=self.on_click_play_btn)
        play_btn.grid(row=3, column=0, padx=(10, 0), sticky="w")
        time_label = ctk.CTkLabel(self, text="00:00 / 00:00")
        time_label.grid(row=3, column=2, padx=(0, 10), pady=10, sticky="e")
        self.time_label = time_label

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

    def progress(self):
        a = pygame.mixer.Sound(self.file_path)
        song_len = a.get_length()
        start_time = time.time()
        while self.is_playing:
            current_pos = time.time() - start_time
            if current_pos >= song_len:
                break
            time.sleep(0.001)
            self.progressbar.set(current_pos / song_len)
            self.time_label.configure(
                text=f"{convert_seconds_to_string(current_pos)} / {convert_seconds_to_string(song_len)}"
            )
        self.is_playing = False
        self.progressbar.set(0)
        self.time_label.configure(text=f"00:00 / {convert_seconds_to_string(song_len)}")

    def on_click_upload_btn(self):
        file_path = filedialog.askopenfilename(filetypes=(("Audio Files", ".wav .ogg .flac"), ("All Files", "*.*")))
        if file_path == "":
            return
        file_name = file_path.split('/')[-1]
        self.file_path = file_path
        self.file_entry.delete(0, END)
        self.file_entry.insert(0, file_name)
        a = pygame.mixer.Sound(self.file_path)
        song_len = a.get_length()
        self.time_label.configure(text=f"00:00 / {convert_seconds_to_string(song_len)}")
        is_spoof = self.is_spoof_voice(file_path)
        self.result_label.configure(text=f"This is the {'fake' if is_spoof else 'real'} voice!", justify='center')

    def on_click_play_btn(self):
        if self.file_path is None:
            self.on_click_upload_btn()
        if self.is_playing:
            return
        self.is_playing = True
        t = Thread(target=self.progress)
        t.start()
        pygame.mixer.music.load(self.file_path)
        pygame.mixer.music.play(loops=0)


if __name__ == "__main__":
    app = App()
    app.mainloop()
