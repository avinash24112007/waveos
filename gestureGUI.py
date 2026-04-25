import sys
import os

# Get the directory where the exe/script is located
if getattr(sys, 'frozen', False):
    # Running as PyInstaller exe
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as normal Python script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


from src.utils.app_state import AppState, state

def capture_loop(state: AppState):
    from inference import (
    hand_landmarker,
    detect_landmarks,
    draw_hand_landmark_on_frame,
    extract_keypoints,
    predict_motion_pose,
    predict_static_pose,
    execute_command,
    detect_change_gui,
    CHROME_PROFILE,
    SPOTIFY_PROFILE,
    DEFAULT_PROFILE,
    MOTION_ACTIONS,
    STATIC_ACTIONS
)
    from src.utils.landmark_utils import hand_landmarker, detect_landmarks, draw_hand_landmark_on_frame
    from src.utils.keypoint_utils import extract_keypoints
    from src.detection.action_detect import predict_motion_pose
    from src.detection.static_detect import predict_static_pose
    from src.commands.executor import execute_command
    import cv2, time, mediapipe as mp, numpy as np
    from collections import deque
    import datetime

    cap = cv2.VideoCapture(0)
    sequence = deque(maxlen=30)
    prev_kp = np.zeros(126)
    prev_time = time.time()

    while not state.stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(mp.ImageFormat.SRGB, data=rgb_frame)
        ts = int(time.time() * 1000)
        hand_result = detect_landmarks(mp_frame, hand_landmarker, ts)
        prediction_kp = extract_keypoints(hand_result)
        movement = np.linalg.norm(prediction_kp - prev_kp)

        static_label, static_conf = None, 0.0
        motion_label, motion_conf = None, 0.0

        if movement > 0.1:
            sequence.append(prediction_kp)
            # motion_label, motion_conf = predict_motion_pose(sequence, action_model, MOTION_ACTIONS)# tensorflow
            motion_label, motion_conf = predict_motion_pose(sequence, MOTION_ACTIONS) # onnx
        else:
            sequence.append(prediction_kp)
            # static_label, static_conf = predict_static_pose(prediction_kp, static_model, STATIC_ACTIONS) # tensorflow
            static_label, static_conf = predict_static_pose(prediction_kp, STATIC_ACTIONS) # onnx

        prev_kp = prediction_kp
        draw_hand_landmark_on_frame(frame, hand_result)

        current_app = detect_change_gui()
        match current_app:
            case 'edge' | 'chrome':
                execute_command(static_label, CHROME_PROFILE)
                execute_command(motion_label, CHROME_PROFILE)
            case 'spotify':
                execute_command(static_label, SPOTIFY_PROFILE)
                execute_command(motion_label, SPOTIFY_PROFILE)
            case _:
                execute_command(static_label, DEFAULT_PROFILE)
                execute_command(motion_label, DEFAULT_PROFILE)
        if static_label and static_label != 'NEUTRAL':
            state.command_log.appendleft(f"{datetime.datetime.now().strftime('%H:%M:%S')} {current_app} → {static_label}")
        elif motion_label and motion_label != 'NEUTRAL':
            state.command_log.appendleft(f"{datetime.datetime.now().strftime('%H:%M:%S')} {current_app} → {motion_label}")
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        with state.lock:
            state.frame          = frame.copy()
            state.static_label   = static_label or "—"
            state.static_conf    = static_conf or 0.0
            state.motion_label   = motion_label or "—"
            state.motion_conf    = motion_conf or 0.0
            state.active_profile = current_app or "—"
            state.fps            = fps

    cap.release()

import customtkinter as ctk
import cv2
import threading
import time
import numpy as np
from PIL import Image, ImageTk
from collections import deque
# ─── APP THEME ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
 
BG        = "#0a0a0f"
PANEL     = "#12121a"
CARD      = "#1a1a26"
ACCENT    = "#7c3aed"       # violet
ACCENT2   = "#06b6d4"       # cyan
SUCCESS   = "#10b981"
WARNING   = "#f59e0b"
DANGER    = "#ef4444"
TEXT      = "#f1f5f9"
MUTED     = "#64748b"
BORDER    = "#1e1e2e"
 
FONT_TITLE  = ("Courier New", 22, "bold")
FONT_LABEL  = ("Courier New", 11, "bold")
FONT_BODY   = ("Courier New", 10)
FONT_SMALL  = ("Courier New", 9)
FONT_MONO   = ("Courier New", 10)
FONT_BIG    = ("Courier New", 32, "bold")
 
 
class GestureControlApp(ctk.CTk):
    def __init__(self):
        super().__init__()
 
        self.title("GESTURE CONTROL")
        self.geometry("1100x700")
        self.minsize(900, 600)
        self.configure(fg_color=BG)
        self.resizable(True, True)
 
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._update_ui)  # ~30 fps UI refresh
 
    # ── UI BUILDER ────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
 
        self._build_header()
        self._build_feed()
        self._build_sidebar()
        self._build_footer()
 
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=56)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)
 
        # Logo
        ctk.CTkLabel(
            header,
            text="◈  GESTURE CONTROL",
            font=FONT_TITLE,
            text_color=ACCENT,
        ).grid(row=0, column=0, padx=20, pady=12, sticky="w")
 
        # Status pill
        self.status_label = ctk.CTkLabel(
            header,
            text="● OFFLINE",
            font=FONT_LABEL,
            text_color=MUTED,
        )
        self.status_label.grid(row=0, column=1, padx=20, sticky="e")
 
        # Toggle button
        self.toggle_btn = ctk.CTkButton(
            header,
            text="START",
            width=100,
            height=34,
            font=FONT_LABEL,
            fg_color=ACCENT,
            hover_color="#6d28d9",
            corner_radius=6,
            command=self._toggle_detection,
        )
        self.toggle_btn.grid(row=0, column=2, padx=20, pady=10, sticky="e")
 
    def _build_feed(self):
        feed_frame = ctk.CTkFrame(self, fg_color=CARD, corner_radius=12)
        feed_frame.grid(row=1, column=0, padx=(16, 8), pady=8, sticky="nsew")
        feed_frame.grid_rowconfigure(0, weight=0)
        feed_frame.grid_rowconfigure(1, weight=1)
        feed_frame.grid_columnconfigure(0, weight=1)
 
        # Feed header
        feed_header = ctk.CTkFrame(feed_frame, fg_color=PANEL, corner_radius=8, height=36)
        feed_header.grid(row=0, column=0, padx=8, pady=(8, 0), sticky="ew")
        feed_header.grid_propagate(False)
        feed_header.grid_columnconfigure(1, weight=1)
 
        ctk.CTkLabel(feed_header, text="LIVE FEED", font=FONT_LABEL, text_color=ACCENT2).grid(
            row=0, column=0, padx=12, pady=6, sticky="w"
        )
        self.fps_label = ctk.CTkLabel(feed_header, text="FPS: —", font=FONT_SMALL, text_color=MUTED)
        self.fps_label.grid(row=0, column=1, padx=12, pady=6, sticky="e")
 
        # Feed canvas
        self.feed_label = ctk.CTkLabel(feed_frame, text="", fg_color=BG, corner_radius=8)
        self.feed_label.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
 
        # Placeholder
        self.placeholder = ctk.CTkLabel(
            feed_frame,
            text="◈\n\nCAMERA OFFLINE\nPress START to begin",
            font=("Courier New", 14),
            text_color=MUTED,
        )
        self.placeholder.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
 
    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        sidebar.grid(row=1, column=1, padx=(8, 16), pady=8, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(4, weight=1)
 
        # Active Profile Card
        self._card_profile(sidebar, row=0)
 
        # Static Prediction Card
        self._card_prediction(sidebar, "STATIC GESTURE", "static", row=1)
 
        # Motion Prediction Card
        self._card_prediction(sidebar, "MOTION GESTURE", "motion", row=2)
 
        # Last Command
        self._card_last_command(sidebar, row=3)
 
        # Command Log
        self._card_log(sidebar, row=4)
 
    def _card_profile(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        card.grid(row=row, column=0, pady=(0, 8), sticky="ew")
        card.grid_columnconfigure(1, weight=1)
 
        ctk.CTkLabel(card, text="ACTIVE PROFILE", font=FONT_SMALL, text_color=MUTED).grid(
            row=0, column=0, columnspan=2, padx=14, pady=(10, 2), sticky="w"
        )
        self.profile_label = ctk.CTkLabel(
            card, text="—", font=("Courier New", 18, "bold"), text_color=ACCENT2
        )
        self.profile_label.grid(row=1, column=0, columnspan=2, padx=14, pady=(0, 10), sticky="w")
 
    def _card_prediction(self, parent, title, kind, row):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        card.grid(row=row, column=0, pady=(0, 8), sticky="ew")
        card.grid_columnconfigure(0, weight=1)
 
        ctk.CTkLabel(card, text=title, font=FONT_SMALL, text_color=MUTED).grid(
            row=0, column=0, columnspan=2, padx=14, pady=(10, 2), sticky="w"
        )
 
        label = ctk.CTkLabel(
            card, text="—", font=("Courier New", 16, "bold"),
            text_color=ACCENT if kind == "static" else WARNING
        )
        label.grid(row=1, column=0, padx=14, pady=(0, 4), sticky="w")
 
        conf_bar = ctk.CTkProgressBar(card, height=6, corner_radius=3, fg_color=BORDER,
                                       progress_color=ACCENT if kind == "static" else WARNING)
        conf_bar.grid(row=2, column=0, padx=14, pady=(0, 4), sticky="ew")
        conf_bar.set(0)
 
        conf_text = ctk.CTkLabel(card, text="0.00", font=FONT_SMALL, text_color=MUTED)
        conf_text.grid(row=2, column=1, padx=(4, 14), pady=(0, 4), sticky="e")
 
        ctk.CTkLabel(card, text="", height=4).grid(row=3, column=0)
 
        if kind == "static":
            self.static_label_widget = label
            self.static_conf_bar     = conf_bar
            self.static_conf_text    = conf_text
        else:
            self.motion_label_widget = label
            self.motion_conf_bar     = conf_bar
            self.motion_conf_text    = conf_text
 
    def _card_last_command(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        card.grid(row=row, column=0, pady=(0, 8), sticky="ew")
 
        ctk.CTkLabel(card, text="LAST COMMAND", font=FONT_SMALL, text_color=MUTED).grid(
            row=0, column=0, padx=14, pady=(10, 2), sticky="w"
        )
        self.last_cmd_label = ctk.CTkLabel(
            card, text="—", font=("Courier New", 13, "bold"), text_color=SUCCESS
        )
        self.last_cmd_label.grid(row=1, column=0, padx=14, pady=(0, 10), sticky="w")
 
    def _card_log(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        card.grid(row=row, column=0, pady=(0, 0), sticky="nsew")
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)
 
        ctk.CTkLabel(card, text="COMMAND LOG", font=FONT_SMALL, text_color=MUTED).grid(
            row=0, column=0, padx=14, pady=(10, 4), sticky="w"
        )
        self.log_text = ctk.CTkTextbox(
            card,
            font=FONT_MONO,
            fg_color=BG,
            text_color=TEXT,
            corner_radius=6,
            state="disabled",
            wrap="none",
        )
        self.log_text.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="nsew")
 
    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=36)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        footer.grid_propagate(False)
        footer.grid_columnconfigure(1, weight=1)
 
        ctk.CTkLabel(
            footer,
            text="MediaPipe  ·  TensorFlow/Keras  ·  Conv1D + LSTM",
            font=FONT_SMALL,
            text_color=MUTED,
        ).grid(row=0, column=0, padx=16, pady=8, sticky="w")
 
        ctk.CTkLabel(
            footer,
            text="Avinash · KU · 2026",
            font=FONT_SMALL,
            text_color=MUTED,
        ).grid(row=0, column=1, padx=16, pady=8, sticky="e")
 
    # ── TOGGLE ────────────────────────────────────────────────────────────────
    def _toggle_detection(self):
        if not state.is_running:
            state.is_running = True
            state.stop_event.clear()
            state.thread = threading.Thread(target=capture_loop, args=(state,), daemon=True)
            state.thread.start()
 
            self.toggle_btn.configure(text="STOP", fg_color=DANGER, hover_color="#dc2626")
            self.status_label.configure(text="● LIVE", text_color=SUCCESS)
            self.placeholder.grid_remove()
        else:
            state.is_running = False
            state.stop_event.set()
 
            self.toggle_btn.configure(text="START", fg_color=ACCENT, hover_color="#6d28d9")
            self.status_label.configure(text="● OFFLINE", text_color=MUTED)
            self.placeholder.grid()
 
    # ── UI UPDATE LOOP ────────────────────────────────────────────────────────
    def _update_ui(self):
        with state.lock:
            frame          = state.frame
            static_label   = state.static_label
            static_conf    = state.static_conf
            motion_label   = state.motion_label
            motion_conf    = state.motion_conf
            active_profile = state.active_profile
            fps            = state.fps
            log = list(state.command_log)
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for entry in log:
            self.log_text.insert("end", entry + "\n")
        self.log_text.configure(state="disabled")
 
        # Feed
        if frame is not None:
            w = self.feed_label.winfo_width()
            h = self.feed_label.winfo_height()
            if w > 10 and h > 10:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (w, h))
                img = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(img)
                self.feed_label.configure(image=photo, text="")
                self.feed_label.image = photo # type: ignore
 
        # FPS
        self.fps_label.configure(text=f"FPS: {fps:.0f}")
 
        # Profile
        self.profile_label.configure(text=active_profile.upper() if active_profile else "—")
 
        # Static prediction
        self.static_label_widget.configure(text=static_label)
        self.static_conf_bar.set(static_conf)
        self.static_conf_text.configure(text=f"{static_conf:.2f}")
 
        # Motion prediction
        self.motion_label_widget.configure(text=motion_label)
        self.motion_conf_bar.set(motion_conf)
        self.motion_conf_text.configure(text=f"{motion_conf:.2f}")
 
        self.after(33, self._update_ui)
 
    # ── CLOSE ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        state.stop_event.set()
        state.is_running = False
        self.destroy()
 
 
# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = GestureControlApp()
    app.mainloop()