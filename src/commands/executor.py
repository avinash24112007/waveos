import pyautogui, time

from src.utils.gui_utils import get_current_app

def volume_up():
    pyautogui.press('volumeup', presses=3)

def volume_down():
    pyautogui.press('volumedown', presses =3)

# Chrome/ Edge profile
def scroll_up(): pyautogui.scroll(30)

def scroll_down(): pyautogui.scroll(-30)

def refresh():  pyautogui.hotkey('ctrl', 'r')

def go_back(): pyautogui.hotkey('alt', 'left')

def next_tab(): pyautogui.hotkey('ctrl', 'shift', 'tab')

def prev_tab(): pyautogui.hotkey('ctrl', 'tab')


#Spotify Profiles
def next_track(): pyautogui.hotkey('ctrl', 'right')

def prev_track(): pyautogui.hotkey('ctrl', 'left')

def pause_play(): pyautogui.hotkey('space')

def screenshot(): pyautogui.hotkey('win', 'shift', 's')


#VS Code Profiles
def next_code_tab(): pyautogui.hotkey('ctrl', 'pagedown')
def prev_code_tab(): pyautogui.hotkey('ctrl', 'pageup')
def open_new_terminal(): pyautogui.hotkey('ctrl', 'shift', '`')
def split_editor(): pyautogui.hotkey('ctrl', '\\')

last_executed_cmd = 0
COOLDOWN_CMD = 1.0

def execute_command(gesture, profile):
    global last_executed_cmd

    now = time.time()

    if now - last_executed_cmd < COOLDOWN_CMD:
        return

    command = profile.get(gesture)
    if command:
        command()
        last_executed_cmd = now
        

last_executed_GUI = 0.0
COOLDOWN_GUI = 1.0
last_known_app = None
def detect_change_gui():

    global last_executed_GUI, last_known_app

    now = time.time()


    if now - last_executed_GUI < COOLDOWN_GUI:
        return last_known_app
    
    last_known_app = get_current_app()
    last_executed_GUI = now
    return last_known_app

