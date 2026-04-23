import pyautogui



def next_tab(): pyautogui.hotkey('ctrl', 'shift', 'tab')

def prev_tab(): pyautogui.hotkey('ctrl', 'tab')

def volume_up():
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)  # type: ignore
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    current_vol = volume.GetMasterVolumeLevelScalar()  # type: ignore
    volume.SetMasterVolumeLevelScalar(min(1.0, current_vol + 0.1), None)  # type: ignore


def scroll_up(): pyautogui.scroll(3)

def scroll_down(): pyautogui.scroll(-3)

def refresh():  pyautogui.hotkey('ctrl', 'r')

def go_back(): pyautogui.hotkey('alt', 'left')
