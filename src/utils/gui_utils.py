import win32gui, psutil, win32process

def get_current_app():
    try:

        app_name = {
            'msedge': 'edge',
            'chrome': 'chrome',
            'explorer': 'explorer',
            'spotify': 'spotify',
            'code': 'code'

        }
        # 1. Get the active window ID
        hwnd = win32gui.GetForegroundWindow()
        
        # 2. Get the Process ID (PID) tied to that window
        # THIS is where win32process is used!
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        
        # 3. Get the actual .exe name using the PID
        process = psutil.Process(pid)
        exe_name = process.name().lower()  
        
        # 4. Your clean 'if' logic
        for process_name, return_name in app_name.items():
            if process_name in exe_name:
                return return_name
        
        return exe_name 
        
    except Exception:
        return None
    
