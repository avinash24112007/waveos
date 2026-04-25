# app.py — inference only, no training code
from gestureGUI import GestureControlApp

if __name__ == "__main__":
    app = GestureControlApp()
    app.mainloop()