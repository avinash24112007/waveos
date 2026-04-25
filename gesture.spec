a = Analysis(
    ['app.py'],  # ← changed from gestureGUI.py
    pathex=[],
    binaries=[
        (r'C:\Projects\Copmuter control by Human Gesture\.venv\Lib\site-packages\mediapipe\tasks\c\libmediapipe.dll',
         'mediapipe/tasks/c'),
    ],
    datas=[
        ('assets', 'assets'),
        ('models', 'models'),
        ('src', 'src'),
    ],
    hiddenimports=[
        'mediapipe.tasks.c',
        'mediapipe.tasks.python.core.mediapipe_c_bindings',
        'mediapipe.tasks.python.vision',
    ],
    runtime_hooks=['mediapipe_hook.py'],
    hookspath=[],
    excludes=['tensorflow', 'keras', 'tf2onnx', 'tensorboard'],
    noarchive=False,

)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='GestureControl',
    debug=False,
    console=True,
    onefile=True,
)