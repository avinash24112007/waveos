from src.commands.executor import next_tab, prev_tab, refresh, scroll_down, scroll_up,  go_back
from src.commands.executor import next_track, prev_track, pause_play, volume_up, volume_down, screenshot
from src.commands.executor import next_code_tab, prev_code_tab, open_new_terminal, split_editor
CHROME_PROFILE = {
    'SWIPE_RIGHT': next_tab,
    'SWIPE_LEFT': prev_tab,
    'FIST': go_back,
    'OPEN_PALM': scroll_up,
    'PEACE': scroll_down,
}


SPOTIFY_PROFILE = {
    'SWIPE_RIGHT': next_track,
    'SWIPE_LEFT': prev_track,
    'FIST': pause_play,
    'OPEN_PALM': volume_up,
    'PEACE': volume_down,
}

DEFAULT_PROFILE = {
    'OPEN_PALM': volume_up,
    'PEACE': volume_down,
    'FIST': screenshot 
}

VS_CODE_PROFILES = {
    'SWIPE_RIGHT': prev_code_tab,
    'SWIPE_LEFT' : next_code_tab,
    'OPEN_PALM': open_new_terminal,
    'PEACE': split_editor
}
