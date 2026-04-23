from src.commands.executor import next_tab, prev_tab, refresh, scroll_down, scroll_up,  go_back

CHROME_PROFILE = {
    'SWIPE_RIGHT': next_tab,
    'SWIPE_LEFT': prev_tab,
    'CIRCLE': refresh,
    'FIST': go_back,
    'OPEN_PALM': scroll_up,
    'PEACE': scroll_down,
}