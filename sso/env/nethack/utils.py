from typing import Dict, Union, List
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv


NLE_LANG = NLELanguageObsv()


ACTION_TEMPLATES = [
    "k: move north",
    "l: move east",
    "j: move south",
    "h: move west",
    "y: move northwest",
    "u: move northeast",
    "b: move southwest",
    "n: move southeast",
    ",: pick up at current location",
    "d: drop item",
    "o: open door",
    "t: throw item",
    "e: eat food",
    "w: weild weapon",
    "W: wear armor",
    "T: take off armor",
    "P: put on accessory",
    "R: remove accessory",
    "q: quaff/drink potion",
    "a: apply/use item",
    "z: zap wand"
]


def get_message(obs) -> str:
    message = NLE_LANG.text_message(obs["tty_chars"]).decode("latin-1")
    if message is None:
        message = ""
    return message.replace("\n", "; ")


def get_vision(obs) -> str:
    vision =  NLE_LANG.text_glyphs(obs["glyphs"], obs["blstats"]).decode("latin-1")
    for dir1 in ["east", "west", "north", "south"]:
        for dir2 in ["northwest", "northeast", "southwest", "southeast"]:
            vision = vision.replace(dir1 + dir2, dir2)
    return vision


def get_lang_obs(obs: Dict, as_list: bool = False, use_stats: bool = False) -> Union[str, List[str]]:    
    text_fields = {
        "text_glyphs": get_vision(obs),
        "text_message": get_message(obs),
        "text_blstats": NLE_LANG.text_blstats(obs["blstats"]).decode("latin-1"),
        "text_inventory": NLE_LANG.text_inventory(obs["inv_strs"], obs["inv_letters"]).decode("latin-1"),
        "text_cursor": NLE_LANG.text_cursor(obs["glyphs"], obs["blstats"], obs["tty_cursor"]).decode("latin-1"),
    }

    lang_obs = ["You have " + x[3:] for x in text_fields["text_inventory"].split("\n") if x]
    if use_stats:
        lang_obs += [x for x in text_fields["text_blstats"].split("\n") if x]
        lang_obs += [text_fields["text_cursor"]]
    lang_obs += ["You see a " + x for x in text_fields["text_glyphs"].split("\n") if x]
    if text_fields["text_message"]:
        lang_obs += [text_fields["text_message"]]
    if as_list:
        return lang_obs
    else:
        return ". ".join(lang_obs)
