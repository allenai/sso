from minihack import MiniHackSkill
from minihack.envs import register

from sso.env.nethack.utils import get_message


class MiniHackLavacross(MiniHackSkill):
    TASK_DESCRIPTION = "find a safe way to cross the lava and navigate to the stairs down. There is an item that can help you behind a locked door."

    def __init__(self, *args, **kwargs):
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.|...L.....|
|+-...L.....|
|.....L.....|
|.....L.....|
|.....L.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$left_bank = selection:fillrect (1,3,5,5)
$right_bank = selection:fillrect (7,1,11,5)
IF [50%] {
    OBJECT:('=',"levitation"),(1,1),blessed
} ELSE {
    OBJECT:('!',"levitation"),(1,1),blessed
}
OBJECT:('(',"skeleton key"),rndcoord($left_bank),blessed,0,name:"The Master Key of Thievery"
DOOR:locked,(1,2)
STAIR:rndcoord($right_bank),down
BRANCH:(1,1,5,5),(1,1,2,2)
"""
        self.picked_up_key = False
        self.unlocked_door = False
        self.used_item = False
        super().__init__(*args, des_file=des_file, max_episode_steps=50, character="rog-hum-cha-mal", **kwargs)

    def reset(self, *args, **kwargs):
        self.picked_up_key = False
        self.unlocked_door = False
        self.used_item = False
        return super().reset(*args, **kwargs)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = 0
        message = get_message(obs)
        if message:
            if not self.picked_up_key and "- a key named The Master Key of Thievery." in message:
                reward = .1
                self.picked_up_key = True
            if not self.unlocked_door and "You succeed in unlocking the door." in message:
                reward = .2
                self.unlocked_door = True
            if not self.used_item and any(x in message for x in [
                "a ring of levitation (on right hand)",
                "You start to float in the air!",
            ]):
                reward = .3
                self.used_item = True
        if "end_status" in info and info["end_status"] == 2:
            reward = .4
        return obs, reward, done, info


class MiniHackLavacross2(MiniHackSkill):
    TASK_DESCRIPTION = "find and navigate to the stairs down. During your search you will need to unlock a door using a key and safely cross lava using a magic ring or boots."

    def __init__(self, *args, **kwargs):
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
---------
|...L...|
|...L...|
|...L...|
|...LLLL|
|...|...|
|...+...|
|...|...|
---------
ENDMAP
REGION:(0,0,8,8),lit,"ordinary"
$left_room = selection:fillrect (1,1,3,7)
$bottom_room = selection:fillrect (5,5,7,7)
$top_room = selection:fillrect (5,1,7,3)
IF [50%] {
    OBJECT:('=',"levitation"),rndcoord($bottom_room),blessed
} ELSE {
    OBJECT:('!',"levitation"),rndcoord($bottom_room),blessed
}
OBJECT:('(',"skeleton key"),rndcoord($left_room),blessed,0,name:"The Master Key of Thievery"
DOOR:locked,(4,6)
STAIR:rndcoord($top_room),down
BRANCH:(1,1,3,7),(0,0,0,0)
"""
        self.picked_up_key = False
        self.unlocked_door = False
        self.used_item = False
        super().__init__(*args, des_file=des_file, max_episode_steps=50, character="rog-hum-cha-mal", **kwargs)

    def reset(self, *args, **kwargs):
        self.picked_up_key = False
        self.unlocked_door = False
        self.used_item = False
        return super().reset(*args, **kwargs)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = 0
        message = get_message(obs)
        if message:
            if not self.picked_up_key and "- a key named The Master Key of Thievery." in message:
                reward = .1
                self.picked_up_key = True
            if not self.unlocked_door and "You succeed in unlocking the door." in message:
                reward = .2
                self.unlocked_door = True
            if not self.used_item and any(x in message for x in [
                "a ring of levitation (on right hand)",
                "You start to float in the air!",
            ]):
                reward = .3
                self.used_item = True
        if "end_status" in info and info["end_status"] == 2:
            reward = .4
        return obs, reward, done, info


register(
    id="MiniHack-KeyLavaCross-v0",
    entry_point="sso.env.nethack.maps:MiniHackLavacross",
)

register(
    id="MiniHack-KeyLavaCross2-v0",
    entry_point="sso.env.nethack.maps:MiniHackLavacross2",
)
