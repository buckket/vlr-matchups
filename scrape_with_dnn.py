import asyncio
import concurrent.futures
import copy
import itertools
import json
import logging
import random
import subprocess
import time
import uuid

import aiohttp
import cv2
import numpy as np
import torch
import twitchAPI.helper
from twitchAPI.twitch import Twitch

import settings


class Streamer:
    def __init__(self, name, view_count, language):
        self.name = name
        self.view_count = view_count
        self.language = language

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Game:

    def __init__(self, streamer):
        self.streamer = streamer
        self.img = None
        self.img_data = None

        self.ocr_success = False
        self.ocr_fails = 0

        self.correction = 0

        self.score_own = 0
        self.score_enemy = 0

        self.team_own = set()
        self.team_enemy = set()

        self.playing_with = set()
        self.playing_against = set()

        self.uuid = ""

    def __hash__(self):
        return hash(self.streamer)

    def __eq__(self, other):
        return self.streamer == other.streamer

    def __lt__(self, other):
        return self.streamer.view_count < other.streamer.view_count

    def __invert__(self):
        self.team_own, self.team_enemy = self.team_enemy, self.team_own
        self.score_own, self.score_enemy = self.score_enemy, self.score_own
        return self

    def to_dict(self):
        json_dict = {
            "streamer": {
                "name": self.streamer.name,
                "view_count": self.streamer.view_count,
                "language": self.streamer.language},
            "playing_with": sorted(list(self.playing_with)),
            "playing_against": sorted(list(self.playing_against))
        }
        return json_dict

    def reset_playing(self):
        self.playing_with = set()
        self.playing_against = set()

    def reset_team(self):
        self.team_own = set()
        self.team_enemy = set()

    async def fetch_image(self, session):
        # Generate random 16:9 resolution to trick Twitch into generating a fresh preview image
        random_x = random.randint(1920, 2048)
        random_y = round(random_x * 9 / 16)

        url = "https://static-cdn.jtvnw.net/previews-ttv/live_user_{}-{}x{}.jpg".format(self.streamer.name,
                                                                                        random_x,
                                                                                        random_y)

        async with session.get(url) as resp:
            if resp.status == 200:
                self.img_data = await resp.read()
            else:
                logging.error("Error fetching {} ({}): {}".format(self.streamer.name, resp.status, await resp.text()))
        return True

    def load_image(self):
        if self.img_data is None:
            return False

        self.img = cv2.imdecode(np.frombuffer(self.img_data, dtype="uint8"), cv2.IMREAD_COLOR)
        if self.img is None:
            return False
        self.img_data = None

        self.img = cv2.resize(self.img, (1920, 1080), interpolation=cv2.INTER_AREA)

        m = np.zeros((640, 640, 3), dtype=np.uint8)
        m[0:200, :, :3] = cv2.vconcat([self.img[0:100, 220:860], self.img[0:100, 1060:-220]])
        self.img = m[:, :, ::-1]

        self.uuid = uuid.uuid1()
        return True

    def unload_image(self):
        self.img = None
        self.img_data = None

    def detect(self, model):
        results = model(self.img)

        cv2.imshow("result", np.squeeze(results.render())[:200, :, ::-1])
        cv2.waitKey(1000)

        df = results.pandas().xyxy[0]

        score_own = ""
        score_own_conf = 0.0

        score_enemy = ""
        score_enemy_conf = 0.0

        for index, row in df.iterrows():
            # print("{} {}".format(row["class"], row["confidence"]))
            if row["class"] > 29:
                if row["name"].endswith("_flipped"):
                    self.team_enemy.add(row["name"][:-8])
                else:
                    self.team_own.add(row["name"])
            else:
                if row["ymin"] >= 100 and row["confidence"] >= score_enemy_conf:
                    score_enemy = str(row["class"])
                    score_enemy_conf = row["confidence"]
                elif row["ymin"] < 100 and row["confidence"] >= score_own_conf:
                    score_own = str(row["class"])
                    score_own_conf = row["confidence"]

        if len(self.team_own) > 5 or len(self.team_enemy) > 5:
            logging.warning("More than 5 agents detected, resetting")
            logging.debug(self.team_own)
            logging.debug(self.team_enemy)
            self.reset_team()

        valid_score = self.update_score(score_own, score_enemy)
        if valid_score and not self.ocr_success:
            self.ocr_success = True
            self.ocr_fails = 0
        if not valid_score:
            self.ocr_fails += 1
            if self.ocr_success and self.ocr_fails >= 3:
                logging.warning("OCR failed three times in a row, resetting")
                self.ocr_success = False
                self.reset_team()
        return valid_score

    def update_score(self, str_own, str_enemy):
        if not (str_own and str_enemy):
            return False
        try:
            own = abs(int(str_own))
            enemy = abs(int(str_enemy))
        except ValueError:
            return False

        if own > 13 or enemy > 13 and abs(own - enemy) >= 2:
            return False

        if own < self.score_own or enemy < self.score_enemy:
            logging.info("Lower score detected, resetting")
            self.reset_team()

        self.score_own = own
        self.score_enemy = enemy
        return True

    def print(self):
        print("{} ({}):\n\tOwn Team: {}\n\t"
              "Enemy team: {}\n\t"
              "Score: {} - {}\n\t"
              "Valid OCR: {}\n".format(self.streamer.name, self.streamer.view_count,
                                       ", ".join(sorted(self.team_own)),
                                       ", ".join(sorted(self.team_enemy)),
                                       self.score_own,
                                       self.score_enemy,
                                       self.ocr_success))

    def is_comparable(self):
        if not self.ocr_success or not self.team_own or not self.team_enemy:
            return False
        return True

    def check_same_team(self, other):
        score = 1.0

        if not (self.ocr_success and other.ocr_success):
            return 0.0

        if abs(self.score_own - other.score_own) > 4:
            return 0.0
        else:
            score *= -(abs(self.score_own - other.score_own)) / 10 + 1

        if abs(self.score_enemy - other.score_enemy) > 4:
            return 0.0
        else:
            score *= -(abs(self.score_enemy - other.score_enemy)) / 10 + 1

        known_agents = min(len(self.team_own), len(other.team_own))
        if len(self.team_own & other.team_own) == known_agents and known_agents > 0:
            score *= (known_agents * 0.2)
        else:
            return 0.0

        known_agents = min(len(self.team_enemy), len(other.team_enemy))
        if len(self.team_enemy & other.team_enemy) == known_agents and known_agents > 0:
            score *= (known_agents * 0.2)
        else:
            return 0.0

        return score


async def get_game_id():
    twitch = await Twitch(settings.TWITCH_APP_ID, settings.TWITCH_APP_SECRET)
    game = await twitchAPI.helper.first(twitch.get_games(names="valorant"))
    return game.id


async def get_streamer_dict(game_id, lang, limit):
    twitch = await Twitch(settings.TWITCH_APP_ID, settings.TWITCH_APP_SECRET)
    # streams_gen = twitch.get_streams(game_id=game_id, language=lang)
    streams_gen = twitch.get_streams(game_id=game_id)

    streams = {}
    async for stream in streams_gen:
        streams[stream.user_login] = Streamer(stream.user_login, stream.viewer_count, stream.language)
        if len(streams) > limit:
            return streams


async def fetch_images_with_concurrency(n, games):
    connector = aiohttp.TCPConnector(limit=n)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def fetch(game):
            await game.fetch_image(session)

        await asyncio.gather(*(fetch(x) for x in games))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    game_id = asyncio.run(get_game_id())

    games = []

    model = torch.hub.load('ultralytics/yolov5', 'custom', 'model/best-v2.onnx')
    # model.conf = 0.50

    while True:
        old_streamer_list = {x.streamer for x in games}
        new_streamer_dict = asyncio.run(get_streamer_dict(game_id, lang="en", limit=100))
        # new_streamer_dict = {x.removesuffix(".jpg"): Streamer(x.removesuffix(".jpg"), 0, "en") for x in
        #                     glob.glob("*.jpg", root_dir="tmp")}

        new_streamer_list = {x for x in new_streamer_dict.values()}

        to_remove = old_streamer_list - new_streamer_list
        print("Removing: " + ", ".join(x.name for x in to_remove))

        to_keep = old_streamer_list & new_streamer_list
        print("Keeping: " + ", ".join(x.name for x in to_keep))

        to_add = new_streamer_list - old_streamer_list
        print("Adding: " + ", ".join(x.name for x in to_add))

        games_new = []
        for game in games:
            if game.streamer in to_keep:
                game.streamer = new_streamer_dict[game.streamer.name]
                games_new.append(game)
        games = games_new
        games.extend([Game(x) for x in to_add])

        asyncio.run(fetch_images_with_concurrency(100, games))


        def load_image(game):
            game.load_image()
            return game


        def work_on_game(game):
            game.detect(model)
            game.unload_image()
            game.print()
            game.reset_playing()
            return game


        ocr_failed = 0

        games_new = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for x in executor.map(load_image, games, chunksize=10):
                if x.img is not None:
                    games_new.append(x)
        games = games_new

        games_new = []
        for game in games:
            work_on_game(game)
            if not game.ocr_success:
                ocr_failed += 1
            games_new.append(game)
        games = games_new

        logging.info("OCR success rate: {:.2f}".format((len(games) - ocr_failed) / len(games)))

        comparable_games = [x for x in games if x.is_comparable()]
        matches_found = set()

        for a, b in itertools.combinations(comparable_games, 2):
            b_copy = copy.deepcopy(b)

            same_team = a.check_same_team(b)
            enemy_team = a.check_same_team(~b_copy)
            if same_team > 0.40 and same_team >= enemy_team:
                print("{} playing with {} (Score: {:.2f})".format(a.streamer.name, b.streamer.name, same_team))
                a.playing_with.add(b.streamer.name)
                b.playing_with.add(a.streamer.name)
                matches_found.add(a)
                matches_found.add(b)
            if enemy_team > 0.40 and enemy_team >= same_team:
                print("{} playing against {} (Score: {:.2f})".format(a.streamer.name, b.streamer.name, enemy_team))
                a.playing_against.add(b.streamer.name)
                b.playing_against.add(a.streamer.name)
                matches_found.add(a)
                matches_found.add(b)

        matches_sorted = sorted(matches_found, reverse=True)
        for match in matches_sorted:
            print("{} ({}):".format(match.streamer.name, match.streamer.view_count))
            if match.playing_with:
                print("\t- playing with: {}".format(", ".join(match.playing_with)))
            if match.playing_against:
                print("\t- playing against: {}".format(", ".join(match.playing_against)))
            print("\n")

        with open("data.json", "w") as outfile:
            outfile.write(json.dumps([x.to_dict() for x in matches_sorted], indent=4))

        subprocess.run(["python", "gen_site.py"])
        time.sleep(60)
