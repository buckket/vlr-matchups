import asyncio
import copy
import glob
import itertools
import json
import logging
import random
import subprocess

import aiofiles
import aiohttp
import cv2
import numpy as np
import twitchAPI.helper
from PIL import Image
from tesserocr import PyTessBaseAPI
from twitchAPI.twitch import Twitch

import settings

AGENTS = None


class Agent:
    def __init__(self, file_name):
        self.name = file_name.removesuffix(".webp")

        img_agent = cv2.imread("agents/{}".format(file_name), cv2.IMREAD_UNCHANGED)
        tmpl_small = cv2.resize(img_agent, (40, 40), interpolation=cv2.INTER_AREA)

        _, self.mask = cv2.threshold(tmpl_small[:, :, 3], 128, 255, cv2.THRESH_BINARY)
        self.img = cv2.cvtColor(tmpl_small, cv2.COLOR_BGRA2BGR)


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

        self.ocr_success = False
        self.ocr_fails = 0

        self.score_own = 0
        self.score_enemy = 0

        self.team_own = set()
        self.team_enemy = set()

        self.playing_with = set()
        self.playing_against = set()

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
                f = await aiofiles.open("tmp/{}.jpg".format(self.streamer.name), mode="wb")
                await f.write(await resp.read())
                await f.close()
            else:
                logging.error("Error fetching {} ({}): {}".format(self.streamer.name, resp.status, await resp.text()))

    def load_image(self):
        self.img = cv2.imread("tmp/{}.jpg".format(self.streamer.name), cv2.IMREAD_COLOR)
        if self.img is None:
            return False
        self.img = cv2.resize(self.img, (1920, 1080), interpolation=cv2.INTER_AREA)
        return True

    def unload_image(self):
        self.img = None

    def update_agents(self, threshold=0.75):
        if not self.ocr_success:
            return

        y, x = self.img.shape[:2]

        img_own = self.img[0:100, 350:x // 2 - 80]
        img_enemy = cv2.flip(self.img[0:100, x // 2 + 80:x - 350], 1)

        for agent in AGENTS:
            w, h = agent.img.shape[:-1]

            res = cv2.matchTemplate(img_own, agent.img, cv2.TM_CCOEFF_NORMED, None, agent.mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # if max_val >= 0.1:
            #    print("{}: {}".format(agent.name, max_val))

            if max_val >= threshold and max_val != np.inf:
                cv2.rectangle(img_own, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
                self.team_own.add(agent.name)

            res = cv2.matchTemplate(img_enemy, agent.img, cv2.TM_CCOEFF_NORMED, None, agent.mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # if max_val >= 0.1:
            #    print("{}: {}".format(agent.name, max_val))

            if max_val >= threshold and max_val != np.inf:
                # cv2.rectangle(img_enemy, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
                self.team_enemy.add(agent.name)

        if len(self.team_own) > 5 or len(self.team_enemy) > 5:
            logging.warning("More than 5 agents detected, resetting")
            logging.debug(self.team_own)
            logging.debug(self.team_enemy)
            self.reset_team()
            self.update_agents(threshold + 0.05)

    def ocr(self, tess_api):
        text = self.img[30:70, 750:-750]
        text = cv2.resize(text, (text.shape[1] * 2, text.shape[0] * 2))

        text = cv2.GaussianBlur(text, (3, 3), 0)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        text = cv2.filter2D(text, -1, kernel)
        text = cv2.bilateralFilter(text, 9, 20, 20)
        text = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)

        def run_pytesseract(img_tesseract):
            tess_api.SetImage(Image.fromarray(img_tesseract))
            tess_api.Recognize()
            try:
                words = tess_api.MapWordConfidences()
            except RuntimeError:
                return ""

            for x in words:
                if x[1] < 70.0:
                    return ""

            text = "".join(["".join(x[0].split()) for x in words])
            return text

        def binary_threshold(img, threshold=210, retries=0):
            if retries > 5 or threshold > 254:
                return "", 0

            _, text_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

            if np.mean(text_bin) >= 255:
                return "", 0

            kernel = np.ones((3, 3), np.uint8)
            text_bin = cv2.erode(text_bin, kernel, iterations=1)

            tess = run_pytesseract(text_bin)
            if not tess:
                return binary_threshold(img, threshold + 4, retries + 1)

            c_x = 0
            try:
                M = cv2.moments(~text_bin)
                c_x = int(M["m10"] / M["m00"])
            except ZeroDivisionError:
                return "", 0

            return tess, c_x

        img_own = text[:, :300]
        img_enemy = text[:, -300:]

        own, own_x = binary_threshold(img_own, 223)
        if own:
            enemy, enemy_x = binary_threshold(img_enemy, 223)
        else:
            enemy = ""

        ret = self.update_score(own, enemy)
        if ret:
            correction = round(abs(own_x - enemy_x) * 0.1) * 20
            if correction > 20:
                logging.info("Images is stretched,  resizing")
                self.img = cv2.resize(self.img, (1920 - correction, 1080), interpolation=cv2.INTER_AREA)
                self.img = cv2.copyMakeBorder(self.img, 0, 0,
                                              correction // 2,
                                              correction // 2,
                                              cv2.BORDER_CONSTANT,
                                              value=(0, 0, 0))
        if ret and not self.ocr_success:
            self.ocr_success = True
            self.ocr_fails = 0
        if not ret:
            self.ocr_fails += 1
            if self.ocr_success and self.ocr_fails >= 3:
                logging.warning("OCR failed three times in a row, resetting")
                self.ocr_success = False
                self.reset_team()
        return ret

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
    semaphore = asyncio.Semaphore(n)

    async with aiohttp.ClientSession() as session:
        async def fetch(game):
            async with semaphore:
                await game.fetch_image(session)

        await asyncio.gather(*(fetch(x) for x in games))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    game_id = asyncio.run(get_game_id())

    agent_images = glob.glob("*.webp", root_dir="agents")
    AGENTS = [Agent(x) for x in agent_images]

    tess_api = PyTessBaseAPI(oem=0, psm=6)
    tess_api.SetVariable("tessedit_char_whitelist", "0123456789")

    games = []

    while True:
        old_streamer_list = {x.streamer for x in games}
        new_streamer_dict = asyncio.run(get_streamer_dict(game_id, lang="en", limit=100))
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

        asyncio.run(fetch_images_with_concurrency(20, games))

        ocr_failed = 0
        for game in games:
            game.load_image()
            game.ocr(tess_api)
            game.update_agents()
            game.unload_image()
            game.print()
            game.reset_playing()
            if not game.ocr_success:
                ocr_failed += 1

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
